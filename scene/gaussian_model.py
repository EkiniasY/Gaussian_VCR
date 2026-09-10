#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import torch
import numpy as np
from utils.general_utils import inverse_sigmoid, get_expon_lr_func, build_rotation
from torch import nn
import os
from utils.system_utils import mkdir_p
from plyfile import PlyData, PlyElement
from utils.sh_utils import RGB2SH
from simple_knn._C import distCUDA2, distIndex2
from utils.graphics_utils import BasicPointCloud
from utils.general_utils import strip_symmetric, build_scaling_rotation, pcast_i16_to_f32
from diff_gaussian_rasterization._C import calculate_colours_variance, kmeans_cuda, find_minimum_projected_pixel_size
import numpy as np
import math
from collections import OrderedDict

class Codebook():
    def __init__(self, ids, centers):
        self.ids = ids
        self.centers = centers
    
    def evaluate(self):
        return self.centers[self.ids.flatten().long()]

def generate_codebook(values, inverse_activation_fn=lambda x: x, num_clusters=256, tol=0.0001):
    shape = values.shape
    values = values.flatten().view(-1, 1)
    centers = values[torch.randint(values.shape[0], (num_clusters, 1), device="cuda").squeeze()].view(-1,1)

    ids, centers = kmeans_cuda(values, centers.squeeze(), tol, 500)
    ids = ids.byte().squeeze().view(shape)
    centers = centers.view(-1,1)

    return Codebook(ids.cuda(), inverse_activation_fn(centers.cuda()))

def _kmeans_pytorch(values, num_clusters, max_iter=100, tol=1e-4):
    """PyTorch k-means for arbitrary cluster counts (CUDA k-means limited to 256)."""
    N = values.shape[0]
    values_1d = values.flatten().view(-1, 1)
    indices = torch.randperm(N, device='cuda')[:num_clusters]
    centers = values_1d[indices].clone()

    for _ in range(max_iter):
        dists = torch.cdist(values_1d, centers)
        ids = dists.argmin(dim=1)
        new_centers = torch.zeros_like(centers)
        for c in range(num_clusters):
            mask = (ids == c)
            if mask.sum() > 0:
                new_centers[c] = values_1d[mask].mean()
            else:
                new_centers[c] = centers[c]
        new_centers[torch.isnan(new_centers)] = 0
        shift = (new_centers - centers).abs().sum().item()
        centers = new_centers
        if shift < tol:
            break

    dists = torch.cdist(values_1d, centers)
    ids = dists.argmin(dim=1)
    return Codebook(ids.byte().squeeze().view(values.shape).cuda(), centers.cuda())

class GaussianModel:

    def setup_functions(self):
        def build_covariance_from_scaling_rotation(scaling, scaling_modifier, rotation):
            L = build_scaling_rotation(scaling_modifier * scaling, rotation)
            actual_covariance = L @ L.transpose(1, 2)
            symm = strip_symmetric(actual_covariance)
            return symm
        
        self.scaling_activation = torch.exp
        self.scaling_inverse_activation = torch.log

        self.covariance_activation = build_covariance_from_scaling_rotation

        self.opacity_activation = torch.sigmoid
        self.inverse_opacity_activation = inverse_sigmoid

        self.rotation_activation = torch.nn.functional.normalize


    def __init__(self, sh_degree : int, variable_sh_bands : bool = False):
        self.active_sh_degree = 0
        self.variable_sh_bands = variable_sh_bands
        self.max_sh_degree = sh_degree  
        self._xyz = torch.empty(0)
        self._features_dc = torch.empty(0)
        if variable_sh_bands:
            # List that stores the individual non 0 band SH features
            # For implementation reasons, the first tensor will always have a shape of Nx0x3
            self._features_rest = list(torch.empty(0)) * (self.max_sh_degree + 1)
        else:
            self._features_rest = torch.empty(0)
        self._scaling = torch.empty(0)
        self._rotation = torch.empty(0)
        self._opacity = torch.empty(0)
        self.max_radii2D = torch.empty(0)
        self._degrees = torch.empty(0)
        self.xyz_gradient_accum = torch.empty(0)
        self.denom = torch.empty(0)
        self.optimizer = None
        self.percent_dense = 0
        self.spatial_lr_scale = 0
        self._codebook_dict = None
        self.setup_functions()

    def capture(self):
        return (
            self.active_sh_degree,
            self._xyz,
            self._features_dc,
            self._features_rest,
            self._scaling,
            self._rotation,
            self._opacity,
            self.max_radii2D,
            self.xyz_gradient_accum,
            self.denom,
            self.optimizer.state_dict(),
            self.spatial_lr_scale,
            self._degrees,
        )
    
    def restore(self, model_args, training_args):
        (self.active_sh_degree, 
        self._xyz, 
        self._features_dc, 
        self._features_rest,
        self._scaling, 
        self._rotation, 
        self._opacity,
        self.max_radii2D, 
        xyz_gradient_accum, 
        denom,
        opt_dict, 
        self.spatial_lr_scale,
        self._degrees) = model_args
        self.training_setup(training_args)
        self.xyz_gradient_accum = xyz_gradient_accum
        self.denom = denom
        self.optimizer.load_state_dict(opt_dict)

    @property
    def per_band_count(self):
        result = list()
        if self.variable_sh_bands:
            for tensor in self._features_rest:
                result.append(tensor.shape[0])
        return result

    @property
    def num_primitives(self):
        return self._xyz.shape[0]

    @property
    def get_scaling(self):
        return self.scaling_activation(self._scaling)
    
    @property
    def get_rotation(self):
        return self.rotation_activation(self._rotation)
    
    @property
    def get_xyz(self):
        return self._xyz
    
    @property
    def get_features(self):
        if self.variable_sh_bands:
            features = list()
            index_start = 0
            for idx, sh_tensor in enumerate(self._features_rest):
                index_end = index_start + self.per_band_count[idx]
                features.append(torch.cat((self._features_dc[index_start: index_end], sh_tensor), dim=1))
                index_start = index_end
        else:
            features = torch.cat((self._features_dc, self._features_rest), dim=1)
        return features

    @property
    def get_opacity(self):
        return self.opacity_activation(self._opacity)
    
    def get_covariance(self, scaling_modifier = 1):
        return self.covariance_activation(self.get_scaling, scaling_modifier, self._rotation)

    def oneupSHdegree(self):
        if self.active_sh_degree < self.max_sh_degree:
            self.active_sh_degree += 1
            self._degrees += 1

    def create_from_pcd(self, pcd : BasicPointCloud, spatial_lr_scale : float):
        self.spatial_lr_scale = spatial_lr_scale
        fused_point_cloud = torch.tensor(np.asarray(pcd.points)).float().cuda()
        fused_color = RGB2SH(torch.tensor(np.asarray(pcd.colors)).float().cuda())
        features = torch.zeros((fused_color.shape[0], 3, (self.max_sh_degree + 1) ** 2)).float().cuda()
        features[:, :3, 0] = fused_color

        print("Number of points at initialisation : ", fused_point_cloud.shape[0])

        dist2 = torch.clamp_min(distCUDA2(torch.from_numpy(np.asarray(pcd.points)).float().cuda()), 0.0000001)
        dist2 = torch.sqrt(dist2)
        scales = torch.log(dist2)[...,None].repeat(1, 3)
        rots = torch.zeros((fused_point_cloud.shape[0], 4), device="cuda")
        rots[:, 0] = 1

        opacities = inverse_sigmoid(0.1 * torch.ones((fused_point_cloud.shape[0], 1), dtype=torch.float, device="cuda"))

        self._xyz = nn.Parameter(fused_point_cloud.requires_grad_(True))
        self._features_dc = nn.Parameter(features[:,:,0:1].transpose(1, 2).contiguous().requires_grad_(True))
        self._features_rest = nn.Parameter(features[:,:,1:].transpose(1, 2).contiguous().requires_grad_(True))
        self._scaling = nn.Parameter(scales.requires_grad_(True))
        self._rotation = nn.Parameter(rots.requires_grad_(True))
        self._opacity = nn.Parameter(opacities.requires_grad_(True))
        self.max_radii2D = torch.zeros((self.num_primitives), device="cuda")
        self._degrees = torch.zeros((self.num_primitives, 1), device="cuda", dtype=torch.int32)

    def training_setup(self, training_args):
        self.percent_dense = training_args.percent_dense
        self.xyz_gradient_accum = torch.zeros((self.num_primitives, 1), device="cuda")
        self.denom = torch.zeros((self.num_primitives, 1), device="cuda")

        l = [
            {'params': [self._xyz], 'lr': training_args.position_lr_init * self.spatial_lr_scale, "name": "xyz"},
            {'params': [self._features_dc], 'lr': training_args.feature_lr, "name": "f_dc"},
            {'params': [self._features_rest], 'lr': training_args.feature_lr / 20.0, "name": "f_rest"},
            {'params': [self._opacity], 'lr': training_args.opacity_lr, "name": "opacity"},
            {'params': [self._scaling], 'lr': training_args.scaling_lr, "name": "scaling"},
            {'params': [self._rotation], 'lr': training_args.rotation_lr, "name": "rotation"}
        ]

        self.optimizer = torch.optim.Adam(l, lr=0.0, eps=1e-15)
        self.xyz_scheduler_args = get_expon_lr_func(lr_init=training_args.position_lr_init*self.spatial_lr_scale,
                                                    lr_final=training_args.position_lr_final*self.spatial_lr_scale,
                                                    lr_delay_mult=training_args.position_lr_delay_mult,
                                                    max_steps=training_args.position_lr_max_steps)

    def update_learning_rate(self, iteration):
        ''' Learning rate scheduling per step '''
        for param_group in self.optimizer.param_groups:
            if param_group["name"] == "xyz":
                lr = self.xyz_scheduler_args(iteration)
                param_group['lr'] = lr
                return lr

    def construct_list_of_attributes(self, rest_coeffs=45):
        return ['x', 'y', 'z',
                'f_dc_0','f_dc_1','f_dc_2',
                *[f"f_rest_{i}" for i in range(rest_coeffs)],
                'opacity',
                'scale_0','scale_1','scale_2',
                'rot_0','rot_1','rot_2','rot_3']

    def save_ply(self, path, quantised=False, half_float=False):
        float_type = 'int16' if half_float else 'f4'
        attribute_type = 'u1' if quantised else float_type
        max_sh_coeffs = (self.max_sh_degree + 1) ** 2 - 1
        mkdir_p(os.path.dirname(path))
        elements_list = []

        if quantised:
            # Read codebook dict to extract ids and centers
            if self._codebook_dict is None:
                print("Clustering codebook missing. Returning without saving")
                return

            # Filter to standard keys only (exclude B2's _xyz, etc.)
            standard_keys = ['features_dc'] + [f'features_rest_{i}' for i in range(max_sh_coeffs)] + \
                            ['opacity', 'scaling', 'rotation_re', 'rotation_im']
            cb_dict = OrderedDict((k, v) for k, v in self._codebook_dict.items() if k in standard_keys)

            opacity = cb_dict["opacity"].ids
            scaling = cb_dict["scaling"].ids
            rot = torch.cat((cb_dict["rotation_re"].ids,
                            cb_dict["rotation_im"].ids),
                            dim=1)
            features_dc = cb_dict["features_dc"].ids
            features_rest = torch.stack([cb_dict[f"features_rest_{i}"].ids
                                        for i in range(max_sh_coeffs)
                                        ], dim=1).squeeze()

            dtype_full = [(k, float_type) for k in cb_dict.keys()]
            codebooks = np.empty(256, dtype=dtype_full)

            centers_numpy_list = []
            for v in cb_dict.values():
                c = v.centers.detach().cpu().numpy()
                # Pad to 256 rows if needed (for tiered codebooks)
                if c.shape[0] < 256:
                    pad = np.repeat(c[-1:], 256 - c.shape[0], axis=0)
                    c = np.concatenate([c, pad], axis=0)
                centers_numpy_list.append(c)

            if half_float:
                # No float 16 for plydata, so we just pointer cast everything to int16
                for i in range(len(centers_numpy_list)):
                    centers_numpy_list[i] = np.cast[np.float16](centers_numpy_list[i]).view(dtype=np.int16)
                
            codebooks[:] = list(map(tuple, np.concatenate([ar for ar in centers_numpy_list], axis=1)))
                
        else:
            opacity = self._opacity
            scaling = self._scaling
            rot = self._rotation
            features_dc = self._features_dc
            features_rest = self._features_rest

        for sh_degree in range(self.max_sh_degree + 1):
            coeffs_num = (sh_degree+1)**2 - 1
            degrees_mask = (self._degrees == sh_degree).squeeze()

            #  Position is not quantised
            if half_float:
                xyz = self._xyz[degrees_mask].detach().cpu().half().view(dtype=torch.int16).numpy()
            else:
                xyz = self._xyz[degrees_mask].detach().cpu().numpy()

            f_dc = features_dc[degrees_mask].detach().contiguous().cpu().view(-1,3).numpy()
            # Transpose so that to save rest featrues as rrr ggg bbb instead of rgb rgb rgb
            if self.variable_sh_bands:
                f_rest = features_rest[sh_degree][:, :coeffs_num].detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
            else:
                f_rest = features_rest[degrees_mask][:, :coeffs_num].detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
            opacities = opacity[degrees_mask].detach().cpu().numpy()
            scale = scaling[degrees_mask].detach().cpu().numpy()
            rotation = rot[degrees_mask].detach().cpu().numpy()

            dtype_full = [(attribute, float_type) 
                          if attribute in ['x', 'y', 'z'] else (attribute, attribute_type) 
                          for attribute in self.construct_list_of_attributes(coeffs_num * 3)]
            elements = np.empty(degrees_mask.sum(), dtype=dtype_full)

            attributes = np.concatenate((xyz, f_dc, f_rest, opacities, scale, rotation), axis=1)
            elements[:] = list(map(tuple, attributes))
            elements_list.append(PlyElement.describe(elements, f'vertex_{sh_degree}'))
        if quantised:
            elements_list.append(PlyElement.describe(codebooks, f'codebook_centers'))
        PlyData(elements_list).write(path)

    def reset_opacity(self):
        opacities_new = inverse_sigmoid(torch.min(self.get_opacity, torch.ones_like(self.get_opacity)*0.01))
        optimizable_tensors = self.replace_tensor_to_optimizer(opacities_new, "opacity")
        self._opacity = optimizable_tensors["opacity"]

    def _parse_vertex_group(self,
                            vertex_group,
                            sh_degree,
                            float_type,
                            attribute_type,
                            max_coeffs_num,
                            quantised,
                            half_precision,                            
                            codebook_centers_torch=None):
        coeffs_num = (sh_degree+1)**2 - 1
        num_primitives = vertex_group.count

        xyz = np.stack((np.asarray(vertex_group["x"], dtype=float_type),
                        np.asarray(vertex_group["y"], dtype=float_type),
                        np.asarray(vertex_group["z"], dtype=float_type)), axis=1)

        opacity = np.asarray(vertex_group["opacity"], dtype=attribute_type)[..., np.newaxis]
    
        # Stacks the separate components of a vector attribute into a joint numpy array
        # Defined just to avoid visual clutter
        def stack_vector_attribute(name, count):
            return np.stack([np.asarray(vertex_group[f"{name}_{i}"], dtype=attribute_type)
                            for i in range(count)], axis=1)

        features_dc = stack_vector_attribute("f_dc", 3).reshape(-1, 1, 3)
        scaling = stack_vector_attribute("scale", 3)
        rotation = stack_vector_attribute("rot", 4)
        
        # Take care of error when trying to stack 0 arrays
        if sh_degree > 0:
            features_rest = stack_vector_attribute("f_rest", coeffs_num*3).reshape((num_primitives, 3, coeffs_num))
        else:
            features_rest = np.empty((num_primitives, 3, 0), dtype=attribute_type)

        if not self.variable_sh_bands:
            # Using full tensors (P x 15) even for points that don't require it
            features_rest = np.concatenate(
                (features_rest,
                    np.zeros((num_primitives, 3, max_coeffs_num - coeffs_num), dtype=attribute_type)), axis=2)

        degrees = np.ones(num_primitives, dtype=np.int32)[..., np.newaxis] * sh_degree

        xyz = torch.from_numpy(xyz).cuda()
        if half_precision:
            xyz = pcast_i16_to_f32(xyz)
        features_dc = torch.from_numpy(features_dc).contiguous().cuda()
        features_rest = torch.from_numpy(features_rest).contiguous().cuda()
        opacity = torch.from_numpy(opacity).cuda()
        scaling = torch.from_numpy(scaling).cuda()
        rotation = torch.from_numpy(rotation).cuda()
        degrees = torch.from_numpy(degrees).cuda()

        # If quantisation has been used, it is needed to index the centers
        if quantised:
            features_dc = codebook_centers_torch['features_dc'][features_dc.view(-1).long()].view(-1, 1, 3)

            # This is needed as we might have padded the features_rest tensor with zeros before
            reshape_channels = coeffs_num if self.variable_sh_bands else max_coeffs_num            
            # The gather operation indexes a 256x15 tensor with a (P*3)features_rest index tensor,
            # in a column-wise fashion
            # Basically this is equivalent to indexing a single codebook with a P*3 index
            # features_rest times inside a loop
            features_rest = codebook_centers_torch['features_rest'].gather(0, features_rest.view(num_primitives*3, reshape_channels).long()).view(num_primitives, 3, reshape_channels)
            opacity = codebook_centers_torch['opacity'][opacity.long()]
            scaling = codebook_centers_torch['scaling'][scaling.view(num_primitives*3).long()].view(num_primitives, 3)
            # Index the real and imaginary part separately
            rotation = torch.cat((
                codebook_centers_torch['rotation_re'][rotation[:, 0:1].long()],
                codebook_centers_torch['rotation_im'][rotation[:, 1:].reshape(num_primitives*3).long()].view(num_primitives,3)
                ), dim=1)

        return {'xyz': xyz,
                'opacity': opacity,
                'features_dc': features_dc,
                'features_rest': features_rest,
                'scaling': scaling,
                'rotation': rotation,
                'degrees': degrees
        }

    def load_ply(self, path, half_float=False, quantised=False):
        plydata = PlyData.read(path)

        xyz_list = []
        features_dc_list = []
        features_rest_list = []
        opacity_list = []
        scaling_list = []
        rotation_list = []
        degrees_list = []

        float_type = 'int16' if half_float else 'f4'
        attribute_type = 'u1' if quantised else float_type
        max_coeffs_num = (self.max_sh_degree+1)**2 - 1

        codebook_centers_torch = None
        if quantised:
            # Parse the codebooks.
            # The layout is 256 x 20, where 256 is the number of centers and 20 number of codebooks
            # In the future we could have different number of centers
            codebook_centers = plydata.elements[-1]

            codebook_centers_torch = OrderedDict()
            codebook_centers_torch['features_dc'] = torch.from_numpy(np.asarray(codebook_centers['features_dc'], dtype=float_type)).cuda()
            codebook_centers_torch['features_rest'] = torch.from_numpy(np.concatenate(
                [
                    np.asarray(codebook_centers[f'features_rest_{i}'], dtype=float_type)[..., np.newaxis]
                    for i in range(max_coeffs_num)
                ], axis=1)).cuda()
            codebook_centers_torch['opacity'] = torch.from_numpy(np.asarray(codebook_centers['opacity'], dtype=float_type)).cuda()
            codebook_centers_torch['scaling'] = torch.from_numpy(np.asarray(codebook_centers['scaling'], dtype=float_type)).cuda()
            codebook_centers_torch['rotation_re'] = torch.from_numpy(np.asarray(codebook_centers['rotation_re'], dtype=float_type)).cuda()
            codebook_centers_torch['rotation_im'] = torch.from_numpy(np.asarray(codebook_centers['rotation_im'], dtype=float_type)).cuda()

            # If use half precision then we have to pointer cast the int16 to float16
            # and then cast them to floats, as that's the format that our renderer accepts
            if half_float:
                for k, v in codebook_centers_torch.items():
                    codebook_centers_torch[k] = pcast_i16_to_f32(v)

        # Iterate over the point clouds that are stored on top level of plyfile
        # to get the various fields values 
        for sh_degree in range(0, self.max_sh_degree+1):
            attribues_dict = self._parse_vertex_group(plydata.elements[sh_degree],
                                                      sh_degree,
                                                      float_type,
                                                      attribute_type,
                                                      max_coeffs_num,
                                                      quantised,
                                                      half_float,
                                                      codebook_centers_torch)

            xyz_list.append(attribues_dict['xyz'])
            features_dc_list.append(attribues_dict['features_dc'])
            features_rest_list.append(attribues_dict['features_rest'].transpose(1,2))
            opacity_list.append(attribues_dict['opacity'])
            scaling_list.append(attribues_dict['scaling'])
            rotation_list.append(attribues_dict['rotation'])
            degrees_list.append(attribues_dict['degrees'])

        # Concatenate the tensors into one, to be used in our program
        # TODO: allow multiple PCDs to be rendered/optimise and skip this step
        xyz = torch.cat((xyz_list), dim=0)
        features_dc = torch.cat((features_dc_list), dim=0)
        if not self.variable_sh_bands:
            features_rest = torch.cat((features_rest_list), dim=0)
        else:
            features_rest = features_rest_list
        opacity = torch.cat((opacity_list), dim=0)
        scaling = torch.cat((scaling_list), dim=0)
        rotation = torch.cat((rotation_list), dim=0)
        
        self._xyz = nn.Parameter(xyz.requires_grad_(True))
        self._features_dc = nn.Parameter(features_dc.requires_grad_(True))
        if not self.variable_sh_bands:
            self._features_rest = nn.Parameter(features_rest.requires_grad_(True))
        else:
            for tensor in features_rest_list:
                tensor.requires_grad_(True)
            self._features_rest = features_rest_list
        self._opacity = nn.Parameter(opacity.requires_grad_(True))
        self._scaling = nn.Parameter(scaling.requires_grad_(True))
        self._rotation = nn.Parameter(rotation.requires_grad_(True))
        self._degrees = torch.cat((degrees_list), dim=0)

        self.active_sh_degree = self.max_sh_degree

    # Replaces a parameter group's tensor defined by name with a given tensor
    # and zeroes-out the internal optimiser state
    def replace_tensor_to_optimizer(self, tensor, name):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            if group["name"] == name:
                stored_state = self.optimizer.state.get(group['params'][0], None)
                stored_state["exp_avg"] = torch.zeros_like(tensor)
                stored_state["exp_avg_sq"] = torch.zeros_like(tensor)

                del self.optimizer.state[group['params'][0]]
                group["params"][0] = nn.Parameter(tensor.requires_grad_(True))
                self.optimizer.state[group['params'][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
        return optimizable_tensors

    def _prune_optimizer(self, mask, store_grads=False):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            stored_state = self.optimizer.state.get(group['params'][0], None)
            if stored_state is not None:
                stored_state["exp_avg"] = stored_state["exp_avg"][mask]
                stored_state["exp_avg_sq"] = stored_state["exp_avg_sq"][mask]

                del self.optimizer.state[group['params'][0]]
                if store_grads:
                    grad = group["params"][0].grad[mask]
                group["params"][0] = nn.Parameter((group["params"][0][mask].requires_grad_(True)))
                if store_grads:
                    group["params"][0].grad = grad
                self.optimizer.state[group['params'][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
            else:
                group["params"][0] = nn.Parameter(group["params"][0][mask].requires_grad_(True))
                optimizable_tensors[group["name"]] = group["params"][0]
        return optimizable_tensors

    def _compute_ecr_and_view_stability(self, cameras):
        """Compute ECR and view stability for each Gaussian across all cameras.
        Returns: ecr (N,1), view_stability (N,1)"""
        xyz = self._xyz.detach()
        N = xyz.shape[0]
        opacity = self.get_opacity.detach()

        # Per-camera pixel contribution estimation
        pixel_counts = []
        for cam in cameras:
            w2ndc = cam.full_proj_transform  # (4,4)
            # Project to NDC
            ones = torch.ones((N, 1), device='cuda')
            xyz_h = torch.cat([xyz, ones], dim=1)  # (N,4)
            ndc = (xyz_h @ w2ndc)  # (N,4)
            ndc = ndc[:, :3] / ndc[:, 3:4]  # (N,3)

            # Frustum culling
            in_frustum = (
                (ndc[:, 0].abs() <= 1) & (ndc[:, 1].abs() <= 1) &
                (ndc[:, 2] >= 0) & (ndc[:, 2] <= 1)
            )

            # Estimate 2D pixel extent from 3D scaling
            scales = self.get_scaling.detach()  # (N,3)
            max_scale = scales.max(dim=1).values  # (N,)
            # Pixel coverage ~ (max_scale * focal_factor)^2
            focal_factor = min(cam.image_width, cam.image_height) * 0.5
            pix = (max_scale * focal_factor) ** 2
            pix[~in_frustum] = 0
            pix = pix * opacity.squeeze()
            pixel_counts.append(pix)

        # (N, num_cameras)
        pixel_counts = torch.stack(pixel_counts, dim=1)

        # ECR: ratio of cameras where Gaussian contributes
        visible_count = (pixel_counts > 0).float().sum(dim=1, keepdim=True)
        total_cameras = len(cameras)
        ecr = visible_count / total_cameras

        # View stability: CoV of per-camera pixel counts (lower = more stable)
        mean_pixels = pixel_counts.mean(dim=1, keepdim=True)
        std_pixels = pixel_counts.std(dim=1, keepdim=True)
        view_stability = torch.zeros_like(mean_pixels)
        mask = mean_pixels.squeeze() > 0
        view_stability[mask] = std_pixels[mask] / mean_pixels[mask]

        return ecr, view_stability

    def mercy_points(self, densification_statistics_dict, lambda_mercy=2, mercy_minimum=2, mercy_type='redundancy_opacity'):
        mean = self._splatted_num_accum.squeeze().float().mean(dim=0, keepdim=True)
        std =  self._splatted_num_accum.squeeze().float().var(dim=0, keepdim=True).sqrt()

        threshold = max((mean + lambda_mercy*std).item(), mercy_minimum)

        mask = (self._splatted_num_accum > threshold).squeeze()

        if mercy_type == 'redundancy_opacity':
            # Prune redundant points based on 50% lowest opacity
            mask[mask.clone()] = self.get_opacity[mask].squeeze() < self.get_opacity[mask].median()
        elif mercy_type == 'redundancy_random':
            # Prune 50% redundant points at random
            mask[mask.clone()] = torch.rand(mask[mask].shape, device="cuda").squeeze() < 0.5
        elif mercy_type == 'opacity':
            # Prune based just on opacity
            threshold = self.get_opacity.quantile(0.045)
            mask = (self.get_opacity < threshold).squeeze()
        elif mercy_type == 'redundancy_opacity_opacity':
            # Prune based on opacity and on redundancy + opacity (options 1 and 3)
            mask[mask.clone()] = self.get_opacity[mask].squeeze() < self.get_opacity[mask].median()
            threshold = torch.min(self.get_opacity.quantile(0.03), torch.tensor([0.05], device="cuda"))
            mask = torch.logical_or(mask, (self.get_opacity < threshold).squeeze())
        elif mercy_type in ['ecr_only', 'view_stability_only', 'ecr_view_stability']:
            # B1: View-aware pruning using ECR and view stability
            cameras = getattr(self, '_cameras', None)
            if cameras is None:
                print(f"Warning: mercy_type={mercy_type} requires cameras. Skipping prune.")
                return
            ecr, view_stability = self._compute_ecr_and_view_stability(cameras)
            theta_ecr = 0.3    # VCR threshold: below this = occluded
            theta_vs = 0.3     # CVF threshold: above this = stable-low-contribution
            if mercy_type == 'ecr_only':
                mask = (ecr < theta_ecr).squeeze()
            elif mercy_type == 'view_stability_only':
                mask = (view_stability > theta_vs).squeeze()
            else:  # ecr_view_stability
                mask = ((ecr < theta_ecr) & (view_stability > theta_vs)).squeeze()

        self.prune_points(mask)
        densification_statistics_dict["n_points_mercied"] = mask.sum()
        densification_statistics_dict["redundancy_threshold"] = mean + lambda_mercy*std
        densification_statistics_dict["opacity_threshold"] = threshold if mercy_type in ['redundancy_opacity_opacity', 'opacity'] else 0

    def prune_points(self, mask, store_grads=False):
        valid_points_mask = ~mask
        optimizable_tensors = self._prune_optimizer(valid_points_mask, store_grads=store_grads)

        self._xyz = optimizable_tensors["xyz"]
        self._features_dc = optimizable_tensors["f_dc"]
        self._features_rest = optimizable_tensors["f_rest"]
        self._opacity = optimizable_tensors["opacity"]
        self._scaling = optimizable_tensors["scaling"]
        self._rotation = optimizable_tensors["rotation"]
        self._degrees = self._degrees[valid_points_mask]

        self.xyz_gradient_accum = self.xyz_gradient_accum[valid_points_mask]

        self.denom = self.denom[valid_points_mask]
        self.max_radii2D = self.max_radii2D[valid_points_mask]

    def cat_tensors_to_optimizer(self, tensors_dict, store_grads=False):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            assert len(group["params"]) == 1
            extension_tensor = tensors_dict[group["name"]]
            stored_state = self.optimizer.state.get(group['params'][0], None)
            if stored_state is not None:

                stored_state["exp_avg"] = torch.cat((stored_state["exp_avg"], torch.zeros_like(extension_tensor)), dim=0)
                stored_state["exp_avg_sq"] = torch.cat((stored_state["exp_avg_sq"], torch.zeros_like(extension_tensor)), dim=0)

                del self.optimizer.state[group['params'][0]]
                if store_grads:
                    grad = torch.cat((group["params"][0].grad, torch.zeros_like(extension_tensor)), dim=0)
                group["params"][0] = nn.Parameter(torch.cat((group["params"][0], extension_tensor), dim=0).requires_grad_(True))
                if store_grads:
                    group["params"][0].grad = grad
                self.optimizer.state[group['params'][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
            else:
                if store_grads:
                    grad = torch.cat((group["params"][0].grad, torch.zeros_like(extension_tensor)), dim=0)
                group["params"][0] = nn.Parameter(torch.cat((group["params"][0], extension_tensor), dim=0).requires_grad_(True))
                if store_grads:
                    group["params"][0].grad = grad
                optimizable_tensors[group["name"]] = group["params"][0]

        return optimizable_tensors

    def densification_postfix(self, new_xyz, new_features_dc, new_features_rest, new_opacities, new_scaling, new_rotation, new_degrees, store_grads=False):
        d = {"xyz": new_xyz,
        "f_dc": new_features_dc,
        "f_rest": new_features_rest,
        "opacity": new_opacities,
        "scaling" : new_scaling,
        "rotation" : new_rotation}

        optimizable_tensors = self.cat_tensors_to_optimizer(d, store_grads=store_grads)
        self._xyz = optimizable_tensors["xyz"]
        self._features_dc = optimizable_tensors["f_dc"]
        self._features_rest = optimizable_tensors["f_rest"]
        self._opacity = optimizable_tensors["opacity"]
        self._scaling = optimizable_tensors["scaling"]
        self._rotation = optimizable_tensors["rotation"]
        self._degrees = torch.cat((self._degrees, new_degrees), dim=0)

        self.xyz_gradient_accum = torch.zeros((self.num_primitives, 1), device="cuda")
        self.density_gradient_accum = torch.zeros((self.num_primitives, 1), device="cuda")
        self.denom = torch.zeros((self.num_primitives, 1), device="cuda")
        self.max_radii2D = torch.zeros((self.num_primitives), device="cuda")

    def densify_and_split(self, grads, grad_threshold, scene_extent, N=2, store_grads=False):
        # The pads are used to ignore the densification that happened before (densify_and_clone)
        n_init_points = self.num_primitives
        # Extract points that satisfy the gradient condition
        padded_grad = torch.zeros((n_init_points), device="cuda")
        padded_grad[:grads.shape[0]] = grads.squeeze()
        selected_pts_mask = padded_grad >= grad_threshold
        selected_pts_mask = torch.logical_and(selected_pts_mask,
                                              torch.max(self.get_scaling, dim=1).values > self.percent_dense*scene_extent)
        n_points_split = selected_pts_mask.sum().item()

        stds = self.get_scaling[selected_pts_mask].repeat(N,1)
        means = torch.zeros((stds.size(0), 3), device="cuda")
        samples = torch.normal(mean=means, std=stds)
        rots = build_rotation(self._rotation[selected_pts_mask]).repeat(N,1,1)
        new_xyz = torch.bmm(rots, samples.unsqueeze(-1)).squeeze(-1) + self.get_xyz[selected_pts_mask].repeat(N, 1)
        new_scaling = self.scaling_inverse_activation(self.get_scaling[selected_pts_mask].repeat(N,1) / (0.8*N))
        new_rotation = self._rotation[selected_pts_mask].repeat(N,1)
        new_features_dc = self._features_dc[selected_pts_mask].repeat(N,1,1)
        new_features_rest = self._features_rest[selected_pts_mask].repeat(N,1,1)
        new_opacity = self._opacity[selected_pts_mask].repeat(N,1)
        new_degrees = self._degrees[selected_pts_mask].repeat(N,1)

        self.densification_postfix(new_xyz, new_features_dc, new_features_rest, new_opacity, new_scaling, new_rotation, new_degrees, store_grads=store_grads)

        prune_filter = torch.cat((selected_pts_mask, torch.zeros(N * selected_pts_mask.sum(), device="cuda", dtype=bool)))
        self.prune_points(prune_filter, store_grads)
        return n_points_split

    def densify_and_clone(self, grads, grad_threshold, scene_extent, store_grads=False):
        # Extract points that satisfy the gradient condition
        selected_pts_mask = grads.squeeze() >= grad_threshold
        selected_pts_mask = torch.logical_and(selected_pts_mask,
                                              torch.max(self.get_scaling, dim=1).values <= self.percent_dense*scene_extent)
        
        n_points_cloned = selected_pts_mask.sum().item()
        
        new_xyz = self._xyz[selected_pts_mask]
        new_features_dc = self._features_dc[selected_pts_mask]
        new_features_rest = self._features_rest[selected_pts_mask]
        new_opacities = self._opacity[selected_pts_mask]
        new_scaling = self._scaling[selected_pts_mask]
        new_rotation = self._rotation[selected_pts_mask]
        new_degrees = self._degrees[selected_pts_mask]

        self.densification_postfix(new_xyz, new_features_dc, new_features_rest, new_opacities, new_scaling, new_rotation, new_degrees, store_grads=store_grads)
        return n_points_cloned

    def densify_and_prune(self, max_grad, min_opacity, extent, max_screen_size, densification_statistics_dict, store_grads=False):
        grads = self.xyz_gradient_accum / self.denom
        grads[grads.isnan()] = 0.0

        n_points_cloned = self.densify_and_clone(grads, max_grad, extent, store_grads=store_grads)
        n_points_split = self.densify_and_split(grads, max_grad, extent, store_grads=store_grads)

        self.prune(min_opacity, extent, max_screen_size, densification_statistics_dict, store_grads=store_grads)

        torch.cuda.empty_cache()

        densification_statistics_dict["n_points_cloned"] = n_points_cloned
        densification_statistics_dict["n_points_split"] = n_points_split

    def prune(self, min_opacity, extent, max_screen_size, densification_statistics_dict, store_grads=False):
        prune_mask = (self.get_opacity < min_opacity).squeeze()
        if max_screen_size:
            big_points_vs = self.max_radii2D > max_screen_size
            big_points_ws = self.get_scaling.max(dim=1).values > 0.1 * extent
            prune_mask = torch.logical_or(torch.logical_or(prune_mask, big_points_vs), big_points_ws)
        densification_statistics_dict["n_points_pruned"] = prune_mask.sum()
        self.prune_points(prune_mask, store_grads=store_grads)

    def add_densification_stats(self, viewspace_point_tensor, update_filter):
        self.xyz_gradient_accum += torch.norm(viewspace_point_tensor.grad[:, :2], dim=-1, keepdim=True)
        self.denom += update_filter.unsqueeze(1)

    def _low_variance_colour_culling(self, threshold, weighted_variance, weighted_mean):
        original_degrees = torch.zeros_like(self._degrees)
        original_degrees.copy_(self._degrees)

        # Uniform colour culling
        weighted_colour_std = weighted_variance.sqrt()
        weighted_colour_std[weighted_colour_std.isnan()] = 0
        weighted_colour_std = weighted_colour_std.mean(dim=2).squeeze()

        std_mask = weighted_colour_std < threshold
        self._features_dc[std_mask] = (weighted_mean[std_mask] - 0.5) / 0.28209479177387814
        self._degrees[std_mask] = 0
        self._features_rest[std_mask] = 0

    def _low_distance_colour_culling(self, threshold, colour_distances):
        colour_distances[colour_distances.isnan()] = 0

        # Loop from active_sh_degree - 1 to 0, since the comparisons
        # are always done based on the max band that corresponds to active_sh_degree
        for sh_degree in range(self.active_sh_degree - 1, 0, -1):
            coeffs_num = (sh_degree+1)**2 - 1
            mask = colour_distances[:, sh_degree] < threshold
            self._degrees[mask] = torch.min(
                    torch.tensor([sh_degree], device="cuda", dtype=int),
                    self._degrees[mask]
                ).int()
            
            # Zero-out the associated SH coefficients for clarity,
            # as they won't be used in rasterisation due to the degrees field
            self._features_rest[mask, coeffs_num:] = 0

    def cull_sh_bands(self, cameras, threshold=0*np.sqrt(3)/255, std_threshold=0.):
        camera_positions = torch.stack([cam.camera_center for cam in cameras], dim=0)
        camera_viewmatrices = torch.stack([cam.world_view_transform for cam in cameras], dim=0)
        camera_projmatrices = torch.stack([cam.full_proj_transform for cam in cameras], dim=0)
        camera_fovx = torch.tensor([camera.FoVx for camera in cameras], device="cuda", dtype=torch.float32)
        camera_fovy = torch.tensor([camera.FoVy for camera in cameras], device="cuda", dtype=torch.float32)
        image_height = torch.tensor([camera.image_height for camera in cameras], device="cuda", dtype=torch.int32)
        image_width = torch.tensor([camera.image_width for camera in cameras], device="cuda", dtype=torch.int32)

        # Wrapping in a function since it's called with the same parameters twice
        def run_calculate_colours_variance():
            return calculate_colours_variance(
                camera_positions,
                self.get_xyz,
                self._opacity,
                self.get_scaling,
                self.get_rotation,
                camera_viewmatrices,
                camera_projmatrices,
                torch.tan(camera_fovx*0.5),
                torch.tan(camera_fovy*0.5),
                image_height,
                image_width,
                self.get_features,
                self._degrees,
                self.active_sh_degree)
        
        _, weighted_variance, weighted_mean = run_calculate_colours_variance()
        self._low_variance_colour_culling(std_threshold, weighted_variance, weighted_mean)

        # Recalculate to account for the changed values
        colour_distances, _, _ = run_calculate_colours_variance()
        self._low_distance_colour_culling(threshold, colour_distances)

    def produce_clusters(self, num_clusters=256, store_dict_path=None):
        max_coeffs_num = (self.max_sh_degree + 1)**2 - 1
        codebook_dict = OrderedDict({})

        codebook_dict["features_dc"] = generate_codebook(self._features_dc.detach()[:, 0],
                                                         num_clusters=num_clusters, tol=0.001)
        for sh_degree in range(max_coeffs_num):
                codebook_dict[f"features_rest_{sh_degree}"] = generate_codebook(
                    self._features_rest.detach()[:, sh_degree], num_clusters=num_clusters)

        codebook_dict["opacity"] = generate_codebook(self.get_opacity.detach(),
                                                     self.inverse_opacity_activation, num_clusters=num_clusters)
        codebook_dict["scaling"] = generate_codebook(self.get_scaling.detach(),
                                                     self.scaling_inverse_activation, num_clusters=num_clusters)
        codebook_dict["rotation_re"] = generate_codebook(self.get_rotation.detach()[:, 0:1],
                                                         num_clusters=num_clusters)
        codebook_dict["rotation_im"] = generate_codebook(self.get_rotation.detach()[:, 1:],
                                                         num_clusters=num_clusters)
        if store_dict_path is not None:
            torch.save(codebook_dict, os.path.join(store_dict_path, 'codebook.pt'))
        
        self._codebook_dict = codebook_dict

    def apply_clustering(self, codebook_dict=None):
        max_coeffs_num = (self.max_sh_degree + 1)**2 - 1
        if codebook_dict is None:
            return

        opacity = codebook_dict["opacity"].evaluate().requires_grad_(True)
        scaling = codebook_dict["scaling"].evaluate().view(-1, 3).requires_grad_(True)
        rotation = torch.cat((codebook_dict["rotation_re"].evaluate(),
                            codebook_dict["rotation_im"].evaluate().view(-1, 3)),
                            dim=1).squeeze().requires_grad_(True)
        features_dc = codebook_dict["features_dc"].evaluate().view(-1, 1, 3).requires_grad_(True)
        features_rest = []
        for sh_degree in range(max_coeffs_num):
            features_rest.append(codebook_dict[f"features_rest_{sh_degree}"].evaluate().view(-1, 3))

        features_rest = torch.stack([*features_rest], dim=1).squeeze().requires_grad_(True)

        with torch.no_grad():
            self._opacity = opacity
            self._scaling = scaling
            self._rotation = rotation
            self._features_dc = features_dc
            self._features_rest = features_rest

    # ─── B2: Cross-attribute prediction residual quantization ──────────────

    def _build_knn_graph(self, xyz_quantized, k=8):
        """Build KNN graph from quantized positions. Returns (N, k) indices."""
        _, indices = distIndex2(xyz_quantized.float(), k)
        return indices.view(-1, k)

    def _knn_predict(self, values, knn_indices, weights):
        """Predict values using weighted average of KNN neighbors.
        values: (N, C), knn_indices: (N, k), weights: (N, k)
        Returns: predicted (N, C)"""
        neighbor_values = values[knn_indices.long()]
        w = weights.unsqueeze(-1)
        predicted = (neighbor_values * w).sum(dim=1)
        return predicted

    def _compute_knn_weights(self, xyz, knn_indices, k=8):
        """Compute inverse-distance weights for KNN prediction."""
        neighbor_xyz = xyz[knn_indices.long()]
        dists = torch.norm(neighbor_xyz - xyz.unsqueeze(1), dim=2)
        weights = 1.0 / (dists + 1e-8)
        weights = weights / weights.sum(dim=1, keepdim=True)
        return weights

    def produce_clusters_residual(self, k=8, num_clusters=256, store_dict_path=None):
        """B2: Cross-attribute prediction residual quantization.
        Cascade: xyz -> features_dc -> features_rest -> opacity -> scaling -> rotation.
        Each step predicts from KNN neighbors of already-quantized predecessors.
        """
        max_coeffs_num = (self.max_sh_degree + 1) ** 2 - 1
        codebook_dict = OrderedDict({})

        # Step 1: Quantize xyz
        xyz_codebook = generate_codebook(self._xyz.detach(), num_clusters=num_clusters)
        codebook_dict['_xyz'] = xyz_codebook
        xyz_quantized = xyz_codebook.evaluate().detach()

        # Build KNN graph from quantized positions
        knn_indices = self._build_knn_graph(xyz_quantized, k)
        knn_weights = self._compute_knn_weights(xyz_quantized, knn_indices, k)

        # Step 2: Predict and quantize features_dc
        dc_values = self._features_dc.detach()[:, 0]  # (N, 3)
        dc_predicted = self._knn_predict(dc_values, knn_indices, knn_weights)
        dc_residual = dc_values - dc_predicted
        codebook_dict['features_dc'] = generate_codebook(dc_residual, num_clusters=num_clusters, tol=0.001)

        # Step 3: Predict and quantize features_rest
        for sh_degree in range(max_coeffs_num):
            rest_values = self._features_rest.detach()[:, sh_degree]  # (N, 3)
            rest_predicted = self._knn_predict(rest_values, knn_indices, knn_weights)
            rest_residual = rest_values - rest_predicted
            codebook_dict[f'features_rest_{sh_degree}'] = generate_codebook(
                rest_residual, num_clusters=num_clusters)

        # Step 4: Predict and quantize opacity
        opacity_values = self.get_opacity.detach()  # (N, 1)
        opacity_predicted = self._knn_predict(opacity_values, knn_indices, knn_weights)
        opacity_residual = opacity_values - opacity_predicted
        codebook_dict['opacity'] = generate_codebook(
            opacity_residual, self.inverse_opacity_activation, num_clusters=num_clusters)

        # Step 5: Predict and quantize scaling
        scaling_values = self.get_scaling.detach()  # (N, 3)
        scaling_predicted = self._knn_predict(scaling_values, knn_indices, knn_weights)
        scaling_residual = scaling_values - scaling_predicted
        codebook_dict['scaling'] = generate_codebook(
            scaling_residual, self.scaling_inverse_activation, num_clusters=num_clusters)

        # Step 6: Predict and quantize rotation
        rotation_values = self.get_rotation.detach()  # (N, 4)
        rotation_predicted = self._knn_predict(rotation_values, knn_indices, knn_weights)
        rotation_residual = rotation_values - rotation_predicted
        codebook_dict['rotation_re'] = generate_codebook(
            rotation_residual[:, 0:1], num_clusters=num_clusters)
        codebook_dict['rotation_im'] = generate_codebook(
            rotation_residual[:, 1:], num_clusters=num_clusters)

        self._residual_meta = {
            'k': k,
            'xyz_quantized': xyz_quantized.detach(),
            'knn_indices': knn_indices.detach(),
            'knn_weights': knn_weights.detach(),
        }

        if store_dict_path is not None:
            torch.save(codebook_dict, os.path.join(store_dict_path, 'codebook_residual.pt'))

        self._codebook_dict = codebook_dict

    def apply_residual_clustering(self, codebook_dict=None):
        """B2: Reconstruct model parameters from residual codebooks."""
        if codebook_dict is None:
            return
        meta = self._residual_meta
        knn_indices = meta['knn_indices']
        knn_weights = meta['knn_weights']
        max_coeffs_num = (self.max_sh_degree + 1) ** 2 - 1

        # Reconstruct xyz
        xyz = codebook_dict['_xyz'].evaluate().detach()
        self._xyz = nn.Parameter(xyz.requires_grad_(True))

        # Reconstruct features_dc: prediction + residual
        dc_pred = self._knn_predict(self._features_dc.detach()[:, 0], knn_indices, knn_weights)
        dc_residual = codebook_dict['features_dc'].evaluate().view(-1, 3).detach()
        dc_reconstructed = dc_pred + dc_residual

        # Reconstruct features_rest
        features_rest_list = []
        for sh_degree in range(max_coeffs_num):
            rest_pred = self._knn_predict(self._features_rest.detach()[:, sh_degree], knn_indices, knn_weights)
            rest_residual = codebook_dict[f'features_rest_{sh_degree}'].evaluate().view(-1, 3).detach()
            features_rest_list.append((rest_pred + rest_residual).view(-1, 3))

        # Reconstruct opacity
        opacity_pred = self._knn_predict(self.get_opacity.detach(), knn_indices, knn_weights)
        opacity_residual = codebook_dict['opacity'].evaluate().detach()
        opacity = opacity_pred + opacity_residual

        # Reconstruct scaling
        scaling_pred = self._knn_predict(self.get_scaling.detach(), knn_indices, knn_weights)
        scaling_residual = codebook_dict['scaling'].evaluate().view(-1, 3).detach()
        scaling = scaling_pred + scaling_residual

        # Reconstruct rotation
        rot_pred = self._knn_predict(self.get_rotation.detach(), knn_indices, knn_weights)
        rot_re_residual = codebook_dict['rotation_re'].evaluate().detach()
        rot_im_residual = codebook_dict['rotation_im'].evaluate().view(-1, 3).detach()
        rotation = torch.cat([rot_pred[:, 0:1] + rot_re_residual,
                              rot_pred[:, 1:] + rot_im_residual], dim=1)

        with torch.no_grad():
            self._features_dc = dc_reconstructed.requires_grad_(True).view(-1, 1, 3)
            self._features_rest = torch.stack(features_rest_list, dim=1).squeeze().requires_grad_(True)
            self._opacity = opacity.requires_grad_(True)
            self._scaling = scaling.requires_grad_(True)
            self._rotation = rotation.requires_grad_(True)

    # ─── B3: Contribution-score-based tiered quantization ─────────────────

    def compute_contribution_scores(self, cameras):
        """Compute rendering contribution score for each Gaussian.
        Score(i) = sum_v [P(i,v) * sigma(opacity_i)]
        """
        xyz = self._xyz.detach()
        N = xyz.shape[0]
        opacity = self.get_opacity.detach().squeeze()  # (N,)

        scores = torch.zeros(N, device='cuda')
        for cam in cameras:
            w2ndc = cam.full_proj_transform
            ones = torch.ones((N, 1), device='cuda')
            xyz_h = torch.cat([xyz, ones], dim=1)
            ndc = (xyz_h @ w2ndc)
            ndc = ndc[:, :3] / ndc[:, 3:4]

            in_frustum = (
                (ndc[:, 0].abs() <= 1) & (ndc[:, 1].abs() <= 1) &
                (ndc[:, 2] >= 0) & (ndc[:, 2] <= 1)
            )

            scales = self.get_scaling.detach()
            max_scale = scales.max(dim=1).values
            focal_factor = min(cam.image_width, cam.image_height) * 0.5
            pix = (max_scale * focal_factor) ** 2
            pix[~in_frustum] = 0

            scores += pix * opacity

        return scores

    def produce_clusters_contribution_tiered(self, cameras, store_dict_path=None):
        """B3: Adaptive bit-width quantization based on rendering contribution.
        Top 20% -> 4096 clusters (12-bit), middle 60% -> 256 clusters (8-bit),
        bottom 20% -> 16 clusters (4-bit).
        """
        scores = self.compute_contribution_scores(cameras)
        q20 = torch.quantile(scores, 0.2)
        q80 = torch.quantile(scores, 0.8)

        tier_low = scores < q20
        tier_high = scores >= q80
        tier_mid = ~tier_low & ~tier_high

        tier_sizes = {'low': 16, 'mid': 256, 'high': 4096}
        tiers = {'low': tier_low, 'mid': tier_mid, 'high': tier_high}

        self._tier_masks = tiers
        self._tier_sizes = tier_sizes

        max_coeffs_num = (self.max_sh_degree + 1) ** 2 - 1
        codebook_dict = OrderedDict({})

        for tier_name, tier_mask in tiers.items():
            nc = tier_sizes[tier_name]
            if tier_mask.sum() == 0:
                continue

            def tier_codebook(values, inv_fn=lambda x: x, nc=nc, mask=tier_mask):
                v = values[mask]
                if nc <= 256:
                    return generate_codebook(v, inv_fn, num_clusters=nc)
                else:
                    return _kmeans_pytorch(v, nc)

            codebook_dict[f'_tier_{tier_name}_features_dc'] = tier_codebook(
                self._features_dc.detach()[:, 0], tol=0.001)
            for sh_d in range(max_coeffs_num):
                codebook_dict[f'_tier_{tier_name}_features_rest_{sh_d}'] = tier_codebook(
                    self._features_rest.detach()[:, sh_d])
            codebook_dict[f'_tier_{tier_name}_opacity'] = tier_codebook(
                self.get_opacity.detach(), self.inverse_opacity_activation)
            codebook_dict[f'_tier_{tier_name}_scaling'] = tier_codebook(
                self.get_scaling.detach(), self.scaling_inverse_activation)
            codebook_dict[f'_tier_{tier_name}_rotation_re'] = tier_codebook(
                self.get_rotation.detach()[:, 0:1])
            codebook_dict[f'_tier_{tier_name}_rotation_im'] = tier_codebook(
                self.get_rotation.detach()[:, 1:])

        unified = self._build_unified_codebook_from_tiers(codebook_dict, tiers, tier_sizes)

        if store_dict_path is not None:
            torch.save(codebook_dict, os.path.join(store_dict_path, 'codebook_tiered.pt'))

        self._codebook_dict = unified
        self._tiered_codebook_dict = codebook_dict

    def _build_unified_codebook_from_tiers(self, tiered_dict, tiers, tier_sizes):
        """Convert tiered codebooks into unified format for apply_clustering.
        Each tier gets 256 slots; indices are remapped accordingly."""
        max_coeffs_num = (self.max_sh_degree + 1) ** 2 - 1
        N = self.num_primitives
        unified = OrderedDict()

        attr_keys = ['features_dc'] + [f'features_rest_{i}' for i in range(max_coeffs_num)] + \
                     ['opacity', 'scaling', 'rotation_re', 'rotation_im']

        for attr_key in attr_keys:
            offset = 0
            combined_ids = torch.zeros(N, dtype=torch.uint8, device='cuda')
            center_list = []

            for tier_name, tier_mask in tiers.items():
                tier_key = f'_tier_{tier_name}_{attr_key}'
                if tier_key not in tiered_dict:
                    continue
                cb = tiered_dict[tier_key]
                nc = cb.centers.shape[0]
                # Remap tier-local ids to combined ids
                tier_ids = cb.ids.flatten()
                combined_ids[tier_mask] = tier_ids.byte() + offset
                # Pad centers to 256 for this tier's slot
                padded = torch.zeros(256, 1, device='cuda')
                padded[:nc] = cb.centers
                center_list.append(padded)
                offset += 256

            if center_list:
                combined_center = torch.cat(center_list, dim=0)
                final_centers = combined_center[:256]
                unified[attr_key] = Codebook(combined_ids, final_centers)

        return unified

    # ─── B4: Frequency-based tiered quantization ──────────────────────────

    def compute_sh_energy_ratio(self):
        """Compute SH energy ratio: high_order_energy / total_energy."""
        features_rest = self._features_rest.detach()
        if features_rest.dim() == 2:
            features_rest = features_rest.view(-1, 3, 15).transpose(1, 2)

        dc_energy = (self._features_dc.detach() ** 2).sum(dim=(1, 2))

        coeff_offset = 0
        high_order_energy = torch.zeros(dc_energy.shape[0], device='cuda')
        for band in range(1, self.max_sh_degree + 1):
            n_coeffs = 2 * band + 1
            band_coeffs = features_rest[:, coeff_offset:coeff_offset + n_coeffs, :]
            high_order_energy += (band_coeffs ** 2).sum(dim=(1, 2))
            coeff_offset += n_coeffs

        total_energy = dc_energy + high_order_energy
        ratio = torch.zeros_like(total_energy)
        mask = total_energy > 0
        ratio[mask] = high_order_energy[mask] / total_energy[mask]
        return ratio

    def produce_clusters_frequency_tiered(self, store_dict_path=None):
        """B4: Frequency-based tiered quantization using SH energy ratio.
        Structure layer (scale > 80th percentile AND SH ratio < 0.3): 16 clusters
        Detail layer (remaining): 256 clusters
        """
        sh_ratio = self.compute_sh_energy_ratio()
        scales = self.get_scaling.detach()
        max_scale = scales.max(dim=1).values
        scale_threshold = torch.quantile(max_scale, 0.8)

        structure_mask = (max_scale > scale_threshold) & (sh_ratio < 0.3)
        detail_mask = ~structure_mask

        n_struct = structure_mask.sum().item()
        n_detail = detail_mask.sum().item()
        total = n_struct + n_detail
        print(f"Frequency tiering: structure={n_struct} ({100*n_struct/total:.1f}%), "
              f"detail={n_detail} ({100*n_detail/total:.1f}%)")

        max_coeffs_num = (self.max_sh_degree + 1) ** 2 - 1
        codebook_dict = OrderedDict({})

        # Structure layer: 16 clusters
        if n_struct > 0:
            def struct_cb(values, inv_fn=lambda x: x):
                return generate_codebook(values[structure_mask], inv_fn, num_clusters=16)

            codebook_dict['_struct_features_dc'] = struct_cb(
                self._features_dc.detach()[:, 0], tol=0.001)
            for sh_d in range(max_coeffs_num):
                codebook_dict[f'_struct_features_rest_{sh_d}'] = struct_cb(
                    self._features_rest.detach()[:, sh_d])
            codebook_dict['_struct_opacity'] = struct_cb(
                self.get_opacity.detach(), self.inverse_opacity_activation)
            codebook_dict['_struct_scaling'] = struct_cb(
                self.get_scaling.detach(), self.scaling_inverse_activation)
            codebook_dict['_struct_rotation_re'] = struct_cb(
                self.get_rotation.detach()[:, 0:1])
            codebook_dict['_struct_rotation_im'] = struct_cb(
                self.get_rotation.detach()[:, 1:])

        # Detail layer: 256 clusters
        if n_detail > 0:
            def detail_cb(values, inv_fn=lambda x: x):
                return generate_codebook(values[detail_mask], inv_fn, num_clusters=256)

            codebook_dict['_detail_features_dc'] = detail_cb(
                self._features_dc.detach()[:, 0], tol=0.001)
            for sh_d in range(max_coeffs_num):
                codebook_dict[f'_detail_features_rest_{sh_d}'] = detail_cb(
                    self._features_rest.detach()[:, sh_d])
            codebook_dict['_detail_opacity'] = detail_cb(
                self.get_opacity.detach(), self.inverse_opacity_activation)
            codebook_dict['_detail_scaling'] = detail_cb(
                self.get_scaling.detach(), self.scaling_inverse_activation)
            codebook_dict['_detail_rotation_re'] = detail_cb(
                self.get_rotation.detach()[:, 0:1])
            codebook_dict['_detail_rotation_im'] = detail_cb(
                self.get_rotation.detach()[:, 1:])

        unified = self._build_unified_codebook_from_tiers(
            codebook_dict,
            {'struct': structure_mask, 'detail': detail_mask},
            {'struct': 16, 'detail': 256}
        )

        self._frequency_masks = {'structure': structure_mask, 'detail': detail_mask}
        self._sh_energy_ratio = sh_ratio

        if store_dict_path is not None:
            torch.save(codebook_dict, os.path.join(store_dict_path, 'codebook_frequency.pt'))

        self._codebook_dict = unified
        self._frequency_codebook_dict = codebook_dict