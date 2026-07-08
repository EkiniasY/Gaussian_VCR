"""Benchmark rendering FPS for a trained 3DGS model."""

import os
import torch
import time
import numpy as np
from argparse import ArgumentParser
from scene import Scene
from gaussian_renderer import render, GaussianModel
from utils.general_utils import safe_state
from arguments import ModelParams, PipelineParams, get_combined_args

MODELS_CONFIGURATION = {
    'baseline': {'quantised': False, 'half_float': False, 'name': 'point_cloud.ply'},
    'quantised': {'quantised': True, 'half_float': False, 'name': 'point_cloud_quantised.ply'},
    'quantised_half': {'quantised': True, 'half_float': True, 'name': 'point_cloud_quantised_half.ply'},
}


def benchmark_single(gaussians, views, pipeline, background, model_name):
    print(f"\nPly file : {model_name}")
    print(f"Num Gaussians: {gaussians.get_xyz.shape[0]}")
    print(f"Num test views: {len(views)}")
    print(f"Image size: {views[0].image_width}x{views[0].image_height}")

    # Warmup: render a few frames first
    print("\nWarming up (5 frames)...")
    for i in range(min(5, len(views))):
        render(views[i], gaussians, pipeline, background)
    torch.cuda.synchronize()

    # Benchmark: measure each frame individually
    print(f"Benchmarking {len(views)} frames...")
    frame_times = []
    for view in views:
        torch.cuda.synchronize()
        start = time.perf_counter()
        render(view, gaussians, pipeline, background)
        torch.cuda.synchronize()
        end = time.perf_counter()
        frame_times.append((end - start) * 1000)  # ms

    frame_times = np.array(frame_times)

    print("\n" + "=" * 50)
    print("RESULTS")
    print("=" * 50)
    print(f"Ply file      : {model_name}")
    print(f"Num Gaussians : {gaussians.get_xyz.shape[0]}")
    print(f"Resolution    : {views[0].image_width}x{views[0].image_height}")
    print(f"Frames        : {len(frame_times)}")
    print(f"Avg frame time: {frame_times.mean():.2f} ms")
    print(f"Std frame time: {frame_times.std():.2f} ms")
    print(f"Min frame time: {frame_times.min():.2f} ms")
    print(f"Max frame time: {frame_times.max():.2f} ms")
    print(f"Avg FPS       : {1000.0 / frame_times.mean():.1f}")
    print(f"Median FPS    : {1000.0 / np.median(frame_times):.1f}")
    print(f"Total time    : {frame_times.sum() / 1000:.2f} s")
    print("=" * 50)


def benchmark(dataset, iteration, pipeline, models):
    with torch.no_grad():
        gaussians = GaussianModel(dataset.sh_degree)
        scene = Scene(dataset, gaussians, load_iteration=iteration, shuffle=False)

        bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
        background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

        views = scene.getTestCameras()
        if len(views) == 0:
            views = scene.getTrainCameras()

        print(f"\nModel path: {dataset.model_path}")

        for model in models:
            cfg = MODELS_CONFIGURATION[model]
            ply_path = os.path.join(scene.model_path,
                                    "point_cloud",
                                    "iteration_" + str(scene.loaded_iter),
                                    cfg['name'])
            gaussians.load_ply(ply_path, quantised=cfg['quantised'], half_float=cfg['half_float'])
            benchmark_single(gaussians, views, pipeline, background, cfg['name'])


if __name__ == "__main__":
    parser = ArgumentParser(description="3DGS Rendering Benchmark")
    model = ModelParams(parser, sentinel=True)
    pipeline = PipelineParams(parser)
    parser.add_argument("--iteration", default=-1, type=int)
    parser.add_argument("--models", default=['baseline'], nargs="+",
                        choices=MODELS_CONFIGURATION.keys(),
                        help="Which PLY models to benchmark (default: baseline)")
    parser.add_argument("--quiet", action="store_true")
    args = get_combined_args(parser)
    safe_state(args.quiet)
    benchmark(model.extract(args), args.iteration, pipeline.extract(args), args.models)
