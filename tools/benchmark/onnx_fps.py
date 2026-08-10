"""
Measure latency and FPS of an exported DEIMv2 ONNX model with onnxruntime.

Timings are end-to-end for ``session.run`` with numpy inputs, i.e. they include host-to-device and
device-to-host copies and the postprocessing baked into the exported graph.

Usage::

    CUDA_VISIBLE_DEVICES=0 python tools/benchmark/onnx_fps.py --onnx outputs/deimv2_dinov2_product/best_stg2.onnx
"""

import argparse
import time
from typing import NamedTuple

import numpy as np
import torch  # noqa: F401  (imported first so its bundled cuDNN/cuBLAS libraries are loadable by onnxruntime)
import onnxruntime as ort


class BenchResult(NamedTuple):
    """Latency and throughput of one benchmark run."""

    latency_ms: float
    fps: float


def bench(session: ort.InferenceSession, batch: int, size: int, warmup: int, iters: int) -> BenchResult:
    """Benchmark ``session`` on random inputs of the given batch size.

    :param session: Inference session for a model with ``images`` / ``orig_target_sizes`` inputs.
    :param batch: Batch size.
    :param size: Square input resolution in pixels.
    :param warmup: Number of untimed warmup runs.
    :param iters: Number of timed runs.
    :return: Mean latency per call (ms) and images per second.
    """
    feed = {
        'images': np.random.rand(batch, 3, size, size).astype(np.float32),
        'orig_target_sizes': np.array([[size, size]] * batch, dtype=np.int64),
    }
    for _ in range(warmup):
        session.run(None, feed)
    start = time.perf_counter()
    for _ in range(iters):
        session.run(None, feed)
    elapsed = time.perf_counter() - start
    return BenchResult(latency_ms=elapsed / iters * 1000, fps=iters * batch / elapsed)


def main() -> None:
    """Parse arguments and run the benchmark for each requested batch size."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--onnx', type=str, required=True, help='path to the ONNX model')
    parser.add_argument('--batch-sizes', type=int, nargs='+', default=[1, 4, 8])
    parser.add_argument('--size', type=int, default=640, help='input resolution')
    parser.add_argument('--warmup', type=int, default=50)
    parser.add_argument('--iters', type=int, default=200)
    parser.add_argument('--cpu', action='store_true', help='benchmark on CPU instead of CUDA')
    args = parser.parse_args()

    providers = ['CPUExecutionProvider'] if args.cpu else ['CUDAExecutionProvider']
    session = ort.InferenceSession(args.onnx, providers=providers)
    print('provider:', session.get_providers()[0])
    for batch in args.batch_sizes:
        result = bench(session, batch, args.size, args.warmup, args.iters)
        print(f'batch={batch}:  latency={result.latency_ms:7.2f} ms  fps={result.fps:7.1f}')


if __name__ == '__main__':
    main()
