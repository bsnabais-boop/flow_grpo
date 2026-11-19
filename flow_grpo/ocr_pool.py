"""PaddleOCR in a pool of CPU worker processes (2026-11-18).

The training loop used to run PaddleOCR in the main process, one image at a time, while the GPU waited (SLURM: ~1 of 8
CPU cores busy). ParallelOcr spreads the images of a batch over N spawned worker processes, each with its own
PaddleOCR instance, and returns futures so the caller can keep sampling on the GPU while the OCR runs.

Scores are identical to flow_grpo.ocr.OcrScorer (the workers call it); only where and when they are computed changes.
"""
import multiprocessing as mp
import os
from concurrent.futures import ProcessPoolExecutor

_SCORER = None


def _init_worker(threads):
    # one CPU thread per worker: parallelism comes from the processes, not from Paddle's thread pool
    for k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        os.environ[k] = str(threads)
    os.environ["CUDA_VISIBLE_DEVICES"] = ""  # workers never touch the GPU
    global _SCORER
    from flow_grpo.ocr import OcrScorer
    _SCORER = OcrScorer()


def _score_one(img, target):
    return float(_SCORER([img], [target])[0])


class ParallelOcr:
    def __init__(self, workers=8, threads_per_worker=1):
        ctx = mp.get_context("spawn")  # the parent has CUDA initialised: fork is unsafe
        self.pool = ProcessPoolExecutor(max_workers=workers, mp_context=ctx, initializer=_init_worker,
                                        initargs=(threads_per_worker,))
        # warm-up: build every worker's PaddleOCR now, not inside the first epoch
        import numpy as np
        blank = np.zeros((64, 64, 3), dtype=np.uint8)
        list(self.pool.map(_score_one, [blank] * workers, ["x"] * workers))

    def submit(self, images_nhwc_uint8, targets):
        """Returns a list of futures, one float score per image."""
        return [self.pool.submit(_score_one, img, t) for img, t in zip(images_nhwc_uint8, targets)]

    def shutdown(self):
        self.pool.shutdown(wait=False, cancel_futures=True)
