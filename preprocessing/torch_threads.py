"""
preprocessing/torch_threads.py

Pins PyTorch's intra-op thread count.

Why this exists.  Running several experiment scripts at once produced a
reproducible hang: the process stopped at the first torch call that follows a
heavy scikit-learn/XGBoost/CatBoost section -- in practice the
`torch.tensor(...)` at the end of `build_node_features_no_leakage` -- with
every thread parked in a futex and no further output.  `build_weighted_graph`
already carries a comment about the same class of failure ("avoids
PyTorch/OpenBLAS thread deadlock that occurs when torch ops follow heavy numpy
ops"), so this is a known hazard in this pipeline, not a new one.

The cause is two OpenMP runtimes in one process: libgomp, pulled in by
scikit-learn/XGBoost/CatBoost, and the one torch ships.  Under CPU
oversubscription their spin-waiting thread pools can deadlock against each
other.  The failure is a race, so it only shows up under load -- the same
region that hung inside a concurrent sweep completes in 157s when run alone,
which is exactly what makes it so easy to mistake for slowness.

Pinning torch to a single intra-op thread removes its pool from the
contention.  Nothing is lost here: the graphs are ~3,000 nodes, where torch's
threading overhead exceeds any gain, and a fixed thread count also makes
floating-point reduction order stable from run to run, which the paper's
reproducibility claims depend on.

Set TORCH_NUM_THREADS to override.
"""

import os


def limit_torch_threads(default: int = 1) -> None:
    """Set torch's intra-op thread count. Safe to call more than once."""
    try:
        import torch
    except ImportError:
        return
    try:
        n = max(1, int(os.environ.get("TORCH_NUM_THREADS", default)))
    except ValueError:
        n = default
    torch.set_num_threads(n)
