from typing import Dict, Optional, Sequence

import numpy as np


def _ensure_rng(seed: Optional[int] = None) -> np.random.Generator:
    return np.random.default_rng(seed)


def compute_residual_stats(
    residuals: Sequence[float], quantiles: Sequence[float] = (0.1, 0.5, 0.9)
) -> Dict[str, float]:
    arr = np.asarray(residuals, dtype=float)
    stats: Dict[str, float] = {}
    stats["mean"] = float(np.mean(arr))
    stats["std"] = float(np.std(arr, ddof=1)) if arr.size > 1 else 0.0
    for q in quantiles:
        stats[f"q{int(q*100):02d}"] = float(np.quantile(arr, q))
    return stats


def sample_residuals(
    n: int,
    residuals: Sequence[float],
    method: str = "bootstrap",
    seed: Optional[int] = None,
) -> np.ndarray:
    """
    Draw residuals for Monte Carlo difficulty generation.

    method: currently only supports simple bootstrap.
    """
    arr = np.asarray(residuals, dtype=float)
    if arr.size == 0:
        raise ValueError("No residuals available to sample.")

    rng = _ensure_rng(seed)
    if method != "bootstrap":
        raise ValueError(f"Unsupported sampling method: {method}")

    return rng.choice(arr, size=n, replace=True)
