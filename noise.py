# =============================================================================
# noise.py  –  Noise injection for robustness experiments
# =============================================================================
"""Additive Gaussian noise injection.

Public API
----------
inject_noise(data, n_vars, std_frac, noise_fraction, seed) -> np.ndarray
"""

from __future__ import annotations

import numpy as np


def inject_noise(
    data: np.ndarray,
    n_vars: int,
    std_frac: float,
    noise_fraction: float = 1.0,
    seed: int = 42,
) -> np.ndarray:
    """Inject additive Gaussian noise into the first ``n_vars`` columns.

    Noise magnitude for column *j* is ``std_frac * std(data[:, j])``,
    matching the signal-level of each variable independently.

    Parameters
    ----------
    data           : clean data array of shape (timesteps, variables)
    n_vars         : number of columns to corrupt (starting from index 0)
    std_frac       : noise σ expressed as a fraction of each variable's std
    noise_fraction : fraction of *rows* to corrupt; 1.0 corrupts all rows,
                     0.1 corrupts only the last 10 % (useful for partial tests)
    seed           : random seed for reproducibility

    Returns
    -------
    noisy : copy of ``data`` with noise added to the selected columns/rows

    Raises
    ------
    AssertionError if columns beyond ``n_vars`` are accidentally modified.

    Example
    -------
    >>> clean = np.ones((1000, 50))
    >>> noisy = inject_noise(clean, n_vars=40, std_frac=0.05)
    >>> assert noisy.shape == clean.shape
    """
    rng    = np.random.default_rng(seed)
    noisy  = data.copy()
    N, D   = data.shape
    n_vars = min(n_vars, D)
    start  = int((1.0 - noise_fraction) * N)

    for j in range(n_vars):
        sigma = std_frac * float(np.std(data[:, j]))
        noisy[start:, j] += rng.normal(0.0, sigma, N - start)

    # Safety check: columns beyond n_vars must remain untouched
    if D > n_vars:
        assert np.allclose(noisy[:, n_vars:], data[:, n_vars:]), (
            f"ERROR: columns {n_vars}–{D - 1} were accidentally modified."
        )

    return noisy
