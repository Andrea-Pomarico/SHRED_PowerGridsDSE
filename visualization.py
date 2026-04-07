# =============================================================================
# visualization.py  –  Plotting helpers for the DSE experiment
# =============================================================================
"""All matplotlib figures used in the notebook live here.

Public API
----------
plot_noise_check(clean, noisy, var_id)
plot_training_curve(val_errors)
plot_reconstruction(real, reconstructed, var_idx, x_window, split_label, key)
plot_all_variables(real, reconstructed, vars_to_plot, x_window, split_label, key)
"""

from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt


# ── Noise verification ────────────────────────────────────────────────────────

def plot_noise_check(
    clean: np.ndarray,
    noisy: np.ndarray,
    var_id: int = 76,
) -> None:
    """Overlay clean and noisy signals for a quick sanity check.

    Parameters
    ----------
    clean  : original data array (timesteps, variables)
    noisy  : noise-injected data array (same shape)
    var_id : column index to plot
    """
    plt.figure(figsize=(14, 3))
    plt.plot(clean[:, var_id], label="Clean",       alpha=0.85, color="steelblue")
    plt.plot(noisy[:, var_id], label="Noisy",       alpha=0.65, color="tomato")
    plt.title(f"Variable {var_id} – clean vs noisy")
    plt.xlabel("Concatenated timestep")
    plt.ylabel("Value")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()


# ── Training curve ────────────────────────────────────────────────────────────

def plot_training_curve(val_errors: list[float]) -> None:
    """Plot the per-epoch validation MSE returned by ``shred.fit()``.

    Parameters
    ----------
    val_errors : list of validation MSE values (one per epoch)
    """
    plt.figure(figsize=(8, 3))
    plt.plot(val_errors, color="steelblue", label="Validation MSE")
    plt.xlabel("Epoch")
    plt.ylabel("MSE")
    plt.title("SHRED training curve")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()


# ── Single-variable reconstruction ───────────────────────────────────────────

def plot_reconstruction(
    real: np.ndarray,
    reconstructed: np.ndarray,
    var_idx: int,
    x_window: tuple[int, int] | None = None,
    split_label: str = "test",
    key: str = "PSSE",
) -> None:
    """Plot ground-truth vs SHRED reconstruction for one variable.

    Parameters
    ----------
    real          : ground-truth array (timesteps, variables)
    reconstructed : reconstructed array (same shape)
    var_idx       : column index to plot
    x_window      : (start, end) timestep window; None = full signal
    split_label   : "train", "val", or "test" (used in title/legend)
    key           : dataset identifier string (used in title)
    """
    real_sig = real[:, var_idx]
    rec_sig  = reconstructed[:, var_idx]

    plt.figure(figsize=(14, 3))
    plt.plot(real_sig, label=f"Ground truth ({split_label})", color="green")
    plt.plot(rec_sig,  label="SHRED reconstruction",          color="blue", alpha=0.7)

    if x_window is not None:
        lo, hi = x_window
        plt.xlim(lo, hi)
        y_win  = np.concatenate([real_sig[lo:hi], rec_sig[lo:hi]])
        y_min, y_max = float(y_win.min()), float(y_win.max())
        margin = max(0.05 * (y_max - y_min), 1e-6)
        plt.ylim(y_min - margin, y_max + margin)

    plt.title(f"Variable {var_idx} – real vs reconstructed ({key})")
    plt.xlabel("Timestep")
    plt.ylabel("Value")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()


# ── Multi-variable loop ───────────────────────────────────────────────────────

def plot_all_variables(
    real: np.ndarray,
    reconstructed: np.ndarray,
    vars_to_plot=range(138),
    x_window: tuple[int, int] | None = None,
    split_label: str = "test",
    key: str = "PSSE",
) -> None:
    """Call ``plot_reconstruction`` for every variable in ``vars_to_plot``.

    Parameters
    ----------
    real          : ground-truth array (timesteps, variables)
    reconstructed : reconstructed array (same shape)
    vars_to_plot  : iterable of column indices to plot
    x_window      : (start, end) timestep window; None = full signal
    split_label   : "train", "val", or "test"
    key           : dataset identifier string
    """
    assert real.shape == reconstructed.shape, (
        f"Shape mismatch: real {real.shape} vs reconstructed {reconstructed.shape}"
    )
    for i in vars_to_plot:
        print(f"Plotting variable {i}")
        plot_reconstruction(real, reconstructed, i, x_window, split_label, key)
