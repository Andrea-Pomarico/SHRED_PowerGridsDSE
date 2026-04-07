# =============================================================================
# shred_pipeline.py  –  SHRED training, reconstruction, and evaluation
# =============================================================================
"""Wrapper functions around the PyShred API.

Public API
----------
setup_manager(X_input, stationary, lags, train_size, val_size, test_size)
    -> DataManager

prepare_datasets(manager)
    -> (train_dataset, val_dataset, test_dataset)

build_shred()
    -> SHRED

train_shred(shred, train_dataset, val_dataset, epochs, batch_size, lr)
    -> list[float]

evaluate_shred(shred, train_dataset, val_dataset, test_dataset)
    -> dict[str, float]

reconstruct(engine, sensor_measurements)
    -> (np.ndarray, str)

full_evaluation(engine, manager, X_flat)
    -> dict[str, object]
"""

from __future__ import annotations

import numpy as np
from pyshred import DataManager, SHRED, SHREDEngine


# ── DataManager ───────────────────────────────────────────────────────────────

def setup_manager(
    X_input: np.ndarray,
    stationary: list[tuple[int, ...]],
    lags: int = 35,
    train_size: float = 0.8,
    val_size: float = 0.1,
    test_size: float = 0.1,
    dataset_id: str = "PSSE",
) -> DataManager:
    """Create and populate a PyShred DataManager.

    Parameters
    ----------
    X_input    : flattened data array (total_timesteps, n_variables)
    stationary : list of sensor index tuples, e.g. [(3,), (4,), ...]
    lags       : number of past timesteps used by the LSTM
    train_size : fraction of data for training
    val_size   : fraction of data for validation
    test_size  : fraction of data for testing
    dataset_id : identifier string for this dataset inside the manager

    Returns
    -------
    Configured DataManager (``prepare()`` not yet called).
    """
    manager = DataManager(
        lags       = lags,
        train_size = train_size,
        val_size   = val_size,
        test_size  = test_size,
    )
    manager.add_data(
        data       = X_input,
        id         = dataset_id,
        stationary = stationary,
        compress   = False,
    )
    return manager


def prepare_datasets(manager: DataManager) -> tuple:
    """Call ``manager.prepare()`` and return (train, val, test) datasets."""
    return manager.prepare()


# ── SHRED model ───────────────────────────────────────────────────────────────

def build_shred(
    sequence_model: str = "LSTM",
    decoder_model: str = "MLP",
    latent_forecaster=None,
) -> SHRED:
    """Instantiate a SHRED model.

    Parameters
    ----------
    sequence_model    : recurrent architecture ("LSTM" or "GRU")
    decoder_model     : decoder architecture ("MLP")
    latent_forecaster : optional latent forecaster (None = no forecasting)

    Returns
    -------
    Un-trained SHRED instance.
    """
    return SHRED(
        sequence_model    = sequence_model,
        decoder_model     = decoder_model,
        latent_forecaster = latent_forecaster,
    )


def train_shred(
    shred: SHRED,
    train_dataset,
    val_dataset,
    epochs: int = 500,
    batch_size: int = 1024,
    lr: float = 1e-4,
) -> list[float]:
    """Fit SHRED and return the per-epoch validation MSE history.

    Parameters
    ----------
    shred         : SHRED model to train (modified in place)
    train_dataset : training dataset from DataManager.prepare()
    val_dataset   : validation dataset from DataManager.prepare()
    epochs        : number of training epochs
    batch_size    : mini-batch size
    lr            : learning rate

    Returns
    -------
    val_errors : list of validation MSE values, one per epoch
    """
    val_errors = shred.fit(
        train_dataset = train_dataset,
        val_dataset   = val_dataset,
        num_epochs    = epochs,
        batch_size    = batch_size,
        lr            = lr,
    )
    return val_errors


def evaluate_shred(
    shred: SHRED,
    train_dataset,
    val_dataset,
    test_dataset,
) -> dict[str, float]:
    """Evaluate SHRED MSE on train / val / test splits.

    Returns
    -------
    dict with keys "train_mse", "val_mse", "test_mse"
    """
    return {
        "train_mse": float(shred.evaluate(dataset=train_dataset)),
        "val_mse":   float(shred.evaluate(dataset=val_dataset)),
        "test_mse":  float(shred.evaluate(dataset=test_dataset)),
    }


# ── Reconstruction ────────────────────────────────────────────────────────────

def reconstruct(
    engine: SHREDEngine,
    sensor_measurements: np.ndarray,
) -> tuple[np.ndarray, str]:
    """Run sensor → latent → decode pipeline for one split.

    Parameters
    ----------
    engine              : SHREDEngine wrapping the trained manager + shred
    sensor_measurements : sensor measurement array for the split

    Returns
    -------
    (X_reconstructed_array, dataset_key)
      – X_reconstructed_array : shape (timesteps, n_variables)
      – dataset_key           : the PyShred dataset identifier string
    """
    latent        = engine.sensor_to_latent(sensor_measurements)
    reconstructed = engine.decode(latent)
    key           = list(reconstructed.keys())[0]
    return reconstructed[key], key


def full_evaluation(
    engine: SHREDEngine,
    manager: DataManager,
    X_flat: np.ndarray,
    dataset_id: str = "PSSE",
) -> dict[str, object]:
    """Reconstruct all three splits and compute per-split errors.

    Parameters
    ----------
    engine     : SHREDEngine
    manager    : fitted DataManager (provides sensor measurement arrays)
    X_flat     : original flattened data (total_timesteps, n_variables)
    dataset_id : dataset identifier used in manager

    Returns
    -------
    dict with keys:
      "train_arr", "train_real", "train_error",
      "val_error",
      "test_arr",  "test_real",  "test_error",
      "key"
    """
    # ── sizes ──────────────────────────────────────────────────────────────
    t_train = len(manager.train_sensor_measurements)
    t_val   = len(manager.val_sensor_measurements)
    t_test  = len(manager.test_sensor_measurements)

    # ── reconstruction ─────────────────────────────────────────────────────
    train_arr, key = reconstruct(engine, manager.train_sensor_measurements)
    test_arr,  _   = reconstruct(engine, manager.test_sensor_measurements)

    # Align ground-truth slices
    train_real = X_flat[:t_train]
    test_real  = X_flat[-t_test:]

    # ── per-split errors via SHREDEngine ───────────────────────────────────
    train_error = engine.evaluate(
        manager.train_sensor_measurements,
        {dataset_id: train_real},
    )
    val_error = engine.evaluate(
        manager.val_sensor_measurements,
        {dataset_id: X_flat[t_train: t_train + t_val]},
    )
    test_error = engine.evaluate(
        manager.test_sensor_measurements,
        {dataset_id: test_real},
    )

    return {
        "train_arr":   train_arr,
        "train_real":  train_real,
        "train_error": train_error,
        "val_error":   val_error,
        "test_arr":    test_arr,
        "test_real":   test_real,
        "test_error":  test_error,
        "key":         key,
    }
