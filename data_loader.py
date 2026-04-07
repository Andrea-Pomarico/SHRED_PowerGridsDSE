# =============================================================================
# data_loader.py  –  DIgSILENT PowerFactory CSV loading and dataset construction
# =============================================================================
"""Utilities for reading DIgSILENT PowerFactory simulation CSVs and assembling the
(X_dataset, y_dataset) arrays used by the SHRED pipeline.

Public API
----------
make_variable_lists(n_bus, n_gen) -> (list_bus, list_omega, list_p, list_q)
zigzag_order(lst)                 -> list
load_simulation(...)              -> pd.DataFrame | None
build_dataset(...)                -> (np.ndarray, np.ndarray)
"""

from __future__ import annotations

import os

import numpy as np
import pandas as pd


# ── Variable name helpers ─────────────────────────────────────────────────────

def make_variable_lists(n_bus: int = 39, n_gen: int = 10) -> tuple[list, list, list, list]:
    """Return (List_Bus, List_omega, List_P_gen, List_Q_gen) for the 39-bus system."""
    list_bus   = [f"Bus {i:02d}"   for i in range(1, n_bus + 1)]
    list_omega = [f"G {i:02d}"     for i in range(1, n_gen + 1)]
    list_p     = [f"G {i:02d}.1"   for i in range(1, n_gen + 1)]
    list_q     = [f"G {i:02d}.2"   for i in range(1, n_gen + 1)]
    return list_bus, list_omega, list_p, list_q


# ── Zig-zag ordering ─────────────────────────────────────────────────────────

def zigzag_order(lst: list) -> list:
    """Return elements in alternating front/back order.

    This ensures train/val/test splits naturally cover both short and long
    clearing times rather than being cut at a single temporal boundary.

    Example
    -------
    >>> zigzag_order(["2", "3", "4", "5", "6", "7", "8"])
    ['2', '8', '3', '7', '4', '6', '5']
    """
    lst = lst.copy()
    result: list = []
    while lst:
        result.append(lst.pop(0))
        if lst:
            result.append(lst.pop(-1))
    return result


# ── Single-simulation loader ──────────────────────────────────────────────────

def load_simulation(
    data_root: str,
    topology: str,
    line_out: str,
    fault: str,
    tempo: str,
    t_min: float = 0.0,
    t_max: float = 10.0,
) -> pd.DataFrame | None:
    """Load one DIgSILENT PowerFactory simulation CSV and return a cleaned DataFrame.

    Parameters
    ----------
    data_root : root data directory (contains N/ and N_1/ sub-folders)
    topology  : "N" (all lines in service) or "N_1" (one line out)
    line_out  : line removed from service (only used when topology == "N_1")
    fault     : fault location, e.g. "Line 01 - 02"
    tempo     : clearing time string, e.g. "8"
    t_min     : lower bound of time window to keep (seconds)
    t_max     : upper bound of time window to keep (seconds)

    Returns
    -------
    DataFrame with numeric values, or None if the file is missing / empty.
    """
    if topology == "N_1":
        path = os.path.join(data_root, topology, line_out, fault, f"{tempo}.csv")
    else:
        path = os.path.join(data_root, topology, fault, f"{tempo}.csv")

    if not os.path.exists(path):
        return None

    df = pd.read_csv(path)
    df = df.iloc[1:].reset_index(drop=True)           # drop DIgSILENT PowerFactory header row
    df = df.apply(pd.to_numeric, errors="coerce")
    df = df[(df.iloc[:, 0] >= t_min) & (df.iloc[:, 0] <= t_max)].reset_index(drop=True)

    return df if len(df) > 0 else None


# ── Full dataset builder ──────────────────────────────────────────────────────

def build_dataset(
    data_root: str,
    topology: str,
    line_out: str,
    list_fault: list[str],
    list_tempo: list[str],
    list_bus: list[str],
    list_p_gen: list[str],
    list_q_gen: list[str],
    verbose: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    """Iterate over all (tempo, fault) pairs and build X and y arrays.

    The clearing times are visited in zig-zag order so that the concatenated
    sequence spans all scenario types uniformly (see ``zigzag_order``).

    Parameters
    ----------
    data_root  : root data directory
    topology   : "N" or "N_1"
    line_out   : line removed (used only for "N_1")
    list_fault : list of fault location strings
    list_tempo : list of clearing-time strings
    list_bus   : column names for bus voltages
    list_p_gen : column names for generator active power
    list_q_gen : column names for generator reactive power
    verbose    : print progress if True

    Returns
    -------
    X_dataset : shape (n_scenarios, n_features)  – static input features
    y_dataset : shape (n_scenarios, n_timesteps, n_variables)  – time-series
    """
    tempo_ordered = zigzag_order(list_tempo)
    if verbose:
        print("Zig-zag time ordering:", tempo_ordered)

    X_list: list[np.ndarray] = []
    y_list: list[np.ndarray] = []
    skipped = 0

    for tempo in tempo_ordered:
        for fault in list_fault:
            df = load_simulation(data_root, topology, line_out, fault, tempo)

            if df is None:
                if verbose:
                    print(f"  [SKIP] tempo={tempo}, fault={fault}")
                skipped += 1
                continue

            # One-hot fault vector
            fault_vec = np.zeros(len(list_fault), dtype=int)
            fault_vec[list_fault.index(fault)] = 1

            # Static input feature: initial conditions + fault info
            x = np.concatenate([
                df[list_bus].iloc[0].values,    # initial bus voltages
                df[list_p_gen].iloc[0].values,  # initial P gen
                df[list_q_gen].iloc[0].values,  # initial Q gen
                fault_vec,                       # one-hot fault location
                np.array([int(tempo)]),          # clearing time
            ])

            # Time-series output: all variables, all timesteps
            y = df.iloc[:, 1:].to_numpy()

            X_list.append(x)
            y_list.append(y)

    X_dataset = np.array(X_list)
    y_dataset = np.array(y_list)

    if verbose:
        print(f"\nDataset built. Skipped: {skipped}")
        print(f"  X_dataset : {X_dataset.shape}  (scenarios × features)")
        print(f"  y_dataset : {y_dataset.shape}  (scenarios × timesteps × variables)")

    return X_dataset, y_dataset
