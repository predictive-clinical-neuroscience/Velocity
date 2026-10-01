"""High-level orchestration: load data and compute z-gains."""
from __future__ import annotations

import pickle
from pathlib import Path
from typing import Any

import pandas as pd

from .config import ZGainConfig
from .core import compute_all_z_gains


def load_input_data(path: str | Path) -> pd.DataFrame:
    """Load a dataset from a CSV or pickle file into a DataFrame."""
    path = Path(path)
    if path.suffix in {".pkl", ".pickle"}:
        return pd.read_pickle(path)
    if path.suffix == ".csv":
        return pd.read_csv(path)
    raise ValueError(f"Unsupported data file type: {path.suffix}")


def load_velocity_matrix(path: str | Path) -> Any:
    """Load the age-by-age predicted correlation matrix used to compute z-gains.

    Expects a pickle containing either the matrix directly, or a dict with
    an ``A_sparse_predict`` key (as produced by the velocity model fitting
    step). Sparse matrices are densified.
    """
    with open(path, "rb") as f:
        data = pickle.load(f)

    R = data["A_sparse_predict"] if isinstance(data, dict) else data
    if hasattr(R, "toarray"):
        R = R.toarray()
    return R


def run_zgain_analysis(config: ZGainConfig, df: pd.DataFrame | None = None) -> dict:
    """Run the full z-gain pipeline for a single measure and write results to
    ``config.output_dir``.

    If ``df`` is not provided, it is loaded from ``config.data_path``.
    Returns a dict with the computed gains DataFrame and the output file path.
    """
    if df is None:
        df = load_input_data(config.data_path)
    R = load_velocity_matrix(config.velocity_matrix_path)
    config.output_dir.mkdir(parents=True, exist_ok=True)

    gains = compute_all_z_gains(
        df, R, config.measure,
        id_col=config.id_col, age_col=config.age_col, visit_col=config.visit_col,
    )

    gains_path = config.output_dir / f"z_gains_{config.measure}.pkl"
    gains.to_pickle(gains_path)

    return {
        "gains": gains,
        "gains_path": gains_path,
    }
