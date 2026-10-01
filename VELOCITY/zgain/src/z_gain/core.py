"""Core z-gain math: dataset-agnostic, operates on plain DataFrames."""
from __future__ import annotations

import logging
from itertools import combinations
from typing import Any

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def calculate_z_gain(Z2: float, Z1: float, age2: float, age1: float, R: np.ndarray) -> float:
    """Compute the z-gain between two timepoints, correcting for the expected
    autocorrelation of z-scores between ``age1`` and ``age2``.

    ``R`` is a dense age-by-age correlation matrix (as produced by the
    velocity/normative model), indexable by integer age.
    """
    if age2 <= age1:
        raise ValueError(f"age2 must be greater than age1. Got age1={age1}, age2={age2}")

    r = R[int(age2), int(age1)]
    if r < 0.3:
        raise ValueError(f"Really low r value here: r = {r}")

    sigma = np.sqrt(1 - r**2)
    return (Z2 - r * Z1) / sigma


def compute_all_z_gains(
    df: pd.DataFrame,
    R: np.ndarray,
    measure: str,
    *,
    id_col: str = "ID_subject",
    age_col: str = "age",
    visit_col: str = "ID_visit",
) -> pd.DataFrame:
    """Compute z-gains for every pair of timepoints of every subject in ``df``.

    Returns a DataFrame with standardized columns (``id``, ``age1``, ``age2``,
    ``t1_index``, ``t2_index``, ``z_gain``, ``Z1``, ``Z2``) regardless of the
    input column names, so downstream code does not need to know about the
    source dataset's schema.
    """
    results = []

    for subject, group in df.groupby(id_col):
        group_sorted = group.sort_values([age_col, visit_col])
        group_unique = group_sorted.drop_duplicates(subset=age_col, keep="first").reset_index(drop=True)

        for i, j in combinations(range(len(group_unique)), 2):
            age1, age2 = group_unique.loc[i, age_col], group_unique.loc[j, age_col]
            try:
                Z1, Z2 = group_unique.loc[i, measure], group_unique.loc[j, measure]
                gain = calculate_z_gain(Z2, Z1, age2, age1, R)
                results.append({
                    "id": subject,
                    "age1": age1, "age2": age2,
                    "t1_index": i, "t2_index": j,
                    "z_gain": gain,
                    "Z1": Z1, "Z2": Z2,
                })
            except Exception as exc:
                logger.warning("Skipped subject %s from t1=%s to t2=%s: %s", subject, age1, age2, exc)
                continue

    return pd.DataFrame(results)
