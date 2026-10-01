"""Build and validate the input DataFrame that :mod:`z_gain` expects.

Input contract
--------------
:func:`z_gain.core.compute_all_z_gains` is deliberately schema-light: it needs
one long-format row per subject *per visit*, with four columns.

================  ========  ====================================================
role              default   requirement
================  ========  ====================================================
subject id        ``ID_subject``  any hashable; rows are grouped on it
age               ``age``   numeric, finite, ``>= 0``; **used as an integer
                            index into the velocity matrix** (``int(age)``), so
                            it must fall inside that matrix's age range
visit id          ``ID_visit``    numeric or string; only used to break ties
                            when two rows share an age
measure           (none)    the **z-score** for the measure being analysed --
                            not the raw volume/thickness
================  ========  ====================================================

Any other columns (diagnosis, sex, site, ...) are carried along harmlessly but
ignored by the z-gain computation; join them back onto the output on ``id``.

How rows become pairs
---------------------
Within each subject, rows are sorted by ``(age, visit)``, rows that repeat an
age are dropped keeping the first, and a z-gain is emitted for every ordered
pair of the survivors. So a subject with 3 distinct ages yields 3 gains, and a
subject with a single visit yields none.

Silent drops to know about
--------------------------
``compute_all_z_gains`` catches per-pair exceptions and logs a warning, so a
malformed frame produces a *smaller* result rather than an error. Pairs are
dropped when the age is NaN or lands outside the velocity matrix, and when the
age-to-age correlation is below 0.3. NaN z-scores are worse: they raise
nothing and flow through as NaN ``z_gain`` values. :func:`validate_zgain_frame`
exists to surface all of this before you run.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from itertools import combinations
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
import pandas as pd
import os

logger = logging.getLogger(__name__)

#: Correlations below this are rejected by :func:`z_gain.core.calculate_z_gain`.
MIN_CORRELATION = 0.3


# --------------------------------------------------------------------------
# validation
# --------------------------------------------------------------------------
@dataclass
class ValidationReport:
    """Outcome of :func:`validate_zgain_frame`.

    ``errors`` are problems that make the frame unusable; ``warnings`` are rows
    or pairs that will be silently dropped or turn into NaN. ``stats`` holds
    counts you can log for a run.
    """

    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    stats: dict[str, Any] = field(default_factory=dict)

    @property
    def ok(self) -> bool:
        return not self.errors

    def raise_if_invalid(self) -> "ValidationReport":
        """Raise ``ValueError`` listing every error, or return self."""
        if self.errors:
            raise ValueError(
                "Input frame is not usable for a z-gain run:\n  - "
                + "\n  - ".join(self.errors)
            )
        return self

    def summary(self) -> str:
        lines = [f"subjects={self.stats.get('n_subjects', '?')}, "
                 f"rows={self.stats.get('n_rows', '?')}, "
                 f"expected gains={self.stats.get('n_pairs_expected', '?')}"]
        for e in self.errors:
            lines.append(f"ERROR   {e}")
        for w in self.warnings:
            lines.append(f"WARNING {w}")
        return "\n".join(lines)

    def __str__(self) -> str:  # pragma: no cover - convenience
        return self.summary()


def validate_zgain_frame(
    df: pd.DataFrame,
    measure: str,
    *,
    id_col: str = "ID_subject",
    age_col: str = "age",
    visit_col: str = "ID_visit",
    R: np.ndarray | None = None,
) -> ValidationReport:
    """Check ``df`` against the input contract and predict what will be dropped.

    Pass ``R`` (the dense age-by-age matrix from
    :func:`z_gain.pipeline.load_velocity_matrix`) to additionally count the
    pairs that will fall below the ``r >= 0.3`` floor. Returns a
    :class:`ValidationReport`; nothing is raised unless you ask for it via
    :meth:`ValidationReport.raise_if_invalid`.
    """
    rep = ValidationReport()
    rep.stats["n_rows"] = len(df)

    missing = [c for c in (id_col, age_col, visit_col, measure) if c not in df.columns]
    if missing:
        rep.errors.append(
            f"missing column(s) {missing}; frame has {list(df.columns)[:15]}"
        )
        return rep

    if df.empty:
        rep.errors.append("frame is empty")
        return rep

    # --- measure ---------------------------------------------------------
    measure_vals = df[measure]
    if not pd.api.types.is_numeric_dtype(measure_vals):
        rep.errors.append(f"measure column {measure!r} is not numeric (dtype {measure_vals.dtype})")
    else:
        n_nan = int(measure_vals.isna().sum())
        if n_nan:
            rep.warnings.append(
                f"{n_nan} NaN value(s) in {measure!r}; these do NOT raise and will "
                "produce NaN z_gain rows in the output"
            )
        finite = measure_vals.dropna()
        if len(finite) and (finite.abs() > 10).any():
            rep.warnings.append(
                f"{int((finite.abs() > 10).sum())} value(s) in {measure!r} exceed |10| -- "
                "this column should hold z-scores, not raw measurements"
            )

    # --- id / visit ------------------------------------------------------
    if df[id_col].isna().any():
        rep.errors.append(f"{int(df[id_col].isna().sum())} NaN subject id(s) in {id_col!r}")
    if df[visit_col].isna().any():
        rep.warnings.append(
            f"{int(df[visit_col].isna().sum())} NaN value(s) in {visit_col!r}; "
            "visit is only used to order rows that share an age"
        )

    # --- age -------------------------------------------------------------
    if not pd.api.types.is_numeric_dtype(df[age_col]):
        rep.errors.append(f"age column {age_col!r} is not numeric (dtype {df[age_col].dtype})")
        return rep

    ages = df[age_col]
    n_bad_age = int(ages.isna().sum() + np.isinf(ages.fillna(0)).sum())
    if n_bad_age:
        rep.warnings.append(f"{n_bad_age} NaN/inf age(s); those rows yield no pairs")
    if (ages.dropna() < 0).any():
        rep.errors.append(
            f"{int((ages.dropna() < 0).sum())} negative age(s); ages index the velocity "
            "matrix and a negative index wraps around silently to the wrong correlation"
        )

    valid_ages = ages.dropna()
    valid_ages = valid_ages[np.isfinite(valid_ages)]
    if len(valid_ages):
        rep.stats["age_range"] = (float(valid_ages.min()), float(valid_ages.max()))
        if R is not None:
            n_ages = min(R.shape)
            out = valid_ages[valid_ages.astype(int) >= n_ages]
            if len(out):
                rep.errors.append(
                    f"{len(out)} row(s) have age >= {n_ages}, outside the velocity "
                    f"matrix (shape {R.shape}); those pairs are dropped"
                )

    # --- pair structure --------------------------------------------------
    grouped = df.groupby(id_col)[age_col]
    n_unique_ages = grouped.nunique()
    rep.stats["n_subjects"] = int(len(n_unique_ages))

    n_dup_rows = int(len(df) - n_unique_ages.sum())
    if n_dup_rows > 0:
        rep.warnings.append(
            f"{n_dup_rows} row(s) repeat an age within a subject; only the first "
            "(by visit order) is kept"
        )

    single = n_unique_ages[n_unique_ages < 2]
    if len(single):
        rep.warnings.append(
            f"{len(single)} subject(s) have fewer than 2 distinct ages and contribute no gains"
        )

    rep.stats["n_pairs_expected"] = int((n_unique_ages * (n_unique_ages - 1) // 2).sum())

    # --- correlation floor ----------------------------------------------
    if R is not None and rep.ok:
        n_low = 0
        n_checked = 0
        for _, group in df.groupby(id_col):
            uniq = (group.sort_values([age_col, visit_col])
                         .drop_duplicates(subset=age_col, keep="first"))
            a = uniq[age_col].to_numpy()
            a = a[np.isfinite(a)]
            for i, j in combinations(range(len(a)), 2):
                n_checked += 1
                if R[int(a[j]), int(a[i])] < MIN_CORRELATION:
                    n_low += 1
        rep.stats["n_pairs_below_min_r"] = n_low
        if n_low:
            rep.warnings.append(
                f"{n_low}/{n_checked} pair(s) have r < {MIN_CORRELATION} and will be "
                "dropped (ages too far apart for this velocity matrix)"
            )

    return rep


# --------------------------------------------------------------------------
# construction
# --------------------------------------------------------------------------
def build_zgain_frame(
    z_scores: pd.DataFrame,
    covariates: pd.DataFrame | None,
    measure: str,
    *,
    id_col: str = "ID_subject",
    age_col: str = "age",
    visit_col: str = "ID_visit",
    align: str = "position",
    merge_on: Sequence[str] | None = None,
    keep_cols: Iterable[str] = (),
    validate: bool = True,
) -> pd.DataFrame:
    """Assemble a z-gain input frame from a z-score table and a covariate table.

    Normative-model output (e.g. ``Z_test.csv``) usually carries one column per
    measure but no subject/age/visit columns -- those live in the test-set
    covariate pickle that was fed to the model. This joins the two.

    Parameters
    ----------
    z_scores
        Table containing ``measure`` as a column of z-scores.
    covariates
        Table containing ``id_col``, ``age_col``, ``visit_col``. Pass ``None``
        if ``z_scores`` already has them.
    align
        ``"position"`` concatenates row-by-row on position, which is correct
        **only if both tables are in the same row order as the model input**
        (the usual case: sort ``z_scores`` by its ``observations`` column and
        reset the index on both first). ``"merge"`` does an inner join on
        ``merge_on`` instead, and is safer whenever the keys exist on both sides.
    keep_cols
        Extra covariate columns to carry through (diagnosis, sex, site, ...).
        They are ignored by the computation but useful for grouping the output.
    validate
        Run :func:`validate_zgain_frame` on the result and raise on errors.

    Returns
    -------
    DataFrame with ``[id_col, age_col, visit_col, measure, *keep_cols]``.
    """
    if measure not in z_scores.columns:
        raise KeyError(
            f"measure {measure!r} not in z_scores columns; got {list(z_scores.columns)[:15]}"
        )

    keep_cols = list(keep_cols)

    if covariates is None:
        frame = z_scores.copy()
    elif align == "position":
        if len(z_scores) != len(covariates):
            raise ValueError(
                f"positional alignment needs equal lengths, got {len(z_scores)} "
                f"z-score rows and {len(covariates)} covariate rows; use "
                "align='merge' instead"
            )
        logger.warning(
            "Aligning z-scores and covariates by row position; this is only correct "
            "if both tables are still in the model's input order."
        )
        cov_cols = [c for c in [id_col, age_col, visit_col, *keep_cols] if c in covariates.columns]
        frame = pd.concat(
            [z_scores[[measure]].reset_index(drop=True),
             covariates[cov_cols].reset_index(drop=True)],
            axis=1,
        )
    elif align == "merge":
        keys = list(merge_on) if merge_on else [id_col, visit_col]
        missing = ([k for k in keys if k not in z_scores.columns]
                   + [k for k in keys if k not in covariates.columns])
        if missing:
            raise KeyError(f"merge key(s) {sorted(set(missing))} missing from one of the tables")
        cov_cols = [c for c in [id_col, age_col, visit_col, *keep_cols] if c in covariates.columns]
        frame = z_scores[[*keys, measure]].merge(
            covariates[list(dict.fromkeys([*keys, *cov_cols]))], on=keys, how="inner"
        )
        if frame.empty:
            raise ValueError(f"merge on {keys} produced no rows; check key dtypes match")
    else:
        raise ValueError(f"align must be 'position' or 'merge', got {align!r}")

    required = [id_col, age_col, visit_col, measure]
    missing = [c for c in required if c not in frame.columns]
    if missing:
        raise KeyError(f"assembled frame is missing {missing}")

    frame = frame[required + [c for c in keep_cols if c in frame.columns]]

    if validate:
        validate_zgain_frame(
            frame, measure, id_col=id_col, age_col=age_col, visit_col=visit_col
        ).raise_if_invalid()

    return frame


# --------------------------------------------------------------------------
# runnable example
# --------------------------------------------------------------------------
def example_velocity_matrix(max_age: int = 100, rho: float = 0.99) -> np.ndarray:
    """An AR(1) age-by-age correlation matrix, ``R[i, j] = rho ** |i - j|``.

    Stand-in for the real matrix so the pipeline can be exercised without model
    output. With the default ``rho`` the ``r >= 0.3`` floor bites at roughly a
    120-year gap, so no pair is dropped; lower ``rho`` to see drops happen.
    """
    ages = np.arange(max_age + 1)
    return rho ** np.abs(ages[:, None] - ages[None, :])


def example_frame(
    n_subjects: int = 20,
    n_visits: int = 3,
    *,
    measure: str = "lh_bankssts",
    id_col: str = "ID_subject",
    age_col: str = "age",
    visit_col: str = "ID_visit",
    seed: int = 0,
) -> pd.DataFrame:
    """A small synthetic frame in the expected layout, for testing and docs."""
    rng = np.random.default_rng(seed)
    rows = []
    for s in range(n_subjects):
        age = float(rng.integers(55, 80))
        z = float(rng.normal())
        for v in range(n_visits):
            rows.append({
                id_col: f"sub-{s:03d}",
                age_col: age + 2.0 * v,
                visit_col: v,
                measure: z - 0.1 * v + float(rng.normal(scale=0.2)),
            })
    return pd.DataFrame(rows)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    try:
        from .core import compute_all_z_gains
    except ImportError:
        # Run as a plain script (IDE "run file", ``python data.py``) rather than
        # as ``python -m z_gain.data``: no parent package, so put ``src`` on the
        # path and import absolutely.
        import sys
        sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
        from z_gain.core import compute_all_z_gains

    MEASURE = "Left-Lateral-Ventricle"
    #df = example_frame(measure=MEASURE)
    #R = example_velocity_matrix()
    
    df = z_scores_ADNI
    
    velocity_path = "/project_cephfs/3022017.06/projects/lifespan_hbr/johbay/Velocity/CODE_new/PCNtoolkit/examples/resources/hbr_SHASH/save_dir_SC_all_regions/results/Velocity/"
   
    velocity_data = pd.read_pickle(os.path.join(velocity_path,MEASURE,"velocity_objects.pkl"))
    R = velocity_data['A_sparse_predict']
  
    

    print("Input frame:")
    print(df.head(6).to_string(index=False))
    print()
    print("Validation:")
    print(validate_zgain_frame(df, MEASURE, R=R).summary())
    print()
    gains = compute_all_z_gains(df, R, MEASURE)
    print(f"Output: {len(gains)} gains")
    print(gains.head(6).to_string(index=False))
