#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Compute z-gains on your own longitudinal dataset.

A z-gain measures how much a subject's normative-model z-score changed between
two timepoints, corrected for how strongly z-scores at those two ages are
expected to correlate anyway. Without that correction, a subject who simply
regresses toward the mean looks like they changed.

----------------------------------------------------------------------------
WHAT YOU NEED
----------------------------------------------------------------------------
1. A table of z-scores, CSV or pickle, in LONG format: one row per subject per
   visit. It needs four columns (names are configurable below):

       ID_subject   subject identifier, repeated across that subject's visits
       age          age at scan, numeric. NOTE: this is used as an INTEGER
                    index into the velocity matrix, so ages are effectively
                    rounded down to whole years
       ID_visit     visit identifier; only used to order rows sharing an age
       <measure>    the Z-SCORE for the measure, not the raw volume/thickness

   Any other columns (diagnosis, sex, site) ride along and are ignored.

       ID_subject  age  ID_visit  Right-Hippocampus
         sub-0001   65         1              -1.88
         sub-0001   66         2              -1.70
         sub-0001   68         3              -1.86
         sub-0002   71         1               0.42

2. A velocity matrix: an age-by-age correlation matrix saying how strongly
   z-scores at age i and age j are expected to correlate. A pickle holding
   either the array itself, or a dict with an "A_sparse_predict" key. Supply
   one file for all measures, or a directory laid out as
   <dir>/<measure>/velocity_objects.pkl for a per-measure matrix.

   This comes from fitting the velocity model to your normative model. If you
   do not have one yet, start with --demo to see the pipeline work on
   synthetic data.

----------------------------------------------------------------------------
INSTALL
----------------------------------------------------------------------------
    pip install -e zgain

----------------------------------------------------------------------------
RUN
----------------------------------------------------------------------------
    # 1. check your install with synthetic data, no inputs needed
    python run_velocity_template.py --demo

    # 2. see what your data looks like to the pipeline, without computing
    python run_velocity_template.py --data my_z.csv --velocity R.pkl \
        --measure Right-Hippocampus --validate-only

    # 3. compute one measure
    python run_velocity_template.py --data my_z.csv --velocity R.pkl \
        --measure Right-Hippocampus --output-dir results/

    # 4. compute several
    python run_velocity_template.py --data my_z.csv --velocity matrices/ \
        --measure Right-Hippocampus Left-Amygdala --output-dir results/

Or edit the CONFIG block below and run the file with no arguments.
Output: <output-dir>/z_gains_<measure>.pkl, one row per pair of timepoints,
with columns id, age1, age2, t1_index, t2_index, z_gain, Z1, Z2.
"""
# %% 1. CONFIG -- edit these, or override them with the command line
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import pandas as pd

# Path to your z-score table (.csv, .pkl). None means "require --data".
DATA_PATH: str | Path | None = None

# Path to your velocity matrix: a .pkl file, or a directory containing
# <measure>/velocity_objects.pkl per measure.
VELOCITY_PATH: str | Path | None = None

# Which measure column(s) to analyse. None means every numeric column that is
# not one of the id/age/visit columns.
MEASURES: list[str] | None = None

# Where to write results. None means compute but do not write.
OUTPUT_DIR: str | Path | None = "z_gain_results"

# Your column names, if they differ from the defaults.
ID_COL = "ID_subject"
AGE_COL = "age"
VISIT_COL = "ID_visit"

# ---------------------------------------------------------------------------

# Lets you run this file straight from the repo without installing first.
_SRC = Path(__file__).resolve().parent / "zgain" / "src"
if _SRC.is_dir() and str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from z_gain import (  # noqa: E402
    build_zgain_frame,
    compute_all_z_gains,
    example_frame,
    example_velocity_matrix,
    load_input_data,
    load_velocity_matrix,
    validate_zgain_frame,
)

log = logging.getLogger("run_velocity")


# %% 2. loading your data
def load_data(path: str | Path) -> pd.DataFrame:
    """Load your z-score table.

    Reads .csv and .pkl out of the box. If your data needs assembling first --
    joining model output to a covariate table, merging in diagnosis labels,
    renaming columns -- do it here and return the finished long-format frame.
    Everything downstream only cares about the four required columns.
    """
    return load_input_data(path)


def resolve_velocity(path: str | Path, measure: str):
    """Load the velocity matrix for one measure.

    Accepts a single file used for every measure, or a directory laid out as
    <dir>/<measure>/velocity_objects.pkl.
    """
    path = Path(path)
    if path.is_dir():
        candidate = path / measure / "velocity_objects.pkl"
        if not candidate.exists():
            raise FileNotFoundError(
                f"No velocity matrix for {measure!r} at {candidate}. Expected "
                f"<dir>/<measure>/velocity_objects.pkl, or pass a single .pkl file."
            )
        path = candidate
    return load_velocity_matrix(path)


def infer_measures(df: pd.DataFrame, id_col: str = ID_COL, age_col: str = AGE_COL,
                   visit_col: str = VISIT_COL) -> list[str]:
    """Every numeric column that is not an id, age or visit column."""
    skip = {id_col, age_col, visit_col}
    return [c for c in df.columns
            if c not in skip and pd.api.types.is_numeric_dtype(df[c])]


# %% 3. the run
def run(df: pd.DataFrame, velocity_path: str | Path, measures: list[str],
        output_dir: str | Path | None = None, validate_only: bool = False,
        *, id_col: str = ID_COL, age_col: str = AGE_COL, visit_col: str = VISIT_COL) -> dict:
    """Compute z-gains for each measure. Returns {measure: gains DataFrame}."""
    cols = dict(id_col=id_col, age_col=age_col, visit_col=visit_col)
    results = {}

    for measure in measures:
        try:
            R = resolve_velocity(velocity_path, measure)

            # Select the four required columns and fail loudly if any are missing.
            frame = build_zgain_frame(df, None, measure, validate=False, **cols)

            # Say up front what will be dropped. Pairs with a NaN or
            # out-of-range age, or an age-to-age correlation below 0.3, are
            # skipped silently by the computation itself.
            report = validate_zgain_frame(frame, measure, R=R, **cols)
            print(f"\n=== {measure} ===")
            print(report.summary())
            if not report.ok:
                print("  -> skipped, see errors above")
                continue
            if validate_only:
                continue

            gains = compute_all_z_gains(frame, R, measure, **cols)
            results[measure] = gains
            print(f"  -> {len(gains)} z-gains")

            if output_dir is not None:
                out = Path(output_dir)
                out.mkdir(parents=True, exist_ok=True)
                dest = out / f"z_gains_{measure}.pkl"
                gains.to_pickle(dest)
                print(f"  -> wrote {dest}")
        except Exception as exc:
            print(f"\n=== {measure} ===")
            print(f"  -> FAILED, {type(exc).__name__}: {exc}")

    return results


def run_demo() -> None:
    """Exercise the whole pipeline on synthetic data, so you can check the
    install and see the expected input and output shapes."""
    measure = "example_measure"
    cols = dict(id_col=ID_COL, age_col=AGE_COL, visit_col=VISIT_COL)
    df = example_frame(measure=measure, **cols)
    R = example_velocity_matrix()

    print("Synthetic input -- your table should look like this:\n")
    print(df.head(6).to_string(index=False))
    print("\nValidation report:")
    print(validate_zgain_frame(df, measure, R=R, **cols).summary())
    gains = compute_all_z_gains(df, R, measure, **cols)
    print(f"\nOutput -- {len(gains)} z-gains:\n")
    print(gains.head(6).to_string(index=False))
    print("\nNow rerun with --data and --velocity pointing at your own files.")


# %% 4. command line
def main(argv=None) -> int:
    p = argparse.ArgumentParser(
        description="Compute z-gains on a longitudinal dataset.",
        epilog="Run with --demo first to check the install.",
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data", default=DATA_PATH, help="z-score table (.csv or .pkl)")
    p.add_argument("--velocity", default=VELOCITY_PATH,
                   help="velocity matrix .pkl, or a directory of <measure>/velocity_objects.pkl")
    p.add_argument("--measure", nargs="*", default=MEASURES,
                   help="measure column(s); default is every numeric non-id column")
    p.add_argument("--output-dir", default=OUTPUT_DIR, help="where to write results")
    p.add_argument("--id-col", default=ID_COL)
    p.add_argument("--age-col", default=AGE_COL)
    p.add_argument("--visit-col", default=VISIT_COL)
    p.add_argument("--validate-only", action="store_true",
                   help="report what would be dropped, without computing or writing")
    p.add_argument("--demo", action="store_true", help="run on synthetic data and exit")
    args = p.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    if args.demo:
        run_demo()
        return 0

    cols = dict(id_col=args.id_col, age_col=args.age_col, visit_col=args.visit_col)

    if not args.data or not args.velocity:
        p.error("--data and --velocity are required (or set them in the CONFIG block). "
                "Try --demo to see the pipeline run on synthetic data.")

    df = load_data(args.data)

    missing = [c for c in (args.id_col, args.age_col, args.visit_col) if c not in df.columns]
    if missing:
        p.error(f"column(s) {missing} not found in {args.data}.\n"
                f"  Your columns are: {list(df.columns)}\n"
                f"  Set --id-col / --age-col / --visit-col to match, or rename them in load_data().")
    print(f"Loaded {len(df)} rows, {df[args.id_col].nunique()} subjects from {args.data}")

    measures = args.measure or infer_measures(df, **cols)
    if not measures:
        p.error(f"No measure columns found. Columns are: {list(df.columns)}")
    print(f"Measures to process: {len(measures)}")

    results = run(df, args.velocity, measures, args.output_dir, args.validate_only, **cols)
    return 0 if (results or args.validate_only) else 1


if __name__ == "__main__":
    sys.exit(main())
