"""Command-line entry point: ``z-gain --data-path ... --velocity-matrix-path ...``"""
from __future__ import annotations

import argparse
from typing import Optional, Sequence

from .config import ZGainConfig
from .pipeline import run_zgain_analysis


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Compute normative-model z-gains for a longitudinal dataset."
    )
    parser.add_argument("--data-path", required=True, help="CSV or pickle file with subject/age/visit/measure columns.")
    parser.add_argument("--velocity-matrix-path", required=True, help="Pickle with the age-by-age predicted correlation matrix.")
    parser.add_argument("--output-dir", required=True, help="Directory to write gains to.")
    parser.add_argument("--measure", required=True, help="Name of the column to compute z-gains for.")
    parser.add_argument("--id-col", default="ID_subject")
    parser.add_argument("--age-col", default="age")
    parser.add_argument("--visit-col", default="ID_visit")
    parser.add_argument("--dataset-name", default="dataset")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = build_parser().parse_args(argv)
    config = ZGainConfig(
        data_path=args.data_path,
        velocity_matrix_path=args.velocity_matrix_path,
        output_dir=args.output_dir,
        measure=args.measure,
        id_col=args.id_col,
        age_col=args.age_col,
        visit_col=args.visit_col,
        dataset_name=args.dataset_name,
    )
    results = run_zgain_analysis(config)
    print(f"Wrote z-gains to {results['gains_path']}")


if __name__ == "__main__":
    main()
