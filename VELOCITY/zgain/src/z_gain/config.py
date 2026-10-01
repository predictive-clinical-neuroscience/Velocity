"""Configuration for a z-gain analysis run."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass
class ZGainConfig:
    """All dataset-specific knobs for :func:`z_gain.pipeline.run_zgain_analysis`.

    ``data_path`` must point to a CSV or pickle file that can be loaded into a
    DataFrame containing, at minimum: a subject id column, an age column, a
    visit id column, and the measure column to analyze.
    """

    data_path: Path
    velocity_matrix_path: Path
    output_dir: Path
    measure: str

    id_col: str = "ID_subject"
    age_col: str = "age"
    visit_col: str = "ID_visit"

    dataset_name: str = "dataset"

    def __post_init__(self) -> None:
        self.data_path = Path(self.data_path)
        self.velocity_matrix_path = Path(self.velocity_matrix_path)
        self.output_dir = Path(self.output_dir)
