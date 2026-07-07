# ============================================================================
# Configuration Module for K-NN Classification Pipeline
# ============================================================================
# Centralizes every tunable parameter so that users can adjust behavior
# without modifying business logic.  All paths are resolved relative to the
# project root (one level above 'src/'), making the config portable.
# ============================================================================

import os
from dataclasses import dataclass, field
from typing import List


@dataclass(frozen=True)
class PathConfig:
    """Immutable path configuration derived from the project layout.

    Attributes:
        project_root: Absolute path to the Classification project directory.
        data_dir:     Absolute path to the 'data/' folder.
        src_dir:      Absolute path to the 'src/' folder.
        output_dir:   Absolute path where results and logs are written.
        dataset_file: Full path to the CSV dataset.
    """

    # Resolve project root as parent of 'src/'
    project_root: str = os.path.abspath(
        os.path.join(os.path.dirname(__file__), os.pardir)
    )
    data_dir: str = ""
    src_dir: str = ""
    output_dir: str = ""
    dataset_file: str = ""

    def __post_init__(self) -> None:
        """Derive dependent paths from project_root after init."""
        # 'frozen=True' requires object.__setattr__ for post-init mutation
        object.__setattr__(
            self, "data_dir", os.path.join(self.project_root, "data")
        )
        object.__setattr__(
            self, "src_dir", os.path.join(self.project_root, "src")
        )
        object.__setattr__(
            self, "output_dir", os.path.join(self.project_root, "output")
        )
        object.__setattr__(
            self,
            "dataset_file",
            os.path.join(self.project_root, "data", "ai_student_impact_dataset (1).csv"),
        )


@dataclass(frozen=True)
class DataConfig:
    """Controls which columns to use and how to split the data.

    Attributes:
        target_column:      Name of the column to predict (classification label).
        drop_columns:       Columns excluded from features (identifiers, leakage).
        test_size:          Fraction of data reserved for test evaluation.
        random_state:       Seed for reproducible train/test splits.
        stratify:           Whether to use stratified splitting on the target.
        max_samples:        Maximum number of samples to use (None for all).
    """

    target_column: str = "Burnout_Risk_Level"

    # 'Student_ID' is an identifier (no predictive value).
    # 'Post_Semester_GPA' and 'Skill_Retention_Score' are post-outcome
    # measurements that would cause data leakage if used as features.
    drop_columns: List[str] = field(
        default_factory=lambda: [
            "Student_ID",
            "Post_Semester_GPA",
            "Skill_Retention_Score",
        ]
    )

    test_size: float = 0.20
    random_state: int = 42
    stratify: bool = True
    max_samples: int = None


@dataclass(frozen=True)
class ModelConfig:
    """Hyperparameters for the K-Nearest Neighbors classifier.

    Attributes:
        n_neighbors:   Number of neighbors (K) to consider for voting.
        weights:       Weight function: 'uniform' or 'distance'.
        metric:        Distance metric for neighbor computation.
        algorithm:     Algorithm used to compute nearest neighbors.
        n_jobs:        Number of parallel jobs (1 or None avoids Windows multiprocessing overhead).
        device:        Target execution device: 'cpu', 'gpu', or 'auto'.
    """

    n_neighbors: int = 7
    weights: str = "distance"
    metric: str = "minkowski"
    algorithm: str = "auto"
    n_jobs: int = 1
    device: str = "auto"


@dataclass(frozen=True)
class LoggingConfig:
    """Controls the logging behavior of the pipeline.

    Attributes:
        log_level:     Minimum severity level (DEBUG, INFO, WARNING, etc.).
        log_to_file:   Whether to persist logs to disk in the output directory.
        log_filename:  Name of the log file written to the output directory.
    """

    log_level: str = "DEBUG"
    log_to_file: bool = True
    log_filename: str = "knn_classification.log"


@dataclass(frozen=True)
class PipelineConfig:
    """Top-level aggregation of all sub-configurations.

    Usage:
        config = PipelineConfig()       # all defaults
        config = PipelineConfig(        # override specific knobs
            model=ModelConfig(n_neighbors=5)
        )
    """

    paths: PathConfig = field(default_factory=PathConfig)
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)
