# ============================================================================
# Data Loader Module for K-NN Classification Pipeline
# ============================================================================
# Handles all data I/O, validation, exploration, and preprocessing.
# Follows the Service Layer pattern: the DataLoaderService class owns the
# entire data preparation lifecycle and exposes a clean interface to callers.
# ============================================================================

import logging
from typing import Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler

from config import DataConfig, PathConfig


class DataLoaderService:
    """Service responsible for loading, validating, and preprocessing the dataset.

    Responsibilities:
        1. Read the CSV from disk and validate its schema.
        2. Log an exploratory summary (shape, types, nulls, class distribution).
        3. Encode categorical features and scale numeric features.
        4. Split into train/test sets with optional stratification.

    Design decision -- Why return numpy arrays instead of DataFrames?
    scikit-learn estimators work natively with numpy arrays; converting
    early avoids repeated implicit conversions inside the model pipeline.
    """

    def __init__(
        self,
        path_config: PathConfig,
        data_config: DataConfig,
        logger: logging.Logger,
    ) -> None:
        """Initialize the data loader with configuration and logger.

        Args:
            path_config: Path settings for locating the dataset file.
            data_config: Data settings for columns, splits, and target.
            logger:      Logger instance for detailed progress reporting.
        """
        self._path_config = path_config
        self._data_config = data_config
        self._logger = logger

        # These are populated during preprocessing and exposed via properties
        # so downstream consumers (e.g., visualization) can map indices back
        # to human-readable feature names and label names.
        self._feature_names: list = []
        self._label_names: list = []
        self._label_encoder: LabelEncoder = LabelEncoder()
        self._scaler: StandardScaler = StandardScaler()

    # -- Public properties ---------------------------------------------------

    @property
    def feature_names(self) -> list:
        """Column names of the final feature matrix (post-encoding)."""
        return self._feature_names

    @property
    def label_names(self) -> list:
        """Ordered class labels produced by the label encoder."""
        return self._label_names

    # -- Public workflow methods ---------------------------------------------

    def load_and_prepare(
        self,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Execute the full data pipeline: load -> validate -> preprocess -> split.

        Returns:
            A tuple of (X_train, X_test, y_train, y_test) as numpy arrays.

        Raises:
            FileNotFoundError:  If the dataset CSV does not exist.
            KeyError:           If expected columns are missing from the CSV.
            ValueError:         If the dataset is empty after loading.
        """
        dataframe = self._load_csv()
        self._validate_schema(dataframe)

        # Why downsample here?
        # Downsampling the dataset early in the pipeline (right after validation)
        # reduces downstream processing time for summarize, preprocess, and
        # model training, facilitating fast iteration during development.
        dataframe = self._downsample(dataframe)

        self._log_data_summary(dataframe)
        features, target = self._preprocess(dataframe)
        return self._split(features, target)

    # -- Private implementation methods --------------------------------------

    def _load_csv(self) -> pd.DataFrame:
        """Read the CSV file and perform basic integrity checks.

        Returns:
            Raw pandas DataFrame.

        Raises:
            FileNotFoundError: If the configured dataset path is invalid.
            ValueError:        If the file is empty.
        """
        self._logger.info(
            "Loading dataset from: %s", self._path_config.dataset_file
        )
        dataframe = pd.read_csv(self._path_config.dataset_file)

        if dataframe.empty:
            raise ValueError(
                f"Dataset is empty: {self._path_config.dataset_file}"
            )

        self._logger.info(
            "Dataset loaded successfully -- shape: %s", dataframe.shape
        )
        return dataframe

    def _validate_schema(self, dataframe: pd.DataFrame) -> None:
        """Ensure the target column and droppable columns exist in the data.

        Args:
            dataframe: The raw DataFrame to validate.

        Raises:
            KeyError: If the target column is missing.
            KeyError: If any column listed in 'drop_columns' is missing.
        """
        if self._data_config.target_column not in dataframe.columns:
            raise KeyError(
                f"Target column '{self._data_config.target_column}' "
                f"not found in dataset. Available columns: "
                f"{list(dataframe.columns)}"
            )

        missing_drops = [
            col
            for col in self._data_config.drop_columns
            if col not in dataframe.columns
        ]
        if missing_drops:
            raise KeyError(
                f"Columns configured for dropping are absent: {missing_drops}"
            )

        self._logger.info("Schema validation passed.")

    def _downsample(self, dataframe: pd.DataFrame) -> pd.DataFrame:
        """Perform stratified downsampling to limit dataset size if configured.

        Args:
            dataframe: The raw DataFrame to downsample.

        Returns:
            A downsampled DataFrame with balanced class representation.
        """
        max_samples = self._data_config.max_samples
        if max_samples is None or max_samples >= len(dataframe):
            return dataframe

        self._logger.info(
            "Downsampling dataset from %d to %d samples...",
            len(dataframe),
            max_samples,
        )

        # Why perform stratified downsampling?
        # Stratification ensures that the relative class frequencies in the target
        # column remain consistent after downsampling. This prevents introducing
        # class imbalance bias that could skew classifier evaluation metrics.
        target = self._data_config.target_column
        fraction = max_samples / len(dataframe)

        downsampled = dataframe.groupby(target, group_keys=False).apply(
            lambda x: x.sample(
                n=int(np.round(len(x) * fraction)),
                random_state=self._data_config.random_state
            )
        )

        self._logger.info(
            "Downsampling complete. New dataset shape: %s", downsampled.shape
        )
        return downsampled

    def _log_data_summary(self, dataframe: pd.DataFrame) -> None:
        """Log a detailed exploratory overview of the dataset.

        Includes shape, column types, missing value counts, and the
        target class distribution -- useful for diagnosing class imbalance.

        Args:
            dataframe: The raw DataFrame to summarize.
        """
        self._logger.info("=" * 70)
        self._logger.info("DATASET SUMMARY")
        self._logger.info("=" * 70)
        self._logger.info("Rows: %d  |  Columns: %d", *dataframe.shape)

        # Column-level details
        self._logger.info("-" * 70)
        self._logger.info("%-35s  %-15s  %s", "Column", "Dtype", "Nulls")
        self._logger.info("-" * 70)
        for col in dataframe.columns:
            null_count = dataframe[col].isna().sum()
            self._logger.info(
                "%-35s  %-15s  %d", col, str(dataframe[col].dtype), null_count
            )

        # Target distribution (critical for classification diagnostics)
        self._logger.info("-" * 70)
        self._logger.info("TARGET CLASS DISTRIBUTION ('%s'):", self._data_config.target_column)
        target_counts = dataframe[self._data_config.target_column].value_counts()
        total = len(dataframe)
        for label, count in target_counts.items():
            pct = (count / total) * 100
            self._logger.info("  %-15s : %6d  (%.1f%%)", label, count, pct)
        self._logger.info("=" * 70)

    def _preprocess(
        self, dataframe: pd.DataFrame
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Transform raw data into model-ready numeric arrays.

        Steps:
            1. Separate target from features.
            2. Drop non-predictive columns (identifiers, leakage columns).
            3. Handle missing values with median/mode imputation.
            4. Encode the target labels to integers.
            5. Encode categorical features via LabelEncoder per column.
            6. Standardize all features (zero mean, unit variance) because
               K-NN is distance-based and sensitive to feature scale.

        Args:
            dataframe: Validated raw DataFrame.

        Returns:
            Tuple of (features_array, target_array).
        """
        self._logger.info("Starting preprocessing pipeline...")

        # Step 1 & 2: Separate and drop
        target_series = dataframe[self._data_config.target_column].copy()
        features_df = dataframe.drop(
            columns=[self._data_config.target_column] + self._data_config.drop_columns
        )
        self._logger.info(
            "Dropped columns: %s",
            [self._data_config.target_column] + self._data_config.drop_columns,
        )
        self._logger.info("Feature matrix shape: %s", features_df.shape)

        # Step 3: Impute missing values
        numeric_cols = features_df.select_dtypes(include=["number"]).columns.tolist()
        categorical_cols = features_df.select_dtypes(exclude=["number"]).columns.tolist()

        for col in numeric_cols:
            null_count = features_df[col].isna().sum()
            if null_count > 0:
                median_val = features_df[col].median()
                features_df[col] = features_df[col].fillna(median_val)
                self._logger.info(
                    "Imputed %d nulls in '%s' with median=%.4f",
                    null_count, col, median_val,
                )

        for col in categorical_cols:
            null_count = features_df[col].isna().sum()
            if null_count > 0:
                mode_val = features_df[col].mode()[0]
                features_df[col] = features_df[col].fillna(mode_val)
                self._logger.info(
                    "Imputed %d nulls in '%s' with mode='%s'",
                    null_count, col, mode_val,
                )

        # Step 4: Encode target labels
        target_encoded = self._label_encoder.fit_transform(target_series)
        self._label_names = list(self._label_encoder.classes_)
        self._logger.info("Target labels encoded: %s", self._label_names)

        # Step 5: Encode categorical features
        # Each categorical column gets its own LabelEncoder.
        # We log the mapping for auditability.
        self._logger.info("Encoding categorical features...")
        for col in categorical_cols:
            col_encoder = LabelEncoder()
            features_df[col] = col_encoder.fit_transform(features_df[col])
            unique_map = dict(
                zip(
                    col_encoder.classes_,
                    col_encoder.transform(col_encoder.classes_),
                )
            )
            self._logger.debug(
                "  %-30s -> %s", col, unique_map
            )

        self._feature_names = features_df.columns.tolist()
        self._logger.info("Final feature columns: %s", self._feature_names)

        # Step 6: Standardize
        # K-NN computes distances; features on different scales would
        # dominate the distance metric if left unscaled.
        features_scaled = self._scaler.fit_transform(features_df.values)
        self._logger.info(
            "Features standardized (mean~0, std~1). Shape: %s",
            features_scaled.shape,
        )

        return features_scaled, target_encoded

    def _split(
        self, features: np.ndarray, target: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Partition data into training and test sets.

        Uses stratified splitting by default so that each class is
        proportionally represented in both subsets -- important for
        imbalanced classification problems.

        Args:
            features: Preprocessed feature matrix.
            target:   Encoded target vector.

        Returns:
            Tuple of (X_train, X_test, y_train, y_test).
        """
        stratify_arg = target if self._data_config.stratify else None

        x_train, x_test, y_train, y_test = train_test_split(
            features,
            target,
            test_size=self._data_config.test_size,
            random_state=self._data_config.random_state,
            stratify=stratify_arg,
        )

        self._logger.info(
            "Train/Test split complete -- "
            "Train: %d samples | Test: %d samples | Test ratio: %.0f%%",
            x_train.shape[0],
            x_test.shape[0],
            self._data_config.test_size * 100,
        )

        return x_train, x_test, y_train, y_test
