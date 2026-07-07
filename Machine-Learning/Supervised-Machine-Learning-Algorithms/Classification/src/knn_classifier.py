# ============================================================================
# K-NN Classifier Service for Classification Pipeline
# ============================================================================
# Owns the entire model lifecycle: training, prediction, evaluation, and
# structured result reporting.  Follows the Service Layer pattern so that
# model logic is isolated from data preparation and orchestration.
# ============================================================================

import logging
import os
import time
from typing import List

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.neighbors import KNeighborsClassifier

from config import ModelConfig, PathConfig


class KNNClassifierService:
    """Service encapsulating K-Nearest Neighbors classification.

    Responsibilities:
        1. Train a KNeighborsClassifier with configurable hyperparameters.
        2. Generate predictions on unseen test data.
        3. Compute and log a comprehensive evaluation report.
        4. Persist the structured results summary to disk.

    Design decision -- Why wrap sklearn in a service class?
    Wrapping allows us to inject configuration and logging uniformly,
    enforce a consistent reporting format, and make the model easily
    swappable (Open/Closed Principle) without changing the caller.
    """

    def __init__(
        self,
        model_config: ModelConfig,
        path_config: PathConfig,
        label_names: List[str],
        feature_names: List[str],
        logger: logging.Logger,
    ) -> None:
        """Initialize the classifier service.

        Args:
            model_config:  KNN hyperparameters (K, weights, metric, etc.).
            path_config:   Path settings for saving output artifacts.
            label_names:   Human-readable class labels in encoder order.
            feature_names: Feature column names for reporting.
            logger:        Logger instance for progress and results.
        """
        self._model_config = model_config
        self._path_config = path_config
        self._label_names = label_names
        self._feature_names = feature_names
        self._logger = logger
        self._model: KNeighborsClassifier = self._build_model()

    # -- Public workflow methods ---------------------------------------------

    def train(self, x_train: np.ndarray, y_train: np.ndarray) -> None:
        """Fit the K-NN model on the training data.

        Args:
            x_train: Training feature matrix (n_samples, n_features).
            y_train: Training target vector (n_samples,).
        """
        self._logger.info("=" * 70)
        self._logger.info("MODEL TRAINING")
        self._logger.info("=" * 70)
        self._log_hyperparameters()

        self._logger.info(
            "Training on %d samples with %d features...",
            x_train.shape[0],
            x_train.shape[1],
        )

        start_time = time.perf_counter()
        self._model.fit(x_train, y_train)
        elapsed = time.perf_counter() - start_time

        self._logger.info(
            "Training completed in %.3f seconds.", elapsed
        )

    def evaluate(
        self, x_test: np.ndarray, y_test: np.ndarray
    ) -> dict:
        """Predict on the test set and compute evaluation metrics.

        Args:
            x_test: Test feature matrix.
            y_test: True test labels.

        Returns:
            Dictionary with keys:
              - accuracy, precision_macro, recall_macro, f1_macro
              - confusion_matrix (2D list)
              - classification_report (formatted string)
              - predictions (array of predicted labels)
        """
        self._logger.info("=" * 70)
        self._logger.info("MODEL EVALUATION")
        self._logger.info("=" * 70)

        start_time = time.perf_counter()
        predictions = self._model.predict(x_test)
        predict_elapsed = time.perf_counter() - start_time
        self._logger.info(
            "Prediction on %d test samples completed in %.3f seconds.",
            x_test.shape[0],
            predict_elapsed,
        )

        # Compute metrics
        accuracy = accuracy_score(y_test, predictions)
        precision = precision_score(
            y_test, predictions, average="macro", zero_division=0
        )
        recall = recall_score(
            y_test, predictions, average="macro", zero_division=0
        )
        f1 = f1_score(
            y_test, predictions, average="macro", zero_division=0
        )
        conf_matrix = confusion_matrix(y_test, predictions)
        class_report = classification_report(
            y_test,
            predictions,
            target_names=self._label_names,
            zero_division=0,
        )

        # Build results dictionary
        results = {
            "accuracy": accuracy,
            "precision_macro": precision,
            "recall_macro": recall,
            "f1_macro": f1,
            "confusion_matrix": conf_matrix.tolist(),
            "classification_report": class_report,
            "predictions": predictions,
        }

        self._log_evaluation(results)
        self._save_results(results)

        return results

    # -- Private helpers -----------------------------------------------------

    def _build_model(self) -> KNeighborsClassifier:
        """Construct the KNN estimator from the model configuration.

        Returns:
            An unfitted KNeighborsClassifier instance.
        """
        return KNeighborsClassifier(
            n_neighbors=self._model_config.n_neighbors,
            weights=self._model_config.weights,
            metric=self._model_config.metric,
            algorithm=self._model_config.algorithm,
            n_jobs=self._model_config.n_jobs,
        )

    def _log_hyperparameters(self) -> None:
        """Log all model hyperparameters for reproducibility."""
        self._logger.info("K-NN Hyperparameters:")
        self._logger.info("  n_neighbors : %d", self._model_config.n_neighbors)
        self._logger.info("  weights     : %s", self._model_config.weights)
        self._logger.info("  metric      : %s", self._model_config.metric)
        self._logger.info("  algorithm   : %s", self._model_config.algorithm)
        self._logger.info("  n_jobs      : %d", self._model_config.n_jobs)

    def _log_evaluation(self, results: dict) -> None:
        """Log evaluation metrics in a structured, human-readable format.

        Args:
            results: Dictionary produced by the evaluate() method.
        """
        self._logger.info("-" * 70)
        self._logger.info("OVERALL METRICS:")
        self._logger.info(
            "  Accuracy         : %.4f  (%.2f%%)",
            results["accuracy"],
            results["accuracy"] * 100,
        )
        self._logger.info(
            "  Precision (macro): %.4f",
            results["precision_macro"],
        )
        self._logger.info(
            "  Recall    (macro): %.4f",
            results["recall_macro"],
        )
        self._logger.info(
            "  F1-Score  (macro): %.4f",
            results["f1_macro"],
        )

        # Confusion matrix
        self._logger.info("-" * 70)
        self._logger.info("CONFUSION MATRIX:")
        # Header row
        header = "%-15s" % "Actual\\Pred"
        for name in self._label_names:
            header += "  %-10s" % name
        self._logger.info(header)

        for idx, row in enumerate(results["confusion_matrix"]):
            row_str = "%-15s" % self._label_names[idx]
            for val in row:
                row_str += "  %-10d" % val
            self._logger.info(row_str)

        # Per-class report
        self._logger.info("-" * 70)
        self._logger.info("PER-CLASS CLASSIFICATION REPORT:")
        for line in results["classification_report"].split("\n"):
            if line.strip():
                self._logger.info("  %s", line)
        self._logger.info("=" * 70)

    def _save_results(self, results: dict) -> None:
        """Persist the evaluation summary to a text file in the output dir.

        The file is formatted for easy reading by humans and can also be
        parsed by downstream scripts for experiment tracking.

        Args:
            results: Dictionary produced by the evaluate() method.
        """
        os.makedirs(self._path_config.output_dir, exist_ok=True)
        output_path = os.path.join(
            self._path_config.output_dir, "classification_results.txt"
        )

        with open(output_path, "w", encoding="utf-8") as fh:
            fh.write("=" * 70 + "\n")
            fh.write("K-NN CLASSIFICATION RESULTS\n")
            fh.write("=" * 70 + "\n\n")

            fh.write("HYPERPARAMETERS\n")
            fh.write("-" * 40 + "\n")
            fh.write(f"  n_neighbors : {self._model_config.n_neighbors}\n")
            fh.write(f"  weights     : {self._model_config.weights}\n")
            fh.write(f"  metric      : {self._model_config.metric}\n")
            fh.write(f"  algorithm   : {self._model_config.algorithm}\n\n")

            fh.write("FEATURES USED\n")
            fh.write("-" * 40 + "\n")
            for i, name in enumerate(self._feature_names, start=1):
                fh.write(f"  {i:2d}. {name}\n")
            fh.write("\n")

            fh.write("OVERALL METRICS\n")
            fh.write("-" * 40 + "\n")
            fh.write(
                f"  Accuracy         : {results['accuracy']:.4f}  "
                f"({results['accuracy'] * 100:.2f}%)\n"
            )
            fh.write(
                f"  Precision (macro): {results['precision_macro']:.4f}\n"
            )
            fh.write(
                f"  Recall    (macro): {results['recall_macro']:.4f}\n"
            )
            fh.write(
                f"  F1-Score  (macro): {results['f1_macro']:.4f}\n\n"
            )

            fh.write("CONFUSION MATRIX\n")
            fh.write("-" * 40 + "\n")
            header = f"{'Actual\\Pred':<15}"
            for name in self._label_names:
                header += f"  {name:<10}"
            fh.write(header + "\n")
            for idx, row in enumerate(results["confusion_matrix"]):
                row_str = f"{self._label_names[idx]:<15}"
                for val in row:
                    row_str += f"  {val:<10}"
                fh.write(row_str + "\n")
            fh.write("\n")

            fh.write("PER-CLASS CLASSIFICATION REPORT\n")
            fh.write("-" * 40 + "\n")
            fh.write(results["classification_report"])
            fh.write("\n" + "=" * 70 + "\n")

        self._logger.info("Results saved to: %s", output_path)
