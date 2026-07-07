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
from typing import List, Union

# Set non-interactive backend for matplotlib to avoid GUI initialization issues
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
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

# Why import cuml dynamically?
# RAPIDS cuML requires NVIDIA GPUs and specific CUDA setups. Importing it inside
# a try-except block allows the script to remain portable on CPU-only machines.
try:
    from cuml.neighbors import KNeighborsClassifier as GPUKNeighborsClassifier
    _HAS_GPU_KNN = True
except ImportError:
    _HAS_GPU_KNN = False


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
        self._generate_plots(results, y_test)
        self._save_analysis(results, y_test)

        return results

    # -- Private helpers -----------------------------------------------------

    def _build_model(self) -> Union[KNeighborsClassifier, 'GPUKNeighborsClassifier']:
        """Construct the KNN estimator from the model configuration.

        Checks the configured device parameter and the availability of
        GPU-accelerated RAPIDS cuML. If GPU execution is requested and
        available, it initializes the cuML KNN classifier. Otherwise, it falls
        back to scikit-learn's CPU implementation.

        Returns:
            An unfitted estimator (KNeighborsClassifier or GPUKNeighborsClassifier).
        """
        device = self._model_config.device.lower()
        use_gpu = False

        if device == "gpu":
            if _HAS_GPU_KNN:
                use_gpu = True
            else:
                self._logger.warning(
                    "GPU execution requested, but RAPIDS cuML is not available. "
                    "Falling back to CPU execution."
                )
        elif device == "auto":
            if _HAS_GPU_KNN:
                use_gpu = True
                self._logger.info("GPU detected and selected for K-NN acceleration.")
            else:
                self._logger.info("Using CPU execution (cuML not available or no GPU).")
        else:
            self._logger.info("Using CPU execution as configured.")

        # Why select between CPU and GPU dynamically?
        # This provides maximum performance where GPUs and cuML are available,
        # while maintaining portability on developer workstations without GPU setups.
        if use_gpu:
            self._logger.info("Instantiating GPU K-NN Classifier (RAPIDS cuML).")
            return GPUKNeighborsClassifier(
                n_neighbors=self._model_config.n_neighbors,
                weights=self._model_config.weights,
                metric=self._model_config.metric,
                algorithm=self._model_config.algorithm,
            )

        self._logger.info("Instantiating CPU K-NN Classifier (scikit-learn).")
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
            col_header = 'Actual\\Pred'
            header = f"{col_header:<15}"
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

    def _generate_plots(self, results: dict, y_test: np.ndarray) -> None:
        """Generate and save the Confusion Matrix heatmap and metrics bar chart.

        Args:
            results: Dictionary containing the predictions and evaluation metrics.
            y_test:  True labels for comparison and per-class calculation.
        """
        self._logger.info("Generating evaluation plots...")
        os.makedirs(self._path_config.output_dir, exist_ok=True)

        # -------------------------------------------------------------
        # Plot 1: Confusion Matrix Heatmap
        # -------------------------------------------------------------
        # Why show confusion matrix as a heatmap?
        # Visualizing the true positives, false positives, and false negatives in a
        # structured color-mapped grid makes it easy to spot which classes are
        # most frequently confused (e.g. Moderate vs High).
        conf_matrix = np.array(results["confusion_matrix"])
        plt.figure(figsize=(8, 6))
        sns.heatmap(
            conf_matrix,
            annot=True,
            fmt="d",
            cmap="Blues",
            xticklabels=self._label_names,
            yticklabels=self._label_names,
        )
        plt.xlabel("Predicted Labels")
        plt.ylabel("Actual Labels")
        plt.title("Confusion Matrix Heatmap")
        plt.tight_layout()
        cm_path = os.path.join(self._path_config.output_dir, "confusion_matrix.png")
        plt.savefig(cm_path, dpi=300)
        plt.close()
        self._logger.info("Confusion matrix heatmap saved to: %s", cm_path)

        # -------------------------------------------------------------
        # Plot 2: Per-Class Metrics Bar Chart
        # -------------------------------------------------------------
        # Why visualize metrics as grouped bars?
        # A grouped bar chart allows quick visual comparison of precision, recall,
        # and F1-score across different target levels, highlighting asymmetric
        # class classification performance.
        precisions = precision_score(y_test, results["predictions"], average=None, zero_division=0)
        recalls = recall_score(y_test, results["predictions"], average=None, zero_division=0)
        f1_scores = f1_score(y_test, results["predictions"], average=None, zero_division=0)

        data = []
        for idx, name in enumerate(self._label_names):
            data.append({"Class": name, "Metric": "Precision", "Value": precisions[idx]})
            data.append({"Class": name, "Metric": "Recall", "Value": recalls[idx]})
            data.append({"Class": name, "Metric": "F1-Score", "Value": f1_scores[idx]})
        
        df_metrics = pd.DataFrame(data)

        plt.figure(figsize=(10, 6))
        sns.barplot(x="Class", y="Value", hue="Metric", data=df_metrics, palette="muted")
        plt.ylim(0, 1.05)
        plt.title("Per-Class Classification Metrics")
        plt.ylabel("Score")
        plt.xlabel("Class")
        plt.legend(loc="lower right")
        plt.tight_layout()
        report_fig_path = os.path.join(self._path_config.output_dir, "classification_report.png")
        plt.savefig(report_fig_path, dpi=300)
        plt.close()
        self._logger.info("Classification metrics bar chart saved to: %s", report_fig_path)

    def _save_analysis(self, results: dict, y_test: np.ndarray) -> None:
        """Generate and save a technical markdown report explaining K-NN performance.

        Args:
            results: The evaluation results dictionary.
            y_test:  True test labels.
        """
        self._logger.info("Writing classification explanation report...")
        os.makedirs(self._path_config.output_dir, exist_ok=True)
        report_path = os.path.join(
            self._path_config.output_dir, "classification_analysis.md"
        )

        precisions = precision_score(y_test, results["predictions"], average=None, zero_division=0)
        recalls = recall_score(y_test, results["predictions"], average=None, zero_division=0)
        f1_scores = f1_score(y_test, results["predictions"], average=None, zero_division=0)

        with open(report_path, "w", encoding="utf-8") as fh:
            fh.write("# K-Nearest Neighbors Classification Analysis Report\n\n")
            fh.write("## 1. Executive Summary\n")
            fh.write("This report provides a formal explanation of the K-Nearest Neighbors (K-NN) model's performance on predicting student burnout risk levels. ")
            fh.write(f"The model achieved an overall accuracy of **{results['accuracy'] * 100:.2f}%** on the test dataset. ")
            fh.write("Below is a detailed analysis of the performance metrics and confusion matrix from a classification algorithm perspective.\n\n")

            fh.write("## 2. Evaluation Metrics Breakdown\n")
            fh.write("In classification, using accuracy alone can be misleading, especially if classes are imbalanced. ")
            fh.write("Therefore, we analyze Precision, Recall, and F1-Score for each target class:\n\n")

            fh.write("| Class | Precision | Recall | F1-Score |\n")
            fh.write("|-------|-----------|--------|----------|\n")
            for idx, name in enumerate(self._label_names):
                fh.write(
                    f"| {name} | {precisions[idx]:.4f} | {recalls[idx]:.4f} | {f1_scores[idx]:.4f} |\n"
                )
            fh.write("\n")

            fh.write("### Theoretical Interpretation of Metrics:\n")
            fh.write("- **Precision (Positive Predictive Value)**: Explains what fraction of students predicted to have a specific burnout level actually had it. ")
            fh.write("High precision indicates a low false-positive rate.\n")
            fh.write("- **Recall (Sensitivity / True Positive Rate)**: Explains what fraction of students who actually had a specific burnout level were correctly identified by the model. ")
            fh.write("High recall indicates a low false-negative rate.\n")
            fh.write("- **F1-Score**: The harmonic mean of Precision and Recall. It provides a balanced metric of the model's reliability for that specific class.\n\n")

            fh.write("## 3. Confusion Matrix Analysis\n")
            fh.write("The confusion matrix (visualized in `confusion_matrix.png`) shows the actual vs. predicted counts:\n\n")
            
            fh.write("| Actual \\ Predicted |")
            for name in self._label_names:
                fh.write(f" {name} |")
            fh.write("\n| --- |")
            for _ in self._label_names:
                fh.write(" --- |")
            fh.write("\n")
            for idx, row in enumerate(results["confusion_matrix"]):
                fh.write(f"| **{self._label_names[idx]}** |")
                for val in row:
                    fh.write(f" {val} |")
                fh.write("\n")
            fh.write("\n")

            fh.write("### Why do misclassifications happen in K-NN?\n")
            fh.write("K-NN is an instance-based learner that constructs decision boundaries based on spatial proximity in the standardized feature space. ")
            fh.write("Misclassifications occur when the features of a test sample from one class (e.g., Moderate Risk) place it closer to training points of a different class (e.g., High Risk) than its own class. ")
            fh.write("This spatial overlap happens because of noise in the features, or because the boundaries between burnout levels are naturally continuous rather than discrete (e.g., a student on the border of Moderate and High risk).\n\n")

            fh.write("## 4. Influence of Hyperparameters and Feature Scaling\n")
            fh.write(f"- **K Value ({self._model_config.n_neighbors} Neighbors)**: A choice of K={self._model_config.n_neighbors} balances the model's complexity. ")
            fh.write("A smaller K would make the boundaries too complex (overfitting to local noise), while a larger K would smooth the boundary too much, ignoring minority classes.\n")
            fh.write(f"- **Distance-Weighted Voting (`{self._model_config.weights}`)**: By weighting votes inversely by distance, closer neighbors have a greater influence on prediction than neighbors further away, allowing for more nuanced decision boundaries.\n")
            fh.write("- **Feature Standardization**: Features are scaled (mean=0, variance=1) prior to distance computation. Without scaling, features with larger numeric ranges (like study hours vs. sleep hours) would dominate the Minkowski distance calculation, making the model blind to the remaining features.\n")

        self._logger.info("Explanation report saved to: %s", report_path)
