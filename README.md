# Machine Learning and Deep Learning Algorithms

A structured, production-grade repository of Machine Learning, Deep Learning, and Computer Vision algorithm implementations. Each algorithm is built as a standalone,
modular pipeline following SOLID principles, the Service Layer pattern, and professional software engineering practices.

---

## Overview

This repository serves as a hands-on reference for core ML/DL/CV algorithms, implemented from the ground up with clean architecture. Rather than one-off scripts, every algorithm is packaged as a fully orchestrated pipeline with:

- Centralized, immutable configuration via Python dataclasses.
- Structured logging with both console and file output.
- Dedicated service classes for data loading, preprocessing, model training, and
  evaluation (separation of concerns).
- Comprehensive evaluation reports persisted to disk.

The project is organized by learning paradigm, making it easy to navigate and extend with new algorithms over time.

---

## Tech Stack

| Layer              | Technology                                          |
| ------------------ | --------------------------------------------------- |
| Language           | Python 3.10+                                        |
| Package Management | [uv](https://docs.astral.sh/uv/) / pip              |
| Data Handling      | pandas, NumPy                                       |
| Machine Learning   | scikit-learn (KNN, train/test split, preprocessing) |
| Visualization      | matplotlib, seaborn                                 |
| Configuration      | Python dataclasses (frozen, immutable)               |

---

## Setup and Installation

### Prerequisites

- **Python 3.10** or higher installed on your system.
- **uv** (recommended) or **pip** for dependency management.

### Installation with uv (Recommended)

```bash
# 1. Clone the repository
git clone https://github.com/Meet-Uddeshi/Machine-Learning-And-Deep-Learning-Algorithms.git
cd Machine-Learning-And-Deep-Learning-Algorithms

# 2. Create a virtual environment and install dependencies
uv sync
```

### Installation with pip

```bash
# 1. Clone the repository
git clone https://github.com/Meet-Uddeshi/Machine-Learning-And-Deep-Learning-Algorithms.git
cd Machine-Learning-And-Deep-Learning-Algorithms

# 2. Create and activate a virtual environment
python -m venv .venv

# On Windows
.venv\Scripts\activate

# On macOS / Linux
source .venv/bin/activate

# 3. Install dependencies
pip install -r requirements.txt
```
---

## Features

- **Modular Service Layer Architecture** -- Each concern (configuration, logging,
  data loading, model training, evaluation) is isolated into its own class, following the Single Responsibility Principle.
- **Immutable Configuration** -- All pipeline parameters are defined as frozen dataclasses, preventing accidental mutation and ensuring reproducibility.
- **Comprehensive Data Validation** -- Schema checks verify that required columns exist before processing begins; missing values are imputed automatically.
- **Automated Preprocessing** -- Categorical encoding, feature standardization, and stratified train/test splitting are handled end-to-end by the data service.
- **Detailed Evaluation Reports** -- Accuracy, precision, recall, F1-score, confusion matrices, and per-class breakdowns are logged.
- **Structured Logging** -- Every pipeline step is logged with timestamps, severity levels, and module names for full traceability and debugging.
- **Reproducibility** -- Fixed random seeds, explicit hyperparameter logging, and deterministic splits ensure consistent results across runs.
- **Extensible Design** -- Adding a new algorithm means creating a new service class under the appropriate learning paradigm directory; the architecture scales cleanly.