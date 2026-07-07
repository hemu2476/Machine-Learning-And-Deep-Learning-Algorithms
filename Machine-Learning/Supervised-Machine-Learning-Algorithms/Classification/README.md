# Classification - K-Nearest Neighbors (K-NN)

> Supervised Machine Learning | Classification Algorithm

---

## Table of Contents

1. [What is Classification?](#1-what-is-classification)
2. [Theoretical Explanation](#2-theoretical-explanation)
3. [Mathematical Operations](#3-mathematical-operations)
4. [Real-World Example](#4-real-world-example)
5. [Worked K-NN Sum (Step-by-Step)](#5-worked-k-nn-sum-step-by-step)
6. [Program Flowchart](#6-program-flowchart)

---

## 1. What is Classification?

Classification is a type of **supervised machine learning** where the goal is to assign an input data point to one of a fixed set of **discrete categories (classes)**.

The algorithm learns from a **labelled training dataset** - data where the correct class is already known. After learning, it applies that knowledge to predict the class of **new, unseen data points**.

### Key Characteristics

| Property           | Description                                                          |
|--------------------|----------------------------------------------------------------------|
| Task type          | Supervised Learning                                                  |
| Output type        | Discrete class label (e.g., "High Risk", "Low Risk", "Cat", "Dog")  |
| Training data      | Labelled (input + known correct class)                               |
| Prediction         | Assigns one class label from a predefined set                        |
| Evaluation metrics | Accuracy, Precision, Recall, F1-Score, Confusion Matrix              |

### Binary vs. Multi-Class Classification

- **Binary Classification** - only two possible classes (e.g., spam / not spam, pass / fail).
- **Multi-Class Classification** - three or more possible classes (e.g., Low Risk / Medium Risk / High Risk).

---

## 2. Theoretical Explanation

### How K-Nearest Neighbors (K-NN) Works

K-NN is one of the simplest and most intuitive classification algorithms. Its core idea is:

> "A data point belongs to the same class as the majority of its K nearest neighbors."

K-NN is a **lazy learning** algorithm - it does not build an explicit model during training. Instead, it memorizes the entire training dataset. The actual computation happens at **prediction time**, when it searches for the closest neighbors.

### Step-by-Step Logic

```
Step 1: Store all training data points with their labels.

Step 2: When a new (unknown) point arrives:
        - Calculate the distance from this point to every training point.

Step 3: Sort all training points by distance (nearest first).

Step 4: Pick the K closest training points (the K nearest neighbors).

Step 5: Count how many neighbors belong to each class.

Step 6: Assign the class that appears most often (majority vote).
        - If weights='distance', closer neighbors get a higher vote weight.

Step 7: Return the winning class as the prediction.
```

### Why Distance Matters

K-NN relies entirely on **distance** to determine similarity. If two data points are close together in the feature space, they are considered similar and likely belong to the same class.

Because of this, **feature scaling is mandatory**. A feature measured in thousands (e.g., salary) would dominate distance calculations over a feature measured in ones (e.g., a rating from 1 to 5). StandardScaler is applied in this project to eliminate that bias.

### The Role of K

The value of K controls the trade-off between bias and variance:

| K value    | Behavior                                                                             |
|------------|--------------------------------------------------------------------------------------|
| Very small (K=1) | Extremely sensitive to noise; model overfits to individual training points     |
| Very large (K=N) | Decision is dominated by global majority class; model underfits                |
| Optimal K  | Found by testing multiple values (e.g., cross-validation); balances bias and variance |

This project uses **K = 7** as configured in `config.py`.

---

## 3. Mathematical Operations

### Distance Metric: Minkowski Distance

This project uses the **Minkowski distance** (configured via `metric='minkowski'`). It is a general formula that covers both Euclidean and Manhattan distance.

For two points **A** and **B** with `n` features:

```
Minkowski Distance = ( |A1 - B1|^p  +  |A2 - B2|^p  +  ...  +  |An - Bn|^p ) ^ (1/p)
```

When **p = 2**, this becomes the standard **Euclidean Distance**:

```
Euclidean Distance = ( (A1 - B1)^2  +  (A2 - B2)^2  +  ...  +  (An - Bn)^2 ) ^ (1/2)
```

When **p = 1**, this becomes the **Manhattan Distance**:

```
Manhattan Distance = |A1 - B1|  +  |A2 - B2|  +  ...  +  |An - Bn|
```

### Feature Standardization (Z-Score Scaling)

Before distance is computed, every feature is scaled so it has a **mean of 0** and a **standard deviation of 1**.

```
Scaled Value = (Original Value  -  Mean of Feature)  /  Standard Deviation of Feature
```

Example: If `Study_Hours_Per_Day` has a mean of 5 and a standard deviation of 2:

```
A student with 7 hours  -->  Scaled Value = (7 - 5) / 2  =  1.0
A student with 3 hours  -->  Scaled Value = (3 - 5) / 2  = -1.0
```

### Distance-Weighted Voting

When `weights='distance'` is set, each neighbor votes with a weight inversely proportional to its distance. Closer neighbors count more.

```
Vote Weight of Neighbor i  =  1  /  Distance(Query, Neighbor_i)
```

The predicted class is the one with the highest total weight:

```
Predicted Class  =  argmax( sum of weights for each class )
```

### Evaluation Metrics

**Accuracy** - What fraction of all predictions were correct?

```
Accuracy  =  Number of Correct Predictions  /  Total Number of Predictions
```

**Precision** - Of all points predicted as a class, how many actually belong to it?

```
Precision  =  True Positives  /  (True Positives  +  False Positives)
```

**Recall** - Of all actual points in a class, how many were correctly predicted?

```
Recall  =  True Positives  /  (True Positives  +  False Negatives)
```

**F1-Score** - Harmonic mean of Precision and Recall (best when classes are imbalanced):

```
F1-Score  =  2  x  (Precision  x  Recall)  /  (Precision  +  Recall)
```

---

## 4. Real-World Example

### Student Burnout Risk Prediction

This project predicts the **Burnout Risk Level** of university students based on their academic habits, AI tool usage, and well-being indicators.

**Dataset:** `ai_student_impact_dataset.csv`

**Target Variable:** `Burnout_Risk_Level`  
Possible classes: `Low`, `Moderate`, `High`

**Features Used (examples):**

| Feature                        | Description                                   |
|--------------------------------|-----------------------------------------------|
| `Study_Hours_Per_Day`          | Average daily study hours                     |
| `AI_Tool_Usage_Hours`          | Daily hours spent using AI tools              |
| `Sleep_Hours_Per_Night`        | Average nightly sleep hours                   |
| `Social_Media_Hours_Per_Day`   | Daily social media usage                      |
| `Assignment_Completion_Rate`   | Percentage of assignments completed           |
| `Mental_Health_Score`          | Self-reported mental health rating            |
| `Stress_Level`                 | Self-reported stress rating                   |

**Columns excluded from features:**

- `Student_ID` - a unique identifier with no predictive value.
- `Post_Semester_GPA` - a post-outcome measurement (data leakage risk).
- `Skill_Retention_Score` - a post-outcome measurement (data leakage risk).

**Why K-NN is a good fit here:**
- Burnout risk follows natural clusters. Students with similar study hours, sleep, and stress levels tend to fall in the same risk category.
- K-NN can learn non-linear, complex decision boundaries without assuming a specific data distribution.
- The dataset has multiple numeric features, and distance-based similarity is a meaningful measure across these dimensions.

---

## 5. Worked K-NN Sum (Step-by-Step)

The following is a simplified, hand-calculated demonstration of how K-NN classifies a new student using **K = 3** and **Euclidean Distance**.

### The Training Data (5 known students, 2 features for simplicity)

| Student | Study Hours (scaled) | Sleep Hours (scaled) | Burnout Risk |
|---------|----------------------|----------------------|--------------|
| S1      | 2.0                  | 1.5                  | Low          |
| S2      | 3.0                  | 0.5                  | Moderate     |
| S3      | 5.0                  | -0.5                 | High         |
| S4      | 4.5                  | -1.0                 | High         |
| S5      | 1.5                  | 2.0                  | Low          |

### The New (Unknown) Student

| Study Hours (scaled) | Sleep Hours (scaled) |
|----------------------|----------------------|
| 4.0                  | 0.0                  |

### Step 1: Calculate Euclidean Distance to Every Training Point

**Formula:**

```
Distance  =  ( (Study_New - Study_Train)^2  +  (Sleep_New - Sleep_Train)^2 ) ^ (1/2)
```

**Distance to S1:**

```
= ( (4.0 - 2.0)^2  +  (0.0 - 1.5)^2 ) ^ (1/2)
= ( (2.0)^2        +  (-1.5)^2       ) ^ (1/2)
= ( 4.0            +  2.25           ) ^ (1/2)
= ( 6.25 ) ^ (1/2)
= 2.50
```

**Distance to S2:**

```
= ( (4.0 - 3.0)^2  +  (0.0 - 0.5)^2 ) ^ (1/2)
= ( (1.0)^2        +  (-0.5)^2       ) ^ (1/2)
= ( 1.0            +  0.25           ) ^ (1/2)
= ( 1.25 ) ^ (1/2)
= 1.12
```

**Distance to S3:**

```
= ( (4.0 - 5.0)^2  +  (0.0 - (-0.5))^2 ) ^ (1/2)
= ( (-1.0)^2       +  (0.5)^2           ) ^ (1/2)
= ( 1.0            +  0.25              ) ^ (1/2)
= ( 1.25 ) ^ (1/2)
= 1.12
```

**Distance to S4:**

```
= ( (4.0 - 4.5)^2  +  (0.0 - (-1.0))^2 ) ^ (1/2)
= ( (-0.5)^2       +  (1.0)^2           ) ^ (1/2)
= ( 0.25           +  1.0               ) ^ (1/2)
= ( 1.25 ) ^ (1/2)
= 1.12
```

**Distance to S5:**

```
= ( (4.0 - 1.5)^2  +  (0.0 - 2.0)^2 ) ^ (1/2)
= ( (2.5)^2        +  (-2.0)^2       ) ^ (1/2)
= ( 6.25           +  4.0            ) ^ (1/2)
= ( 10.25 ) ^ (1/2)
= 3.20
```

### Step 2: Rank All Distances (Nearest First)

| Rank | Student | Distance | Burnout Risk |
|------|---------|----------|--------------|
| 1    | S2      | 1.12     | Moderate     |
| 2    | S3      | 1.12     | High         |
| 3    | S4      | 1.12     | High         |
| 4    | S1      | 2.50     | Low          |
| 5    | S5      | 3.20     | Low          |

### Step 3: Select the K = 3 Nearest Neighbors

The 3 nearest neighbors are: **S2 (Moderate), S3 (High), S4 (High)**

### Step 4: Count Class Votes

| Class    | Vote Count |
|----------|------------|
| Moderate | 1          |
| High     | 2          |
| Low      | 0          |

### Step 5: Majority Vote - Final Prediction

```
High = 2 votes   <-- Winner
Moderate = 1 vote
Low = 0 votes

Predicted Burnout Risk Level for the new student = HIGH
```

This makes intuitive sense: the student studies 4 hours per day with very little sleep, similar to other high-risk students in the dataset.

---

## 6. Program Flowchart

The following flowchart describes the complete execution flow of the K-NN Classification Pipeline as implemented in this project.

```
+-----------------------------------------------------+
|               START: main.py runs                   |
+-----------------------------------------------------+
                         |
                         v
+-----------------------------------------------------+
|  Step 1: Load PipelineConfig                        |
|  - PathConfig  (dataset path, output path)          |
|  - DataConfig  (target column, drop columns, split) |
|  - ModelConfig (K=7, weights=distance, metric=      |
|                 minkowski, algorithm=auto)           |
|  - LoggingConfig (level=DEBUG, log to file=True)    |
+-----------------------------------------------------+
                         |
                         v
+-----------------------------------------------------+
|  Step 2: Initialize Logger (LoggerFactory)          |
|  - Console handler (INFO level output)              |
|  - File handler  (knn_classification.log)           |
+-----------------------------------------------------+
                         |
                         v
+-----------------------------------------------------+
|  Step 3: DataLoaderService.load_and_prepare()       |
+-----------------------------------------------------+
                         |
         +---------------+-------------------+
         |                                   |
         v                                   v
+------------------+               +-------------------+
|  _load_csv()     |               |  File not found?  |
|  Read CSV from   |  -- Error -->  |  Raise            |
|  data/ folder    |               |  FileNotFoundError |
+------------------+               +-------------------+
         |
         v
+------------------+
|  _validate_schema|
|  Check target    |
|  column exists   |
|  Check drop cols |
|  exist in CSV    |
+------------------+
         |
         v
+------------------+
|  _log_data_      |
|  summary()       |
|  Log shape,      |
|  dtypes, nulls,  |
|  class balance   |
+------------------+
         |
         v
+--------------------------------------------------+
|  _preprocess()                                   |
|  1. Separate target from features                |
|  2. Drop Student_ID, Post_Semester_GPA,          |
|     Skill_Retention_Score                        |
|  3. Impute nulls (median for numeric,            |
|     mode for categorical)                        |
|  4. LabelEncode target (Low=0, Moderate=1,       |
|     High=2)                                      |
|  5. LabelEncode each categorical feature column  |
|  6. StandardScaler: scale features to            |
|     mean=0, std=1                                |
+--------------------------------------------------+
         |
         v
+--------------------------------------------------+
|  _split()                                        |
|  Stratified Train/Test split                     |
|  - 80% Training data                             |
|  - 20% Test data                                 |
|  - Random state = 42 (reproducible)              |
+--------------------------------------------------+
         |
         v
+-----------------------------------------------------+
|  Step 4: KNNClassifierService.train()               |
|  - Build KNeighborsClassifier with config params    |
|  - Call model.fit(X_train, y_train)                 |
|  - Log training time                                |
+-----------------------------------------------------+
                         |
                         v
+-----------------------------------------------------+
|  Step 5: KNNClassifierService.evaluate()            |
+-----------------------------------------------------+
                         |
                         v
+--------------------------------------------------+
|  For each test point in X_test:                  |
|  - Compute Minkowski distance to all             |
|    training points                               |
|  - Sort by distance                              |
|  - Select K=7 nearest neighbors                  |
|  - Aggregate weighted votes (1/distance)         |
|  - Assign class with highest vote weight         |
+--------------------------------------------------+
                         |
                         v
+--------------------------------------------------+
|  Compute Evaluation Metrics                      |
|  - Accuracy Score                                |
|  - Precision (macro average)                     |
|  - Recall    (macro average)                     |
|  - F1-Score  (macro average)                     |
|  - Confusion Matrix (3x3 for 3 classes)          |
|  - Per-class Classification Report               |
+--------------------------------------------------+
                         |
                         v
+--------------------------------------------------+
|  _log_evaluation()                               |
|  Print all metrics to console and log file       |
+--------------------------------------------------+
                         |
                         v
+--------------------------------------------------+
|  _save_results()                                 |
|  Write classification_results.txt to output/    |
+--------------------------------------------------+
                         |
                         v
+-----------------------------------------------------+
|               END: Pipeline Complete               |
|  Total elapsed time is logged                      |
+-----------------------------------------------------+
```

### Module Responsibility Map

```
main.py
  |
  +-- config.py          (PipelineConfig, PathConfig, DataConfig,
  |                        ModelConfig, LoggingConfig)
  |
  +-- logger.py          (LoggerFactory - console)
  |
  +-- data_loader.py     (DataLoaderService - load, validate,
  |                        preprocess, split)
  |
  +-- knn_classifier.py  (KNNClassifierService - train, predict,
                           evaluate, save results)
```

---

### Configuration

All tunable parameters are located in `src/config.py`. No changes to business logic are needed. Edit the dataclass defaults to adjust:

| Parameter        | Location          | Default            | Description                         |
|------------------|-------------------|--------------------|-------------------------------------|
| `n_neighbors`    | `ModelConfig`     | `7`                | Number of K nearest neighbors       |
| `weights`        | `ModelConfig`     | `'distance'`       | Voting strategy                     |
| `metric`         | `ModelConfig`     | `'minkowski'`      | Distance formula                    |
| `test_size`      | `DataConfig`      | `0.20`             | Fraction of data used for testing   |
| `target_column`  | `DataConfig`      | `Burnout_Risk_Level` | Column being predicted            |