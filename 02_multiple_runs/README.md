# MLflow Practice 02 — Multiple Runs

## Overview

This practice demonstrates how MLflow can track **multiple machine learning experiments within a single Experiment**.

In Practice 01, one training run was recorded.

Here, the same machine learning workflow is executed multiple times with different hyperparameter values. Each execution is recorded as a separate MLflow Run.

## Objective

The goal is to understand the relationship between:

```text
Experiment
    │
    ├── Run 1
    ├── Run 2
    ├── Run 3
    └── Run 4
```

Each Run represents one experiment with a particular configuration.

## Model

This practice uses:

* Dataset: Iris
* Model: Logistic Regression
* Library: scikit-learn
* Evaluation metric: Accuracy

The model's `C` hyperparameter is changed between runs.

## Hyperparameter

The following values are tested:

```python
C = [0.01, 0.1, 1.0, 10.0]
```

`C` controls the strength of regularization in Logistic Regression.

The purpose of changing `C` here is not to find the best value, but to demonstrate how different configurations can be tracked and compared.

## Experiment Structure

The MLflow Experiment contains four Runs:

```text
mlflow-practice-02
│
├── Run 1
│   ├── C = 0.01
│   ├── Accuracy
│   └── Model
│
├── Run 2
│   ├── C = 0.1
│   ├── Accuracy
│   └── Model
│
├── Run 3
│   ├── C = 1.0
│   ├── Accuracy
│   └── Model
│
└── Run 4
    ├── C = 10.0
    ├── Accuracy
    └── Model
```

## How It Works

The key part of the implementation is the loop:

```python
for c in c_values:
    model = LogisticRegression(
        C=c,
        max_iter=200,
    )

    with mlflow.start_run():
        ...
```

Each iteration:

1. Creates a model with a different `C` value.
2. Trains the model.
3. Evaluates the model.
4. Starts an MLflow Run.
5. Logs the parameters.
6. Logs the metric.
7. Logs the trained model.

As a result, one Python script produces multiple MLflow Runs.

## Parameters vs. Metrics

Each Run records both the configuration and the result.

### Parameters

```text
model
C
max_iter
```

These describe how the experiment was configured.

### Metric

```text
accuracy
```

This describes the model's evaluation result.

The distinction can be summarized as:

```text
Parameters → What did we use?

Metrics → What did we get?
```

## Run the Practice

Activate the virtual environment from the project root:

```bash
source .venv/bin/activate
```

Then run:

```bash
cd 02_multiple_runs
python train.py
```

The script prints the result of each Run:

```text
C=0.01: Accuracy=...
C=0.1: Accuracy=...
C=1.0: Accuracy=...
C=10.0: Accuracy=...
```

## Inspect the Results

Start the MLflow UI from the project root:

```bash
mlflow server --port 5000
```

Then open:

```text
http://localhost:5000
```

Open the `mlflow-practice-02` Experiment.

Four Runs should be visible.

The Runs can be compared using their parameters and metrics.

## Key Learning

MLflow becomes more useful when experiments are repeated with different configurations.

Without experiment tracking, results might look like:

```text
Run the script
    ↓
See accuracy
    ↓
Change C
    ↓
Run again
    ↓
Forget what was used previously
```

With MLflow:

```text
Experiment
    │
    ├── C=0.01 → Accuracy
    ├── C=0.1  → Accuracy
    ├── C=1.0  → Accuracy
    └── C=10.0 → Accuracy
```

The configuration and results of each experiment are preserved as separate Runs.

This provides the foundation for systematic experiment comparison and later hyperparameter optimization.

## Next Step

The next practice will introduce **MLflow Autologging**, which can automatically record parameters, metrics, and model information without manually calling each logging function.
