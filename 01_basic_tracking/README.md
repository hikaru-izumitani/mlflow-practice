# MLflow Practice 01 — Basic Tracking

## Overview

This practice introduces the basic workflow of **MLflow Tracking**.

The goal is to understand how MLflow records information about a machine learning experiment, including:

* Parameters
* Metrics
* Trained models
* Individual experiment runs

The machine learning model itself is intentionally simple. The main focus is understanding **what MLflow records and how an MLflow Run works**.

## Model

This practice uses:

* Dataset: Iris
* Model: Logistic Regression
* Library: scikit-learn

The dataset is split into training and test sets, and the model is evaluated using accuracy.

## MLflow Workflow

The basic workflow is:

```text
Python Training Script
        │
        ▼
mlflow.start_run()
        │
        ├── log_param()
        │
        ├── log_metric()
        │
        └── log_model()
        │
        ▼
      MLflow Run
        │
        ▼
   Local Tracking Data
```

Each execution of the training script creates a new **Run**.

## What Is Logged?

### Parameters

Parameters describe the configuration used for the experiment.

This practice logs:

```text
model = LogisticRegression
max_iter = 200
test_size = 0.2
```

They answer the question:

> How was this experiment configured?

### Metrics

Metrics describe the numerical results of the experiment.

This practice logs:

```text
accuracy
```

It answers the question:

> How well did the model perform?

### Model Artifact

The trained Logistic Regression model is also logged as an MLflow artifact.

This makes it possible to keep the trained model together with the experiment information that produced it.

## Key MLflow Concepts

### Experiment

An **Experiment** groups related MLflow Runs.

This practice creates the experiment:

```text
mlflow-practice-01
```

### Run

A **Run** represents one execution of an ML experiment.

For example:

```text
Run 1
├── Parameters
├── Metrics
└── Model

Run 2
├── Parameters
├── Metrics
└── Model
```

Running the training script multiple times creates multiple Runs.

### Parameters vs. Metrics

A useful distinction is:

```text
Parameters → What did we use?

Metrics    → What did we get?
```

For example:

```text
Parameter:
max_iter = 200

Metric:
accuracy = 0.9667
```

## Run the Practice

From the project root, activate the virtual environment:

```bash
source .venv/bin/activate
```

Then run:

```bash
cd 01_basic_tracking
python train.py
```

The script prints the model accuracy and the MLflow Run ID.

Example:

```text
Accuracy: 0.9667
Run ID: <run-id>
```

## Inspect the Tracking Data

MLflow creates local tracking data when the experiment is executed.

From the project root:

```bash
find mlruns -maxdepth 3 -type f | sort
```

The `mlruns/` directory contains the information recorded by MLflow.

It is intentionally included in `.gitignore` because it is local experiment-tracking data rather than source code.

## MLflow UI

The experiment can also be inspected through the MLflow UI.

From the project root:

```bash
mlflow server --port 5000
```

Then open:

```text
http://localhost:5000
```

The UI allows the Run, parameters, metrics, and logged model to be inspected visually.

## What I Learned

This practice demonstrates that MLflow is not responsible for training the machine learning model itself.

Instead, the training code performs the machine learning work while MLflow records information about the process.

In simplified form:

```text
scikit-learn
    │
    │ trains the model
    ▼
Machine Learning Model
    │
    │ MLflow records the experiment
    ▼
MLflow Tracking
    │
    ├── Parameters
    ├── Metrics
    └── Model Artifact
```

This separation is the foundation for comparing experiments and managing machine learning workflows.

## Next Step

The next practice will create **multiple MLflow Runs** with different configurations and compare their results.
