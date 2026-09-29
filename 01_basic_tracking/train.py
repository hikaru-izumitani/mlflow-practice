import mlflow
import mlflow.sklearn

from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split


mlflow.set_experiment("mlflow-practice-01")

iris = load_iris()

X_train, X_test, y_train, y_test = train_test_split(
    iris.data,
    iris.target,
    test_size=0.2,
    random_state=42,
)

model = LogisticRegression(max_iter=200)

with mlflow.start_run():
    model.fit(X_train, y_train)

    predictions = model.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)

    mlflow.log_param("model", "LogisticRegression")
    mlflow.log_param("max_iter", 200)
    mlflow.log_param("test_size", 0.2)

    mlflow.log_metric("accuracy", accuracy)

    mlflow.sklearn.log_model(
        model,
        name="model",
    )

    print(f"Accuracy: {accuracy:.4f}")
    print(f"Run ID: {mlflow.active_run().info.run_id}")