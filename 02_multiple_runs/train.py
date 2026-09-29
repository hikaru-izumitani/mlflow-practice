import mlflow
import mlflow.sklearn

from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split


mlflow.set_experiment("mlflow-practice-02")

iris = load_iris()

X_train, X_test, y_train, y_test = train_test_split(
    iris.data,
    iris.target,
    test_size=0.2,
    random_state=42,
)

c_values = [0.01, 0.1, 1.0, 10.0]

for c in c_values:
    model = LogisticRegression(
        C=c,
        max_iter=200,
    )

    with mlflow.start_run():
        model.fit(X_train, y_train)

        predictions = model.predict(X_test)
        accuracy = accuracy_score(y_test, predictions)

        mlflow.log_param("model", "LogisticRegression")
        mlflow.log_param("C", c)
        mlflow.log_param("max_iter", 200)

        mlflow.log_metric("accuracy", accuracy)

        mlflow.sklearn.log_model(
            model,
            name="model",
        )

        print(f"C={c}: Accuracy={accuracy:.4f}")