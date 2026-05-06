import os
from pathlib import Path

import pandas as pd
import joblib
import mlflow
import mlflow.sklearn

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score


def train() -> float:
    """Train the model on processed data and save artifacts."""
    mlflow.set_experiment("content-moderation")

    df = pd.read_csv("data/processed.csv")
    X = df["clean_text"]
    y = df["toxic_combined"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    vectorizer = TfidfVectorizer(max_features=5000)
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)

    os.makedirs("models", exist_ok=True)

    with mlflow.start_run():
        model = LogisticRegression(max_iter=200)
        model.fit(X_train_vec, y_train)

        preds = model.predict(X_test_vec)
        acc = accuracy_score(y_test, preds)

        mlflow.log_param("model", "logistic_regression")
        mlflow.log_metric("accuracy", acc)

        joblib.dump(model, "models/model.pkl")
        joblib.dump(vectorizer, "models/vectorizer.pkl")

        mlflow.log_artifact("models/model.pkl")
        mlflow.log_artifact("models/vectorizer.pkl")

        print(f"Accuracy: {acc}")

    return acc


if __name__ == "__main__":
    train()