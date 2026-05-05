import pandas as pd
import joblib
import mlflow
import mlflow.sklearn

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score


# Set experiment
mlflow.set_experiment("content-moderation")

# Load data
df = pd.read_csv("data/processed.csv")

X = df["clean_text"]
y = df["toxic_combined"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Vectorization
vectorizer = TfidfVectorizer(max_features=5000)
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

with mlflow.start_run():

    model = LogisticRegression(max_iter=200)
    model.fit(X_train_vec, y_train)

    preds = model.predict(X_test_vec)
    acc = accuracy_score(y_test, preds)

    # Log parameters + metrics
    mlflow.log_param("model", "logistic_regression")
    mlflow.log_metric("accuracy", acc)

    # Save model
    joblib.dump(model, "models/model.pkl")
    joblib.dump(vectorizer, "models/vectorizer.pkl")

    # Log artifacts
    mlflow.log_artifact("models/model.pkl")
    mlflow.log_artifact("models/vectorizer.pkl")

    print(f" Accuracy: {acc}")