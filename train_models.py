
import os
import json
import joblib
import numpy as np
import pandas as pd

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, roc_auc_score, precision_score, recall_score,
    f1_score, matthews_corrcoef
)

RANDOM_STATE = 42
TEST_SIZE = 0.20
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "model")
os.makedirs(MODEL_DIR, exist_ok=True)

def get_dataset():
    data = load_breast_cancer(as_frame=True)
    X = data.data.copy()
    y = data.target.copy()
    X.columns = [c.replace(" ", "_") for c in X.columns]
    return X, y

def build_models():
    return {
        "Logistic Regression": Pipeline([
            ("scaler", StandardScaler()),
            ("model", LogisticRegression(max_iter=5000, random_state=RANDOM_STATE))
        ]),
        "Decision Tree": DecisionTreeClassifier(
            max_depth=5, min_samples_leaf=3, random_state=RANDOM_STATE
        ),
        "KNN": Pipeline([
            ("scaler", StandardScaler()),
            ("model", KNeighborsClassifier(n_neighbors=7))
        ]),
        "Naive Bayes": GaussianNB(),
        "Random Forest": RandomForestClassifier(
            n_estimators=300, max_depth=8, min_samples_leaf=2,
            random_state=RANDOM_STATE, n_jobs=-1
        )
    }

def main():
    X, y = get_dataset()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )

    # Save exactly the held-out test data required by the assignment.
    test_df = X_test.copy()
    test_df["target"] = y_test.to_numpy()
    test_df.to_csv(os.path.join(BASE_DIR, "test_data.csv"), index=False)

    models = build_models()
    results = []

    for name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:, 1]

        row = {
            "Model": name,
            "Accuracy": accuracy_score(y_test, y_pred),
            "AUC": roc_auc_score(y_test, y_prob),
            "Precision": precision_score(y_test, y_pred, zero_division=0),
            "Recall": recall_score(y_test, y_pred, zero_division=0),
            "F1": f1_score(y_test, y_pred, zero_division=0),
            "MCC": matthews_corrcoef(y_test, y_pred),
        }
        results.append(row)
        joblib.dump(model, os.path.join(MODEL_DIR, name.lower().replace(" ", "_") + ".joblib"))

    results_df = pd.DataFrame(results)
    results_df.to_csv(os.path.join(BASE_DIR, "metrics.csv"), index=False)

    metadata = {
        "dataset": "UCI Breast Cancer Wisconsin (Diagnostic)",
        "instances": int(X.shape[0]),
        "features": int(X.shape[1]),
        "target_column": "target",
        "target_mapping": {"0": "malignant", "1": "benign"},
        "random_state": RANDOM_STATE,
        "test_size": TEST_SIZE,
        "feature_columns": list(X.columns),
        "models": list(models.keys())
    }
    with open(os.path.join(BASE_DIR, "metadata.json"), "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    print("\nTraining completed.")
    print(results_df.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    print("\nFiles created: test_data.csv, metrics.csv, metadata.json and model/*.joblib")

if __name__ == "__main__":
    main()
