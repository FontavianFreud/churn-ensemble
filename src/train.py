from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, confusion_matrix, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

ARTIFACT_DIR = Path("artifacts")
MODEL_PATH = ARTIFACT_DIR / "model.joblib"
METRICS_PATH = ARTIFACT_DIR / "metrics.json"

DATA_PATH = Path("data/raw/telco.csv")


def main() -> None:
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(DATA_PATH)

    # Target
    y = (df["Churn"] == "Yes").astype(int)

    # Basic cleanup
    X = df.drop(columns=["Churn"])

    # Drop obvious identifier
    if "customerID" in X.columns:
        X = X.drop(columns=["customerID"])

    # Convert TotalCharges to numeric (it sometimes contains spaces)
    if "TotalCharges" in X.columns:
        X["TotalCharges"] = pd.to_numeric(X["TotalCharges"], errors="coerce")

    # Split train/val/test (60/20/20), stratified
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.4, random_state=42, stratify=y
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp
    )

    # Column types
    num_cols = X.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = X.select_dtypes(exclude=[np.number]).columns.tolist()

    numeric_pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    categorical_pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_pipe, num_cols),
            ("cat", categorical_pipe, cat_cols),
        ]
    )

    model = LogisticRegression(max_iter=2000)

    clf = Pipeline(
        steps=[
            ("preprocess", preprocessor),
            ("model", model),
        ]
    )

    clf.fit(X_train, y_train)

    # Predict probabilities
    val_probs = clf.predict_proba(X_val)[:, 1]
    test_probs = clf.predict_proba(X_test)[:, 1]

    # Metrics
    val_roc = roc_auc_score(y_val, val_probs)
    test_roc = roc_auc_score(y_test, test_probs)

    val_pr = average_precision_score(y_val, val_probs)
    test_pr = average_precision_score(y_test, test_probs)

    threshold = 0.5
    test_pred = (test_probs >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_test, test_pred).ravel()

    metrics = {
        "val_roc_auc": float(val_roc),
        "test_roc_auc": float(test_roc),
        "val_pr_auc": float(val_pr),
        "test_pr_auc": float(test_pr),
        "threshold": float(threshold),
        "confusion_matrix": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
        "n_train": int(len(X_train)),
        "n_val": int(len(X_val)),
        "n_test": int(len(X_test)),
    }

    joblib.dump(clf, MODEL_PATH)
    METRICS_PATH.write_text(json.dumps(metrics, indent=2))

    print("Saved model to:", MODEL_PATH)
    print("Saved metrics to:", METRICS_PATH)
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()