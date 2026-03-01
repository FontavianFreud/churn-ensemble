from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier, VotingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

import joblib


DATA_URL = "https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/master/data/Telco-Customer-Churn.csv"
DATA_PATH = Path("data/raw/telco.csv")
ARTIFACT_DIR = Path("artifacts")
MODEL_PATH = ARTIFACT_DIR / "week4_candidate_calibrated_soft_platt.joblib"

MODEL_VERSION = "week4_soft_platt_v1"


def make_ohe():
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        return OneHotEncoder(handle_unknown="ignore", sparse=False)


def download_if_missing() -> None:
    if DATA_PATH.exists():
        return
    DATA_PATH.parent.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(DATA_URL)
    df.to_csv(DATA_PATH, index=False)


def load_and_prepare() -> tuple[pd.DataFrame, pd.Series]:
    df = pd.read_csv(DATA_PATH)
    y = (df["Churn"] == "Yes").astype(int)
    X = df.drop(columns=["Churn"])

    if "customerID" in X.columns:
        X = X.drop(columns=["customerID"])

    if "TotalCharges" in X.columns:
        X["TotalCharges"] = pd.to_numeric(X["TotalCharges"], errors="coerce")

    return X, y


def build_preprocessor(X: pd.DataFrame) -> ColumnTransformer:
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
            ("onehot", make_ohe()),
        ]
    )

    return ColumnTransformer(
        transformers=[
            ("num", numeric_pipe, num_cols),
            ("cat", categorical_pipe, cat_cols),
        ]
    )


def main() -> None:
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    download_if_missing()

    X, y = load_and_prepare()

    # Train on 80% of data for artifact build (simple and reproducible)
    X_train, _, y_train, _ = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    preprocessor = build_preprocessor(X_train)

    estimators = [
        ("logreg", Pipeline([("preprocess", preprocessor), ("model", LogisticRegression(max_iter=2000))])),
        ("rf", Pipeline([("preprocess", preprocessor), ("model", RandomForestClassifier(
            n_estimators=400, random_state=42, n_jobs=-1, min_samples_leaf=5
        ))])),
        ("gb", Pipeline([("preprocess", preprocessor), ("model", GradientBoostingClassifier(random_state=42))])),
    ]

    soft_ens = VotingClassifier(estimators=estimators, voting="soft")

    # Platt calibration via CV (works across sklearn versions; avoids cv="prefit")
    cal = CalibratedClassifierCV(estimator=soft_ens, method="sigmoid", cv=3)
    cal.fit(X_train, y_train)

    joblib.dump({"model": cal, "model_version": MODEL_VERSION}, MODEL_PATH)
    print(f"Saved model artifact: {MODEL_PATH}")


if __name__ == "__main__":
    main()