from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


DATA_PATH = Path("data/raw/telco.csv")
ARTIFACT_DIR = Path("artifacts")
REPORT_DIR = Path("reports")


def make_ohe():
    # Compatibility across sklearn versions
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        return OneHotEncoder(handle_unknown="ignore", sparse=False)


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


def eval_probs(y_true: np.ndarray, probs: np.ndarray, threshold: float) -> dict:
    preds = (probs >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, preds).ravel()
    return {
        "roc_auc": float(roc_auc_score(y_true, probs)),
        "pr_auc": float(average_precision_score(y_true, probs)),
        "precision": float(precision_score(y_true, preds, zero_division=0)),
        "recall": float(recall_score(y_true, preds, zero_division=0)),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
    }


def choose_threshold_for_recall(y_val: np.ndarray, val_probs: np.ndarray, target_recall: float) -> tuple[float, dict]:
    # Pick the lowest threshold that achieves at least target recall,
    # and among those, pick the one with highest precision.
    thresholds = np.linspace(0.01, 0.99, 99)
    candidates = []
    for t in thresholds:
        metrics = eval_probs(y_val, val_probs, threshold=float(t))
        if metrics["recall"] >= target_recall:
            candidates.append((t, metrics))

    if not candidates:
        # If target recall not achievable, fall back to 0.5
        return 0.5, eval_probs(y_val, val_probs, threshold=0.5)

    # Sort by precision desc, then threshold asc (prefer lower thresholds if tie)
    candidates.sort(key=lambda x: (-x[1]["precision"], x[0]))
    best_t, best_metrics = candidates[0]
    return float(best_t), best_metrics


def segment_analysis(df_val: pd.DataFrame, y_val: pd.Series, probs: np.ndarray, threshold: float) -> pd.DataFrame:
    out = df_val.copy()
    out["y_true"] = y_val.values
    out["prob"] = probs
    out["pred"] = (probs >= threshold).astype(int)

    # Tenure buckets
    out["tenure_bucket"] = pd.cut(
        out["tenure"],
        bins=[-0.1, 6, 12, 24, 48, 72, 1e9],
        labels=["0-6", "6-12", "12-24", "24-48", "48-72", "72+"],
    )

    group_cols = ["tenure_bucket", "Contract", "InternetService", "PaymentMethod"]
    rows = []

    for col in group_cols:
        for group, g in out.groupby(col, dropna=False):
            tp = int(((g["pred"] == 1) & (g["y_true"] == 1)).sum())
            fp = int(((g["pred"] == 1) & (g["y_true"] == 0)).sum())
            fn = int(((g["pred"] == 0) & (g["y_true"] == 1)).sum())
            tn = int(((g["pred"] == 0) & (g["y_true"] == 0)).sum())

            precision = tp / (tp + fp) if (tp + fp) else 0.0
            recall = tp / (tp + fn) if (tp + fn) else 0.0
            churn_rate = float(g["y_true"].mean())

            rows.append(
                {
                    "segment_col": col,
                    "segment": str(group),
                    "n": int(len(g)),
                    "churn_rate": churn_rate,
                    "precision": precision,
                    "recall": recall,
                    "tp": tp,
                    "fp": fp,
                    "fn": fn,
                    "tn": tn,
                }
            )

    return pd.DataFrame(rows).sort_values(["segment_col", "recall", "n"], ascending=[True, True, False])


def main() -> None:
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    X, y = load_and_prepare()

    # Keep a copy of raw X for segment analysis later
    X_raw = X.copy()

    # Split train/val/test (same as Week 2)
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.4, random_state=42, stratify=y
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp
    )

    # Raw val frame for segment analysis
    X_raw_train, X_raw_temp, _, _ = train_test_split(
        X_raw, y, test_size=0.4, random_state=42, stratify=y
    )
    X_raw_val, X_raw_test, _, _ = train_test_split(
        X_raw_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp
    )

    preprocessor = build_preprocessor(X_train)

    models = {
        "logreg": LogisticRegression(max_iter=2000),
        "rf": RandomForestClassifier(
            n_estimators=400, random_state=42, n_jobs=-1, min_samples_leaf=5
        ),
        "gb": GradientBoostingClassifier(random_state=42),
    }

    results = []
    fitted = {}

    # Compare on validation (use val for decisions)
    for name, model in models.items():
        clf = Pipeline(steps=[("preprocess", preprocessor), ("model", model)])
        clf.fit(X_train, y_train)

        val_probs = clf.predict_proba(X_val)[:, 1]
        metrics_05 = eval_probs(y_val.values, val_probs, threshold=0.5)

        results.append(
            {
                "model": name,
                "val_roc_auc": metrics_05["roc_auc"],
                "val_pr_auc": metrics_05["pr_auc"],
                "val_precision@0.5": metrics_05["precision"],
                "val_recall@0.5": metrics_05["recall"],
            }
        )
        fitted[name] = clf

    results_df = pd.DataFrame(results).sort_values("val_pr_auc", ascending=False)
    print("\n=== Week 3 Model Comparison (Validation) ===")
    print(results_df.to_string(index=False))

    # Choose best by PR-AUC (good for imbalanced churn)
    best_name = results_df.iloc[0]["model"]
    best_clf = fitted[best_name]

    # Threshold tuned for higher recall (validation only)
    val_probs_best = best_clf.predict_proba(X_val)[:, 1]
    chosen_t, chosen_metrics = choose_threshold_for_recall(
        y_val.values, val_probs_best, target_recall=0.75
    )

    print(f"\nBest model by val PR-AUC: {best_name}")
    print(f"Chosen threshold for recall>=0.75 on validation: {chosen_t:.2f}")
    print("Validation metrics at chosen threshold:")
    print(json.dumps(chosen_metrics, indent=2))

    # Error analysis on validation at chosen threshold
    seg_df = segment_analysis(X_raw_val, y_val, val_probs_best, threshold=chosen_t)
    seg_path = REPORT_DIR / "week3_error_analysis_val.csv"
    seg_df.to_csv(seg_path, index=False)
    print(f"\nSaved segment error analysis to: {seg_path}")

    # Save comparison table
    compare_path = REPORT_DIR / "week3_model_compare_val.csv"
    results_df.to_csv(compare_path, index=False)
    print(f"Saved model comparison table to: {compare_path}")

    # Save best model artifact (for Week 4 ensemble work)
    best_model_path = ARTIFACT_DIR / f"week3_best_{best_name}.joblib"
    import joblib
    joblib.dump(best_clf, best_model_path)
    print(f"Saved best model to: {best_model_path}")


if __name__ == "__main__":
    main()