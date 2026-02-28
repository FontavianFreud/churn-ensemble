from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier, VotingClassifier
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
REPORT_DIR = Path("reports")
ARTIFACT_DIR = Path("artifacts")


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


def make_base_estimators(preprocessor: ColumnTransformer):
    # Create fresh estimators each time (important: avoid sharing fitted state)
    logreg = Pipeline(
        steps=[
            ("preprocess", preprocessor),
            ("model", LogisticRegression(max_iter=2000)),
        ]
    )
    rf = Pipeline(
        steps=[
            ("preprocess", preprocessor),
            ("model", RandomForestClassifier(
                n_estimators=400, random_state=42, n_jobs=-1, min_samples_leaf=5
            )),
        ]
    )
    gb = Pipeline(
        steps=[
            ("preprocess", preprocessor),
            ("model", GradientBoostingClassifier(random_state=42)),
        ]
    )
    return [("logreg", logreg), ("rf", rf), ("gb", gb)]


def make_calibrator(estimator, method: str, cv: int):
    # sklearn changed arg name from base_estimator -> estimator across versions
    try:
        return CalibratedClassifierCV(estimator=estimator, method=method, cv=cv)
    except TypeError:
        return CalibratedClassifierCV(base_estimator=estimator, method=method, cv=cv)


def eval_ranking(y_true: np.ndarray, probs: np.ndarray) -> dict:
    return {
        "roc_auc": float(roc_auc_score(y_true, probs)),
        "pr_auc": float(average_precision_score(y_true, probs)),
    }


def eval_classification(y_true: np.ndarray, preds: np.ndarray) -> dict:
    tn, fp, fn, tp = confusion_matrix(y_true, preds).ravel()
    precision = precision_score(y_true, preds, zero_division=0)
    recall = recall_score(y_true, preds, zero_division=0)
    return {
        "precision": float(precision),
        "recall": float(recall),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
    }


def choose_threshold_cost_sensitive(y_val: np.ndarray, val_probs: np.ndarray, cost_fp: float, cost_fn: float):
    thresholds = np.linspace(0.01, 0.99, 99)
    rows = []
    best_t = 0.5
    best_cost = float("inf")

    for t in thresholds:
        preds = (val_probs >= t).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_val, preds).ravel()
        cost = fp * cost_fp + fn * cost_fn

        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / (tp + fn) if (tp + fn) else 0.0

        rows.append({
            "threshold": float(t),
            "precision": float(precision),
            "recall": float(recall),
            "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
            "cost": float(cost),
        })

        if cost < best_cost:
            best_cost = cost
            best_t = float(t)

    return best_t, pd.DataFrame(rows)


def plot_reliability(y_true: np.ndarray, probs_before: np.ndarray, probs_after: np.ndarray, title: str, out_path: Path):
    frac_pos_b, mean_pred_b = calibration_curve(y_true, probs_before, n_bins=10, strategy="uniform")
    frac_pos_a, mean_pred_a = calibration_curve(y_true, probs_after, n_bins=10, strategy="uniform")

    plt.figure()
    plt.plot(mean_pred_b, frac_pos_b, marker="o", label="before")
    plt.plot(mean_pred_a, frac_pos_a, marker="o", label="after")
    plt.plot([0, 1], [0, 1], linestyle="--", label="perfect")
    plt.xlabel("Mean predicted probability")
    plt.ylabel("Fraction of positives")
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


def main() -> None:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)

    X, y = load_and_prepare()

    # Split: train/val/test (60/20/20), stratified
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.4, random_state=42, stratify=y
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp
    )

    preprocessor = build_preprocessor(X_train)

    # --- Hard vs Soft voting (uncalibrated) ---
    hard_ens = VotingClassifier(estimators=make_base_estimators(preprocessor), voting="hard")
    soft_ens = VotingClassifier(estimators=make_base_estimators(preprocessor), voting="soft")

    hard_ens.fit(X_train, y_train)
    soft_ens.fit(X_train, y_train)

    hard_test_preds = hard_ens.predict(X_test)
    soft_test_probs_before = soft_ens.predict_proba(X_test)[:, 1]

    hard_metrics = eval_classification(y_test.values, hard_test_preds)

    # Also evaluate soft at 0.5 for a fair “decision” comparison
    soft_test_preds_05 = (soft_test_probs_before >= 0.5).astype(int)
    soft_metrics_05 = eval_classification(y_test.values, soft_test_preds_05)

    (REPORT_DIR / "week4_hard_vs_soft_test.json").write_text(json.dumps({
        "hard_voting_test": hard_metrics,
        "soft_voting_test_threshold_0.5": soft_metrics_05,
    }, indent=2))

    # --- Calibration (CV on TRAIN) ---
    # Platt scaling = sigmoid, Isotonic = isotonic regression
    # We calibrate logreg and soft ensemble; hard voting has no probabilities to calibrate.
    logreg_for_cal = Pipeline(steps=[("preprocess", preprocessor), ("model", LogisticRegression(max_iter=2000))])
    soft_for_cal = VotingClassifier(estimators=make_base_estimators(preprocessor), voting="soft")

    cal_logreg_platt = make_calibrator(logreg_for_cal, method="sigmoid", cv=3)
    cal_logreg_iso = make_calibrator(Pipeline(steps=[("preprocess", preprocessor), ("model", LogisticRegression(max_iter=2000))]),
                                     method="isotonic", cv=3)

    cal_soft_platt = make_calibrator(soft_for_cal, method="sigmoid", cv=3)
    cal_soft_iso = make_calibrator(VotingClassifier(estimators=make_base_estimators(preprocessor), voting="soft"),
                                   method="isotonic", cv=3)

    cal_logreg_platt.fit(X_train, y_train)
    cal_logreg_iso.fit(X_train, y_train)
    cal_soft_platt.fit(X_train, y_train)
    cal_soft_iso.fit(X_train, y_train)

    # Uncalibrated logreg (for "before" comparison)
    logreg_before = Pipeline(steps=[("preprocess", preprocessor), ("model", LogisticRegression(max_iter=2000))])
    logreg_before.fit(X_train, y_train)

    # --- Probabilities on TEST (before/after) ---
    test_probs_logreg_before = logreg_before.predict_proba(X_test)[:, 1]
    test_probs_logreg_platt = cal_logreg_platt.predict_proba(X_test)[:, 1]
    test_probs_logreg_iso = cal_logreg_iso.predict_proba(X_test)[:, 1]

    test_probs_soft_before = soft_test_probs_before
    test_probs_soft_platt = cal_soft_platt.predict_proba(X_test)[:, 1]
    test_probs_soft_iso = cal_soft_iso.predict_proba(X_test)[:, 1]

    # Ranking metrics table (test)
    rows = []
    for name, probs in [
        ("logreg_before", test_probs_logreg_before),
        ("logreg_platt", test_probs_logreg_platt),
        ("logreg_isotonic", test_probs_logreg_iso),
        ("soft_before", test_probs_soft_before),
        ("soft_platt", test_probs_soft_platt),
        ("soft_isotonic", test_probs_soft_iso),
    ]:
        m = eval_ranking(y_test.values, probs)
        rows.append({"model": name, **m})

    results_df = pd.DataFrame(rows).sort_values("pr_auc", ascending=False)
    results_path = REPORT_DIR / "week4_ranking_compare_test.csv"
    results_df.to_csv(results_path, index=False)
    print(f"Saved: {results_path}")
    print(results_df.to_string(index=False))

    # --- Threshold selection on VALIDATION (cost-sensitive), using calibrated soft-platt probabilities ---
    cost_fp = 1.0
    cost_fn = 5.0

    val_probs_soft_platt = cal_soft_platt.predict_proba(X_val)[:, 1]
    best_t, sweep_df = choose_threshold_cost_sensitive(y_val.values, val_probs_soft_platt, cost_fp, cost_fn)

    sweep_path = REPORT_DIR / "week4_threshold_sweep_val.csv"
    sweep_df.to_csv(sweep_path, index=False)
    print(f"Saved: {sweep_path}")
    print(f"Chosen threshold (min cost, fp={cost_fp}, fn={cost_fn}): {best_t:.2f}")

    # Reliability plots (TEST): before vs platt
    plot_reliability(
        y_test.values,
        probs_before=test_probs_logreg_before,
        probs_after=test_probs_logreg_platt,
        title="Reliability: LogReg (before vs Platt)",
        out_path=REPORT_DIR / "week4_reliability_logreg_platt.png",
    )
    plot_reliability(
        y_test.values,
        probs_before=test_probs_soft_before,
        probs_after=test_probs_soft_platt,
        title="Reliability: Soft Ensemble (before vs Platt)",
        out_path=REPORT_DIR / "week4_reliability_soft_platt.png",
    )
    print("Saved reliability plots to reports/")

    # Save candidate calibrated model (soft + platt)
    import joblib
    joblib.dump(cal_soft_platt, ARTIFACT_DIR / "week4_candidate_calibrated_soft_platt.joblib")

    (REPORT_DIR / "week4_summary.json").write_text(json.dumps({
        "cost_fp": cost_fp,
        "cost_fn": cost_fn,
        "chosen_threshold_val_min_cost": float(best_t),
        "ranking_compare_test_csv": str(results_path),
        "threshold_sweep_val_csv": str(sweep_path),
        "hard_vs_soft_test_json": "reports/week4_hard_vs_soft_test.json",
        "reliability_plots": [
            "reports/week4_reliability_logreg_platt.png",
            "reports/week4_reliability_soft_platt.png",
        ],
    }, indent=2))


if __name__ == "__main__":
    main()