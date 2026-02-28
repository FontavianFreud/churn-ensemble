# Week 4: Ensemble + Calibration + Cost-Sensitive Threshold

## Goal
Improve decision quality for churn prediction by:
- combining multiple models (ensemble)
- calibrating probabilities (confidence should match reality)
- choosing a threshold based on business costs (not default 0.5)

## Ensembles (hard vs soft)
- **Hard voting**: each model outputs a class (0/1); final prediction is majority vote. No direct probabilities.
- **Soft voting**: each model outputs a probability; final probability is the average of probabilities.

## Model results (ranking quality on test)
Best PR-AUC came from the **soft voting ensemble + Platt calibration**.

Top entries (test):
- soft_platt: ROC-AUC 0.833274, PR-AUC 0.630506
- soft_before: ROC-AUC 0.832938, PR-AUC 0.628282
- logreg_before: ROC-AUC 0.832445, PR-AUC 0.622128

Notes:
- Calibration usually changes probability quality more than it changes ROC-AUC/PR-AUC, so small AUC differences are expected.

Saved table:
- `reports/week4_ranking_compare_test.csv`

## Calibration
Methods tried:
- **Platt scaling** (sigmoid)
- **Isotonic regression**

Reliability plots saved (test):
- `reports/week4_reliability_logreg_platt.png`
- `reports/week4_reliability_soft_platt.png`

## Threshold selection (cost-sensitive on validation)
We selected a threshold by minimizing expected cost:
- **Cost(FP) = 1**
- **Cost(FN) = 5**
- Total cost = FP * 1 + FN * 5

Chosen threshold (validation): **0.11**

Metrics at threshold=0.11 (validation):
- Precision: **0.427704**
- Recall: **0.941176**
- Confusion matrix counts:
  - TN = 564
  - FP = 471
  - FN = 22
  - TP = 352
- Expected cost: **581**

Saved sweep:
- `reports/week4_threshold_sweep_val.csv`

## Outputs created
- `src/week4_ensemble_calibration.py`
- `reports/week4_ranking_compare_test.csv`
- `reports/week4_threshold_sweep_val.csv`
- `reports/week4_hard_vs_soft_test.json`
- `reports/week4_reliability_logreg_platt.png`
- `reports/week4_reliability_soft_platt.png`
- `reports/week4_summary.json`
- Candidate model saved locally (ignored by git):
  - `artifacts/week4_candidate_calibrated_soft_platt.joblib`

## Next steps (Week 5)
- Deploy the Week 4 candidate model behind FastAPI (return calibrated probability + decision).
- Dockerize the service.
- Add basic logging/monitoring signals (latency, prediction rate, confidence distribution).