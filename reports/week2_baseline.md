# Week 2 Baseline: Telco Customer Churn (LogReg)

## Goal
Predict whether a customer will churn (Yes/No) given account and service features.

## Data
- Rows: 7043
- Target: Churn (Yes/No), churn rate ≈ 26.5%
- Key preprocessing notes:
  - Dropped `customerID` (identifier, high cardinality)
  - Converted `TotalCharges` from string to numeric; 11 blank entries became missing (NaN) and were imputed

## Baseline model
- Model: Logistic Regression
- Preprocessing:
  - Numeric: median imputation + standard scaling
  - Categorical: most-frequent imputation + one-hot encoding
- Split: train/val/test = 60/20/20, stratified

## Metrics (from artifacts/metrics.json)
- Validation ROC-AUC: 0.860
- Test ROC-AUC: 0.832
- Validation PR-AUC: 0.678
- Test PR-AUC: 0.622
- Confusion matrix (test, threshold=0.5):
  - TN=913, FP=122, FN=177, TP=197

## Interpretation
- Baseline ranking quality is solid (ROC-AUC ~0.83).
- At threshold 0.5, recall is modest (misses many churners). Since business preference is catching more churners, threshold should likely be lowered and/or model improved.

## Next steps (Week 3)
1) Train Random Forest and Gradient Boosting.
2) Compare ROC-AUC + PR-AUC on validation.
3) Do error analysis by segments (tenure buckets, contract type, internet service, payment method).
4) Tune threshold toward higher recall (accept more false positives).