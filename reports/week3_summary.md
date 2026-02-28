# Week 3: Model Comparison + Segment Error Analysis

## What I did
- Trained and compared 3 models on the validation set:
  - Logistic Regression
  - Random Forest
  - Gradient Boosting
- Selected the best model by validation PR-AUC.
- Tuned the classification threshold to prioritize catching churners (higher recall).
- Ran segment-level error analysis on validation by:
  - tenure bucket
  - Contract
  - InternetService
  - PaymentMethod

## Model comparison (validation)
| Model | Val ROC-AUC | Val PR-AUC | Precision@0.5 | Recall@0.5 |
|---|---:|---:|---:|---:|
| logreg | 0.859971 | 0.677947 | 0.698113 | 0.593583 |
| rf     | 0.854850 | 0.671604 | 0.686567 | 0.491979 |
| gb     | 0.853351 | 0.667527 | 0.683274 | 0.513369 |

**Winner (by val PR-AUC):** logreg

## Recall-focused thresholding (validation)
- Target recall: >= 0.75
- Chosen threshold: 0.33

Metrics at threshold=0.33 (validation):
- Precision: 0.550864
- Recall: 0.767380
- TN=801, FP=234, FN=87, TP=287

## Segment error analysis (validation)
Worst recall segments (n >= 100) observed:
- Contract = Two year (very low churn rate, caught 0 of 4 churners)
- InternetService = No (caught 1 of 23 churners)
- Contract = One year (caught 5 of 31 churners)

Notes:
- Low churn-rate segments naturally produce fewer positive examples; recall can look very low even with small FN counts.
- High churn-rate segments (month-to-month, low tenure, fiber optic, electronic check) show much higher recall at the lowered threshold.

## Artifacts produced
- reports/week3_model_compare_val.csv
- reports/week3_error_analysis_val.csv
- artifacts/week3_best_logreg.joblib (not tracked by git)

## Next steps (Week 4)
- Build a soft-voting ensemble (logreg + rf + gb) and calibrate probabilities.
- Continue using validation for threshold decisions (optimize for higher recall).
- Add monitoring-friendly logging outputs (prediction rate, confidence distribution).