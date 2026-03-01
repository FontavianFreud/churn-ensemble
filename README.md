Churn Ensemble API (Soft Voting + Platt Calibration)

End-to-end churn prediction system built on the Telco Customer Churn dataset.

This project demonstrates:

Tabular ML pipeline with mixed numeric + categorical preprocessing

Soft-voting ensemble (LogReg + RandomForest + GradientBoosting)

Probability calibration (Platt scaling / sigmoid) for more meaningful probabilities

Cost-sensitive thresholding (FP cost = 1, FN cost = 5) to prioritize catching churners

FastAPI deployment with basic JSONL logging

Docker packaging

Project structure

notebooks/ — EDA and data inspection

src/ — training/build scripts + FastAPI app

reports/ — weekly summaries, comparisons, reliability plots, threshold sweep outputs

artifacts/ — saved model artifacts (ignored by git)

logs/ — prediction logs (ignored by git)

Data

Dataset: Telco Customer Churn (CSV)

Target: Churn (Yes/No)

Notes:

Dropped customerID (identifier)

Converted TotalCharges to numeric (blank strings become missing and are imputed)

Model

Candidate used for deployment:

Soft-voting ensemble of:

Logistic Regression

Random Forest

Gradient Boosting

Calibrated with Platt scaling (sigmoid) using CV

Decision threshold: 0.11 (chosen by minimizing FP1 + FN5 on validation)

API outputs:

prob_churn: calibrated probability of churn

pred_churn: decision at the chosen threshold

confidence: max(p, 1-p)

Quickstart (WSL / Linux)

Create environment

conda env create -f environment.yml

conda activate mle

Build the model artifact
This trains the calibrated ensemble and writes:
artifacts/week4_candidate_calibrated_soft_platt.joblib

Command:

python src/build_week4_model.py

Run the API locally
Command:

uvicorn src.api:app --reload --host 127.0.0.1 --port 8000

Health check:

http://127.0.0.1:8000/health

Interactive docs:

http://127.0.0.1:8000/docs

Example request
In /docs → POST /predict → Try it out, paste this JSON:

{
"record": {
"gender": "Female",
"SeniorCitizen": 0,
"Partner": "Yes",
"Dependents": "No",
"tenure": 1,
"PhoneService": "No",
"MultipleLines": "No phone service",
"InternetService": "DSL",
"OnlineSecurity": "No",
"OnlineBackup": "Yes",
"DeviceProtection": "No",
"TechSupport": "No",
"StreamingTV": "No",
"StreamingMovies": "No",
"Contract": "Month-to-month",
"PaperlessBilling": "Yes",
"PaymentMethod": "Electronic check",
"MonthlyCharges": 29.85,
"TotalCharges": "29.85"
}
}

Docker

Build image:

docker build -t churn-api:0.1 .

Run container:
If port 8000 is in use, map host port 8001 → container port 8000:

docker run --rm -p 8001:8000 churn-api:0.1

Test health endpoint:

curl -s http://127.0.0.1:8001/health

Logging / monitoring signals

Predictions are logged to:

logs/predictions.jsonl (JSON Lines)

Each prediction includes:

schema validity

probability + decision

confidence

latency

model version

timestamp

Reports

Week 4 artifacts:

reports/week4_ranking_compare_test.csv

reports/week4_threshold_sweep_val.csv

reports/week4_reliability_soft_platt.png

reports/week4_reliability_logreg_platt.png

Notes / limitations

This is a public dataset with limited behavioral features. Real churn systems typically achieve larger gains using product usage, engagement, and support interaction signals. This repo emphasizes correct methodology and MLE-style shipping: reproducible preprocessing, calibration, decision thresholds, and deployability.