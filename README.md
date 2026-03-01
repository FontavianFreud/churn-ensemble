# Churn Ensemble API

An end-to-end churn prediction system built on the IBM Telco Customer Churn dataset.

This project packages a production-style tabular ML workflow around a calibrated soft-voting ensemble, then exposes it through a FastAPI service with lightweight prediction logging and Docker support.

## Why This Project Stands Out

- **Business-aware modeling** with cost-sensitive threshold selection to prioritize catching likely churners
- **Robust tabular pipeline** for mixed numeric and categorical features
- **Probability calibration** using Platt scaling for more usable risk scores
- **Deployment-ready API** built with FastAPI
- **Containerized workflow** for reproducible local runs
- **Readable project artifacts** including threshold sweeps, ranking comparisons, and reliability plots

## Model Snapshot

### Dataset

- **Source:** IBM Telco Customer Churn CSV
- **Target:** `Churn` (`Yes` / `No`)

### Preprocessing

- Drops `customerID` as a pure identifier
- Converts `TotalCharges` to numeric
- Median imputes missing numeric values
- Most-frequent imputes missing categorical values
- Standardizes numeric features
- One-hot encodes categorical features

### Candidate Model Used for Deployment

A **soft-voting ensemble** of:

- Logistic Regression
- Random Forest
- Gradient Boosting

The ensemble is then **calibrated with Platt scaling** (`sigmoid`) using cross-validation.

### Decision Policy

- **Decision threshold:** `0.11`
- **Threshold objective:** minimize `FP * 1 + FN * 5`

This intentionally favors recall for churners because false negatives are treated as materially more expensive than false positives.

## Project Structure

```text
.
├── Dockerfile
├── README.md
├── environment.yml
├── notebooks/
│   └── 01_eda.ipynb
├── reports/
│   ├── week2_baseline.md
│   ├── week3_error_analysis_val.csv
│   ├── week3_model_compare_val.csv
│   ├── week3_summary.md
│   ├── week4_hard_vs_soft_test.json
│   ├── week4_ranking_compare_test.csv
│   ├── week4_reliability_logreg_platt.png
│   ├── week4_reliability_soft_platt.png
│   ├── week4_summary.json
│   ├── week4_summary.md
│   └── week4_threshold_sweep_val.csv
├── src/
│   ├── api.py
│   ├── build_week4_model.py
│   ├── train.py
│   ├── week3_analysis.py
│   └── week4_ensemble_calibration.py
└── tests/
    └── test_sanity.py
```

## Outputs and Runtime Artifacts

The repo creates a few local runtime directories during normal use:

- `artifacts/` for serialized model artifacts
- `logs/` for prediction logs
- `data/raw/` for the downloaded Telco CSV

These are generated automatically when needed and are typically not committed.

## Quickstart

### 1. Create Environment

```bash
conda env create -f environment.yml
conda activate mle
```

### 2. Build the Model Artifact

This script downloads the dataset if it is missing, trains the calibrated ensemble, and writes:

`artifacts/week4_candidate_calibrated_soft_platt.joblib`

```bash
python src/build_week4_model.py
```

### 3. Run the API Locally

```bash
uvicorn src.api:app --reload --host 127.0.0.1 --port 8000
```

Once the server is up:

- **Health check:** `http://127.0.0.1:8000/health`
- **Interactive docs:** `http://127.0.0.1:8000/docs`

## API Usage

### Health Endpoint

`GET /health`

Example response:

```json
{
  "status": "ok",
  "model_version": "week4_soft_platt_v1",
  "threshold": 0.11
}
```

### Prediction Endpoint

`POST /predict`

Example request body:

```json
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
```

Example response shape:

```json
{
  "prob_churn": 0.63,
  "pred_churn": 1,
  "threshold": 0.11,
  "confidence": 0.63,
  "latency_ms": 12.4,
  "model_version": "week4_soft_platt_v1"
}
```

### Output Fields

- `prob_churn`: calibrated probability of churn
- `pred_churn`: thresholded churn decision
- `threshold`: active business decision threshold
- `confidence`: `max(p, 1 - p)`
- `latency_ms`: request processing latency
- `model_version`: artifact version loaded by the API

## Docker

### Build the Image

```bash
docker build -t churn-api:0.1 .
```

### Run the Container

If port `8000` is already in use on your machine, map host port `8001` to container port `8000`:

```bash
docker run --rm -p 8001:8000 churn-api:0.1
```

### Test the Running Container

```bash
curl -s http://127.0.0.1:8001/health
```

## Logging and Monitoring Signals

Predictions are written to:

`logs/predictions.jsonl`

Each event captures lightweight operational signals such as:

- schema validity
- churn probability
- final decision
- confidence
- latency
- model version
- timestamp

This keeps the project simple while still demonstrating production-minded observability.

## Key Reports

Notable Week 4 outputs:

- `reports/week4_ranking_compare_test.csv`
- `reports/week4_threshold_sweep_val.csv`
- `reports/week4_reliability_soft_platt.png`
- `reports/week4_reliability_logreg_platt.png`

These artifacts help evaluate:

- ranking quality
- threshold tradeoffs
- probability calibration quality
- model comparison outcomes

## Technical Highlights

- Uses `ColumnTransformer` pipelines to keep preprocessing reproducible
- Calibrates ensemble probabilities rather than using raw classifier scores
- Separates ranking quality from business decision thresholding
- Serves predictions through a typed FastAPI interface using Pydantic models
- Builds the model artifact on-demand if the API starts before the artifact exists

## Notes and Limitations

This project uses a public churn dataset with relatively limited behavioral signal. In real commercial churn systems, stronger lift often comes from richer product telemetry, engagement history, support interactions, and pricing context.

The focus here is on **sound ML engineering practice**:

- reproducible preprocessing
- interpretable deployment choices
- calibrated probabilities
- explicit thresholding logic
- shipping a model behind an API
