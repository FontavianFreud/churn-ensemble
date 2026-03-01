FROM python:3.11-slim

WORKDIR /app
COPY . /app

RUN pip install --no-cache-dir \
  fastapi uvicorn[standard] pydantic pandas numpy scikit-learn joblib matplotlib

# Build the calibrated ensemble artifact at image build time
RUN python src/build_week4_model.py

EXPOSE 8000
CMD ["uvicorn", "src.api:app", "--host", "0.0.0.0", "--port", "8000"]