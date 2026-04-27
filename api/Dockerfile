FROM python:3.11-slim

WORKDIR /app

# Install system deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    libpq-dev \
    && rm -rf /var/lib/apt/lists/*

# Install Python deps (separate layer for caching)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy app code
COPY api/         ./api/
COPY configs/     ./configs/
COPY data/processed/id2label.json ./data/processed/id2label.json

WORKDIR /app/api

EXPOSE 8000

ENV MODEL_STAGE=Production

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
