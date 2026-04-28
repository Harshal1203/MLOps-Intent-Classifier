# 🚀 MLOps Intent Classifier

> End-to-end MLOps pipeline for multi-class intent classification — fine-tuned transformer model with full CI/CD, experiment tracking, containerized serving, and GenAI-powered synthetic data generation.

---

## 📐 Architecture

```
                        ┌─────────────────────────────────────────────────────┐
                        │               GitHub Actions (CI/CD)                │
                        │  lint → test → trigger pipeline → deploy on promote │
                        └───────────────────────┬─────────────────────────────┘
                                                │
           ┌────────────────────────────────────▼──────────────────────────────────┐
           │                        Kubeflow Pipeline                               │
           │  data_validation → preprocess → train → evaluate → register/promote   │
           └────────┬────────────────────┬──────────────────────┬──────────────────┘
                    │                    │                       │
           ┌────────▼───────┐  ┌────────▼───────┐   ┌──────────▼────────┐
           │  DVC (data     │  │ MLflow (expts  │   │  Model Registry   │
           │  versioning)   │  │  + artifacts)  │   │  Staging → Prod   │
           └────────────────┘  └────────────────┘   └──────────┬────────┘
                                                               │ promote
                                                    ┌──────────▼────────┐
                                                    │ FastAPI on K8s     │
                                                    │ /predict /health   │
                                                    └──────────┬────────┘
                                                               │
                                                    ┌──────────▼────────┐
                                                    │ Monitoring         │
                                                    │ Evidently + Grafana│
                                                    └───────────────────┘
```

## 🗂️ Project Structure

```
mlops-intent-classifier/
├── data/
│   ├── raw/            # Original data (DVC tracked)
│   ├── processed/      # Cleaned, tokenized (DVC tracked)
│   └── synthetic/      # LLM-generated samples (DVC tracked)
├── src/
│   ├── train.py        # PyTorch fine-tuning + MLflow logging
│   ├── evaluate.py     # Metrics, thresholds, reports
│   ├── preprocess.py   # Data cleaning & tokenization
│   └── datagen.py      # LLM synthetic data generator
├── api/
│   ├── main.py         # FastAPI inference server
│   ├── schemas.py      # Pydantic request/response models
│   └── Dockerfile      # Container definition
├── pipeline/
│   └── kubeflow_pipeline.py  # Full Kubeflow DAG
├── .github/workflows/
│   ├── ci.yml          # Lint, test, trigger pipeline
│   └── cd.yml          # Deploy on model promotion
├── k8s/
│   ├── deployment.yaml
│   ├── service.yaml
│   └── hpa.yaml        # Horizontal pod autoscaler
├── monitoring/
│   ├── drift_report.py
│   └── grafana_dashboard.json
├── configs/
│   └── config.yaml     # Centralized config
├── tests/
│   ├── test_train.py
│   ├── test_api.py
│   └── test_preprocess.py
├── notebooks/          # EDA and experimentation
├── dvc.yaml            # DVC pipeline definition
├── MLproject           # MLflow project entry point
├── requirements.txt
└── README.md
```

## 🛠️ Tech Stack

| Layer                  | Technology                        |
| ---------------------- | --------------------------------- |
| Model Training         | PyTorch, HuggingFace Transformers |
| Experiment Tracking    | MLflow                            |
| Pipeline Orchestration | Kubeflow Pipelines                |
| Data Versioning        | DVC                               |
| CI/CD                  | GitHub Actions                    |
| Serving                | FastAPI                           |
| Containerization       | Docker                            |
| Deployment             | Kubernetes (K8s)                  |
| Monitoring             | Evidently AI, Grafana, Prometheus |
| Database               | PostgreSQL                        |
| GenAI Data Gen         | LangChain + Claude/OpenAI API     |

## ⚡ Quickstart

```bash
# 1. Clone and install
git clone https://github.com/YOUR_USERNAME/mlops-intent-classifier.git
cd mlops-intent-classifier
pip install -r requirements.txt

# 2. Pull data
dvc pull

# 3. Run training experiment
python src/train.py --config configs/config.yaml

# 4. View experiments
mlflow ui

# 5. Start API locally
cd api && uvicorn main:app --reload

# 6. Run tests
pytest tests/
```

## 📊 Dataset

Uses the [CLINC150](https://github.com/clinc/oos-eval) dataset — 150 intent classes, ~23,000 samples across 10 domains. Ideal for demonstrating multi-class classification and drift detection.

## 🔁 Pipeline Stages

1. **Data Validation** — schema checks, class distribution, quality gates
2. **Preprocessing** — tokenization, train/val/test split, DVC versioning
3. **Training** — DistilBERT fine-tuning, MLflow run tracking
4. **Evaluation** — accuracy, F1, confusion matrix, latency benchmark
5. **Promotion** — auto-promote to Staging/Production if thresholds pass
6. **Deployment** — CD pipeline re-deploys K8s pods on promotion

## 📈 Monitoring

Predictions are logged to PostgreSQL. Evidently AI runs nightly drift reports comparing incoming data distributions to the training baseline. If drift exceeds threshold, a retrain is triggered automatically.

---

## 🧠 Key Design Decisions

- **DistilBERT over BERT**: 40% smaller, 60% faster inference, ~97% of BERT accuracy for classification tasks
- **MLflow Model Registry**: Decouples promotion logic from deployment — CI/CD only triggers on `Production` stage transitions
- **DVC for data**: Keeps large files out of Git while maintaining full reproducibility via content hashing
- **Synthetic data via LLM**: Addresses class imbalance for rare intents without expensive manual labeling

---

_Built as a portfolio project showcasing end-to-end ML Engineering skills._
