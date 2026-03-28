# AI Decision Copilot for Customer Retention

A local ML-powered decision-support tool that predicts 
customer churn risk and enables interactive exploration 
of results through a local LLM-powered Q&A interface.

---

## Project Overview

| Item | Details |
|------|---------|
| Problem | Customer churn prediction and retention |
| ML Model | XGBoost / LightGBM |
| Explainability | SHAP values |
| LLM | Phi-3 Mini (via Ollama, runs locally) |
| Interface | Command-Line (CLI) |
| Dataset | IBM Telco Customer Churn (Kaggle) |
| Status | Week 1 — Specification complete |

---

## How It Works

**Layer 1 — ML Prediction**
Trains an XGBoost or LightGBM model on customer 
data to predict churn probability. Uses SHAP values 
to explain which factors contributed to each prediction.

**Layer 2 — LLM Q&A Agent**
A local Phi-3 Mini model runs on your machine via 
Ollama. You can ask free-form questions about 
prediction results and customer data.

---

## Example Usage
```bash
# Analyze one customer
python main.py --file customers.csv --customer_id 101

# Get all high risk customers
python main.py --file customers.csv --high-risk

# Summary statistics
python main.py --file customers.csv --summary

# Process full dataset
python main.py --file customers.csv --all
```

---

## Repository Structure
```
/data       Dataset files (CSV)
/src        Python source code
/tests      Unit tests
/outputs    Session logs and prediction results
/docs       Project specification document
```

---

## Documentation

Full project specification is available in the 
[/docs](/docs) folder.

---

## Tech Stack

- Python 3.x
- XGBoost / LightGBM
- SHAP
- Ollama + Phi-3 Mini
- pandas, scikit-learn, pytest
```
