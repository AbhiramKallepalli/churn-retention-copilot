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
| Interface | CLI (Phase 1) + Streamlit Web UI (Phase 2) |
| Dataset | IBM Telco Customer Churn (Kaggle) |
| Status | Week 1 — Specification complete |

---

## How It Works

**Layer 1 — ML Prediction**
Trains an XGBoost or LightGBM model on customer 
data to predict churn probability. Uses SHAP values 
to explain which factors contributed to each 
prediction. The model is trained once, saved to disk, 
and reused for all future predictions.

**Layer 2 — LLM Q&A Agent**
A local Phi-3 Mini model runs on your machine via 
Ollama. You can ask free-form questions about 
prediction results and customer data. The LLM only 
interprets and explains — it does not make predictions.

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

## Interface Plan

**Phase 1 — Command-Line Interface (CLI)**

The initial prototype uses a CLI for rapid development 
and testing of the core ML model and LLM Q&A 
integration. This allows full focus on building and 
validating the pipeline before adding a visual layer.

**Phase 2 — Web Interface (Streamlit)**

A lightweight web-based UI built using Streamlit, 
making the tool accessible to non-technical business 
users without requiring command-line knowledge.

The Streamlit interface includes three pages:

- **Upload and Predict** — Upload a CSV file, run 
  the ML model, view all customers with churn 
  probability and risk category in a table
- **Customer Detail** — Enter a customer ID to view 
  full SHAP analysis and ask questions to the 
  LLM Q&A agent
- **Summary Dashboard** — View risk distribution 
  chart and average monthly charges per risk category

---

## Repository Structure
/data       Dataset files (CSV)
/src        Python source code
/tests      Unit tests
/outputs    Session logs and prediction results
/docs       Project specification document

---

## Tech Stack

- Python 3.x (pandas, scikit-learn, XGBoost, 
  LightGBM, SHAP, pytest)
- Ollama + Phi-3 Mini (local LLM runtime)
- Streamlit (web interface — Phase 2)

---

## Documentation

Full project specification is available in the 
[/docs](/docs) folder.
