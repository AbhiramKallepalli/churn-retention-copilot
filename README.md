# AI Decision Copilot for Customer Retention

A local ML-powered decision-support tool that predicts 
customer churn risk and enables interactive exploration 
of results through a local LLM-powered Q&A interface.

---

## Project Overview

| Item | Details |
|------|---------|
| Problem | Customer churn prediction and retention |
| ML Model | XGBoost |
| Explainability | SHAP values |
| LLM | Phi-3 Mini (via Ollama, runs locally) |
| Interface | CLI + Streamlit Web UI |
| Dataset | IBM Telco Customer Churn (Kaggle) |
| Status | Project Complete — May 2026 |

---

## How It Works

**Layer 1 — ML Prediction**
Trains an XGBoost model on customer data to predict 
churn probability. Uses SHAP values to explain which 
factors contributed to each prediction. The model is 
trained once, saved to disk, and reused for all 
future predictions.

**Layer 2 — LLM Q&A Agent**
A local Phi-3 Mini model runs on your machine via 
Ollama. You can ask free-form questions about 
prediction results and customer data. The LLM only 
interprets and explains — it does not make predictions.

---

## Setup Instructions

### Prerequisites
- Python 3.12+
- Ollama (download from https://ollama.com)

### Installation

**Step 1 — Clone the repository**
```bash
git clone https://github.com/AbhiramKallepalli/churn-retention-copilot
cd churn-retention-copilot
```

**Step 2 — Create virtual environment**
```bash
python -m venv venv
venv\Scripts\activate
```

**Step 3 — Install dependencies**
```bash
pip install -r requirements.txt
```

**Step 4 — Install Ollama and Phi-3 Mini**
```bash
# Download Ollama from https://ollama.com
# Then run:
ollama pull phi3:mini
```

**Step 5 — Train the ML model**
```bash
python src/scorer.py
```

---

## Running the Application

### Option 1 — Web Interface (Recommended)
Double click `run_app.bat`
Browser opens automatically with the Streamlit app.

### Option 2 — Command Line Interface

```bash
# Summary statistics
python src/main.py --file data/WA_Fn-UseC_-Telco-Customer-Churn.csv --summary

# List high risk customers
python src/main.py --file data/WA_Fn-UseC_-Telco-Customer-Churn.csv --high-risk

# Analyze specific customer
python src/main.py --file data/WA_Fn-UseC_-Telco-Customer-Churn.csv --customer_id 9497-QCMMS

# Process all customers
python src/main.py --file data/WA_Fn-UseC_-Telco-Customer-Churn.csv --all
```

---

## Streamlit Web Interface

Three pages available:

- **Upload and Predict** — Upload a CSV file, run 
  the ML model, view all customers with churn 
  probability and risk category in a table
- **Customer Detail** — Enter a customer ID to view 
  full SHAP analysis and ask questions to the 
  LLM Q&A agent
- **Summary Dashboard** — View risk distribution 
  chart and average monthly charges per risk category

---

## Model Performance

| Metric | Value |
|--------|-------|
| Accuracy | 80.20% |
| AUC Score | 84.42% |
| Precision | 65.37% |
| Recall | 54.01% |

---
## Testing

Run the automated test suite:

```bash
pytest tests/test_churn.py -v
```

All 7 tests pass successfully:
- test_data_loads
- test_preprocessing
- test_high_risk_customer
- test_low_risk_customer
- test_missing_customer
- test_model_loads
- test_risk_categories

---

## Repository Structure

/data       Dataset files (CSV)
/src        Python source code
/tests      Unit tests
/outputs    Session logs and prediction results
/docs       Project specification and final report

---

## Tech Stack

- Python 3.x (pandas, scikit-learn, XGBoost, SHAP)
- Ollama + Phi-3 Mini (local LLM runtime)
- Streamlit (web interface)

---

## Documentation

Full project specification and final report are 
available in the [/docs](/docs) folder.



