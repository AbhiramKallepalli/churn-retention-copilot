# AI Decision Copilot for Customer Retention
## Final Project Report

---

## 1. Executive Summary

This project presents the design and implementation of a local AI-powered decision-support tool for customer churn prediction and retention. The system combines a machine learning model (XGBoost) for churn prediction, SHAP values for explainability, and a local Large Language Model (Phi-3 Mini via Ollama) for interactive Q&A. The tool enables business
analysts to identify high-risk customers, understand why they are at risk, and receive AI-generated retention recommendations - all running entirely on a local machine without requiring internet connectivity or cloud services.

The final system achieved an AUC score of 84.42% and accuracy of 80.20% on the IBM Telco Customer Churn dataset,demonstrating strong predictive performance for a first implementation.

---

## 2. Business Problem Statement

Customer churn - the loss of customers to competitors or disengagement - is a persistent and costly challenge across subscription-based industries including telecommunications, SaaS, e-commerce, and financial services. Research  consistently shows that retaining an existing customer costs significantly less than acquiring a new one, yet most organizations lack tools that help frontline analysts make fast, explainable retention decisions.

### 2.1 The Decision Being Supported

The specific decision this project targets is:

> "Should this customer be flagged as high churn risk, and what action should be taken to retain them?"

This decision is:
- **Recurring** - made daily by customer success teams
- **High-impact** - directly tied to revenue
- **Under-supported** - most tools predict churn but do not explain why or recommend actions

### 2.2 Stakeholders

| Stakeholder | Role |
|-------------|------|
| Customer Success Managers | Primary users who monitor account health |
| Business Analysts | Analyze churn trends across segments |
| Marketing Teams | Use recommendations for targeted campaigns |
| Product Managers | Use churn signals for product decisions |

---

## 3. Dataset Description

### 3.1 Dataset Overview

The project uses the IBM Telco Customer Churn dataset sourced from Kaggle:
https://www.kaggle.com/datasets/blastchar/telco-customer-churn

| Attribute | Value |
|-----------|-------|
| Total Records | 7,043 customers |
| Total Features | 21 columns |
| Churn Rate | 26.5% (1,869 churned) |
| Retention Rate | 73.5% (5,174 retained) |

### 3.2 Key Features Used

| Feature | Type | Role |
|---------|------|------|
| tenure | Numeric | Months as customer |
| Contract | Categorical | Contract type |
| MonthlyCharges | Numeric | Monthly bill amount |
| TechSupport | Categorical | Has tech support |
| OnlineSecurity | Categorical | Has online security |
| InternetService | Categorical | Type of internet |
| PaymentMethod | Categorical | Payment method |
| Churn | Binary | Target variable |

### 3.3 Key Data Insights

Analysis of the dataset revealed several important patterns:

- Customers on **month-to-month contracts** churn at a rate of 42.7% compared to only 2.8% for two-year contract customers and 11.3% for one-year contract customers.- Customers with **tenure under 6 months** show significantly higher churn rates.
- **Higher monthly charges** correlate with increased churn risk.
- Customers without **tech support or online security** are more likely to churn.

---

## 4. System Architecture

### 4.1 Two-Layer Design

The system follows a two-layer architecture:

**Layer 1 - ML Prediction Engine**
Handles all churn prediction tasks using XGBoost.Takes preprocessed customer data as input and outputs churn probability
and SHAP-based feature importance scores. This layer is fully deterministic and does not involve the LLM.

**Layer 2 - LLM Q&A Agent**
Handles all user interaction using Phi-3 Mini running locally via Ollama. Receives structured outputs from Layer 1 and generates natural language responses to user questions. Does not make predictions - only interprets and explains.

### 4.2 Data Flow
CSV Data
↓
data_loader.py (load and explore)
↓
preprocessor.py (clean and encode)
↓
scorer.py (XGBoost prediction)
↓
shap_explainer.py (SHAP values)
↓
llm_interface.py (LLM Q&A)
↓
main.py (CLI commands)
↓
streamlit_app.py (Web interface)

### 4.3 Technology Stack

| Component | Technology | Reason |
|-----------|-----------|--------|
| ML Model | XGBoost | High accuracy, handles tabular data well |
| Explainability | SHAP | Industry standard for ML explainability |
| Local LLM | Phi-3 Mini | Optimized for CPU, low memory usage |
| LLM Runtime | Ollama | Simple local LLM deployment on Windows |
| Web Interface | Streamlit | Rapid UI development in pure Python |
| Data Processing | pandas | Industry standard for data manipulation |
| CLI Interface | argparse | Standard Python CLI library |

### 4.4 Design Decisions

**Why XGBoost over other models?**
XGBoost was chosen because it handles tabular data extremely well, is robust to missing values, provides feature importance scores compatible with SHAP, and is recommended for tabular data classification tasks. It achieved 84.42% AUC on the test
set without any hyperparameter tuning.

**Why Phi-3 Mini over larger models?**
The project hardware has 16GB RAM with no dedicated GPU. Phi-3 Mini requires only 2-3GB of memory and runs efficiently on CPU while still providing high quality natural language responses for Q&A tasks.

**Why Ollama over cloud APIs?**
The project specification requires fully offline operation. Ollama enables running LLMs locally without internet connectivity, ensuring data privacy and zero API costs.

**Why Streamlit over traditional web frameworks?**
Streamlit allows building professional web interfaces using only Python, without requiring HTML, CSS, or JavaScript knowledge. This was appropriate given the project scope and timeline.

---

## 5. Implementation Details

### 5.1 Data Preprocessing

The raw dataset required several preprocessing steps before being suitable for ML model training:

1. **TotalCharges fix** - Blank strings converted to numeric values, missing values imputed with column median (11 records affected)

2. **Categorical encoding** - Text categories converted to numeric using one-hot encoding, expanding features from 21 to 42 columns

3. **Binary conversion** - Yes/No columns converted to 1/0 for model compatibility

4. **Target encoding** - Churn column converted from Yes/No to 1/0

5. **ID removal** - customerID removed as it carries no predictive information

### 5.2 Model Training

The XGBoost model was trained with the following configuration:

| Parameter | Value | Reason |
|-----------|-------|--------|
| n_estimators | 100 | Good balance of accuracy and speed |
| max_depth | 4 | Prevents overfitting |
| learning_rate | 0.1 | Standard starting value |
| random_state | 42 | Ensures reproducibility |
| test_size | 0.2 | 80/20 train/test split |
| stratify | y | Maintains class balance in split |

Training set: 5,634 customers
Test set: 1,409 customers

### 5.3 SHAP Implementation

SHAP (SHapley Additive exPlanations) values were calculated using TreeExplainer, which is optimized for tree-based models like XGBoost. For each customer prediction, the top 5 contributing features are extracted and ranked by absolute SHAP value magnitude.

Positive SHAP values increase churn risk.
Negative SHAP values decrease churn risk.

### 5.4 LLM Q&A Implementation

The LLM Q&A agent receives structured context containing customer information and SHAP results. A carefully engineered prompt instructs the model to answer questions based only on provided data, preventing hallucination and ensuring grounded
responses.

Key design decisions:
- Temperature set to 0 for consistent responses
- Prompt explicitly instructs LLM not to fabricate information
- Context includes customer data, churn probability, risk category, and top 5 SHAP factors

### 5.5 Interface Implementation

**CLI Interface**
Four commands implemented using argparse:
- --customer_id: Single customer analysis
- --summary: Dataset statistics
- --all: Process all customers
- --high-risk: List high risk customers

**Streamlit Web Interface**
Three pages implemented:
- Page 1: Upload CSV and get predictions
- Page 2: Customer detail with SHAP chart and Q&A
- Page 3: Summary dashboard with charts

**run_app.bat**
A batch file created for non-technical users to start the application with a simple double-click, without requiring terminal knowledge.

---

## 6. Model Performance Results

### 6.1 Evaluation Metrics

| Metric | Value | Interpretation |
|--------|-------|----------------|
| Accuracy | 80.20% | 80% of predictions correct |
| AUC Score | 84.42% | Strong discriminative ability |
| Precision | 65.37% | 65% of predicted churners actually churned |
| Recall | 54.01% | Identified 54% of actual churners |

### 6.2 Results Interpretation

The AUC score of 84.42% indicates strong model performance. An AUC above 80% is generally considered good for churn prediction tasks. The model significantly outperforms random guessing (AUC = 50%).

The precision of 65.37% means that when the model flags a customer as high risk, it is correct approximately 65% of the time. This is acceptable for a retention use case where the cost of a missed churner (false negative) is higher than the cost of an unnecessary retention offer (false positive).

### 6.3 Risk Distribution

After running predictions on all 7,043 customers:

| Risk Category | Count | Percentage | Avg Monthly Charges |
|---------------|-------|------------|---------------------|
| High Risk | 592 | 8.4% | $81.58 |
| Medium Risk | 2,021 | 28.7% | $73.64 |
| Low Risk | 4,430 | 62.9% | $58.46 |

**Key insight:** Higher monthly charges correlate strongly with higher churn risk, suggesting price sensitivity is a major driver of customer dissatisfaction.

### 6.4 Top Churn Predictors

Based on SHAP analysis across all customers, the most important features driving churn are:

1. **Contract type** - Month-to-month customers are significantly more likely to churn
2. **Tenure** - Short tenure strongly predicts churn, especially under 6 months
3. **Total charges** - Higher cumulative charges increase churn risk
4. **Monthly charges** - High monthly bills contribute to churn
5. **Internet service type** - Fiber optic customers show higher churn rates

---

## 7. Evaluation Results

### 7.1 Test Scenarios

All 6 test scenarios from the project specification were executed and passed:

| Scenario | Description | Result |
|----------|-------------|--------|
| 1 | High risk customer analysis | ✅ Passed |
| 2 | Low risk customer analysis | ✅ Passed |
| 3 | Unknown customer ID | ✅ Passed |
| 4 | Invalid CSV file | ✅ Passed |
| 5 | Summary command | ✅ Passed |
| 6 | High risk list | ✅ Passed |

### 7.2 LLM Q&A Evaluation

LLM responses were evaluated using the 0-3 rubric defined in the project specification:

| Test Question | Score | Notes |
|---------------|-------|-------|
| Why is this customer at risk? | 3 | Correctly cited SHAP factors |
| What retention strategy to recommend? | 3 | Specific actionable recommendations |
| What is the most important factor? | 3 | Correctly identified contract type |
| Is this customer likely to stay? | 3 | Accurate low risk assessment |
| What should we do to retain? | 3 | Detailed retention strategy |

Average LLM alignment score: 3.0 / 3.0

All responses were grounded in actual model outputs and SHAP values. No hallucinated information was detected in any response.

---

## 8. Limitations

### 8.1 Model Limitations

- **Recall of 54%** means the model misses approximately 46% of actual churners. Further hyperparameter tuning could improve this metric.

- **Static model** - The model is trained once and does not automatically retrain as new customer data becomes available.

- **Dataset scope** - The model was trained on telecom data and may not generalize well to other industries without retraining.

### 8.2 System Limitations

- **Local only** - The system runs on a single machine and cannot support multiple concurrent users without deployment to a server.

- **Terminal in background** - Despite the run_app.bat convenience file, a terminal window remains in the taskbar during operation. Full elimination requires server deployment.

- **LLM response time** - Phi-3 Mini on CPU takes 10-30 seconds per response depending on question complexity and hardware.

### 8.3 Data Limitations

- The IBM Telco dataset is a publicly available synthetic dataset. Real-world performance may differ with actual company data.

---

## 9. Lessons Learned

### 9.1 Technical Lessons

1. **Data preprocessing is critical** - Over 50% of development time was spent on data cleaning and preprocessing. Clean data is the foundation of any ML system.

2. **SHAP adds significant value** - Without SHAP explainability, the model would be a black box. SHAP transforms predictions into actionable business insights.

3. **Prompt engineering matters** - The quality of LLM responses depends heavily on how the prompt is structured. Grounding the LLM in structured data prevents hallucination.

4. **Local LLMs are viable** - Phi-3 Mini running via Ollama demonstrated that capable AI can run on consumer hardware without cloud dependencies.

### 9.2 Business Lessons

1. **New customers are highest risk** - The data clearly showed customers in their first 6 months are most likely to churn. Retention efforts should focus on this critical period.

2. **Contract type is the strongest predictor** - Encouraging customers to commit to longer contracts is the single most effective retention strategy based on the data.

3. **Price sensitivity matters** - Higher monthly charges consistently increased churn risk, suggesting competitive pricing and value demonstration are important retention tools.

---

## 10. Future Improvements

### 10.1 Technical Improvements

1. **Model improvement** - Implement hyperparameter tuning using cross-validation to improve recall from 54% to 70%+

2. **Automatic retraining** - Build a pipeline that retrains the model monthly as new customer data becomes available

3. **Cloud deployment** - Deploy to Streamlit Cloud or AWS to make the tool accessible to multiple users via a public URL

4. **Replace Ollama with cloud API** - Use OpenAI or Anthropic API for faster and more capable LLM responses in production

5. **Add RAG pipeline** - Implement Retrieval-Augmented Generation to allow the LLM to reference company knowledge base documents

### 10.2 Feature Improvements

1. **Batch LLM analysis** - Generate AI recommendations for all high-risk customers automatically in batch mode

2. **Email integration** - Automatically send retention offers to high-risk customers based on AI recommendations

3. **Historical tracking** - Track customer risk scores over time to identify deteriorating relationships early

4. **A/B testing** - Test different retention strategies and measure their effectiveness

---

## 11. Conclusion

This project successfully delivered a complete local AI-powered decision-support tool for customer churn prediction and retention. The system combines traditional machine learning (XGBoost) with modern AI capabilities (local LLM) to provide business analysts with fast, explainable, and actionable churn risk assessments.

The tool achieved strong predictive performance with an AUC of 84.42% and demonstrated that capable AI applications can run entirely on local hardware without cloud dependencies. All project requirements defined in the specification were implemented and verified through systematic testing.

The project provided valuable hands-on experience with the full machine learning pipeline from data preprocessing through model deployment, as well as practical experience integrating large language models into real-world business applications.

---

*Report prepared as part of the SPS Corps - Analytics Initiative*
*May 2026*