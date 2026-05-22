import pandas as pd
import numpy as np
import shap
import pickle
import os
import sys

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from src.preprocessor import preprocess_data

def load_model():
    # Load trained model from disk
    with open('outputs/model.pkl', 'rb') as f:
        model = pickle.load(f)

    # Load feature names
    with open('outputs/feature_names.pkl', 'rb') as f:
        feature_names = pickle.load(f)

    return model, feature_names

def get_shap_values(customer_id, filepath):
    print("=" * 50)
    print(f"SHAP ANALYSIS FOR CUSTOMER: {customer_id}")
    print("=" * 50)

    # Step 1 - Load model
    print("\nStep 1: Loading trained model...")
    model, feature_names = load_model()
    print("Model loaded successfully!")

    # Step 2 - Load and preprocess data
    print("\nStep 2: Loading customer data...")
    df_original = pd.read_csv(filepath)
    df_processed = preprocess_data(filepath)

    # Step 3 - Find the customer
    print("\nStep 3: Finding customer...")
    if customer_id not in df_original['customerID'].values:
        print(f"Customer {customer_id} not found!")
        return None

    # Get customer index
    customer_idx = df_original[
        df_original['customerID'] == customer_id
    ].index[0]

    # Get customer row
    customer_row = df_processed.drop(
        'Churn', axis=1
    ).iloc[customer_idx]

    # Get original customer info
    original_info = df_original.iloc[customer_idx]

    print(f"Customer found at index: {customer_idx}")

    # Step 4 - Get churn prediction
    print("\nStep 4: Getting churn prediction...")
    customer_array = customer_row.values.reshape(1, -1)
    churn_prob = model.predict_proba(customer_array)[0][1]
    churn_pred = "HIGH RISK" if churn_prob > 0.7 else \
                 "MEDIUM RISK" if churn_prob > 0.3 else \
                 "LOW RISK"

    print(f"Churn Probability: {churn_prob*100:.1f}%")
    print(f"Risk Category: {churn_pred}")

    # Step 5 - Calculate SHAP values
    print("\nStep 5: Calculating SHAP values...")
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(customer_array)

    # Step 6 - Get top risk factors
    print("\nStep 6: Getting top risk factors...")
    feature_importance = list(zip(
        feature_names,
        shap_values[0]
    ))

    # Sort by absolute SHAP value
    feature_importance.sort(
        key=lambda x: abs(x[1]),
        reverse=True
    )

    # Get top 5 factors
    top_factors = feature_importance[:5]

    # Step 7 - Display results
    print("\n" + "=" * 50)
    print("CUSTOMER RISK ANALYSIS")
    print("=" * 50)
    print(f"Customer ID:       {customer_id}")
    print(f"Tenure:            {original_info['tenure']} months")
    print(f"Contract:          {original_info['Contract']}")
    print(f"Monthly Charges:   ${original_info['MonthlyCharges']}")
    print(f"Internet Service:  {original_info['InternetService']}")
    print("-" * 50)
    print(f"Churn Probability: {churn_prob*100:.1f}%")
    print(f"Risk Category:     {churn_pred}")
    print("-" * 50)
    print("Top Risk Factors (SHAP):")
    for i, (feature, value) in enumerate(top_factors, 1):
        direction = "increases" if value > 0 else "decreases"
        print(f"  {i}. {feature}: {value:+.4f} "
              f"({direction} churn risk)")
    print("=" * 50)

    return {
        'customer_id': customer_id,
        'churn_probability': churn_prob,
        'risk_category': churn_pred,
        'top_factors': top_factors,
        'original_info': original_info
    }

# Test it
if __name__ == "__main__":
    result = get_shap_values(
        '7590-VHVEG',
        'data/WA_Fn-UseC_-Telco-Customer-Churn.csv'
    )