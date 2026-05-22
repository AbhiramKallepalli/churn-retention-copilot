import pandas as pd
import numpy as np
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    precision_score,
    recall_score,
    classification_report
)
import pickle
import os
import sys

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from src.preprocessor import preprocess_data

def train_model(filepath):
    print("=" * 50)
    print("TRAINING XGBOOST MODEL")
    print("=" * 50)

    # Step 1 - Load and preprocess data
    print("\nStep 1: Loading and preprocessing data...")
    df = preprocess_data(filepath)

    # Step 2 - Split into features and target
    print("\nStep 2: Splitting features and target...")
    X = df.drop('Churn', axis=1)
    y = df['Churn']
    print(f"Features: {X.shape[1]} columns")
    print(f"Target: {y.sum()} churned out of {len(y)} customers")

    # Step 3 - Split into train and test sets
    print("\nStep 3: Splitting into train and test sets...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )
    print(f"Training set: {len(X_train)} customers")
    print(f"Test set: {len(X_test)} customers")

    # Step 4 - Train XGBoost model
    print("\nStep 4: Training XGBoost model...")
    model = XGBClassifier(
        n_estimators=100,
        max_depth=4,
        learning_rate=0.1,
        random_state=42,
        eval_metric='logloss',
        use_label_encoder=False
    )
    model.fit(X_train, y_train)
    print("Model trained successfully!")

    # Step 5 - Evaluate model
    print("\nStep 5: Evaluating model...")
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    accuracy = accuracy_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_prob)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)

    print("\n" + "=" * 50)
    print("MODEL PERFORMANCE RESULTS")
    print("=" * 50)
    print(f"Accuracy:  {accuracy:.4f} ({accuracy*100:.2f}%)")
    print(f"AUC Score: {auc:.4f} ({auc*100:.2f}%)")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print("=" * 50)

    # Step 6 - Save model to disk
    print("\nStep 6: Saving model to disk...")
    os.makedirs('outputs', exist_ok=True)
    with open('outputs/model.pkl', 'wb') as f:
        pickle.dump(model, f)

    # Save feature names
    with open('outputs/feature_names.pkl', 'wb') as f:
        pickle.dump(list(X.columns), f)

    print("Model saved to outputs/model.pkl")
    print("Feature names saved to outputs/feature_names.pkl")

    return model, X_test, y_test, list(X.columns)

# Run if executed directly
if __name__ == "__main__":
    model, X_test, y_test, feature_names = train_model(
        'data/WA_Fn-UseC_-Telco-Customer-Churn.csv'
    )
    print("\nModel training complete!")
    print("Ready for SHAP analysis.")