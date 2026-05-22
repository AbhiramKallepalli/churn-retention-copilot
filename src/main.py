import argparse
import sys
import os
import pandas as pd

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from src.preprocessor import preprocess_data
from src.shap_explainer import get_shap_values, load_model
from src.llm_interface import start_qa_session, format_context, ask_llm

def analyze_customer(filepath, customer_id):
    """
    Analyze a single customer
    """
    print(f"\nAnalyzing customer: {customer_id}")
    
    # Get SHAP analysis
    result = get_shap_values(customer_id, filepath)
    
    if result is None:
        print(f"Customer {customer_id} not found.")
        return
    
    # Start Q&A session
    start_qa_session(result)


def process_all(filepath):
    """
    Process all customers and save results
    """
    print("\n" + "=" * 50)
    print("PROCESSING ALL CUSTOMERS")
    print("=" * 50)
    
    # Load original data
    df_original = pd.read_csv(filepath)
    
    # Load model
    model, feature_names = load_model()
    
    # Preprocess data
    df_processed = preprocess_data(filepath)
    X = df_processed.drop('Churn', axis=1)
    
    # Get predictions for all customers
    print("Getting predictions for all customers...")
    probs = model.predict_proba(X)[:, 1]
    
    # Add results to original dataframe
    df_original['ChurnProbability'] = (probs * 100).round(1)
    df_original['RiskCategory'] = df_original[
        'ChurnProbability'
    ].apply(
        lambda x: 'HIGH' if x > 70
        else 'MEDIUM' if x > 30
        else 'LOW'
    )
    
    # Sort by churn probability
    df_results = df_original[[
        'customerID', 'Contract', 'tenure',
        'MonthlyCharges', 'ChurnProbability', 'RiskCategory'
    ]].sort_values('ChurnProbability', ascending=False)
    
    # Save to outputs
    os.makedirs('outputs', exist_ok=True)
    output_path = 'outputs/predictions.csv'
    df_results.to_csv(output_path, index=False)
    
    print(f"\nResults saved to {output_path}")
    print(f"Total customers processed: {len(df_results)}")
    print(f"High risk:   {(df_original['RiskCategory'] == 'HIGH').sum()}")
    print(f"Medium risk: {(df_original['RiskCategory'] == 'MEDIUM').sum()}")
    print(f"Low risk:    {(df_original['RiskCategory'] == 'LOW').sum()}")


def show_summary(filepath):
    """
    Show summary statistics
    """
    print("\n" + "=" * 50)
    print("DATASET SUMMARY")
    print("=" * 50)
    
    # Load original data
    df_original = pd.read_csv(filepath)
    
    # Load model and preprocess
    model, feature_names = load_model()
    df_processed = preprocess_data(filepath)
    X = df_processed.drop('Churn', axis=1)
    
    # Get predictions
    probs = model.predict_proba(X)[:, 1]
    df_original['ChurnProbability'] = (probs * 100).round(1)
    df_original['RiskCategory'] = df_original[
        'ChurnProbability'
    ].apply(
        lambda x: 'HIGH' if x > 70
        else 'MEDIUM' if x > 30
        else 'LOW'
    )
    
    # Summary statistics
    total = len(df_original)
    high = (df_original['RiskCategory'] == 'HIGH').sum()
    medium = (df_original['RiskCategory'] == 'MEDIUM').sum()
    low = (df_original['RiskCategory'] == 'LOW').sum()
    
    print(f"Total Customers:     {total}")
    print(f"High Risk:           {high} ({high/total*100:.1f}%)")
    print(f"Medium Risk:         {medium} ({medium/total*100:.1f}%)")
    print(f"Low Risk:            {low} ({low/total*100:.1f}%)")
    print("-" * 50)
    print("Average Monthly Charges by Risk:")
    for risk in ['HIGH', 'MEDIUM', 'LOW']:
        avg = df_original[
            df_original['RiskCategory'] == risk
        ]['MonthlyCharges'].mean()
        print(f"  {risk}: ${avg:.2f}")
    print("=" * 50)


def show_high_risk(filepath):
    """
    Show all high risk customers
    """
    print("\n" + "=" * 50)
    print("HIGH RISK CUSTOMERS")
    print("=" * 50)
    
    # Load data and model
    df_original = pd.read_csv(filepath)
    model, feature_names = load_model()
    df_processed = preprocess_data(filepath)
    X = df_processed.drop('Churn', axis=1)
    
    # Get predictions
    probs = model.predict_proba(X)[:, 1]
    df_original['ChurnProbability'] = (probs * 100).round(1)
    df_original['RiskCategory'] = df_original[
        'ChurnProbability'
    ].apply(
        lambda x: 'HIGH' if x > 70
        else 'MEDIUM' if x > 30
        else 'LOW'
    )
    
    # Filter high risk only
    high_risk = df_original[
        df_original['RiskCategory'] == 'HIGH'
    ].sort_values('ChurnProbability', ascending=False)
    
    print(f"Total High Risk Customers: {len(high_risk)}")
    print("-" * 50)
    print(f"{'Rank':<5} {'CustomerID':<15} {'Probability':<12} {'Contract':<20} {'Tenure':<8} {'Charges'}")
    print("-" * 50)
    
    for i, (_, row) in enumerate(high_risk.head(20).iterrows(), 1):
        print(
            f"{i:<5} {row['customerID']:<15} "
            f"{row['ChurnProbability']}%{'':<8} "
            f"{row['Contract']:<20} "
            f"{row['tenure']} months{'':<3} "
            f"${row['MonthlyCharges']}"
        )
    
    if len(high_risk) > 20:
        print(f"\n... and {len(high_risk) - 20} more high risk customers")
    print("=" * 50)


def main():
    # Set up argument parser
    parser = argparse.ArgumentParser(
        description='AI Decision Copilot for Customer Retention'
    )
    
    parser.add_argument(
        '--file',
        required=True,
        help='Path to customer CSV file'
    )
    
    parser.add_argument(
        '--customer_id',
        help='Customer ID to analyze'
    )
    
    parser.add_argument(
        '--all',
        action='store_true',
        help='Process all customers'
    )
    
    parser.add_argument(
        '--summary',
        action='store_true',
        help='Show summary statistics'
    )
    
    parser.add_argument(
        '--high-risk',
        action='store_true',
        help='Show high risk customers'
    )
    
    args = parser.parse_args()
    
    # Check file exists
    if not os.path.exists(args.file):
        print(f"Error: File '{args.file}' not found.")
        print("Please check the file path and try again.")
        sys.exit(1)
    
    # Check file is CSV
    if not args.file.endswith('.csv'):
        print("Error: File must be a CSV file.")
        sys.exit(1)
    
    # Check required columns exist
    try:
        df_check = pd.read_csv(args.file)
        required_columns = [
            'customerID', 'tenure', 'Contract',
            'MonthlyCharges', 'TotalCharges', 'Churn'
        ]
        missing_columns = [
            col for col in required_columns
            if col not in df_check.columns
        ]
        if missing_columns:
            print(f"Error: Missing required columns: {missing_columns}")
            print("Please check your CSV file.")
            sys.exit(1)
    except Exception as e:
        print(f"Error reading file: {e}")
        sys.exit(1)
    
    # Check model exists
    if not os.path.exists('outputs/model.pkl'):
        print("Error: Trained model not found.")
        print("Please run scorer.py first to train the model.")
        sys.exit(1)
    
    # Run appropriate command
    try:
        if args.customer_id:
            analyze_customer(args.file, args.customer_id)
        elif args.all:
            process_all(args.file)
        elif args.summary:
            show_summary(args.file)
        elif args.high_risk:
            show_high_risk(args.file)
        else:
            parser.print_help()
    except KeyboardInterrupt:
        print("\nOperation cancelled by user.")
        sys.exit(0)
    except Exception as e:
        print(f"\nAn error occurred: {e}")
        print("Please check your input and try again.")
        sys.exit(1)


if __name__ == "__main__":
    main()