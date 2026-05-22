import pandas as pd

def preprocess_data(filepath):
    # Load data
    df = pd.read_csv(filepath)
    print("Data loaded. Shape:", df.shape)

    # Fix TotalCharges
    # Some values are blank strings
    # We convert them to numbers
    df['TotalCharges'] = pd.to_numeric(
        df['TotalCharges'], errors='coerce'
    )

    # Fill missing TotalCharges with median
    median_value = df['TotalCharges'].median()
    df['TotalCharges'] = df['TotalCharges'].fillna(median_value)
    print("TotalCharges fixed.")

    # Convert Churn to 1 and 0
    # Yes = 1, No = 0
    df['Churn'] = df['Churn'].map({'Yes': 1, 'No': 0})
    print("Churn column converted to 1 and 0.")

    # Drop customerID
    # We do not need it for ML model
    df = df.drop('customerID', axis=1)
    print("customerID removed.")

    # Convert Yes/No columns to 1 and 0
    yes_no_columns = [
        'Partner', 'Dependents', 'PhoneService',
        'PaperlessBilling'
    ]
    for col in yes_no_columns:
        df[col] = df[col].map({'Yes': 1, 'No': 0})
    print("Yes/No columns converted.")

    # Use one-hot encoding for categorical columns
    categorical_columns = [
        'gender', 'MultipleLines', 'InternetService',
        'OnlineSecurity', 'OnlineBackup',
        'DeviceProtection', 'TechSupport',
        'StreamingTV', 'StreamingMovies',
        'Contract', 'PaymentMethod'
    ]
    df = pd.get_dummies(df, columns=categorical_columns)
    print("Categorical columns encoded.")

    print("Preprocessing complete. Shape:", df.shape)
    return df

# Test it
if __name__ == "__main__":
    df = preprocess_data(
        'data/WA_Fn-UseC_-Telco-Customer-Churn.csv'
    )
    print()
    print("Final number of columns:", len(df.columns))
    print()
    print("First 3 rows:")
    print(df.head(3))