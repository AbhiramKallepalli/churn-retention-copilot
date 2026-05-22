import pandas as pd

# Load the dataset
df = pd.read_csv('data/WA_Fn-UseC_-Telco-Customer-Churn.csv')

# Basic information
print("Dataset loaded successfully!")
print(f"Total customers: {len(df)}")
print(f"Total columns: {len(df.columns)}")
print()

# Show churn distribution
print("Churn Distribution:")
print(df['Churn'].value_counts())
print()

# Average monthly charges
print("Average Monthly Charges:")
print(round(df['MonthlyCharges'].mean(), 2))
print()

# Check missing values
print("Missing Values in Each Column:")
print(df.isnull().sum())
print()

# Show first 3 rows
print("First 3 rows of data:")
print(df.head(3))