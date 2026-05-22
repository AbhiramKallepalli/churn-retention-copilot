import sys
import os
import pytest
import pandas as pd

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from src.preprocessor import preprocess_data
from src.shap_explainer import get_shap_values, load_model

FILEPATH = 'data/WA_Fn-UseC_-Telco-Customer-Churn.csv'

def test_data_loads():
    """Test that dataset loads correctly"""
    df = pd.read_csv(FILEPATH)
    assert len(df) == 7043
    assert 'Churn' in df.columns
    assert 'customerID' in df.columns
    print("✅ Data loads correctly")

def test_preprocessing():
    """Test that preprocessing works correctly"""
    df = preprocess_data(FILEPATH)
    assert len(df.columns) == 42
    assert 'Churn' in df.columns
    assert 'customerID' not in df.columns
    print("✅ Preprocessing works correctly")

def test_high_risk_customer():
    """Test high risk customer analysis"""
    result = get_shap_values('9497-QCMMS', FILEPATH)
    assert result is not None
    assert result['churn_probability'] > 0.7
    assert result['risk_category'] == 'HIGH RISK'
    assert len(result['top_factors']) == 5
    print("✅ High risk customer identified correctly")

def test_low_risk_customer():
    """Test low risk customer analysis"""
    result = get_shap_values('5575-GNVDE', FILEPATH)
    assert result is not None
    assert result['churn_probability'] < 0.3
    assert result['risk_category'] == 'LOW RISK'
    print("✅ Low risk customer identified correctly")

def test_missing_customer():
    """Test unknown customer ID handled correctly"""
    result = get_shap_values('99999-XXXXX', FILEPATH)
    assert result is None
    print("✅ Unknown customer handled correctly")

def test_model_loads():
    """Test that trained model loads correctly"""
    model, feature_names = load_model()
    assert model is not None
    assert len(feature_names) == 41
    print("✅ Model loads correctly")

def test_risk_categories():
    """Test risk categories are assigned correctly"""
    result = get_shap_values('9497-QCMMS', FILEPATH)
    prob = result['churn_probability']
    if prob > 0.7:
        assert result['risk_category'] == 'HIGH RISK'
    elif prob > 0.3:
        assert result['risk_category'] == 'MEDIUM RISK'
    else:
        assert result['risk_category'] == 'LOW RISK'
    print("✅ Risk categories assigned correctly")