import ollama
import json

def ask_llm(question, context):
    """
    Send a question to Phi-3 Mini via Ollama
    with customer data as context
    
    question: The user's question
    context: Dictionary with customer data and SHAP results
    """
    
    # Build the prompt
    prompt = f"""You are a customer retention analyst assistant.
You have access to the following customer data and churn analysis results.
Answer the user's question based ONLY on this data.
Do not make up information that is not in the data.

CUSTOMER DATA:
Customer ID: {context.get('customer_id', 'Unknown')}
Tenure: {context.get('tenure', 'Unknown')} months
Contract: {context.get('contract', 'Unknown')}
Monthly Charges: ${context.get('monthly_charges', 'Unknown')}
Internet Service: {context.get('internet_service', 'Unknown')}
Tech Support: {context.get('tech_support', 'Unknown')}
Online Security: {context.get('online_security', 'Unknown')}

CHURN ANALYSIS:
Churn Probability: {context.get('churn_probability', 'Unknown')}%
Risk Category: {context.get('risk_category', 'Unknown')}

TOP RISK FACTORS:
{context.get('top_factors_text', 'No factors available')}

USER QUESTION: {question}

Please provide a clear and concise answer based on the data above.
Include specific recommendations if asked about retention strategies."""

    # Send to Ollama
    response = ollama.chat(
        model='phi3:mini',
        messages=[
            {
                'role': 'user',
                'content': prompt
            }
        ]
    )
    
    return response['message']['content']


def format_context(shap_result):
    """
    Format SHAP results into context
    for the LLM
    """
    if shap_result is None:
        return None
    
    # Format top factors as text
    top_factors_text = ""
    for i, (feature, value) in enumerate(
        shap_result['top_factors'], 1
    ):
        direction = "increases" if value > 0 \
                    else "decreases"
        top_factors_text += (
            f"{i}. {feature}: {value:+.4f} "
            f"({direction} churn risk)\n"
        )
    
    # Get original customer info
    original = shap_result['original_info']
    
    context = {
        'customer_id': shap_result['customer_id'],
        'tenure': original['tenure'],
        'contract': original['Contract'],
        'monthly_charges': original['MonthlyCharges'],
        'internet_service': original['InternetService'],
        'tech_support': original['TechSupport'],
        'online_security': original['OnlineSecurity'],
        'churn_probability': round(
            shap_result['churn_probability'] * 100, 1
        ),
        'risk_category': shap_result['risk_category'],
        'top_factors_text': top_factors_text
    }
    
    return context


def start_qa_session(shap_result):
    """
    Start an interactive Q&A session
    for a customer
    """
    context = format_context(shap_result)
    
    if context is None:
        print("No customer data available.")
        return
    
    print("\n" + "=" * 50)
    print("LLM Q&A SESSION")
    print("=" * 50)
    print(f"Customer: {context['customer_id']}")
    print(f"Churn Probability: {context['churn_probability']}%")
    print(f"Risk Category: {context['risk_category']}")
    print("=" * 50)
    print("You can now ask questions about this customer.")
    print("Type 'exit' to quit.")
    print("=" * 50)
    
    while True:
        # Get user question
        question = input("\nYour question: ").strip()
        
        # Check if user wants to exit
        if question.lower() == 'exit':
            print("Ending Q&A session.")
            break
        
        if not question:
            print("Please enter a question.")
            continue
        
        # Get answer from LLM
        print("\nThinking...")
        answer = ask_llm(question, context)
        
        print("\n" + "-" * 50)
        print("AI Response:")
        print("-" * 50)
        print(answer)
        print("-" * 50)


# Test it
if __name__ == "__main__":
    import sys
    import os
    sys.path.append(
        os.path.dirname(os.path.dirname(__file__))
    )
    from src.shap_explainer import get_shap_values
    
    # Get SHAP results for test customer
    print("Getting SHAP analysis...")
    result = get_shap_values(
        '7590-VHVEG',
        'data/WA_Fn-UseC_-Telco-Customer-Churn.csv'
    )
    
    # Start Q&A session
    start_qa_session(result)