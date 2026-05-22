import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import sys
import os
import time

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from src.preprocessor import preprocess_data
from src.shap_explainer import get_shap_values, load_model
from src.llm_interface import format_context, ask_llm

# Page configuration
st.set_page_config(
    page_title="AI Churn Copilot",
    page_icon="🤖",
    layout="wide"
)

# Title
st.title("🤖 AI Decision Copilot for Customer Retention")
st.markdown("*Predict customer churn and get AI-powered retention insights*")
st.divider()

# Stop button in sidebar
with st.sidebar:
    st.divider()
    if st.button("⏹️ Stop Application", type="secondary"):
        st.session_state['stopped'] = True

if st.session_state.get('stopped'):
    st.success("✅ Application stopped successfully!")
    st.info("You can now close this browser tab.")
    st.balloons()
    st.stop()

# Sidebar navigation
page = st.sidebar.selectbox(
    "Navigate to:",
    [
        "📤 Upload and Predict",
        "🔍 Customer Detail",
        "📊 Summary Dashboard"
    ]
)

# ─────────────────────────────────────────
# PAGE 1 — UPLOAD AND PREDICT
# ─────────────────────────────────────────
if page == "📤 Upload and Predict":
    st.header("📤 Upload and Predict")
    st.write("Upload your customer CSV file to get churn predictions.")

    # File uploader
    uploaded_file = st.file_uploader(
        "Choose a CSV file",
        type="csv"
    )

    if uploaded_file is not None:
        # Load data
        df_original = pd.read_csv(uploaded_file)
        st.success(f"✅ File uploaded successfully! {len(df_original)} customers found.")

        # Save file temporarily
        temp_path = "data/temp_upload.csv"
        df_original.to_csv(temp_path, index=False)

        # Run analysis button
        if st.button("🚀 Run Analysis", type="primary"):
            with st.spinner("Running ML model on all customers..."):
                # Load model
                model, feature_names = load_model()

                # Preprocess
                df_processed = preprocess_data(temp_path)
                X = df_processed.drop('Churn', axis=1)

                # Get predictions
                probs = model.predict_proba(X)[:, 1]
                df_original['ChurnProbability'] = (
                    probs * 100
                ).round(1)
                df_original['RiskCategory'] = df_original[
                    'ChurnProbability'
                ].apply(
                    lambda x: '🔴 HIGH' if x > 70
                    else '🟡 MEDIUM' if x > 30
                    else '🟢 LOW'
                )

                # Sort by risk
                df_results = df_original[[
                    'customerID', 'Contract', 'tenure',
                    'MonthlyCharges',
                    'ChurnProbability', 'RiskCategory'
                ]].sort_values(
                    'ChurnProbability', ascending=False
                )

                # Store in session
                st.session_state['results'] = df_results
                st.session_state['filepath'] = temp_path

            st.success("✅ Analysis complete!")

        # Show results
        if 'results' in st.session_state:
            df_results = st.session_state['results']

            # Risk counts
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric(
                    "Total Customers",
                    len(df_results)
                )
            with col2:
                high = len(df_results[
                    df_results['RiskCategory'] == '🔴 HIGH'
                ])
                st.metric("🔴 High Risk", high)
            with col3:
                medium = len(df_results[
                    df_results['RiskCategory'] == '🟡 MEDIUM'
                ])
                st.metric("🟡 Medium Risk", medium)
            with col4:
                low = len(df_results[
                    df_results['RiskCategory'] == '🟢 LOW'
                ])
                st.metric("🟢 Low Risk", low)

            st.divider()

            # Filter options
            risk_filter = st.selectbox(
                "Filter by Risk:",
                ["All", "🔴 HIGH", "🟡 MEDIUM", "🟢 LOW"]
            )

            if risk_filter != "All":
                df_show = df_results[
                    df_results['RiskCategory'] == risk_filter
                ]
            else:
                df_show = df_results

            # Show table
            st.dataframe(
                df_show,
                use_container_width=True,
                height=400
            )


# ─────────────────────────────────────────
# PAGE 2 — CUSTOMER DETAIL
# ─────────────────────────────────────────
elif page == "🔍 Customer Detail":
    st.header("🔍 Customer Detail Analysis")
    st.write("Enter a customer ID to get full analysis and ask questions.")

    # Input
    col1, col2 = st.columns([3, 1])
    with col1:
        customer_id = st.text_input(
            "Enter Customer ID:",
            placeholder="e.g. 9497-QCMMS"
        )
    with col2:
        analyze_btn = st.button(
            "🔍 Analyze",
            type="primary"
        )

    if analyze_btn and customer_id:
        filepath = "data/WA_Fn-UseC_-Telco-Customer-Churn.csv"

        with st.spinner(f"Analyzing customer {customer_id}..."):
            result = get_shap_values(customer_id, filepath)

        if result is None:
            st.error(f"❌ Customer {customer_id} not found!")
        else:
            # Store result
            st.session_state['customer_result'] = result

    # Show results
    if 'customer_result' in st.session_state:
        result = st.session_state['customer_result']
        original = result['original_info']

        # Risk display
        prob = result['churn_probability'] * 100
        risk = result['risk_category']

        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric(
                "Churn Probability",
                f"{prob:.1f}%"
            )
        with col2:
            st.metric("Risk Category", risk)
        with col3:
            st.metric(
                "Tenure",
                f"{original['tenure']} months"
            )

        st.divider()

        col1, col2 = st.columns(2)

        with col1:
            st.subheader("📋 Customer Info")
            st.write(f"**Contract:** {original['Contract']}")
            st.write(
                f"**Monthly Charges:** "
                f"${original['MonthlyCharges']}"
            )
            st.write(
                f"**Internet Service:** "
                f"{original['InternetService']}"
            )
            st.write(
                f"**Tech Support:** "
                f"{original['TechSupport']}"
            )
            st.write(
                f"**Online Security:** "
                f"{original['OnlineSecurity']}"
            )

        with col2:
            st.subheader("📊 Top Risk Factors (SHAP)")
            factors = result['top_factors']
            features = [f[0].replace('_', ' ')[:20]
                       for f in factors]
            values = [f[1] for f in factors]
            colors = [
                'red' if v > 0 else 'green'
                for v in values
            ]

            fig, ax = plt.subplots(figsize=(6, 4))
            bars = ax.barh(features, values, color=colors)
            ax.axvline(x=0, color='black', linewidth=0.5)
            ax.set_xlabel('SHAP Value')
            ax.set_title('Risk Factor Impact')
            plt.tight_layout()
            st.pyplot(fig)

        st.divider()

        # Q&A Section
        st.subheader("💬 Ask the AI")
        question = st.text_input(
            "Ask a question about this customer:",
            placeholder="e.g. What should we do to retain this customer?"
        )

        if st.button("Ask AI", type="primary"):
            if question:
                context = format_context(result)
                with st.spinner("AI is thinking..."):
                    answer = ask_llm(question, context)

                st.markdown("**AI Response:**")
                st.info(answer)


# ─────────────────────────────────────────
# PAGE 3 — SUMMARY DASHBOARD
# ─────────────────────────────────────────
elif page == "📊 Summary Dashboard":
    st.header("📊 Summary Dashboard")

    filepath = "data/WA_Fn-UseC_-Telco-Customer-Churn.csv"

    with st.spinner("Loading data..."):
        df_original = pd.read_csv(filepath)
        model, feature_names = load_model()
        df_processed = preprocess_data(filepath)
        X = df_processed.drop('Churn', axis=1)
        probs = model.predict_proba(X)[:, 1]
        df_original['ChurnProbability'] = (
            probs * 100
        ).round(1)
        df_original['RiskCategory'] = df_original[
            'ChurnProbability'
        ].apply(
            lambda x: 'HIGH' if x > 70
            else 'MEDIUM' if x > 30
            else 'LOW'
        )

    # Metrics
    total = len(df_original)
    high = (df_original['RiskCategory'] == 'HIGH').sum()
    medium = (df_original['RiskCategory'] == 'MEDIUM').sum()
    low = (df_original['RiskCategory'] == 'LOW').sum()

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Customers", total)
    with col2:
        st.metric("🔴 High Risk", f"{high} ({high/total*100:.1f}%)")
    with col3:
        st.metric("🟡 Medium Risk", f"{medium} ({medium/total*100:.1f}%)")
    with col4:
        st.metric("🟢 Low Risk", f"{low} ({low/total*100:.1f}%)")

    st.divider()

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Risk Distribution")
        fig1, ax1 = plt.subplots(figsize=(5, 5))
        ax1.pie(
            [high, medium, low],
            labels=['High Risk', 'Medium Risk', 'Low Risk'],
            colors=['red', 'orange', 'green'],
            autopct='%1.1f%%',
            startangle=90
        )
        ax1.set_title('Customer Risk Distribution')
        st.pyplot(fig1)

    with col2:
        st.subheader("Average Monthly Charges by Risk")
        avg_charges = df_original.groupby(
            'RiskCategory'
        )['MonthlyCharges'].mean()

        fig2, ax2 = plt.subplots(figsize=(5, 5))
        bars = ax2.bar(
            avg_charges.index,
            avg_charges.values,
            color=['red', 'green', 'orange']
        )
        ax2.set_xlabel('Risk Category')
        ax2.set_ylabel('Average Monthly Charges ($)')
        ax2.set_title('Avg Charges by Risk Category')

        for bar, val in zip(bars, avg_charges.values):
            ax2.text(
                bar.get_x() + bar.get_width()/2,
                bar.get_height() + 0.5,
                f'${val:.2f}',
                ha='center'
            )

        plt.tight_layout()
        st.pyplot(fig2)

    st.divider()
    st.subheader("Contract Type Distribution")
    contract_churn = df_original.groupby(
        'Contract'
    )['RiskCategory'].value_counts().unstack()
    st.bar_chart(contract_churn)