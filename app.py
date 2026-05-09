import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import joblib
import shap
import matplotlib.pyplot as plt

# Page config
st.set_page_config(
    page_title="Fraud Detection System",
    page_icon="💳",
    layout="wide"
)

# Load model
@st.cache_resource
def load_model():
    model = joblib.load('models/fraud_detector.pkl')
    feature_names = joblib.load('models/feature_names.pkl')
    return model, feature_names

model, feature_names = load_model()

# Load data
@st.cache_data
def load_data():
    df = pd.read_csv('data/creditcard.csv')
    df['Hour'] = (df['Time'] / 3600) % 24
    df['Is_Night'] = ((df['Hour'] >= 22) | (df['Hour'] <= 6)).astype(int)
    df['Amount_log'] = np.log1p(df['Amount'])
    df['Amount_Squared'] = df['Amount'] ** 2
    df['Is_Round_Amount'] = (df['Amount'] % 1 == 0).astype(int)
    return df

df = load_data()

# Header
st.title("💳 Payment Fraud Detection System")
st.markdown("**ML-powered real-time fraud detection with explainable AI | XGBoost + SHAP**")
st.divider()

# KPI Metrics
col1, col2, col3, col4, col5 = st.columns(5)

with col1:
    st.metric("Total Transactions", f"{len(df):,}")
with col2:
    st.metric("Fraud Cases", f"{df['Class'].sum():,}")
with col3:
    st.metric("Fraud Rate", f"{(df['Class'].sum()/len(df)*100):.3f}%")
with col4:
    st.metric("Model ROC-AUC", "0.9821")
with col5:
    st.metric("Fraud Caught", "83.7%")

st.divider()

# Sidebar
st.sidebar.title("⚙️ Controls")
page = st.sidebar.radio("Navigate", [
    "📊 Overview",
    "🔍 Run Detection",
    "💡 Model Explainability",
    "💰 Financial Impact"
])

# ─── PAGE 1: OVERVIEW ───
if page == "📊 Overview":
    st.header("Dataset Overview")

    col1, col2 = st.columns(2)

    with col1:
        # Class distribution
        fig = px.bar(
            x=['Legitimate', 'Fraud'],
            y=[284315, 492],
            color=['Legitimate', 'Fraud'],
            color_discrete_map={'Legitimate': '#2ecc71', 'Fraud': '#e74c3c'},
            title="Transaction Count by Class",
            text=[284315, 492]
        )
        fig.update_traces(textposition='outside')
        fig.update_layout(showlegend=False)
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        # Fraud by hour
        fraud_by_hour = df[df['Class']==1].groupby(
            df['Hour'].astype(int))['Class'].count()
        fig2 = px.bar(
            x=fraud_by_hour.index,
            y=fraud_by_hour.values,
            title="Fraud Cases by Hour of Day",
            color=fraud_by_hour.values,
            color_continuous_scale='Reds'
        )
        fig2.update_layout(
            xaxis_title="Hour",
            yaxis_title="Fraud Count",
            coloraxis_showscale=False
        )
        st.plotly_chart(fig2, use_container_width=True)

    # Amount distribution
    fig3 = px.histogram(
        df[df['Amount'] < 1000],
        x='Amount',
        color='Class',
        barmode='overlay',
        title="Transaction Amount Distribution (Fraud vs Legitimate)",
        color_discrete_map={0: '#2ecc71', 1: '#e74c3c'},
        labels={'Class': 'Transaction Type'},
        opacity=0.7
    )
    st.plotly_chart(fig3, use_container_width=True)

# ─── PAGE 2: RUN DETECTION ───
elif page == "🔍 Run Detection":
    st.header("Real-Time Fraud Detection")
    st.markdown("Run the model on a sample of transactions and see predictions.")

    sample_size = st.slider("Sample size", 100, 5000, 1000)

    if st.button("🚀 Run Fraud Detection", type="primary"):
        sample = df.sample(sample_size, random_state=42)
        X_sample = sample.drop(['Class', 'Time', 'Amount'], axis=1)
        X_sample = X_sample[feature_names]

        probabilities = model.predict_proba(X_sample)[:, 1]
        predictions = model.predict(X_sample)

        sample = sample.copy()
        sample['Risk_Score'] = probabilities
        sample['Predicted'] = predictions
        sample['Risk_Level'] = pd.cut(
            probabilities,
            bins=[0, 0.3, 0.7, 1.0],
            labels=['🟢 Low', '🟡 Medium', '🔴 High']
        )

        # Results metrics
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Transactions Scanned", f"{sample_size:,}")
        with col2:
            flagged = predictions.sum()
            st.metric("Flagged as Fraud", f"{flagged:,}")
        with col3:
            st.metric("Flag Rate", f"{(flagged/sample_size*100):.2f}%")

        # Risk distribution
        fig = px.histogram(
            sample,
            x='Risk_Score',
            color='Risk_Level',
            title="Risk Score Distribution",
            color_discrete_map={
                '🟢 Low': '#2ecc71',
                '🟡 Medium': '#f39c12',
                '🔴 High': '#e74c3c'
            }
        )
        st.plotly_chart(fig, use_container_width=True)

        # High risk transactions table
        st.subheader("🔴 High Risk Transactions")
        high_risk = sample[sample['Risk_Score'] > 0.7].sort_values(
            'Risk_Score', ascending=False
        )[['Hour', 'Amount_log', 'Risk_Score', 'Risk_Level', 'Class']].head(20)

        high_risk.columns = ['Hour', 'Log Amount', 'Risk Score', 'Risk Level', 'Actual Class']
        high_risk['Risk Score'] = high_risk['Risk Score'].apply(lambda x: f"{x:.4f}")
        st.dataframe(high_risk, use_container_width=True)

# ─── PAGE 3: EXPLAINABILITY ───
elif page == "💡 Model Explainability":
    st.header("Model Explainability (SHAP)")
    st.markdown("""
    SHAP (SHapley Additive exPlanations) shows **why** the model made each prediction.
    This is critical for regulatory compliance in financial services.
    """)

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Feature Importance")
        st.image('models/shap_importance.png')

    with col2:
        st.subheader("SHAP Summary Plot")
        st.image('models/shap_summary.png')

    st.subheader("Key Findings")
    st.markdown("""
    | Feature | SHAP Importance | Interpretation |
    |---------|----------------|----------------|
    | **V4** | 2.019 | Strongest fraud indicator - behavioral anomaly pattern |
    | **V14** | 2.008 | Second strongest - unusually low values signal fraud |
    | **V12** | 0.894 | Transaction pattern deviation |
    | **V10** | 0.763 | Spending behavior anomaly |
    | **Hour** | ~0.30 | Time-based risk (2-4am = 10x higher fraud rate) |
    """)

# ─── PAGE 4: FINANCIAL IMPACT ───
elif page == "💰 Financial Impact":
    st.header("Financial Impact Analysis")

    st.markdown("### Cost-Benefit Calculator")

    col1, col2 = st.columns(2)

    with col1:
        avg_fraud_amount = st.slider("Average Fraud Amount (€)", 50, 500, 122)
        monthly_transactions = st.slider("Monthly Transactions",
                                          100000, 10000000, 1000000)
        fraud_rate = st.slider("Fraud Rate (%)", 0.1, 1.0, 0.17)

    with col2:
        false_positive_cost = st.slider(
            "Cost per False Positive (€)", 5, 50, 15,
            help="Customer service cost when legitimate transaction is blocked")
        model_catch_rate = 0.837

    # Calculations
    monthly_fraud = int(monthly_transactions * (fraud_rate/100))
    fraud_caught = int(monthly_fraud * model_catch_rate)
    fraud_missed = monthly_fraud - fraud_caught
    false_positives = int(monthly_transactions * 0.000053)

    savings = fraud_caught * avg_fraud_amount
    fp_cost = false_positives * false_positive_cost
    net_savings = savings - fp_cost

    st.divider()
    st.subheader("Monthly Impact Estimate")

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Fraud Cases Detected", f"{fraud_caught:,}")
    with col2:
        st.metric("Gross Savings", f"€{savings:,.0f}")
    with col3:
        st.metric("False Positive Cost", f"€{fp_cost:,.0f}")
    with col4:
        st.metric("Net Monthly Savings", f"€{net_savings:,.0f}",
                  delta=f"€{net_savings:,.0f}")

    # Savings chart
    fig = go.Figure(go.Waterfall(
        name="Financial Impact",
        orientation="v",
        measure=["relative", "relative", "total"],
        x=["Fraud Prevention Savings", "False Positive Costs", "Net Savings"],
        y=[savings, -fp_cost, 0],
        connector={"line": {"color": "rgb(63, 63, 63)"}},
        decreasing={"marker": {"color": "#e74c3c"}},
        increasing={"marker": {"color": "#2ecc71"}},
        totals={"marker": {"color": "#3498db"}}
    ))
    fig.update_layout(title="Monthly Financial Impact Breakdown")
    st.plotly_chart(fig, use_container_width=True)

# Footer
st.divider()
st.markdown("""
<div style='text-align: center; color: gray;'>
    Built by Saif Ullah | XGBoost + SHAP + Streamlit | 
    <a href='https://github.com/saif06910'>GitHub</a>
</div>
""", unsafe_allow_html=True)