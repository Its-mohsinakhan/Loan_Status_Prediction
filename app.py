import streamlit as st
import pandas as pd
import numpy as np
import joblib

# ✅ Must be the first Streamlit command
st.set_page_config(page_title="Loan Status Predictor", layout="centered")

# Load trained model and preprocessors
model = joblib.load("Notebook/loan_model.pkl")
imputer = joblib.load("Notebook/imputer.pkl")
scaler = joblib.load("Notebook/scaler.pkl")

# Optional: Load label encoders if needed
# le_home = joblib.load("le_home.pkl")
# le_purpose = joblib.load("le_purpose.pkl")

st.title("🏦 Loan Status Prediction App")
st.markdown("Enter loan applicant details below to predict whether the loan will be **Fully Paid** or **Charged Off**.")

# 📋 User inputs
with st.form("loan_form"):
    credit_score = st.number_input("Credit Score", min_value=300, max_value=850, value=700)
    annual_income = st.number_input("Annual Income", min_value=0.0, value=50000.0)
    monthly_debt = st.number_input("Monthly Debt", min_value=0.0, value=1500.0)
    loan_amount = st.number_input("Loan Amount", min_value=0.0, value=10000.0)
    term = st.selectbox("Loan Term", ["Short Term", "Long Term"])
    years_in_job = st.selectbox("Years in Current Job", ["< 1 year", "1 year", "2 years", "3 years", "4 years",
                                                         "5 years", "6 years", "7 years", "8 years", "9 years", "10+ years"])
    home_ownership = st.selectbox("Home Ownership", ["Own Home", "Home Mortgage", "Rent", "Other"])
    purpose = st.selectbox("Purpose", ["Debt Consolidation", "Home Improvements", "Other"])

    submit = st.form_submit_button("Predict")

if submit:
    # Feature Engineering
    loan_term_months = 36 if term == "Short Term" else 60
    monthly_installment_est = loan_amount / (loan_term_months + 1)
    loan_to_income = loan_amount / (annual_income + 1)
    dti = monthly_debt / (annual_income + 1)

    def parse_years(val):
        if val == "10+ years":
            return 10
        if val == "< 1 year":
            return 0.5
        return float(val.split()[0])

    years_in_job_num = parse_years(years_in_job)

    # Construct DataFrame
    input_df = pd.DataFrame([{
        "Credit Score": credit_score,
        "Annual Income": annual_income,
        "Monthly Debt": monthly_debt,
        "Current Loan Amount": loan_amount,
        "loan_term_months": loan_term_months,
        "monthly_installment_est": monthly_installment_est,
        "dti": dti,
        "loan_to_income": loan_to_income,
        "years_in_job_num": years_in_job_num,
        "Credit_Score_missing": 0,
        "Annual_Income_missing": 0,
        "Delinquent_missing": 0,
        "Home Ownership": home_ownership,
        "Purpose": purpose
    }])

    # Handle encoding manually (if needed)
    # For demo, convert to simple labels
    input_df["Home Ownership"] = input_df["Home Ownership"].map({
        "Own Home": 0, "Home Mortgage": 1, "Rent": 2, "Other": 3
    })
    input_df["Purpose"] = input_df["Purpose"].map({
        "Debt Consolidation": 0, "Home Improvements": 1, "Other": 2
    })

    # Impute and scale numeric features
    numeric_cols = ["Credit Score", "Annual Income", "Monthly Debt", "Current Loan Amount",
                    "loan_term_months", "monthly_installment_est", "dti", "loan_to_income", "years_in_job_num"]

    input_df[numeric_cols] = imputer.transform(input_df[numeric_cols])
    input_df[numeric_cols] = scaler.transform(input_df[numeric_cols])

    # Predict
    prediction = model.predict(input_df)[0]
    result = "✅ Fully Paid" if prediction == 0 else "❌ Charged Off"
    st.subheader("Prediction Result:")
    st.success(result)

