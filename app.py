import pickle
import pandas as pd
import numpy as np
import streamlit as st

def predict_loan_status(inputs, scaler_path, imputer_path, model_path):
    try:
        # Load the scaler, imputer, model
        with open(scaler_path, 'rb') as f1:
            scaler = pickle.load(f1)
        with open(imputer_path, 'rb') as f2:
            imputer = pickle.load(f2)
        with open(model_path, 'rb') as f3:
            model = pickle.load(f3)

        # Columns used in the model
        cols = [
            "Credit Score", "Annual Income", "Monthly Debt", "Current Loan Amount",
            "loan_term_months", "monthly_installment_est", "dti", "loan_to_income",
            "years_in_job_num", "Credit_Score_missing", "Annual_Income_missing",
            "Delinquent_missing", "Home Ownership", "Purpose"
        ]

        # Create a DataFrame
        x_input = pd.DataFrame([inputs], columns=cols)

        # Impute & scale
        x_input[scaler.feature_names_in_] = imputer.transform(x_input[scaler.feature_names_in_])
        x_input[scaler.feature_names_in_] = scaler.transform(x_input[scaler.feature_names_in_])

        # Predict
        pred = model.predict(x_input)
        prob = model.predict_proba(x_input)[0][pred[0]]

        return pred[0], prob
    except Exception as e:
        st.error(f"Prediction error: {e}")
        return None, None

# ============== Streamlit UI ==============
st.set_page_config(page_title="Loan Status Predictor")
st.title("🏦 Loan Status Prediction")

st.markdown("Fill in the loan application details:")

# Inputs
credit_score = st.number_input("Credit Score", 300, 850, value=700)
annual_income = st.number_input("Annual Income ($)", value=50000.0)
monthly_debt = st.number_input("Monthly Debt ($)", value=1500.0)
loan_amount = st.number_input("Loan Amount ($)", value=10000.0)
term = st.selectbox("Loan Term", ["Short Term", "Long Term"])
years_in_job = st.selectbox("Years in Current Job", ["< 1 year", "1 year", "2 years", "3 years", "4 years",
                                                     "5 years", "6 years", "7 years", "8 years", "9 years", "10+ years"])
home_ownership = st.selectbox("Home Ownership", ["Own Home", "Home Mortgage", "Rent", "Other"])
purpose = st.selectbox("Purpose", ["Debt Consolidation", "Home Improvements", "Other"])

# Feature engineering
loan_term_months = 36 if term == "Short Term" else 60
monthly_installment_est = loan_amount / (loan_term_months + 1)
dti = monthly_debt / (annual_income + 1)
loan_to_income = loan_amount / (annual_income + 1)
years_in_job_num = 0.5 if years_in_job == "< 1 year" else (10 if years_in_job == "10+ years" else float(years_in_job.split()[0]))

# Manual encoding for categoricals
home_map = {"Own Home": 0, "Home Mortgage": 1, "Rent": 2, "Other": 3}
purpose_map = {"Debt Consolidation": 0, "Home Improvements": 1, "Other": 2}

# Prediction button
if st.button("🔍 Predict Loan Status"):
    input_list = [
        credit_score, annual_income, monthly_debt, loan_amount,
        loan_term_months, monthly_installment_est, dti, loan_to_income,
        years_in_job_num, 0, 0, 0,  # no missing values in UI
        home_map[home_ownership],
        purpose_map[purpose]
    ]

    pred, prob = predict_loan_status(
        inputs=input_list,
        scaler_path="Notebook/scaler.pkl",
        imputer_path="Notebook/imputer.pkl",
        model_path="Notebook/model.pkl"
    )

    if pred is not None:
        result = "✅ Fully Paid" if pred == 0 else "❌ Charged Off"
        st.subheader(f"Prediction: {result}")
        st.metric("Prediction Confidence", f"{prob*100:.2f}%")
        st.progress(prob)
    else:
        st.error("Something went wrong during prediction.")
