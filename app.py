import pickle
import pandas as pd
import numpy as np
import streamlit as st

def predict_loan_status(inputs, scaler_path, imputer_path, model_path, features_path):
    try:
        # Load everything
        scaler = pickle.load(open(scaler_path, "rb"))
        imputer = pickle.load(open(imputer_path, "rb"))
        model = pickle.load(open(model_path, "rb"))
        feature_names = pickle.load(open(features_path, "rb"))

        # Create empty input with all model features
        x_input = pd.DataFrame([0]*len(feature_names), index=feature_names).T

        # Fill provided values
        for key, value in inputs.items():
            x_input.at[0, key] = value

        # Preprocess
        x_input[scaler.feature_names_in_] = imputer.transform(x_input[scaler.feature_names_in_])
        x_input[scaler.feature_names_in_] = scaler.transform(x_input[scaler.feature_names_in_])

        # Predict
        pred = model.predict(x_input)[0]
        prob = model.predict_proba(x_input)[0][pred]

        return pred, prob
    except Exception as e:
        st.error(f"Prediction error: {e}")
        return None, None

st.set_page_config(page_title="Loan Status Predictor")
st.title("🏦 Loan Status Prediction")

# UI Inputs
credit_score = st.number_input("Credit Score", 300, 850, value=700)
annual_income = st.number_input("Annual Income ($)", value=50000.0)
monthly_debt = st.number_input("Monthly Debt ($)", value=1500.0)
loan_amount = st.number_input("Loan Amount ($)", value=10000.0)
term = st.selectbox("Loan Term", ["Short Term", "Long Term"])
years_in_job = st.selectbox("Years in Current Job", ["< 1 year", "1 year", "2 years", "3 years", "4 years",
                                                     "5 years", "6 years", "7 years", "8 years", "9 years", "10+ years"])
home_ownership = st.selectbox("Home Ownership", ["Own Home", "Home Mortgage", "Rent", "Other"])
purpose = st.selectbox("Purpose", ["Debt Consolidation", "Home Improvements", "Other"])

# Feature Engineering
loan_term_months = 36 if term == "Short Term" else 60
monthly_installment_est = loan_amount / (loan_term_months + 1)
dti = monthly_debt / (annual_income + 1)
loan_to_income = loan_amount / (annual_income + 1)
years_in_job_num = 0.5 if years_in_job == "< 1 year" else (10 if years_in_job == "10+ years" else float(years_in_job.split()[0]))

# Maps
home_map = {"Own Home": 0, "Home Mortgage": 1, "Rent": 2, "Other": 3}
purpose_map = {"Debt Consolidation": 0, "Home Improvements": 1, "Other": 2}

if st.button("🔍 Predict Loan Status"):
    input_dict = {
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
        "Home Ownership": home_map[home_ownership],
        "Purpose": purpose_map[purpose]
    }

    pred, prob = predict_loan_status(
        inputs=input_dict,
        scaler_path="Notebook/scaler.pkl",
        imputer_path="Notebook/imputer.pkl",
        model_path="Notebook/model.pkl",
        features_path="Notebook/features.pkl"
    )

    if pred is not None:
        st.subheader(f"Prediction: {'✅ Fully Paid' if pred == 0 else '❌ Charged Off'}")
        st.metric("Confidence", f"{prob * 100:.2f}%")
        st.progress(prob)