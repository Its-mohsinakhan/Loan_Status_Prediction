import pandas as pd
import numpy as np
import pickle
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

# Load dataset
df = pd.read_csv("credit_train.csv")
df = df[df["Loan Status"].notna()]
df["Loan Status"] = df["Loan Status"].map({"Fully Paid": 0, "Charged Off": 1})

# Feature engineering
df["dti"] = df["Monthly Debt"] / (df["Annual Income"] + 1)
df["loan_to_income"] = df["Current Loan Amount"] / (df["Annual Income"] + 1)
df["loan_term_months"] = df["Term"].map({"Short Term": 36, "Long Term": 60})
df["monthly_installment_est"] = df["Current Loan Amount"] / (df["loan_term_months"] + 1)

def parse_years(val):
    if pd.isna(val): return np.nan
    if val == "10+ years": return 10
    if val == "< 1 year": return 0.5
    return float(val.split()[0])
df["years_in_job_num"] = df["Years in current job"].apply(parse_years)

df["Credit_Score_missing"] = df["Credit Score"].isnull().astype(int)
df["Annual_Income_missing"] = df["Annual Income"].isnull().astype(int)
df["Delinquent_missing"] = df["Months since last delinquent"].isnull().astype(int)

# Encode categoricals
df["Home Ownership"] = df["Home Ownership"].map({"Own Home": 0, "Home Mortgage": 1, "Rent": 2, "Other": 3})
df["Purpose"] = df["Purpose"].map({"Debt Consolidation": 0, "Home Improvements": 1, "Other": 2})

# Drop unused columns
df.drop(columns=["Loan ID", "Customer ID", "Years in current job", "Term"], inplace=True)

X = df.drop(columns=["Loan Status"])
y = df["Loan Status"]

# Impute and scale
num_cols = X.select_dtypes(include=["float64", "int64"]).columns.tolist()
imputer = SimpleImputer(strategy="mean")
X[num_cols] = imputer.fit_transform(X[num_cols])

scaler = StandardScaler()
X[num_cols] = scaler.fit_transform(X[num_cols])

# Train model
model = RandomForestClassifier(n_estimators=100, class_weight="balanced", random_state=42)
model.fit(X, y)

# Save model and preprocessing tools
os.makedirs("Notebook", exist_ok=True)
pickle.dump(model, open("Notebook/model.pkl", "wb"))
pickle.dump(scaler, open("Notebook/scaler.pkl", "wb"))
pickle.dump(imputer, open("Notebook/imputer.pkl", "wb"))
pickle.dump(X.columns.tolist(), open("Notebook/features.pkl", "wb"))

print("✅ Model, scaler, imputer, and feature list saved.")