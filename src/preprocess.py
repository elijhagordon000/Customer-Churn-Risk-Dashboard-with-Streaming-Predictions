import pandas as pd
# Splitting TotalCharges Values:
df = pd.read_csv('data/raw/WA_Fn-UseC_-Telco-Customer-Churn.csv')

df["TotalCharges"] = df["TotalCharges"].str.strip()
df["TotalCharges"] = df["TotalCharges"].replace("", "0")
df["TotalCharges"] = df["TotalCharges"].astype(float)



