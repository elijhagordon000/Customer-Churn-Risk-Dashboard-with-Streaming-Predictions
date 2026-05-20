import pandas as pd

df = pd.read_csv('data/raw/WA_Fn-UseC_-Telco-Customer-Churn.csv')
# Splitting TotalCharges Values:
df["TotalCharges"] = df["TotalCharges"].str.strip()
df["TotalCharges"] = df["TotalCharges"].replace("", "0")
df["TotalCharges"] = df["TotalCharges"].astype(float)

# Dropping Customer ID because it is just an identifier and does not provide predictive value for churn modeling.
df = df.drop("customerID", axis=1)

# Features
features_df = df.drop("Churn", axis=1)

# Target
target_series = df["Churn"]

# Turning categorical labels into numeric features the model can use.
# Focusing on changing gender , Partner , PhoneService, and PaperlessBilling into binary values

# gender:
features_df["gender"] = features_df["gender"].replace({"Female": 0, "Male": 1})
# Partner:
features_df["Partner"] = features_df["Partner"].replace({"No": 0, "Yes": 1})
# PhoneService
features_df["PhoneService"] = features_df["PhoneService"].replace({"No": 0, "Yes": 1})
# PaperlessBilling:
features_df["PaperlessBilling"] = features_df["PaperlessBilling"].replace({"No": 0, "Yes": 1})

# Categories with more than two options: MultipleLines, OnlineSecurity, OnlineBackup, DeviceProtection, TechSupport, StreamingTV , StreamingMovies

# Multiple Lines: (Using "No phone service" and "No" as 0  and "Yes" as 1)
features_df["MultipleLines"] = features_df["MultipleLines"].replace({"No": 0,"No phone service": 0 , "Yes": 1})




