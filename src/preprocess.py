import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import MinMaxScaler

df = pd.read_csv('data/raw/WA_Fn-UseC_-Telco-Customer-Churn.csv')

# Human readable columns for later Tableau / scored output
readable_df = df[[
    "tenure",
    "MonthlyCharges",
    "TotalCharges",
    "Contract",
    "InternetService"
]].copy()

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
# Focusing on changing gender , Partner , PhoneService, ,PaperlessBilling, and dependents into binary values

# gender:
features_df["gender"] = features_df["gender"].replace({"Female": 0, "Male": 1})
# Partner:
features_df["Partner"] = features_df["Partner"].replace({"No": 0, "Yes": 1})
# PhoneService
features_df["PhoneService"] = features_df["PhoneService"].replace({"No": 0, "Yes": 1})
# PaperlessBilling:
features_df["PaperlessBilling"] = features_df["PaperlessBilling"].replace({"No": 0, "Yes": 1})
#Dependents:
features_df["Dependents"] = features_df["Dependents"].replace({"No": 0, "Yes":1})

# Categories with more than two options: MultipleLines, OnlineSecurity, OnlineBackup, DeviceProtection, TechSupport, StreamingTV , StreamingMovies, InternetService

# Multiple Lines: (Using "No phone service" and "No" as 0  and "Yes" as 1)
features_df["MultipleLines"] = features_df["MultipleLines"].replace({"No": 0,"No phone service": 0 , "Yes": 1})

# Options for internet service are: No, DSL, and Fiber optic.
# Here I will seperate the Internet Service column into Has DSL and Has Fiber Optic columns:
internet_encode = pd.get_dummies(
    features_df["InternetService"],
    prefix="InternetService",
    dtype=int
)
internet_encode = internet_encode.rename(
    columns={"InternetService_Fiber optic": "InternetService_Fiber_optic"}
)
features_df = features_df.drop("InternetService", axis=1)
features_df = pd.concat([features_df, internet_encode], axis=1)
# Online Security
online_security_encode = pd.get_dummies(
    features_df["OnlineSecurity"],
    prefix="OnlineSecurity",
    dtype=int
)

online_security_encode = online_security_encode.rename(
    columns={"OnlineSecurity_No internet service": "OnlineSecurity_No_internet_service"}
)
features_df = features_df.drop("OnlineSecurity", axis=1)
features_df = pd.concat([features_df, online_security_encode], axis=1)
# Online Backup:
online_backup_encode = pd.get_dummies(
    features_df["OnlineBackup"],
    prefix="OnlineBackup",
    dtype=int
)

online_backup_encode = online_backup_encode.rename(
    columns={"OnlineBackup_No internet service": "OnlineBackup_No_internet_service"}
)

features_df = features_df.drop("OnlineBackup", axis=1)
features_df = pd.concat([features_df, online_backup_encode], axis=1)
#Device Protection
device_protection_encode = pd.get_dummies(
    features_df["DeviceProtection"],
    prefix="DeviceProtection",
    dtype=int
)

device_protection_encode = device_protection_encode.rename(
    columns={"DeviceProtection_No internet service": "DeviceProtection_No_internet_service"}
)

features_df = features_df.drop("DeviceProtection", axis=1)
features_df = pd.concat([features_df, device_protection_encode], axis=1)
# TechSupport
tech_support_encode = pd.get_dummies(
    features_df["TechSupport"],
    prefix="TechSupport",
    dtype=int
)

tech_support_encode = tech_support_encode.rename(
    columns={"TechSupport_No internet service": "TechSupport_No_internet_service"}
)

features_df = features_df.drop("TechSupport", axis=1)
features_df = pd.concat([features_df, tech_support_encode], axis=1)
# StreamingTV:
streaming_tv_encode = pd.get_dummies(
    features_df["StreamingTV"],
    prefix="StreamingTV",
    dtype=int
)

streaming_tv_encode = streaming_tv_encode.rename(
    columns={"StreamingTV_No internet service": "StreamingTV_No_internet_service"}
)

features_df = features_df.drop("StreamingTV", axis=1)
features_df = pd.concat([features_df, streaming_tv_encode], axis=1)
# Streaming Movies
streaming_movies_encode = pd.get_dummies(
    features_df["StreamingMovies"],
    prefix="StreamingMovies",
    dtype=int
)

streaming_movies_encode = streaming_movies_encode.rename(
    columns={"StreamingMovies_No internet service": "StreamingMovies_No_internet_service"}
)

features_df = features_df.drop("StreamingMovies", axis=1)
features_df = pd.concat([features_df, streaming_movies_encode], axis=1)

# Contract:
contract_encode = pd.get_dummies(
    features_df["Contract"],
    prefix="Contract",
    dtype=int
)

contract_encode = contract_encode.rename(
    columns={
        "Contract_Month-to-month": "Contract_Month_to_month",
        "Contract_One year": "Contract_One_year",
        "Contract_Two year": "Contract_Two_year"
    }
)
features_df = features_df.drop("Contract", axis=1)
features_df = pd.concat([features_df, contract_encode], axis=1)

# PaymentMethod:
payment_method_encode = pd.get_dummies(
    features_df["PaymentMethod"],
    prefix="PaymentMethod",
    dtype=int
)

payment_method_encode = payment_method_encode.rename(
    columns={
        "PaymentMethod_Electronic check": "PaymentMethod_Electronic_check",
        "PaymentMethod_Mailed check": "PaymentMethod_Mailed_check",
        "PaymentMethod_Bank transfer (automatic)": "PaymentMethod_Bank_transfer_automatic",
        "PaymentMethod_Credit card (automatic)": "PaymentMethod_Credit_card_automatic"
    }
)
features_df = features_df.drop("PaymentMethod", axis=1)
features_df = pd.concat([features_df, payment_method_encode], axis=1)

# Changing target column to numerical values:
target_series = target_series.replace({"Yes": 1, "No": 0})

# Splitting Training and Testing sets
X_train, X_test, Y_train, Y_test, readable_X_train, readable_X_test  = train_test_split(
    features_df,
    target_series,
    readable_df,
    test_size=0.2,
    random_state=42,
    stratify=target_series
)

# Scaling column values (that need scaling; ie not binary values like Yes and No) in X_train:

columns_to_scale = ["tenure", "MonthlyCharges", "TotalCharges"]

ct = ColumnTransformer(
    transformers=[
        ("scaler", MinMaxScaler(), columns_to_scale)
    ],
    remainder="passthrough"
)

X_train = ct.fit_transform(X_train)
X_test = ct.transform(X_test)
#Turning numpy arrays back to dataframes to give the training and testing column names
feature_names = ct.get_feature_names_out()
X_train = pd.DataFrame(X_train, columns=feature_names)
X_test = pd.DataFrame(X_test, columns=feature_names)

X_train.to_csv("data/processed/X_train.csv", index=False)
X_test.to_csv("data/processed/X_test.csv", index=False)
Y_train.to_csv("data/processed/y_train.csv", index=False)
Y_test.to_csv("data/processed/y_test.csv", index=False)
readable_X_train.to_csv("data/processed/readable_X_train.csv", index=False)
readable_X_test.to_csv("data/processed/readable_X_test.csv", index=False)
