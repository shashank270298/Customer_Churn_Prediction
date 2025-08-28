# 1. Importing the libraries

import streamlit as st
import pandas as pd
import pickle
from sklearn.preprocessing import LabelEncoder, StandardScaler


# 2. Loading the saved model

with open("notebooks/churn_model.pkl", "rb") as file:
    model = pickle.load(file)

    
# 3. Streamlit app title 

st.set_page_config(page_title="Customer Churn Prediction", layout="wide")
st.title("📊 Telco Customer Churn Prediction")
st.write("Predict whether a customer is likely to churn based on their details.")


# Sidebar for user input

st.sidebar.header("Enter Customer Details")


# Function to collect from user inputs

def get_user_input():
    gender = st.sidebar.selectbox("Gender", ("Male", "Female"))
    SeniorCitizen = st.sidebar.selectbox("Senior Citizen", (0, 1))
    Partner = st.sidebar.selectbox("Partner", ("Yes", "No"))
    Dependents = st.sidebar.selectbox("Dependents", ("Yes", "No"))
    tenure = st.sidebar.slider("Tenure (Months)", 0, 72, 1)
    PhoneService = st.sidebar.selectbox("Phone Service", ("Yes", "No"))
    MultipleLines = st.sidebar.selectbox("Multiple Lines", ("Yes", "No", "No phone service"))
    InternetService = st.sidebar.selectbox("Internet Service", ("DSL", "Fiber optic", "No"))
    OnlineSecurity = st.sidebar.selectbox("Online Security", ("Yes", "No", "No internet service"))
    OnlineBackup = st.sidebar.selectbox("Online Backup", ("Yes", "No", "No internet service"))
    DeviceProtection = st.sidebar.selectbox("Device Protection", ("Yes", "No", "No internet service"))
    TechSupport = st.sidebar.selectbox("Tech Support", ("Yes", "No", "No internet service"))
    StreamingTV = st.sidebar.selectbox("Streaming TV", ("Yes", "No", "No internet service"))
    StreamingMovies = st.sidebar.selectbox("Streaming Movies", ("Yes", "No", "No internet service"))
    Contract = st.sidebar.selectbox("Contract", ("Month-to-month", "One year", "Two year"))
    PaperlessBilling = st.sidebar.selectbox("Paperless Billing", ("Yes", "No"))
    PaymentMethod = st.sidebar.selectbox("Payment Method", ("Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"))
    MonthlyCharges = st.sidebar.number_input("Monthly Charges", 0, 150, 70)
    TotalCharges = st.sidebar.number_input("Total Charges", 0, 10000, 500)
    
    data = {
        "gender": gender,
        "SeniorCitizen": SeniorCitizen,
        "Partner": Partner,
        "Dependents": Dependents,
        "tenure": tenure,
        "PhoneService": PhoneService,
        "MultipleLines": MultipleLines,
        "InternetService": InternetService,
        "OnlineSecurity": OnlineSecurity,
        "OnlineBackup": OnlineBackup,
        "DeviceProtection": DeviceProtection,
        "TechSupport": TechSupport,
        "StreamingTV": StreamingTV,
        "StreamingMovies": StreamingMovies,
        "Contract": Contract,
        "PaperlessBilling": PaperlessBilling,
        "PaymentMethod": PaymentMethod,
        "MonthlyCharges": MonthlyCharges,
        "TotalCharges": TotalCharges
    }
    
    return pd.DataFrame(data, index=[0])

# Get user input data

input_df = get_user_input()

# Encode categorical variables similar to training 

df_encoded = input_df.copy()
for col in df_encoded.columns:
    if df_encoded[col].dtype == 'object':
        le = LabelEncoder()
        df_encoded[col] = le.fit_transform(df_encoded[col])
        
# Scaling the features

scaler = StandardScaler()
df_encoded = scaler.fit_transform(df_encoded)

# Make Prediction 

if st.sidebar.button("Predict Churn"):
    prediction = model.predict(df_encoded)
    if prediction[0] == 1:
        st.error("⚠️ The customer is likely to churn!")
    else:
        st.success("✅ The customer is likely to stay.")