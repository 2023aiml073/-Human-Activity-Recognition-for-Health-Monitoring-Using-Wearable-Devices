import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Load trained model, scaler, and label encoder
model = joblib.load("model.pkl")
scaler = joblib.load("scaler.pkl")
label_encoder = joblib.load("label_encoder.pkl")

st.set_page_config(page_title="Human Activity Recognition", layout="wide")
st.title("Human Activity Recognition Dashboard")

st.sidebar.header("Upload Sensor Data")
uploaded_file = st.sidebar.file_uploader("Choose a CSV file", type="csv")

if uploaded_file is not None:
    data = pd.read_csv(uploaded_file)
    st.subheader("Uploaded Sensor Data")
    st.write(data.head())

    numeric_cols = data.select_dtypes(include=np.number).columns
    data_scaled = scaler.transform(data[numeric_cols])

    predictions = model.predict(data_scaled)
    predicted_labels = label_encoder.inverse_transform(predictions)
    data['Predicted Activity'] = predicted_labels

    st.subheader("Predicted Activities Summary")
    st.write(data['Predicted Activity'].value_counts().rename("Count"))

    st.subheader("Full Data with Predictions")
    st.dataframe(data)

    if (data['Predicted Activity'] == 'LAYING').sum() > len(data) * 0.6:
        st.warning("Prolonged laying detected! Consider checking in.")

    if (data['Predicted Activity'] == 'WALKING_DOWNSTAIRS').sum() > 0:
        st.info("Detected stair descent activity — monitor for stability.")
else:
    st.info("Upload a sensor data CSV file to get started.")
