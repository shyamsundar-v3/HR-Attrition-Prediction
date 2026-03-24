# app.py

import streamlit as st
import pickle
import pandas as pd

st.title("HR Attrition Prediction")

# Load pipeline
pipeline = pickle.load(open("models/pipeline.pkl", "rb"))

# Inputs (example - adjust based on your features)
age = st.number_input("Age", 18, 65, 30)
business_travel = st.selectbox("Business Travel", ['Non-Travel', 'Travel_Rarely', 'Travel_Frequently'])
department = st.selectbox("Department", ['Sales', 'Research & Development', 'Human Resources'])
distance = st.number_input("Distance From Home", 0, 50, 10)
education = st.selectbox("Education", [1, 2, 3, 4, 5])
education_field = st.selectbox("Education Field", ['Life Sciences', 'Medical', 'Marketing', 'Technical Degree', 'Other', 'Human Resources'])
gender = st.selectbox("Gender", ['Male', 'Female'])
job_role = st.selectbox("Job Role", ['Sales Executive', 'Research Scientist', 'Laboratory Technician', 'Manufacturing Director',
                                    'Healthcare Representative', 'Manager', 'Sales Representative', 'Research Director', 'Human Resources'])
over_time = st.selectbox("OverTime", ['Yes', 'No'])

input_data = pd.DataFrame({
    'Age': [age],
    'BusinessTravel': [business_travel],
    'Department': [department],
    'DistanceFromHome': [distance],
    'Education': [education],
    'EducationField': [education_field],
    'Gender': [gender],
    'JobRole': [job_role],
    'OverTime': [over_time]
})

if st.button("Predict"):
    prediction = pipeline.predict(input_data)[0]
    proba = pipeline.predict_proba(input_data)[0][1]

    if prediction == 1:
        st.error(f"Employee likely to attrite (Probability: {proba:.2f})")
    else:
        st.success(f"Employee unlikely to attrite (Probability: {proba:.2f})")
