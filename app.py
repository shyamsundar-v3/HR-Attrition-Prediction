# app.py

import streamlit as st
import pickle
import pandas as pd

st.title("HR Attrition Prediction")

pipeline = pickle.load(open("models/pipeline.pkl", "rb"))

# Collect inputs for all required features
age = st.number_input("Age", 18, 65, 30)
business_travel = st.selectbox("Business Travel", ['Non-Travel', 'Travel_Rarely', 'Travel_Frequently'])
department = st.selectbox("Department", ['Sales', 'Research & Development', 'Human Resources'])
distance_from_home = st.number_input("Distance From Home", 0, 50, 10)
education = st.selectbox("Education", [1, 2, 3, 4, 5])
education_field = st.selectbox("Education Field", ['Life Sciences', 'Medical', 'Marketing', 'Technical Degree', 'Other', 'Human Resources'])
gender = st.selectbox("Gender", ['Male', 'Female'])
job_role = st.selectbox("Job Role", [
    'Sales Executive', 'Research Scientist', 'Laboratory Technician', 'Manufacturing Director',
    'Healthcare Representative', 'Manager', 'Sales Representative', 'Research Director', 'Human Resources'])
over_time = st.selectbox("OverTime", ['Yes', 'No'])
marital_status = st.selectbox("Marital Status", ['Single', 'Married', 'Divorced'])
stock_option_level = st.number_input("Stock Option Level", 0, 3, 0)
monthly_income = st.number_input("Monthly Income", 1000, 20000, 5000)
years_since_last_promotion = st.number_input("Years Since Last Promotion", 0, 15, 0)
percent_salary_hike = st.number_input("Percent Salary Hike", 0, 30, 15)
years_with_curr_manager = st.number_input("Years With Current Manager", 0, 20, 3)
monthly_rate = st.number_input("Monthly Rate", 1000, 20000, 10000)
job_satisfaction = st.number_input("Job Satisfaction (1-4)", 1, 4, 3)
hourly_rate = st.number_input("Hourly Rate", 20, 100, 50)
job_level = st.number_input("Job Level", 1, 5, 1)
years_at_company = st.number_input("Years At Company", 0, 40, 5)
environment_satisfaction = st.number_input("Environment Satisfaction (1-4)", 1, 4, 3)
relationship_satisfaction = st.number_input("Relationship Satisfaction (1-4)", 1, 4, 3)
years_in_current_role = st.number_input("Years In Current Role", 0, 20, 3)
training_times_last_year = st.number_input("Training Times Last Year", 0, 10, 2)
work_life_balance = st.number_input("Work Life Balance (1-4)", 1, 4, 3)
daily_rate = st.number_input("Daily Rate", 100, 1500, 500)
job_involvement = st.number_input("Job Involvement (1-4)", 1, 4, 3)
performance_rating = st.number_input("Performance Rating (1-4)", 1, 4, 3)
num_companies_worked = st.number_input("Number of Companies Worked", 0, 20, 2)
total_working_years = st.number_input("Total Working Years", 0, 40, 10)

# Create DataFrame from inputs
input_dict = {
    'Age': age,
    'BusinessTravel': business_travel,
    'Department': department,
    'DistanceFromHome': distance_from_home,
    'Education': education,
    'EducationField': education_field,
    'Gender': gender,
    'JobRole': job_role,
    'OverTime': over_time,
    'MaritalStatus': marital_status,
    'StockOptionLevel': stock_option_level,
    'MonthlyIncome': monthly_income,
    'YearsSinceLastPromotion': years_since_last_promotion,
    'PercentSalaryHike': percent_salary_hike,
    'YearsWithCurrManager': years_with_curr_manager,
    'MonthlyRate': monthly_rate,
    'JobSatisfaction': job_satisfaction,
    'HourlyRate': hourly_rate,
    'JobLevel': job_level,
    'YearsAtCompany': years_at_company,
    'EnvironmentSatisfaction': environment_satisfaction,
    'RelationshipSatisfaction': relationship_satisfaction,
    'YearsInCurrentRole': years_in_current_role,
    'TrainingTimesLastYear': training_times_last_year,
    'WorkLifeBalance': work_life_balance,
    'DailyRate': daily_rate,
    'JobInvolvement': job_involvement,
    'PerformanceRating': performance_rating,
    'NumCompaniesWorked': num_companies_worked,
    'TotalWorkingYears': total_working_years
}

input_data = pd.DataFrame([input_dict])

if st.button("Predict"):
    prediction = pipeline.predict(input_data)[0]
    proba = pipeline.predict_proba(input_data)[0][1]

    if prediction == 1:
        st.error(f"Employee likely to attrite (Probability: {proba:.2f})")
    else:
        st.success(f"Employee unlikely to attrite (Probability: {proba:.2f})")

