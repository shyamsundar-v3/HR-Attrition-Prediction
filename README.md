# HR Attrition Prediction 🚀

This is a machine learning web application built with **Streamlit** to predict the probability of employee attrition (whether an employee is likely to leave the company or not) based on HR data. It uses a trained machine learning model and an intuitive frontend for HR professionals to input employee details and receive predictions.

---

## 🔍 Features

- Predicts likelihood of employee attrition using a trained ML model  
- Built with a clean and interactive UI using **Streamlit**  
- Model trained using **Random Forest Classifier** and **GridSearchCV**  
- Modular and scalable project structure  
- Deployed via **Streamlit Cloud**

---

## 📊 Dataset

The dataset used is the **IBM HR Analytics Employee Attrition & Performance** dataset, containing features like:

- Age, Gender, Job Role, Education  
- Monthly Income, Stock Option Level  
- OverTime, Job Satisfaction, Environment Satisfaction  
- And many more...

---

## 🏗️ Project Structure

```text
HR_Attrition_Project/
│
├── app.py
├── models/
│   └── pipeline.pkl
├── preprocessing.py
├── train_model.py
├── requirements.txt
├── runtime.txt
├── data/
│   └── HR_Employee_Attrition-1.csv
└── README.md
```


---

## 🚀 How to Run Locally

### 1. Clone the repository

git clone https://github.com/shyamsundar-v3/HR-Attrition-Prediction.git<br>

cd HR-Attrition-Prediction

### 2. Create a virtual environment (optional but recommended)

python -m venv venv
# On macOS/Linux:
source venv/bin/activate
# On Windows:
venv\Scripts\activate

1. Install dependencies

pip install -r requirements.txt

2. Train the model

python train_model.py

3. Run the Streamlit app

streamlit run app.py



# 📌 Tech Stack

1.Python 

2.Streamlit

3.scikit-learn

4.pandas, numpy, matplotlib, seaborn

# ✨ Future Improvements

-Add SHAP or LIME interpretability for feature importance

-Improve prediction confidence with advanced models (e.g., XGBoost)

-Allow CSV upload for batch predictions

-Visualize employee attrition trends in dashboard
