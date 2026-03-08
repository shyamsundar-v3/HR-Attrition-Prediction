# src/preprocessing.py

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OrdinalEncoder

def load_data(csv_path):
    df = pd.read_csv(csv_path)
    # Drop irrelevant columns
    df.drop(columns=['EmployeeNumber', 'EmployeeCount', 'Over18', 'StandardHours'], inplace=True, errors='ignore')
    return df

def get_features_target_preprocessor(df):
    # Target: convert to 0/1
    df['Attrition'] = df['Attrition'].map({'Yes':1, 'No':0})

    X = df.drop('Attrition', axis=1)
    y = df['Attrition']

    # Identify categorical and numerical columns
    categorical_cols = X.select_dtypes(include=['object']).columns.tolist()
    numerical_cols = X.select_dtypes(include=['int64', 'float64']).columns.tolist()

    # Build ColumnTransformer for preprocessing
    preprocessor = ColumnTransformer(
        transformers=[
            ('cat', OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1), categorical_cols)
            # You can add numeric transformers/scalers if needed here
        ],
        remainder='passthrough'
    )

    return X, y, preprocessor
