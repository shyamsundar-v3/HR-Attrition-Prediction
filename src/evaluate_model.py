# src/evaluate_model.py

import pickle
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve

from preprocessing import load_data, get_features_target_preprocessor

def main():
    # Load pipeline
    with open("models/pipeline.pkl", "rb") as f:
        pipeline = pickle.load(f)

    # Fixed indentation here 
    df = load_data("data/HR_Employee_Attrition-1.csv")
    X, y, _ = get_features_target_preprocessor(df)

    y_pred = pipeline.predict(X)
    y_proba = pipeline.predict_proba(X)[:, 1]

    # Confusion matrix
    cm = confusion_matrix(y, y_pred)
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.show()

    print("Classification Report:\n")
    print(classification_report(y, y_pred))

    # ROC Curve
    roc_auc = roc_auc_score(y, y_proba)
    fpr, tpr, _ = roc_curve(y, y_proba)
    plt.figure()
    plt.plot(fpr, tpr, label=f'ROC curve (area = {roc_auc:.2f})', color='darkorange', lw=2)
    plt.plot([0, 1], [0, 1], linestyle='--', color='navy')
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve")
    plt.legend(loc="lower right")
    plt.show()

   

if __name__ == "__main__":
    main()
