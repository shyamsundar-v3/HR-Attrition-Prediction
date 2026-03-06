# eda/eda_visuals.py

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load data
df = pd.read_csv("data/HR_Employee_Attrition-1.csv")

# Attrition distribution
plt.figure(figsize=(5, 4))
sns.countplot(x='Attrition', data=df)
plt.title("Attrition Count")
plt.savefig("eda/attrition_count.png")
plt.close()

# Age distribution
plt.figure(figsize=(6, 4))
sns.histplot(df['Age'], kde=True)
plt.title("Age Distribution")
plt.savefig("eda/age_distribution.png")
plt.close()

# Attrition vs Job Role
plt.figure(figsize=(10, 5))
sns.countplot(x='JobRole', hue='Attrition', data=df)
plt.title("Attrition by Job Role")
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig("eda/attrition_by_jobrole.png")
plt.close()

# Correlation heatmap
plt.figure(figsize=(12, 8))
corr = df.select_dtypes(include='number').corr()
sns.heatmap(corr, cmap="coolwarm", annot=False)
plt.title("Feature Correlation Heatmap")
plt.savefig("eda/correlation_heatmap.png")
plt.close()
