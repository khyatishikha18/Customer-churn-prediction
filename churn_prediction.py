# churn_prediction.py

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

# Step 1: Load Dataset (sample telecom churn dataset from Kaggle or local CSV)
# Replace 'your_dataset.csv' with your actual file name
data = pd.read_csv('customer_churn.csv')

# Step 2: Basic Exploration
print(data.head())
print("\nData Summary:\n", data.info())
print("\nMissing Values:\n", data.isnull().sum())

# Step 3: Preprocessing
# Drop customerID if present
if 'customerID' in data.columns:
    data.drop('customerID', axis=1, inplace=True)

# Convert 'TotalCharges' to numeric
data['TotalCharges'] = pd.to_numeric(data['TotalCharges'], errors='coerce')

# Fill missing values
data['TotalCharges'].fillna(data['TotalCharges'].median(), inplace=True)

# Encode categorical variables
label_enc = LabelEncoder()
for column in data.select_dtypes(include=['object']).columns:
    data[column] = label_enc.fit_transform(data[column])

# Step 4: Split features and target
X = data.drop('Churn', axis=1)
y = data['Churn']

# Step 5: Scale the data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Step 6: Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Step 7: Model Training
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Step 8: Prediction & Evaluation
y_pred = model.predict(X_test)

print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))
print("Accuracy Score:", accuracy_score(y_test, y_pred))
