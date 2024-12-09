# Impact-Project-Predictive-Maintenance-System-for-Industrial-Equipment
Objective

To develop a predictive maintenance system that minimizes equipment downtime and operational costs by using real-time data to forecast potential failures.

Description

This project focuses on analyzing historical data and real-time sensor readings from industrial machinery to predict equipment failures. By implementing predictive analytics, the system ensures timely maintenance, reduces costs, and improves operational efficiency. This project aligns with JSW Group's focus on innovation and sustainable growth.

Methodology

Data Collection: Gather historical maintenance data and sensor readings from industrial equipment.
Preprocessing: Clean and normalize data to remove inconsistencies.
Feature Engineering: Identify key indicators such as vibration, temperature, and operational hours that correlate with equipment failures.
Model Development: Use machine learning models like Random Forest or Gradient Boosting to predict failure probabilities.
Real-Time Monitoring: Integrate IoT sensors to capture live data for the model.
Dashboard: Build a user-friendly dashboard to display predictions and maintenance schedules.

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
import matplotlib.pyplot as plt
import seaborn as sns

# Sample dataset creation
data = {
    'Temperature': np.random.uniform(50, 100, 1000),
    'Vibration': np.random.uniform(0.5, 2.5, 1000),
    'Pressure': np.random.uniform(1.0, 5.0, 1000),
    'MaintenanceRequired': np.random.choice([0, 1], 1000, p=[0.8, 0.2])  # 0: No, 1: Yes
}
df = pd.DataFrame(data)

# Display dataset
print(df.head())

# Splitting features and labels
X = df[['Temperature', 'Vibration', 'Pressure']]
y = df['MaintenanceRequired']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Handling imbalanced data using SMOTE
smote = SMOTE(random_state=42)
X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)

# Normalizing the data
scaler = StandardScaler()
X_train_resampled = scaler.fit_transform(X_train_resampled)
X_test_scaled = scaler.transform(X_test)  # Ensure X_test is scaled properly

# Initialize Random Forest Classifier with hyperparameter tuning
param_grid = {
    'n_estimators': [50, 100],
    'max_depth': [None, 10, 20],
    'min_samples_split': [2, 5]
}
rf_model = RandomForestClassifier(random_state=42)
grid_search = GridSearchCV(estimator=rf_model, param_grid=param_grid, cv=3)
grid_search.fit(X_train_resampled, y_train_resampled)

# Best model from grid search
best_rf_model = grid_search.best_estimator_

# Train the model
best_rf_model.fit(X_train_resampled, y_train_resampled)

# Predictions
y_pred_rf = best_rf_model.predict(X_test_scaled)

# Evaluate the model
print("Random Forest Confusion Matrix:\n", confusion_matrix(y_test, y_pred_rf))
print("\nRandom Forest Classification Report:\n", classification_report(y_test, y_pred_rf))

# Visualize Confusion Matrix for Random Forest
plt.figure(figsize=(6, 4))
sns.heatmap(confusion_matrix(y_test, y_pred_rf), annot=True, fmt="d", cmap="Blues", xticklabels=['No', 'Yes'], yticklabels=['No', 'Yes'])
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Random Forest Confusion Matrix")
plt.show()

# Compare with Logistic Regression Model
logistic_model = LogisticRegression(random_state=42)
logistic_model.fit(X_train_resampled, y_train_resampled)

# Predictions for Logistic Regression
y_pred_lr = logistic_model.predict(X_test_scaled)

# Evaluate the Logistic Regression model
print("Logistic Regression Confusion Matrix:\n", confusion_matrix(y_test, y_pred_lr))
print("\nLogistic Regression Classification Report:\n", classification_report(y_test, y_pred_lr))

# Visualize Confusion Matrix for Logistic Regression
plt.figure(figsize=(6, 4))
sns.heatmap(confusion_matrix(y_test, y_pred_lr), annot=True, fmt="d", cmap="Blues", xticklabels=['No', 'Yes'], yticklabels=['No', 'Yes'])
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Logistic Regression Confusion Matrix")
plt.show()

# Real-time simulation for Random Forest prediction
new_data = pd.DataFrame({
    'Temperature': [85],
    'Vibration': [1.8],
    'Pressure': [4.0]
})

new_data_scaled = scaler.transform(new_data)  # Ensure new data is scaled properly
prediction_rf = best_rf_model.predict(new_data_scaled)
print(f"Random Forest Prediction (0 = No Maintenance, 1 = Maintenance Required): {prediction_rf[0]}")

# Real-time simulation for Logistic Regression prediction
prediction_lr = logistic_model.predict(new_data_scaled)
print(f"Logistic Regression Prediction (0 = No Maintenance, 1 = Maintenance Required): {prediction_lr[0]}")

OUTPUT

![image](https://github.com/user-attachments/assets/cd2633bd-2c9f-4f96-b703-023eb004daea)

![image](https://github.com/user-attachments/assets/4a6251dc-e5fd-4d84-8742-798a06cd59c4)





