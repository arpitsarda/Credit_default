
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import roc_auc_score
import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.compose import ColumnTransformer
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.pipeline import Pipeline
from sklearn.ensemble import IsolationForest


# Load the training data
train_data = pd.read_csv("train_1.csv")
loan_id_train = train_data['loan_id']

# Drop mixed columns
mixed_type_cols = train_data.select_dtypes(include=['object']).columns
train_data = train_data.drop(mixed_type_cols, axis=1, errors='ignore')

X_train = train_data.drop(['label'], axis=1)  # Features
y_train = train_data['label']

# Handle missing values using mean imputation
imputer = SimpleImputer(strategy='mean')
X_train_imputed = imputer.fit_transform(X_train)

# Scale the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_imputed)

# Outlier removal using Isolation Forest
iso_forest = IsolationForest(contamination=0.05) # Adjust contamination as needed
outlier_pred = iso_forest.fit_predict(X_train_scaled)
X_train_no_outliers = X_train_scaled[outlier_pred == 1]
y_train_no_outliers = y_train[outlier_pred == 1]


# Initialize and train a Random Forest Classifier
model = RandomForestClassifier(random_state=42, n_estimators=1200) # You can adjust hyperparameters here
model.fit(X_train_no_outliers, y_train_no_outliers)

print("Model Training finished")

# Load the test data
test_data = pd.read_csv("test_1.csv")
loan_id_test = test_data['loan_id']

# Drop mixed columns
test_data = test_data.drop(mixed_type_cols, axis=1, errors='ignore')

X_test = test_data  # Features

# Handle missing values in test data using the same imputer fitted on training data
X_test_imputed = imputer.transform(X_test)

# Scaling the test features using the same scaler fitted on training data
X_test_scaled = scaler.transform(X_test_imputed)

# Making predictions on the test data
y_pred_test = model.predict_proba(X_test_scaled)[:, 1]

# Creating submission file for test data
submission_test = pd.DataFrame({'loan_id': loan_id_test, 'prob': y_pred_test})
submission_test.to_csv('submission_test.csv', index=False)

print("Prediction finished")
