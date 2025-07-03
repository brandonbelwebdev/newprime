"""
Reservation Cancellation Prediction using Decision Tree (Gini Criterion)

This script builds a predictive model using a decision tree classifier to determine 
whether a hotel reservation will be cancelled or not. It includes:

- Data cleaning and preprocessing
- Feature selection and categorical encoding
- Training/test split
- Model training using DecisionTreeClassifier
- Prediction and evaluation (accuracy score)

Dependencies:
- numpy, pandas, seaborn, matplotlib
- category_encoders
- scikit-learn

Author: Brandon Costello
"""

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import sys as ss

# Load the dataset
df_rez = pd.read_csv('/Users/brandongwp/Desktop/Reservations.csv')

# Preview dataset
print(df_rez.head())
print(df_rez.isnull().sum())

# Define features and target
x = df_rez.drop(['cancelled', 'Booking_ID', 'booking_status'], axis=1)
y = df_rez['cancelled']

# Train-test split
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(x, y, test_size=0.33, random_state=42)

# Encode categorical features
import category_encoders as ce
cols_to_encode = ['type_of_meal_plan', 'room_type_reserved', 'market_segment_type']
encoder = ce.OrdinalEncoder(cols=cols_to_encode)
X_train = encoder.fit_transform(X_train)
X_test = encoder.transform(X_test)

# Show transformed data
print(X_train.head())
print(X_test.head())
print(Y_train.shape, Y_test.shape)

# Train Decision Tree model
from sklearn.tree import DecisionTreeClassifier
clf_gini = DecisionTreeClassifier(criterion='gini', max_depth=3, random_state=0)
clf_gini.fit(X_train, Y_train)

# Make predictions
y_pred_gini = clf_gini.predict(X_test)
print("First 5 predictions:", y_pred_gini[:5])

# Evaluate accuracy
from sklearn.metrics import accuracy_score
print("Model accuracy score with criterion gini index: {:.4f}".format(accuracy_score(Y_test, y_pred_gini)))
