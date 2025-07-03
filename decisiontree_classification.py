import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Load dataset
df_car = pd.read_csv('/Users/brandonog/Desktop/car_evaluation_class.csv')
print(df_car.columns)
print(df_car.isnull().sum())
X = df_car.drop(['class'], axis=1)
y = df_car['class']

# Split into test and training
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.33, random_state=42
)

# Convert categorical encoders
from sklearn.preprocessing import OrdinalEncoder
encoder = OrdinalEncoder(cols=['buying', 'maint', 'doors', 'persons', 'lug_boot', 'safety'])
X_train = encoder.fit_transform(X_train)
X_test = encoder.transform(X_test)

print(X_train.shape, X_test.shape)

# Train decision tree
from sklearn.tree import DecisionTreeClassifier
clf_gini = DecisionTreeClassifier(criterion='gini', max_depth=3, random_state=0)
clf_gini.fit(X_train, y_train)

# Predict and measure accuracy
y_pred_gini = clf_gini.predict(X_test)

from sklearn.metrics import accuracy_score
print("Model accuracy score with Gini index: {:.4f}".format(accuracy_score(y_test, y_pred_gini)))

# Predict on training set
y_pred_train_gini = clf_gini.predict(X_train)
print("Training set accuracy score: {:.4f}".format(accuracy_score(y_train, y_pred_train_gini)))
print("Test set score: {:.4f}".format(clf_gini.score(X_test, y_test)))
print("Train set score: {:.4f}".format(clf_gini.score(X_train, y_train)))

# Plot tree
plt.figure(figsize=(12, 8))
from sklearn import tree
tree.plot_tree(clf_gini.fit(X_train, y_train))
plt.show()
