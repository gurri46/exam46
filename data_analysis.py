python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import accuracy_score, classification_report

# Load the dataset
data = pd.read_csv('C:/Data.csv')

# Basic analysis (head, tail, describe)
print(data.head())
print(data.tail())
print(data.describe())

# Visualization
plt.figure(figsize=(16, 12))
sns.histplot(data['tenure'], bins=30, kde=True)
plt.title('Distribution of Customer Tenure')
plt.figure()
sns.histplot(data['age'], bins=30, kde=True)
plt.title('Age Distribution')
plt.figure()
sns.histplot(data['income'], bins=30, kde=True)
plt.title('Income Distribution')
plt.figure()
sns.countplot(x='ed', data=data)
plt.title('Education Level Distribution')
plt.show()

# Data Preparation: Checking for duplicates and missing values
print("Duplicates:", data.duplicated().sum())
print("Missing values:\n", data.isnull().sum())

# Splitting data into training and testing
X = data.drop('custcat', axis=1)  # Features
y = data['custcat']  # Target variable
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Applying KNN
knn_pipeline = make_pipeline(StandardScaler(), KNeighborsClassifier(n_neighbors=5))
knn_pipeline.fit(X_train, y_train)
y_pred = knn_pipeline.predict(X_test)

# Checking accuracy
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)
print("Classification Report:\n", classification_report(y_test, y_pred))
