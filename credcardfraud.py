import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report, confusion_matrix

# Read the CSV file
df = pd.read_csv('C:\\Users\\Harry\\Desktop\\creditcardproject\\creditcard\\creditcard.csv')

# Create category mapping
categories = {0: 'Not Fraud', 1: 'Fraud'}

print("\nDistribution of transactions:")

print(df.Class.value_counts().rename(index = categories))

# Prepare the data by dropping the categories we want to exclude, class because its the target variable and amount because its not relevant.
X = df.drop(['Class', 'Amount'], axis=1)  # Features

y = df['Class']  # Target variable

# Split the data into training and testing sets
#x_train are the indepedent variables and y_train are the depedent variables.
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create and train the decision tree classifier
# Setting random_state=42 ensures that the tree structure is always the same every time you run the code. 42 is just a commonly used random seed value; any integer would work.
dt_classifier = DecisionTreeClassifier(random_state=42)

# training the decision tree classifier using dataset.
dt_classifier.fit(X_train, y_train)

# Make predictions
y_pred = dt_classifier.predict(X_test)

# Print classification report
print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=['Not Fraud', 'Fraud']))

# Print confusion matrix
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

# Feature importance
feature_importance = pd.DataFrame({
    'feature': X.columns,
    'importance': dt_classifier.feature_importances_
})
feature_importance = feature_importance.sort_values('importance', ascending=False)

print("\nTop 10 Most Important Features:")
print(feature_importance.head(10))

