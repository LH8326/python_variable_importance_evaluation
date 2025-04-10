import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt

# Read the CSV file
df = pd.read_csv('C:\\Users\\Harry\\Desktop\\creditcardproject\\creditcard\\creditcard.csv')

# Prepare the data by dropping the categories we want to exclude, 'Class' is the target, 'Amount' and 'Time' are not relevant for feature importance.
X = df.drop(['Class', 'Amount', 'Time'], axis=1)  # Features
y = df['Class']  # Target variable

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create and train the decision tree classifier
dt_classifier = DecisionTreeClassifier(random_state=42)
dt_classifier.fit(X_train, y_train)

# Make predictions
y_pred = dt_classifier.predict(X_test)

# Print classification report and confusion matrix
print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=['Not Fraud', 'Fraud']))

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

# Feature importance
feature_importance = pd.DataFrame({
    'feature': X.columns,
    'importance': dt_classifier.feature_importances_
})

# Sort the feature importance in descending order
feature_importance = feature_importance.sort_values('importance', ascending=False)

# Show the top 10 most important features
print("\nTop 10 Most Important Features:")
print(feature_importance.head(10))

# Visualize the top 10 most important features in a bar chart
plt.figure(figsize=(10, 6))
plt.barh(feature_importance['feature'].head(10), feature_importance['importance'].head(10), color='skyblue')
plt.xlabel('Importance')
plt.title('Top 10 Most Important Features in Predicting Fraud')
plt.gca().invert_yaxis()  # Invert y-axis to have the most important feature on top
plt.show()
