import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.tree import plot_tree
import matplotlib.pyplot as plt

# Read the data
df = pd.read_excel(r'Cleaned_Data.xlsx')
X = df[["body_temperature", "heart_rate", "sleeping_duration", "lying_down_duration"]]
Y = df["health_status"]

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.12, random_state=42)

# Feature scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Model - Using RandomForestClassifier
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train_scaled, y_train)

# Predict on test set
y_pred = model.predict(X_test_scaled)
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.2f}")
print("Classification Report:")
print(classification_report(y_test, y_pred))

# Visualize one tree from the Random Forest
#for i in range(100):
estimator = model.estimators_[45]

plt.figure(figsize=(20,10))
plot_tree(estimator, feature_names=X.columns, class_names=['Healthy', 'Unhealthy'], filled=True, rounded=True,proportion=True,node_ids=True,fontsize=10)
plt.show()