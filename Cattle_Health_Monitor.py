import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
# from sklearn.tree import export_graphviz
# import pydotplus
# from IPython.display import Image
# from six import StringIO

df = pd.read_excel(r'Cleaned_Data.xlsx')
X = df[["body_temperature", "heart_rate", "sleeping_duration", "lying_down_duration"]]
Y = df["health_status"]
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=42)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train_scaled, y_train)
y_pred = model.predict(X_test_scaled)
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.2f}")
print("Classification Report:")
print(classification_report(y_test, y_pred))
def validate_input(temp, heart, hr, shr ):
    if not (30 <= temp <= 45):
        raise ValueError("Temperature")
    if not (2 <= shr <= 5):
        raise ValueError("Sleep time")
    if not (8 <= hr <= 13):
        raise ValueError("Laying down")
    if not (40 <= heart <= 120):
        raise ValueError("Heat rate")
    return True
try:
    temperature = float(input("Enter cattle's temperature (°C): ")) 
    heart_rate = int(input("Enter cattle's heart rate (bpm): ")) 
    laying_down = int(input("Enter cattle's movement level: ")) 
    sleeping = int(input("Enter cattle's sleeping time: "))
    new_data = np.array([[temperature, heart_rate, laying_down,sleeping]]) 
    new_data_scaled = scaler.transform(new_data) 
    prediction = model.predict(new_data_scaled)
    if prediction[0] == 1: 
        print("Cattle is healthy.") 
    else: 
        print("Cattle might have health issues.") 
except ValueError as ve: 
    print(ve)

# model = RandomForestClassifier(n_estimators=100, random_state=42)
# model.fit(X_train_scaled, y_train)
# estimator = model.estimators_[0]
# dot_data = StringIO()
# export_graphviz(estimator, out_file=dot_data, 
#                 filled=True, rounded=True, 
#                 special_characters=True, feature_names = X.columns, class_names=['Healthy','Unhealthy'])
# graph = pydotplus.graph_from_dot_data(dot_data.getvalue())
# Image(graph.create_png())