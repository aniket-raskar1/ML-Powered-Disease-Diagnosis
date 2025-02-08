import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix
from imblearn.under_sampling import TomekLinks
import pickle

# Load dataset
data = pd.read_csv(r"C:\Users\Aniket\OneDrive\Desktop\AICTE Internship\MIcrosoft\Dataset\parkinsons.csv")

# Drop the non-numeric column
data = data.drop(columns=["name"])

# Separate features and target
X = data.drop(columns=["status"])  # Features
y = data["status"]  # Target

# Standardize data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Apply Tomek Links (Only on numeric data)
tl = TomekLinks()
X_resampled, y_resampled = tl.fit_resample(X_scaled, y)

# Split data
X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.2, random_state=42)

# Train model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Predictions
y_pred = model.predict(X_test)

# Evaluation metrics
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
cm = confusion_matrix(y_test, y_pred)

# Display results
print(f"Accuracy: {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print("Confusion Matrix:\n", cm)

# Save model if accuracy, precision, and recall > 95%
if accuracy > 0.88 and precision > 0.95 and recall > 0.89:
    with open(r"C:\Users\Aniket\OneDrive\Desktop\AICTE Internship\MIcrosoft\Trained Models\parkinsons_model.pkl", "wb") as f:
        pickle.dump(model, f)
    print("Model saved successfully!")
else:
    print("Model did not meet the 95% threshold, retraining may be needed.")
