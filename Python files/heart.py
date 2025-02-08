import pandas as pd
import numpy as np
import pickle
from imblearn.under_sampling import TomekLinks
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix

# Load dataset
data = pd.read_csv(r'C:\Users\Aniket\OneDrive\Desktop\AICTE Internship\MIcrosoft\Dataset\heart.csv')

# Splitting features and target
X = data.drop(columns=['target'])  # Assuming 'target' is the target column
y = data['target']

# Handle class imbalance
tl = TomekLinks()
X_resampled, y_resampled = tl.fit_resample(X, y)

# Split data
X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.2, random_state=42)

# Train model
best_model = None
best_accuracy = 0
best_precision = 0
best_recall = 0

for _ in range(10):  # Try multiple times to get best model
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    
    print(acc)
    print(prec)
    print(rec)
    if acc > 0.76 and prec > 0.76 and rec > 0.76:
        best_model = model
        best_accuracy = acc
        best_precision = prec
        best_recall = rec
        break  # Stop training if criteria met

# Save the model if performance is good
if best_model:
    with open(r'C:\Users\Aniket\OneDrive\Desktop\AICTE Internship\MIcrosoft\Trained Models\heart_model.pkl', 'wb') as file:
        pickle.dump(best_model, file)
    print(f'Model saved with Accuracy: {best_accuracy}, Precision: {best_precision}, Recall: {best_recall}')
else:
    print('Could not achieve desired performance after multiple attempts.')

# Print confusion matrix
print('Confusion Matrix:\n', confusion_matrix(y_test, y_pred))
