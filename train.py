
import pandas as pd
import joblib

from sklearn.ensemble import IsolationForest
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

# Step 1: Load the cleaned dataset
X = pd.read_csv("cleaned_air_quality.csv")

print("Dataset loaded:", X.shape)

# Step 2: Split data into training and testing sets
X_train, X_test = train_test_split(
    X,
    test_size=0.2,
    random_state=42
)

print("Training rows:", len(X_train))
print("Testing rows:", len(X_test))

# Step 3: Create the ML model
model = IsolationForest(
    n_estimators=100,
    contamination=0.05,
    random_state=42,
    n_jobs=-1
)

# Step 4: Train the model
print("\nTraining model...")
model.fit(X_train)

print("Training completed!")

# Step 5: Predict anomalies in the test set
predictions = model.predict(X_test)

# Isolation Forest returns:
#  1 = normal
# -1 = anomaly
anomaly_count = (predictions == -1).sum()
normal_count = (predictions == 1).sum()

print("\nEvaluation results")
print("Test observations:", len(X_test))
print("Normal observations:", normal_count)
print("Anomalies detected:", anomaly_count)
print(
    "Anomaly percentage:",
    round(anomaly_count / len(X_test) * 100, 2),
    "%"
)

# Step 6: Save the trained model
joblib.dump(model, "anomaly_model.joblib")

# Save the feature names for later use
joblib.dump(list(X.columns), "feature_names.joblib")

print("\nModel saved as anomaly_model.joblib")
print("Feature names saved successfully!")
