
import pandas as pd
import joblib
import matplotlib.pyplot as plt

# Load the cleaned dataset
X = pd.read_csv("cleaned_air_quality.csv")

# Load the trained model
model = joblib.load("anomaly_model.joblib")

# Predict normal observations and anomalies
predictions = model.predict(X)

# Count each prediction type
normal_count = (predictions == 1).sum()
anomaly_count = (predictions == -1).sum()

print("Normal observations:", normal_count)
print("Anomalies detected:", anomaly_count)

# Plot CO readings and highlight anomalies
plt.figure(figsize=(12, 5))

plt.scatter(
    range(len(X)),
    X["CO(GT)"],
    c=["red" if p == -1 else "blue" for p in predictions],
    s=8
)

plt.title("EcoSense AI: Air Quality Anomaly Detection")
plt.xlabel("Observation number")
plt.ylabel("CO sensor reading")
plt.legend(
    handles=[
        plt.Line2D([], [], marker="o", linestyle="", color="blue",
                   label="Normal"),
        plt.Line2D([], [], marker="o", linestyle="", color="red",
                   label="Anomaly")
    ]
)

plt.tight_layout()
plt.savefig("anomaly_plot.png", dpi=150)
plt.show()

print("Graph saved as anomaly_plot.png")
