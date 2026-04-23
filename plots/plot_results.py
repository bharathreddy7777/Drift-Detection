import pandas as pd
import matplotlib.pyplot as plt

# Load results
df = pd.read_csv("results/drift_results.csv")

# -------- Plot 1: Accuracy vs Batch --------
plt.figure()
plt.plot(df["batch"], df["accuracy"])
plt.xlabel("Batch Index")
plt.ylabel("Accuracy")
plt.title("Model Accuracy Across Batches")
plt.show()

# -------- Plot 2: Before vs After Retraining --------
plt.figure()
plt.plot(df["batch"], df["before_retrain"], label="Before Retraining")
plt.plot(df["batch"], df["after_retrain"], label="After Retraining")
plt.xlabel("Batch Index")
plt.ylabel("Accuracy")
plt.title("Effect of Sliding Window Retraining")
plt.legend()
plt.show()

# -------- Plot 3: Drift Detection Timeline --------
plt.figure()
plt.plot(df["batch"], df["ks_drift"], label="KS Drift")
plt.plot(df["batch"], df["psi_drift"], label="PSI Drift")
plt.plot(df["batch"], df["error_drift"], label="Error Drift")
plt.xlabel("Batch Index")
plt.ylabel("Drift Detected (0/1)")
plt.title("Drift Detection Over Time")
plt.legend()
plt.show()
