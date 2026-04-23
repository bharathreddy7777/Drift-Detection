import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
from collections import deque
import os

from training.train_baseline import load_dataset
from drift.data_stream import stream_batches
from drift.covariate_drift import apply_covariate_drift
from drift.concept_drift import apply_concept_drift
from drift.drift_detectors import ks_drift, psi_drift, error_rate_drift
from evaluation.batch_evaluation import evaluate_batch
from training.retrain_model import retrain_model

# ----------------- PAGE CONFIG -----------------
st.set_page_config(
    page_title="ML Drift Monitoring System",
    layout="wide"
)

st.title("📊 Machine Learning Drift Monitoring Dashboard")
st.markdown("Interactive demonstration of **data drift detection and adaptive retraining**")

# ----------------- LOAD DATA -----------------
X_train, X_test, y_train, y_test = load_dataset()
model = joblib.load("models/baseline_rf.pkl")

BASELINE_ACCURACY = 0.9285
BATCH_SIZE = 500
DRIFT_START = 3
WINDOW_SIZE = 5

# ----------------- SIDEBAR -----------------
st.sidebar.header("⚙ Experiment Controls")

enable_covariate = st.sidebar.checkbox("Enable Covariate Drift", True)
enable_concept = st.sidebar.checkbox("Enable Concept Drift", True)

run_button = st.sidebar.button("▶ Run Experiment")

# ----------------- STORAGE -----------------
batch_acc = []
before_retrain = []
after_retrain = []
ks_flags = []
psi_flags = []
error_flags = []

window_X = deque(maxlen=WINDOW_SIZE)
window_y = deque(maxlen=WINDOW_SIZE)

# ----------------- RUN EXPERIMENT -----------------
if run_button:

    st.subheader("📡 Streaming Evaluation")

    progress = st.progress(0)
    status = st.empty()

    batches = list(stream_batches(X_test, y_test, BATCH_SIZE))
    total_batches = len(batches)

    for i, (X_batch, y_batch) in enumerate(batches):

        # Inject drift
        if i >= DRIFT_START:
            if enable_covariate:
                X_batch = apply_covariate_drift(X_batch, drift_strength=0.4)
            if enable_concept:
                y_batch = apply_concept_drift(y_batch, drift_ratio=0.4)

        # Evaluate before retrain
        acc_before = evaluate_batch(model, X_batch, y_batch)

        # Retrain using sliding window
        if len(window_X) == WINDOW_SIZE:
            X_retrain = pd.concat(window_X)
            y_retrain = pd.concat(window_y)
            model = retrain_model(X_retrain, y_retrain)

        # Evaluate after retrain
        acc_after = evaluate_batch(model, X_batch, y_batch)

        # Drift detection
        ref_batch = X_test.iloc[:BATCH_SIZE]
        ks = ks_drift(ref_batch, X_batch)
        psi = psi_drift(ref_batch, X_batch)
        err = error_rate_drift(acc_before, BASELINE_ACCURACY)

        # Store results
        batch_acc.append(acc_before)
        before_retrain.append(acc_before)
        after_retrain.append(acc_after)
        ks_flags.append(int(ks))
        psi_flags.append(int(psi))
        error_flags.append(int(err))

        window_X.append(X_batch)
        window_y.append(y_batch)

        status.info(
            f"Batch {i} | "
            f"Acc: {acc_before:.3f} | "
            f"KS: {ks} | PSI: {psi} | Error Drift: {err}"
        )

        progress.progress((i + 1) / total_batches)

    st.success("✅ Experiment Completed")

    # ----------------- SAVE RESULTS -----------------
    os.makedirs("results", exist_ok=True)

    df = pd.DataFrame({
        "batch": range(len(batch_acc)),
        "accuracy": batch_acc,
        "before_retrain": before_retrain,
        "after_retrain": after_retrain,
        "ks_drift": ks_flags,
        "psi_drift": psi_flags,
        "error_drift": error_flags
    })

    df.to_csv("results/drift_results.csv", index=False)

    # ----------------- VISUALIZATION -----------------
    st.subheader("📈 Accuracy Trend")

    fig, ax = plt.subplots()
    ax.plot(df["batch"], df["before_retrain"], label="Before Retrain", marker="o")
    ax.plot(df["batch"], df["after_retrain"], label="After Retrain", marker="s")
    ax.axvline(DRIFT_START, linestyle="--", color="red", label="Drift Injected")
    ax.set_xlabel("Batch")
    ax.set_ylabel("Accuracy")
    ax.legend()
    st.pyplot(fig)

    st.subheader("🚨 Drift Detection Timeline")

    fig2, ax2 = plt.subplots()
    ax2.plot(df["batch"], df["ks_drift"], label="KS Drift", marker="o")
    ax2.plot(df["batch"], df["psi_drift"], label="PSI Drift", marker="s")
    ax2.plot(df["batch"], df["error_drift"], label="Error Drift", marker="^")
    ax2.set_xlabel("Batch")
    ax2.set_ylabel("Drift Detected (0/1)")
    ax2.legend()
    st.pyplot(fig2)

    st.subheader("📄 Final Results Table")
    st.dataframe(df)
