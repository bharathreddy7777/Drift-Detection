import streamlit as st
import pandas as pd
import numpy as np
import time
import os
from collections import deque

from training.train_baseline_copy import load_and_prepare_dataset, train_baseline_model
from drift.data_stream_copy import stream_batches
from drift.covariate_drift_copy import apply_covariate_drift
from drift.concept_drift_copy import apply_concept_drift
from drift.drift_detectors_copy import ks_drift, psi_drift, error_rate_drift
from evaluation.batch_evaluation_copy import evaluate_batch
from training.retrain_model_copy import retrain_model
from evaluation.plots_copy import (
    create_accuracy_figure,
    create_metrics_figure,
    create_drift_timeline_figure,
    create_drift_heatmap,
    create_psi_scores_figure,
    create_retrain_improvement_figure,
    create_combined_dashboard,
    COLORS,
)

# =====================================================================
#                        PAGE CONFIGURATION
# =====================================================================
st.set_page_config(
    page_title="ML Drift Monitoring System",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# =====================================================================
#                         CUSTOM STYLING
# =====================================================================
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

    /* Global */
    .stApp {
        background: #ffffff;
        font-family: 'Inter', sans-serif;
        color: #1e293b;
    }

    /* Header */
    .main-header {
        text-align: center;
        padding: 2rem 0 1rem;
    }
    .main-header h1 {
        background: linear-gradient(135deg, #4f46e5, #0891b2, #7c3aed);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-size: 2.8rem;
        font-weight: 700;
        letter-spacing: -0.02em;
        margin-bottom: 0.3rem;
    }
    .main-header p {
        color: #64748b;
        font-size: 1.1rem;
        font-weight: 300;
    }

    /* Metric cards */
    .metric-card {
        background: #f8fafc;
        border: 1px solid #e2e8f0;
        border-radius: 16px;
        padding: 1.5rem;
        text-align: center;
        box-shadow: 0 4px 16px rgba(0,0,0,0.06);
        transition: transform 0.2s ease, box-shadow 0.2s ease;
    }
    .metric-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 24px rgba(79,70,229,0.12);
    }
    .metric-value {
        font-size: 2.4rem;
        font-weight: 700;
        background: linear-gradient(135deg, #4f46e5, #0891b2);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    .metric-label {
        color: #64748b;
        font-size: 0.85rem;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        margin-top: 0.3rem;
    }

    /* Status badges */
    .badge-safe {
        background: linear-gradient(135deg, #059669, #10b981);
        color: white;
        padding: 0.3rem 1rem;
        border-radius: 20px;
        font-weight: 600;
        font-size: 0.85rem;
        display: inline-block;
    }
    .badge-drift {
        background: linear-gradient(135deg, #dc2626, #ef4444);
        color: white;
        padding: 0.3rem 1rem;
        border-radius: 20px;
        font-weight: 600;
        font-size: 0.85rem;
        display: inline-block;
        animation: pulse 2s infinite;
    }
    @keyframes pulse {
        0%, 100% { opacity: 1; }
        50% { opacity: 0.7; }
    }

    /* Section headers */
    .section-header {
        color: #1e293b;
        font-size: 1.4rem;
        font-weight: 600;
        padding: 1rem 0 0.5rem;
        border-bottom: 2px solid #4f46e5;
        margin-bottom: 1rem;
    }

    /* Upload area */
    .upload-area {
        border: 2px dashed #4f46e5;
        border-radius: 16px;
        padding: 2rem;
        text-align: center;
        background: #f8fafc;
        margin: 1rem 0;
    }

    /* Sidebar styling */
    section[data-testid="stSidebar"] {
        background: #f8fafc;
        border-right: 1px solid #e2e8f0;
    }

    /* Batch log */
    .batch-log {
        background: #f8fafc;
        border: 1px solid #e2e8f0;
        border-radius: 8px;
        padding: 0.6rem 1rem;
        margin: 0.3rem 0;
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.82rem;
        color: #1e293b;
    }

    /* Hide default streamlit elements */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}

    /* Divider */
    .gradient-divider {
        height: 3px;
        background: linear-gradient(90deg, transparent, #4f46e5, #0891b2, #7c3aed, transparent);
        border: none;
        margin: 2rem 0;
        border-radius: 2px;
    }
</style>
""", unsafe_allow_html=True)


# =====================================================================
#                            HEADER
# =====================================================================
st.markdown("""
<div class="main-header">
    <h1>📊 ML Drift Monitoring System</h1>
    <p>Upload your dataset • Configure batches • Watch live drift detection & adaptive retraining</p>
</div>
<div class="gradient-divider"></div>
""", unsafe_allow_html=True)


# =====================================================================
#                       SIDEBAR — CONTROLS
# =====================================================================
with st.sidebar:
    st.markdown("## 📁 Upload Dataset")
    uploaded_file = st.file_uploader(
        "Drop your CSV file here",
        type=["csv"],
        help="Upload any CSV dataset. The system will auto-detect columns.",
    )

    st.markdown("---")
    st.markdown("## ⚙️ Experiment Controls")

    target_column = None
    n_batches = 10
    drift_start = 3
    drift_strength = 0.4
    concept_ratio = 0.4
    window_size = 5
    enable_covariate = True
    enable_concept = True

    if uploaded_file is not None:
        df_uploaded = pd.read_csv(uploaded_file)
        st.session_state["df_uploaded"] = df_uploaded

        all_columns = df_uploaded.columns.tolist()

        target_column = st.selectbox(
            "🎯 Select Target Column",
            options=all_columns,
            index=len(all_columns) - 1,
            help="The column the model should predict.",
        )

        st.markdown("---")

        n_batches = st.slider(
            "📦 Number of Batches",
            min_value=3, max_value=50, value=10,
            help="How many batches to split the test data into.",
        )

        drift_start = st.slider(
            "⚡ Drift Starts at Batch",
            min_value=1, max_value=n_batches - 1,
            value=max(1, n_batches // 3),
            help="Batch index where drift injection begins.",
        )

        st.markdown("---")
        st.markdown("### 🌊 Drift Settings")

        enable_covariate = st.checkbox("Enable Covariate Drift", True)
        if enable_covariate:
            drift_strength = st.slider(
                "Covariate Drift Strength", 0.05, 1.0, 0.4, 0.05,
            )

        enable_concept = st.checkbox("Enable Concept Drift", True)
        if enable_concept:
            concept_ratio = st.slider(
                "Label Flip Ratio", 0.05, 0.8, 0.4, 0.05,
            )

        st.markdown("---")
        st.markdown("### 🔄 Retraining")

        window_size = st.slider(
            "Sliding Window Size", 2, min(15, n_batches), min(5, n_batches - 1),
            help="Number of past batches used for retraining.",
        )

        st.markdown("---")
        run_button = st.button("▶️  Run Experiment", use_container_width=True, type="primary")
    else:
        run_button = False


# =====================================================================
#                     MAIN CONTENT AREA
# =====================================================================

if uploaded_file is None:
    # ---------- Landing / Upload state ----------
    st.markdown("""
    <div class="upload-area">
        <h3 style="color:#1e293b;">👈 Upload a CSV dataset to get started</h3>
        <p style="color:#64748b;">
            The system will let you pick the target column, choose how many batches
            to split into, and then run a full drift detection experiment with
            <strong>live visualizations</strong>.
        </p>
    </div>
    """, unsafe_allow_html=True)

    # Show feature overview
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("""
        <div class="metric-card">
            <div class="metric-value">📤</div>
            <div class="metric-label">Upload Any CSV</div>
            <p style="color:#64748b; font-size:0.85rem; margin-top:0.5rem;">
                Works with any classification dataset
            </p>
        </div>
        """, unsafe_allow_html=True)
    with col2:
        st.markdown("""
        <div class="metric-card">
            <div class="metric-value">🔬</div>
            <div class="metric-label">3 Drift Detectors</div>
            <p style="color:#64748b; font-size:0.85rem; margin-top:0.5rem;">
                KS Test, PSI, and Error Rate analysis
            </p>
        </div>
        """, unsafe_allow_html=True)
    with col3:
        st.markdown("""
        <div class="metric-card">
            <div class="metric-value">📊</div>
            <div class="metric-label">Live Graphs</div>
            <p style="color:#64748b; font-size:0.85rem; margin-top:0.5rem;">
                Real-time Plotly charts updating per batch
            </p>
        </div>
        """, unsafe_allow_html=True)

    st.stop()

# ---------- Dataset uploaded ----------
df_uploaded = st.session_state.get("df_uploaded")

if df_uploaded is not None and target_column is not None and not run_button:
    st.markdown('<div class="section-header">📋 Dataset Preview</div>', unsafe_allow_html=True)

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value">{df_uploaded.shape[0]:,}</div>
            <div class="metric-label">Rows</div>
        </div>""", unsafe_allow_html=True)
    with c2:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value">{df_uploaded.shape[1]}</div>
            <div class="metric-label">Columns</div>
        </div>""", unsafe_allow_html=True)
    with c3:
        n_classes = df_uploaded[target_column].nunique()
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value">{n_classes}</div>
            <div class="metric-label">Target Classes</div>
        </div>""", unsafe_allow_html=True)
    with c4:
        missing = df_uploaded.isnull().sum().sum()
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value">{missing}</div>
            <div class="metric-label">Missing Values</div>
        </div>""", unsafe_allow_html=True)

    st.dataframe(df_uploaded.head(10), use_container_width=True)

    st.info("👈 Configure experiment settings in the sidebar and click **▶️ Run Experiment**")


# =====================================================================
#                      RUN EXPERIMENT
# =====================================================================
if run_button and df_uploaded is not None and target_column is not None:

    st.markdown('<div class="gradient-divider"></div>', unsafe_allow_html=True)
    st.markdown('<div class="section-header">🔧 Training Baseline Model</div>', unsafe_allow_html=True)

    with st.spinner("Preparing dataset & training baseline Random Forest..."):
        X_train, X_test, y_train, y_test = load_and_prepare_dataset(df_uploaded, target_column)
        model, baseline_metrics = train_baseline_model(X_train, X_test, y_train, y_test)

    baseline_accuracy = baseline_metrics["accuracy"]

    # Show baseline metrics
    bcols = st.columns(4)
    metric_icons = {"accuracy": "🎯", "precision": "✅", "recall": "🔁", "f1_score": "⚖️"}
    for col_widget, (metric_name, metric_val) in zip(bcols, baseline_metrics.items()):
        with col_widget:
            icon = metric_icons.get(metric_name, "📊")
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-value">{metric_val:.2%}</div>
                <div class="metric-label">{icon} Baseline {metric_name.replace('_',' ').title()}</div>
            </div>""", unsafe_allow_html=True)

    st.markdown('<div class="gradient-divider"></div>', unsafe_allow_html=True)

    # ---- Streaming setup ----
    st.markdown('<div class="section-header">📡 Live Batch Processing</div>', unsafe_allow_html=True)

    progress_bar = st.progress(0)
    status_text = st.empty()

    # Live chart placeholders
    chart_col1, chart_col2 = st.columns(2)
    with chart_col1:
        acc_chart_placeholder = st.empty()
    with chart_col2:
        metrics_chart_placeholder = st.empty()

    drift_col1, drift_col2 = st.columns(2)
    with drift_col1:
        drift_chart_placeholder = st.empty()
    with drift_col2:
        psi_chart_placeholder = st.empty()

    batch_log_placeholder = st.empty()

    # Data storage
    results = []
    psi_history = []
    window_X = deque(maxlen=window_size)
    window_y = deque(maxlen=window_size)

    reference_batch = X_test.iloc[:max(1, len(X_test) // n_batches)]
    batches = list(stream_batches(X_test, y_test, n_batches))
    total_batches = len(batches)

    log_lines = []

    # ---- Process each batch ----
    for i, (X_batch, y_batch) in enumerate(batches):

        # Inject drift
        if i >= drift_start:
            if enable_covariate:
                X_batch = apply_covariate_drift(X_batch, drift_strength=drift_strength)
            if enable_concept:
                y_batch = apply_concept_drift(y_batch, drift_ratio=concept_ratio)

        # Evaluate before retrain
        metrics_before = evaluate_batch(model, X_batch, y_batch)

        # Sliding window retrain
        if len(window_X) == window_size:
            X_retrain = pd.concat(list(window_X))
            y_retrain = pd.concat(list(window_y))
            model = retrain_model(X_retrain, y_retrain)

        # Evaluate after retrain
        metrics_after = evaluate_batch(model, X_batch, y_batch)

        # Drift detection
        ks_detected, ks_pval, ks_features = ks_drift(reference_batch, X_batch)
        psi_detected, psi_max, psi_scores = psi_drift(reference_batch, X_batch)
        err_detected, err_drop = error_rate_drift(metrics_before["accuracy"], baseline_accuracy)

        psi_history.append(psi_max)

        results.append({
            "batch": i,
            "accuracy": metrics_before["accuracy"],
            "precision": metrics_before["precision"],
            "recall": metrics_before["recall"],
            "f1_score": metrics_before["f1_score"],
            "acc_before": metrics_before["accuracy"],
            "acc_after": metrics_after["accuracy"],
            "ks_drift": int(ks_detected),
            "psi_drift": int(psi_detected),
            "error_drift": int(err_detected),
            "ks_pval": ks_pval,
            "psi_max": psi_max,
            "err_drop": err_drop,
        })

        window_X.append(X_batch)
        window_y.append(y_batch)

        # ---- Live UI updates ----
        progress_bar.progress((i + 1) / total_batches)

        any_drift = ks_detected or psi_detected or err_detected
        drift_badge = "🔴 DRIFT" if any_drift else "🟢 STABLE"
        status_text.markdown(
            f"**Batch {i}/{total_batches - 1}** — "
            f"Acc: `{metrics_before['accuracy']:.4f}` → `{metrics_after['accuracy']:.4f}` — "
            f"{drift_badge}"
        )

        results_df = pd.DataFrame(results)

        # Update live charts
        with chart_col1:
            acc_chart_placeholder.plotly_chart(
                create_accuracy_figure(results_df, drift_start),
                use_container_width=True, key=f"acc_{i}",
            )
        with chart_col2:
            metrics_chart_placeholder.plotly_chart(
                create_metrics_figure(results_df, drift_start),
                use_container_width=True, key=f"met_{i}",
            )
        with drift_col1:
            drift_chart_placeholder.plotly_chart(
                create_drift_timeline_figure(results_df, drift_start),
                use_container_width=True, key=f"drt_{i}",
            )
        with drift_col2:
            psi_chart_placeholder.plotly_chart(
                create_psi_scores_figure(psi_history),
                use_container_width=True, key=f"psi_{i}",
            )

        # Batch log
        emoji = "⚠️" if any_drift else "✅"
        log_lines.append(
            f"{emoji} Batch {i:2d} | "
            f"Acc: {metrics_before['accuracy']:.4f} → {metrics_after['accuracy']:.4f} | "
            f"KS:{ks_detected}  PSI:{psi_detected}  ERR:{err_detected}"
        )
        batch_log_placeholder.code("\n".join(log_lines[-8:]), language="text")

        time.sleep(0.15)  # small delay for visual effect

    # ---- Experiment Complete ----
    st.markdown('<div class="gradient-divider"></div>', unsafe_allow_html=True)
    st.success("✅ **Experiment Completed!**")

    # =====================================================================
    #                   FINAL RESULTS DASHBOARD
    # =====================================================================
    results_df = pd.DataFrame(results)

    st.markdown('<div class="section-header">📊 Final Analysis Dashboard</div>', unsafe_allow_html=True)

    # Summary metrics
    s1, s2, s3, s4 = st.columns(4)
    avg_acc = results_df["accuracy"].mean()
    drift_batches = results_df["ks_drift"].sum() + results_df["psi_drift"].sum() + results_df["error_drift"].sum()
    max_drop = (results_df["acc_before"] - results_df["acc_after"]).min()
    avg_improvement = (results_df["acc_after"] - results_df["acc_before"]).mean()

    with s1:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value">{avg_acc:.1%}</div>
            <div class="metric-label">Average Accuracy</div>
        </div>""", unsafe_allow_html=True)
    with s2:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value">{int(drift_batches)}</div>
            <div class="metric-label">Total Drift Alerts</div>
        </div>""", unsafe_allow_html=True)
    with s3:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value">{avg_improvement:+.2%}</div>
            <div class="metric-label">Avg Retrain Boost</div>
        </div>""", unsafe_allow_html=True)
    with s4:
        drift_pct = (results_df[["ks_drift", "psi_drift", "error_drift"]].any(axis=1).sum() / len(results_df)) * 100
        badge_class = "badge-drift" if drift_pct > 50 else "badge-safe"
        badge_text = "HIGH DRIFT" if drift_pct > 50 else "STABLE"
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value">{drift_pct:.0f}%</div>
            <div class="metric-label">Batches with Drift</div>
            <div style="margin-top:0.5rem;"><span class="{badge_class}">{badge_text}</span></div>
        </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # Heatmap & Improvement
    hm_col, imp_col = st.columns(2)
    with hm_col:
        st.plotly_chart(create_drift_heatmap(results_df), use_container_width=True)
    with imp_col:
        st.plotly_chart(create_retrain_improvement_figure(results_df), use_container_width=True)

    # Combined dashboard
    st.plotly_chart(
        create_combined_dashboard(results_df, psi_history, drift_start),
        use_container_width=True,
    )

    # ---- Save & Download ----
    st.markdown('<div class="section-header">💾 Export Results</div>', unsafe_allow_html=True)

    os.makedirs("results", exist_ok=True)
    results_df.to_csv("results/drift_results.csv", index=False)

    col_dl1, col_dl2 = st.columns(2)
    with col_dl1:
        csv_data = results_df.to_csv(index=False).encode("utf-8")
        st.download_button(
            "📥 Download Results CSV",
            csv_data, "drift_results.csv", "text/csv",
            use_container_width=True,
        )
    with col_dl2:
        st.dataframe(results_df, use_container_width=True)
