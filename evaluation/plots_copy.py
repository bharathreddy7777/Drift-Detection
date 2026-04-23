import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np


# ----- Color Palette -----
COLORS = {
    "primary": "#4f46e5",       # Indigo
    "secondary": "#7c3aed",     # Violet
    "accent": "#0891b2",        # Cyan
    "success": "#059669",       # Emerald
    "warning": "#d97706",       # Amber
    "danger": "#dc2626",        # Red
    "info": "#2563eb",          # Blue
    "pink": "#db2777",          # Pink
    "bg_dark": "#ffffff",       # White
    "bg_card": "#f8fafc",       # Slate 50
    "text": "#1e293b",          # Slate 800 (black text)
    "text_muted": "#64748b",    # Slate 500
    "grid": "#e2e8f0",          # Slate 200 (light grid)
}

LAYOUT_TEMPLATE = dict(
    paper_bgcolor=COLORS["bg_dark"],
    plot_bgcolor=COLORS["bg_card"],
    font=dict(family="Inter, sans-serif", color=COLORS["text"], size=13),
    margin=dict(l=60, r=30, t=60, b=50),
    legend=dict(
        bgcolor="rgba(255,255,255,0.9)",
        bordercolor=COLORS["grid"],
        borderwidth=1,
        font=dict(size=12),
    ),
)

AXIS_GRID = dict(gridcolor=COLORS["grid"], zerolinecolor=COLORS["grid"])


def create_accuracy_figure(results_df, drift_start_batch=None):
    """
    Creates a live-updating accuracy trend chart with before/after retrain comparison.
    """
    fig = go.Figure()

    # Before retrain line
    fig.add_trace(go.Scatter(
        x=results_df["batch"],
        y=results_df["acc_before"],
        name="Before Retrain",
        mode="lines+markers",
        line=dict(color=COLORS["danger"], width=3),
        marker=dict(size=8, symbol="circle",
                    line=dict(width=2, color="white")),
        fill="tozeroy",
        fillcolor="rgba(239,68,68,0.08)",
    ))

    # After retrain line
    fig.add_trace(go.Scatter(
        x=results_df["batch"],
        y=results_df["acc_after"],
        name="After Retrain",
        mode="lines+markers",
        line=dict(color=COLORS["success"], width=3),
        marker=dict(size=8, symbol="diamond",
                    line=dict(width=2, color="white")),
        fill="tozeroy",
        fillcolor="rgba(16,185,129,0.08)",
    ))

    # Drift injection line
    if drift_start_batch is not None:
        fig.add_vline(
            x=drift_start_batch, line_dash="dash",
            line_color=COLORS["warning"], line_width=2,
            annotation_text="⚡ Drift Injected",
            annotation_position="top right",
            annotation_font=dict(color=COLORS["warning"], size=12),
        )

    fig.update_layout(
        title=dict(text="📈 Accuracy Trend — Before vs After Retraining",
                   font=dict(size=18)),
        xaxis_title="Batch Index",
        yaxis_title="Accuracy",
        hovermode="x unified",
        **LAYOUT_TEMPLATE,
    )
    fig.update_xaxes(**AXIS_GRID)
    fig.update_yaxes(range=[0, 1.05], tickformat=".0%", **AXIS_GRID)

    return fig


def create_metrics_figure(results_df, drift_start_batch=None):
    """
    Creates a multi-metric line chart showing accuracy, precision, recall, F1.
    """
    fig = go.Figure()

    metric_config = [
        ("accuracy", "Accuracy", COLORS["primary"], "circle"),
        ("precision", "Precision", COLORS["accent"], "square"),
        ("recall", "Recall", COLORS["pink"], "triangle-up"),
        ("f1_score", "F1 Score", COLORS["warning"], "star"),
    ]

    for col, name, color, symbol in metric_config:
        if col in results_df.columns:
            fig.add_trace(go.Scatter(
                x=results_df["batch"],
                y=results_df[col],
                name=name,
                mode="lines+markers",
                line=dict(color=color, width=2.5),
                marker=dict(size=7, symbol=symbol,
                            line=dict(width=1.5, color="white")),
            ))

    if drift_start_batch is not None:
        fig.add_vline(
            x=drift_start_batch, line_dash="dash",
            line_color=COLORS["warning"], line_width=2,
            annotation_text="⚡ Drift Injected",
            annotation_position="top right",
            annotation_font=dict(color=COLORS["warning"], size=12),
        )

    fig.update_layout(
        title=dict(text="📊 All Metrics Over Batches", font=dict(size=18)),
        xaxis_title="Batch Index",
        yaxis_title="Score",
        hovermode="x unified",
        **LAYOUT_TEMPLATE,
    )
    fig.update_xaxes(**AXIS_GRID)
    fig.update_yaxes(range=[0, 1.05], tickformat=".0%", **AXIS_GRID)

    return fig


def create_drift_timeline_figure(results_df, drift_start_batch=None):
    """
    Creates a stacked-area style drift detection timeline.
    """
    fig = go.Figure()

    drift_traces = [
        ("ks_drift", "KS Drift", COLORS["accent"]),
        ("psi_drift", "PSI Drift", COLORS["pink"]),
        ("error_drift", "Error Rate Drift", COLORS["danger"]),
    ]

    def hex_to_rgba(hex_color, alpha=0.15):
        hex_color = hex_color.lstrip("#")
        r, g, b = int(hex_color[0:2], 16), int(hex_color[2:4], 16), int(hex_color[4:6], 16)
        return f"rgba({r},{g},{b},{alpha})"

    for col, name, color in drift_traces:
        if col in results_df.columns:
            fig.add_trace(go.Scatter(
                x=results_df["batch"],
                y=results_df[col],
                name=name,
                mode="lines+markers",
                line=dict(color=color, width=3, shape="hv"),
                marker=dict(size=9, symbol="circle",
                            line=dict(width=2, color="white")),
                fill="tozeroy",
                fillcolor=hex_to_rgba(color, 0.15),
            ))

    if drift_start_batch is not None:
        fig.add_vline(
            x=drift_start_batch, line_dash="dash",
            line_color=COLORS["warning"], line_width=2,
        )

    fig.update_layout(
        title=dict(text="🚨 Drift Detection Timeline", font=dict(size=18)),
        xaxis_title="Batch Index",
        yaxis_title="Drift Detected",
        hovermode="x unified",
        **LAYOUT_TEMPLATE,
    )
    fig.update_xaxes(**AXIS_GRID)
    fig.update_yaxes(range=[-0.15, 1.35], tickvals=[0, 1],
                     ticktext=["No Drift", "Drift Detected"], **AXIS_GRID)

    return fig


def create_drift_heatmap(results_df):
    """
    Creates a heatmap of drift detection across batches and detector types.
    """
    drift_cols = ["ks_drift", "psi_drift", "error_drift"]
    available = [c for c in drift_cols if c in results_df.columns]

    if not available:
        return go.Figure()

    z_data = results_df[available].values.T
    labels = ["KS Test", "PSI", "Error Rate"][:len(available)]

    fig = go.Figure(data=go.Heatmap(
        z=z_data,
        x=[f"Batch {i}" for i in results_df["batch"]],
        y=labels,
        colorscale=[
            [0, COLORS["bg_card"]],
            [0.5, COLORS["warning"]],
            [1, COLORS["danger"]],
        ],
        showscale=False,
        hovertemplate="<b>%{y}</b><br>Batch: %{x}<br>Detected: %{z}<extra></extra>",
    ))

    fig.update_layout(
        title=dict(text="🗺️ Drift Detection Heatmap", font=dict(size=18)),
        xaxis_title="Batch",
        yaxis_title="Detector",
        **LAYOUT_TEMPLATE,
    )
    fig.update_xaxes(**AXIS_GRID)
    fig.update_yaxes(**AXIS_GRID)

    return fig


def create_psi_scores_figure(psi_history, threshold=0.2):
    """
    Creates a bar chart showing max PSI score per batch.
    """
    fig = go.Figure()

    batches = list(range(len(psi_history)))
    colors = [COLORS["danger"] if v > threshold else COLORS["success"]
              for v in psi_history]

    fig.add_trace(go.Bar(
        x=batches,
        y=psi_history,
        marker=dict(
            color=colors,
            line=dict(width=1, color="white"),
            opacity=0.85,
        ),
        name="Max PSI",
        hovertemplate="Batch %{x}<br>PSI: %{y:.4f}<extra></extra>",
    ))

    fig.add_hline(
        y=threshold, line_dash="dash",
        line_color=COLORS["warning"], line_width=2,
        annotation_text=f"Threshold ({threshold})",
        annotation_position="top right",
        annotation_font=dict(color=COLORS["warning"]),
    )

    fig.update_layout(
        title=dict(text="📊 PSI Scores Per Batch", font=dict(size=18)),
        xaxis_title="Batch Index",
        yaxis_title="PSI Score",
        **LAYOUT_TEMPLATE,
    )
    fig.update_xaxes(**AXIS_GRID)
    fig.update_yaxes(**AXIS_GRID)

    return fig


def create_retrain_improvement_figure(results_df):
    """
    Creates a waterfall/bar showing accuracy improvement from retraining.
    """
    if "acc_before" not in results_df.columns or "acc_after" not in results_df.columns:
        return go.Figure()

    improvement = results_df["acc_after"] - results_df["acc_before"]

    colors = [COLORS["success"] if v >= 0 else COLORS["danger"] for v in improvement]

    fig = go.Figure()

    fig.add_trace(go.Bar(
        x=results_df["batch"],
        y=improvement,
        marker=dict(color=colors, opacity=0.85,
                    line=dict(width=1, color="white")),
        hovertemplate="Batch %{x}<br>Improvement: %{y:+.4f}<extra></extra>",
    ))

    fig.add_hline(y=0, line_color=COLORS["text_muted"], line_width=1)

    fig.update_layout(
        title=dict(text="🔄 Retraining Accuracy Improvement", font=dict(size=18)),
        xaxis_title="Batch Index",
        yaxis_title="Accuracy Change",
        **LAYOUT_TEMPLATE,
    )
    fig.update_xaxes(**AXIS_GRID)
    fig.update_yaxes(tickformat="+.2%", **AXIS_GRID)

    return fig


def create_combined_dashboard(results_df, psi_history, drift_start_batch=None):
    """
    Creates a 2x2 subplot dashboard with all key visualizations.
    """
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=(
            "Accuracy Trend",
            "All Metrics",
            "Drift Detection",
            "Retraining Impact",
        ),
        vertical_spacing=0.12,
        horizontal_spacing=0.08,
    )

    # Accuracy trend (top-left)
    fig.add_trace(go.Scatter(
        x=results_df["batch"], y=results_df["acc_before"],
        name="Before Retrain", mode="lines+markers",
        line=dict(color=COLORS["danger"], width=2),
        marker=dict(size=6), legendgroup="acc",
    ), row=1, col=1)

    fig.add_trace(go.Scatter(
        x=results_df["batch"], y=results_df["acc_after"],
        name="After Retrain", mode="lines+markers",
        line=dict(color=COLORS["success"], width=2),
        marker=dict(size=6), legendgroup="acc",
    ), row=1, col=1)

    # All metrics (top-right)
    for col, name, color in [
        ("accuracy", "Accuracy", COLORS["primary"]),
        ("precision", "Precision", COLORS["accent"]),
        ("recall", "Recall", COLORS["pink"]),
        ("f1_score", "F1", COLORS["warning"]),
    ]:
        if col in results_df.columns:
            fig.add_trace(go.Scatter(
                x=results_df["batch"], y=results_df[col],
                name=name, mode="lines",
                line=dict(color=color, width=2), legendgroup="metrics",
            ), row=1, col=2)

    # Drift detection (bottom-left)
    for col, name, color in [
        ("ks_drift", "KS", COLORS["accent"]),
        ("psi_drift", "PSI", COLORS["pink"]),
        ("error_drift", "Error", COLORS["danger"]),
    ]:
        if col in results_df.columns:
            fig.add_trace(go.Scatter(
                x=results_df["batch"], y=results_df[col],
                name=name, mode="lines+markers",
                line=dict(color=color, width=2, shape="hv"),
                marker=dict(size=5), legendgroup="drift",
            ), row=2, col=1)

    # Retraining improvement (bottom-right)
    if "acc_before" in results_df.columns and "acc_after" in results_df.columns:
        improvement = results_df["acc_after"] - results_df["acc_before"]
        colors = [COLORS["success"] if v >= 0 else COLORS["danger"] for v in improvement]
        fig.add_trace(go.Bar(
            x=results_df["batch"], y=improvement,
            marker=dict(color=colors, opacity=0.85),
            name="Improvement", legendgroup="impact", showlegend=False,
        ), row=2, col=2)

    fig.update_layout(
        height=700,
        title=dict(text="📋 Complete Drift Analysis Dashboard", font=dict(size=20)),
        showlegend=True,
        **LAYOUT_TEMPLATE,
    )
    fig.update_xaxes(**AXIS_GRID)
    fig.update_yaxes(**AXIS_GRID)

    return fig
