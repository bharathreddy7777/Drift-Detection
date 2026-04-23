"""
Standalone plotting script — reads saved results and generates interactive Plotly charts.
Run: python plots/plot_results_copy.py
"""
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from evaluation.plots_copy import (
    COLORS, LAYOUT_TEMPLATE,
    create_accuracy_figure,
    create_drift_timeline_figure,
    create_drift_heatmap,
    create_retrain_improvement_figure,
)


def main():
    csv_path = "results/drift_results.csv"
    if not os.path.exists(csv_path):
        print(f"Results file not found: {csv_path}")
        print("Run the experiment first via app_copy.py or main_copy.py")
        return

    df = pd.read_csv(csv_path)

    print(f"Loaded {len(df)} batch results from {csv_path}")
    print(df.to_string(index=False))

    # --- Figure 1: Accuracy trend ---
    fig1 = create_accuracy_figure(df)
    fig1.show()

    # --- Figure 2: Drift timeline ---
    fig2 = create_drift_timeline_figure(df)
    fig2.show()

    # --- Figure 3: Drift heatmap ---
    fig3 = create_drift_heatmap(df)
    fig3.show()

    # --- Figure 4: Retraining improvement ---
    fig4 = create_retrain_improvement_figure(df)
    fig4.show()


if __name__ == "__main__":
    main()
