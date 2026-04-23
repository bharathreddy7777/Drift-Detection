def print_metrics(metrics, title="Model Performance"):
    """
    Pretty-prints a dictionary of evaluation metrics.
    """
    print(f"\n{'='*40}")
    print(f"  {title}")
    print(f"{'='*40}")
    for key, value in metrics.items():
        print(f"  {key.capitalize():<12}: {value:.4f}")
    print(f"{'='*40}\n")
