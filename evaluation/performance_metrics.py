def print_metrics(metrics, title="Model Performance"):
    print(f"\n{title}")
    print("-" * 30)
    for key, value in metrics.items():
        print(f"{key.capitalize():<10}: {value:.4f}")
