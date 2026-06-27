from __future__ import annotations

from pathlib import Path

import pandas as pd

from liverct.models.statistical_baseline import run_statistical_baseline


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def print_metrics(metrics_df: pd.DataFrame) -> None:
    """
    Imprime métricas principais no terminal.
    """
    selected_columns = [
        "model",
        "level",
        "split",
        "n",
        "balanced_accuracy",
        "recall_sensitivity",
        "specificity",
        "f1",
        "roc_auc",
        "average_precision",
    ]

    print("\n=== STATISTICAL BASELINE METRICS ===")
    print(metrics_df[selected_columns].to_string(index=False))


def main() -> None:
    interim_dir = PROJECT_ROOT / "data" / "interim"
    output_dir = PROJECT_ROOT / "reports" / "tables"

    audit_path = interim_dir / "image_quality_audit.csv"

    if not audit_path.exists():
        raise FileNotFoundError(
            f"Arquivo não encontrado: {audit_path}\n"
            "Execute antes: python scripts/audit_images.py"
        )

    audit_df = pd.read_csv(audit_path)

    result = run_statistical_baseline(
        audit_df=audit_df,
        output_dir=output_dir,
    )

    metrics_df = result["metrics_df"]
    print_metrics(metrics_df)

    print("\nArquivos gerados:")
    print(f"  - {output_dir / 'statistical_baseline_metrics.csv'}")
    print(f"  - {output_dir / 'statistical_baseline_summary.json'}")
    print(f"  - {output_dir / 'dummy_most_frequent_slice_predictions.csv'}")
    print(f"  - {output_dir / 'dummy_most_frequent_group_predictions.csv'}")
    print(f"  - {output_dir / 'logistic_regression_slice_predictions.csv'}")
    print(f"  - {output_dir / 'logistic_regression_group_predictions.csv'}")


if __name__ == "__main__":
    main()
