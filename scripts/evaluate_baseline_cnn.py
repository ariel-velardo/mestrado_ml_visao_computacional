from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd
import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from liverct.models.cnn_dataset import (  # noqa: E402
    build_dataloader,
    load_split_slices,
    select_split,
    validate_group_split_integrity,
)
from liverct.models.train_cnn import (  # noqa: E402
    evaluate_prediction_tables,
    get_device,
    load_cnn_checkpoint,
    predict_with_model,
)


DOCUMENTED_STATISTICAL_BASELINE_TEST = [
    {
        "reference_model": "logistic_regression",
        "level": "slice",
        "split": "test",
        "reference_balanced_accuracy": 0.7340,
        "reference_recall_sensitivity": 0.7833,
        "reference_specificity": 0.6846,
        "reference_f1": 0.7618,
        "reference_roc_auc": 0.8549,
        "reference_average_precision": 0.8768,
    },
    {
        "reference_model": "logistic_regression",
        "level": "group",
        "split": "test",
        "reference_balanced_accuracy": 0.7899,
        "reference_recall_sensitivity": 0.9130,
        "reference_specificity": 0.6667,
        "reference_f1": 0.8750,
        "reference_roc_auc": 0.8333,
        "reference_average_precision": 0.9057,
    },
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate the trained CNN baseline on the held-out test split.",
    )
    parser.add_argument(
        "--split-csv",
        type=Path,
        default=PROJECT_ROOT / "data" / "interim" / "split_slices.csv",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=PROJECT_ROOT / "models" / "checkpoints" / "baseline_cnn_best.pt",
    )
    parser.add_argument(
        "--reports-dir",
        type=Path,
        default=PROJECT_ROOT / "reports" / "tables",
    )
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--image-size", type=int, default=256)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--threshold", type=float, default=0.5)
    return parser.parse_args()


def build_comparison_table(cnn_metrics_df: pd.DataFrame) -> pd.DataFrame:
    """
    Compare CNN test metrics with the documented statistical baseline.
    """
    reference_df = pd.DataFrame(DOCUMENTED_STATISTICAL_BASELINE_TEST)
    metric_columns = [
        "balanced_accuracy",
        "recall_sensitivity",
        "specificity",
        "f1",
        "roc_auc",
        "average_precision",
    ]

    cnn_test_df = cnn_metrics_df[cnn_metrics_df["split"] == "test"].copy()
    cnn_test_df = cnn_test_df[["level", "split", *metric_columns]]
    cnn_test_df = cnn_test_df.rename(
        columns={column: f"cnn_{column}" for column in metric_columns}
    )

    comparison_df = reference_df.merge(
        cnn_test_df,
        on=["level", "split"],
        how="left",
    )

    for column in metric_columns:
        comparison_df[f"delta_{column}"] = (
            comparison_df[f"cnn_{column}"]
            - comparison_df[f"reference_{column}"]
        )

    comparison_df["reference_source"] = "docs/04_baseline_estatistico_controle.md"
    return comparison_df


def print_metrics(metrics_df: pd.DataFrame) -> None:
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
        "tn",
        "fp",
        "fn",
        "tp",
    ]

    print("\n=== SIMPLE CNN TEST METRICS ===")
    print(metrics_df[selected_columns].to_string(index=False))


def main() -> None:
    args = parse_args()
    args.reports_dir.mkdir(parents=True, exist_ok=True)

    all_df = load_split_slices(args.split_csv)
    validate_group_split_integrity(all_df)
    test_df = select_split(all_df, "test")

    device = get_device()
    model, _ = load_cnn_checkpoint(args.checkpoint, device=device)

    test_loader = build_dataloader(
        test_df,
        batch_size=args.batch_size,
        image_size=args.image_size,
        shuffle=False,
        num_workers=args.num_workers,
    )

    with torch.no_grad():
        slice_pred_df = predict_with_model(
            model=model,
            dataloader=test_loader,
            device=device,
            threshold=args.threshold,
        )

    slice_pred_df, group_pred_df, metrics_df = evaluate_prediction_tables(
        slice_pred_df,
        threshold=args.threshold,
    )
    comparison_df = build_comparison_table(metrics_df)

    slice_pred_path = args.reports_dir / "baseline_cnn_test_slice_predictions.csv"
    group_pred_path = args.reports_dir / "baseline_cnn_test_group_predictions.csv"
    metrics_path = args.reports_dir / "baseline_cnn_test_metrics.csv"
    comparison_path = (
        args.reports_dir / "baseline_cnn_vs_statistical_baseline_test.csv"
    )

    slice_pred_df.to_csv(slice_pred_path, index=False, encoding="utf-8")
    group_pred_df.to_csv(group_pred_path, index=False, encoding="utf-8")
    metrics_df.to_csv(metrics_path, index=False, encoding="utf-8")
    comparison_df.to_csv(comparison_path, index=False, encoding="utf-8")

    print_metrics(metrics_df)

    print("\nComparacao com baseline estatistico documentado:")
    print(comparison_df.to_string(index=False))

    print("\nArquivos gerados:")
    print(f"  - {slice_pred_path}")
    print(f"  - {group_pred_path}")
    print(f"  - {metrics_path}")
    print(f"  - {comparison_path}")


if __name__ == "__main__":
    main()
