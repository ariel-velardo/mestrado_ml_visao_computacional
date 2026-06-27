from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from liverct.models.train_cnn import CNNTrainingConfig, run_cnn_training


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train a simple CNN baseline on train split and validate on val split.",
    )
    parser.add_argument(
        "--split-csv",
        type=Path,
        default=PROJECT_ROOT / "data" / "interim" / "split_slices.csv",
    )
    parser.add_argument(
        "--reports-dir",
        type=Path,
        default=PROJECT_ROOT / "reports" / "tables",
    )
    parser.add_argument(
        "--checkpoint-dir",
        type=Path,
        default=PROJECT_ROOT / "models" / "checkpoints",
    )
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--dropout", type=float, default=0.25)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-workers", type=int, default=0)
    return parser.parse_args()


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
    ]

    print("\n=== SIMPLE CNN VALIDATION METRICS ===")
    print(metrics_df[selected_columns].to_string(index=False))


def main() -> None:
    args = parse_args()

    config = CNNTrainingConfig(
        split_csv_path=args.split_csv,
        reports_dir=args.reports_dir,
        checkpoint_dir=args.checkpoint_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        dropout=args.dropout,
        patience=args.patience,
        seed=args.seed,
        num_workers=args.num_workers,
    )

    result = run_cnn_training(config)
    print_metrics(result["val_metrics_df"])

    print("\nArquivos gerados:")
    for path in result["output_paths"]:
        print(f"  - {path}")
    print(f"  - {result['checkpoint_path']}")


if __name__ == "__main__":
    main()
