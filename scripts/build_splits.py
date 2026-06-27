from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from liverct.data.split_dataset import (
    expand_splits_to_slices,
    save_split_outputs,
    split_groups,
    summarize_splits,
    validate_no_split_leakage,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def print_split_summary(summary: dict) -> None:
    """
    Imprime o resumo dos splits.
    """
    print("\n=== SPLIT SUMMARY ===")

    print("\nGrupos por split e classe:")
    for row in summary["groups_by_split_class"]:
        print(f"  - {row['split']} | {row['class_name']}: {row['n_groups']} grupos")

    print("\nSlices por split e classe:")
    for row in summary["slices_by_split_class"]:
        print(f"  - {row['split']} | {row['class_name']}: {row['n_slices']} slices")

    print("\nLeakage entre splits:")
    leakage = summary["leakage"]
    print(f"  - Train-Val leakage: {leakage['train_val_leakage']}")
    print(f"  - Train-Test leakage: {leakage['train_test_leakage']}")
    print(f"  - Val-Test leakage: {leakage['val_test_leakage']}")


def main() -> None:
    input_dir = PROJECT_ROOT / "data" / "interim"
    output_dir = PROJECT_ROOT / "data" / "interim"

    dataset_index_path = input_dir / "dataset_index.csv"
    group_index_path = input_dir / "group_index.csv"

    if not dataset_index_path.exists():
        raise FileNotFoundError(
            f"Arquivo não encontrado: {dataset_index_path}\n"
            "Execute antes: python scripts/build_dataset_index.py"
        )

    if not group_index_path.exists():
        raise FileNotFoundError(
            f"Arquivo não encontrado: {group_index_path}\n"
            "Execute antes: python scripts/build_dataset_index.py"
        )

    slice_df = pd.read_csv(dataset_index_path)
    group_df = pd.read_csv(group_index_path)

    split_group_df = split_groups(
        group_df=group_df,
        train_size=0.70,
        val_size=0.15,
        test_size=0.15,
        seed=42,
    )

    split_slice_df = expand_splits_to_slices(
        slice_df=slice_df,
        split_group_df=split_group_df,
    )

    leakage = validate_no_split_leakage(split_group_df)

    if any(value > 0 for value in leakage.values()):
        raise RuntimeError(
            "Data leakage detectado entre splits. "
            "Investigue os inferred_group_id antes de modelar."
        )

    save_split_outputs(
        split_group_df=split_group_df,
        split_slice_df=split_slice_df,
        output_dir=output_dir,
    )

    summary = summarize_splits(
        split_group_df=split_group_df,
        split_slice_df=split_slice_df,
    )

    with (output_dir / "split_summary.json").open("w", encoding="utf-8") as file:
        json.dump(summary, file, ensure_ascii=False, indent=2)

    print_split_summary(summary)

    print("\nArquivos gerados:")
    print(f"  - {output_dir / 'split_groups.csv'}")
    print(f"  - {output_dir / 'split_slices.csv'}")
    print(f"  - {output_dir / 'train_groups.csv'}")
    print(f"  - {output_dir / 'val_groups.csv'}")
    print(f"  - {output_dir / 'test_groups.csv'}")
    print(f"  - {output_dir / 'train_slices.csv'}")
    print(f"  - {output_dir / 'val_slices.csv'}")
    print(f"  - {output_dir / 'test_slices.csv'}")
    print(f"  - {output_dir / 'split_summary.json'}")


if __name__ == "__main__":
    main()
