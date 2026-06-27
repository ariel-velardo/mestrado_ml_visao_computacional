from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from liverct.data.audit_images import (
    build_image_quality_audit,
    find_exact_duplicates,
    summarize_image_audit,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def print_audit_summary(summary: dict) -> None:
    """
    Imprime resumo da auditoria de qualidade das imagens.
    """
    print("\n=== IMAGE QUALITY AUDIT ===")
    print(f"Total de imagens: {summary['total_images']}")
    print(f"Arquivos ausentes: {summary['missing_files']}")
    print(f"Imagens ilegíveis: {summary['unreadable_images']}")
    print(f"Grupos de duplicatas exatas: {summary['duplicate_hash_groups']}")
    print(f"Arquivos duplicados exatos: {summary['duplicate_files']}")

    print("\nResumo por classe:")
    for row in summary["by_class"]:
        print(
            f"  - {row['class_name']}: "
            f"n_images={row['n_images']}, "
            f"n_groups={row['n_groups']}, "
            f"mean_intensity_avg={row['mean_intensity_avg']}, "
            f"std_intensity_avg={row['std_intensity_avg']}, "
            f"file_size_avg={row['file_size_avg']}"
        )

    if "by_split_class" in summary:
        print("\nResumo por split e classe:")
        for row in summary["by_split_class"]:
            print(
                f"  - {row['split']} | {row['class_name']}: "
                f"n_images={row['n_images']}, "
                f"n_groups={row['n_groups']}, "
                f"mean_intensity_avg={row['mean_intensity_avg']}, "
                f"std_intensity_avg={row['std_intensity_avg']}, "
                f"file_size_avg={row['file_size_avg']}"
            )

    print("\nDimensões encontradas:")
    for row in summary["dimensions"]:
        print(
            f"  - {int(row['width_audit'])}x{int(row['height_audit'])}: "
            f"{row['n_images']} imagens"
        )


def main() -> None:
    interim_dir = PROJECT_ROOT / "data" / "interim"

    split_slices_path = interim_dir / "split_slices.csv"
    dataset_index_path = interim_dir / "dataset_index.csv"

    if split_slices_path.exists():
        input_path = split_slices_path
    elif dataset_index_path.exists():
        input_path = dataset_index_path
    else:
        raise FileNotFoundError(
            "Nenhum índice encontrado. Execute antes:\n"
            "  python scripts/build_dataset_index.py\n"
            "  python scripts/build_splits.py"
        )

    print(f"Usando índice: {input_path}")

    slice_df = pd.read_csv(input_path)

    audit_df = build_image_quality_audit(slice_df)
    duplicates_df = find_exact_duplicates(audit_df)
    summary = summarize_image_audit(audit_df, duplicates_df)

    audit_df.to_csv(
        interim_dir / "image_quality_audit.csv",
        index=False,
        encoding="utf-8",
    )

    duplicates_df.to_csv(
        interim_dir / "duplicate_hashes.csv",
        index=False,
        encoding="utf-8",
    )

    with (interim_dir / "image_quality_summary.json").open("w", encoding="utf-8") as file:
        json.dump(summary, file, ensure_ascii=False, indent=2)

    print_audit_summary(summary)

    print("\nArquivos gerados:")
    print(f"  - {interim_dir / 'image_quality_audit.csv'}")
    print(f"  - {interim_dir / 'duplicate_hashes.csv'}")
    print(f"  - {interim_dir / 'image_quality_summary.json'}")


if __name__ == "__main__":
    main()
