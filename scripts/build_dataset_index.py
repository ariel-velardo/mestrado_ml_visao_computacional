from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

from liverct.data.index_dataset import (
    build_group_index,
    build_slice_index,
    summarize_index,
    validate_group_label_consistency,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def load_config(config_path: Path) -> dict[str, Any]:
    """
    Carrega o arquivo YAML de configuração local.
    """
    if not config_path.exists():
        raise FileNotFoundError(
            f"Arquivo de configuração não encontrado: {config_path}\n"
            "Crie configs/config.local.yaml a partir de configs/config.example.yaml."
        )

    with config_path.open("r", encoding="utf-8") as file:
        return yaml.safe_load(file)


def get_dataset_config(config: dict[str, Any]) -> dict[str, str]:
    """
    Extrai configurações do dataset com defaults seguros.
    """
    dataset_config = config.get("dataset", {})

    raw_dir = dataset_config.get("raw_dir")
    healthy_folder = dataset_config.get("healthy_folder", "Healthy")
    steatosis_folder = dataset_config.get("steatosis_folder", "Hepatic_Steatosis")

    if not raw_dir:
        raise ValueError(
            "Campo dataset.raw_dir não encontrado em configs/config.local.yaml."
        )

    return {
        "raw_dir": raw_dir,
        "healthy_folder": healthy_folder,
        "steatosis_folder": steatosis_folder,
    }


def save_outputs(
    slice_df: pd.DataFrame,
    group_df: pd.DataFrame,
    summary: dict[str, Any],
    output_dir: Path,
) -> None:
    """
    Salva índices e resumo em data/interim.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    slice_df.to_csv(output_dir / "dataset_index.csv", index=False, encoding="utf-8")
    group_df.to_csv(output_dir / "group_index.csv", index=False, encoding="utf-8")

    with (output_dir / "dataset_summary.json").open("w", encoding="utf-8") as file:
        json.dump(summary, file, ensure_ascii=False, indent=2)


def print_summary(summary: dict[str, Any], conflicts: pd.DataFrame) -> None:
    """
    Imprime um resumo legível da auditoria.
    """
    print("\n=== DATASET INDEX SUMMARY ===")
    print(f"Total de imagens/slices: {summary['total_images']}")
    print(f"Total de grupos inferidos: {summary['total_groups']}")

    print("\nImagens por classe:")
    for class_name, count in summary["images_by_class"].items():
        print(f"  - {class_name}: {count}")

    print("\nGrupos por classe:")
    for class_name, count in summary["groups_by_class"].items():
        print(f"  - {class_name}: {count}")

    print("\nResumo de slices por grupo:")
    for row in summary["slices_by_group_summary"]:
        print(
            f"  - {row['class_name']}: "
            f"min={row['min']}, "
            f"mean={row['mean']}, "
            f"max={row['max']}"
        )

    print("\nDimensões mais frequentes:")
    for row in summary["top_dimensions"]:
        print(f"  - {row['width']}x{row['height']}: {row['count']} imagens")

    print(f"\nImagens ilegíveis: {summary['n_unreadable_images']}")

    print("\nConflitos de classe por inferred_group_id:")
    if conflicts.empty:
        print("  - Nenhum conflito encontrado.")
    else:
        print(conflicts.to_string(index=False))


def main() -> None:
    config_path = PROJECT_ROOT / "configs" / "config.local.yaml"
    output_dir = PROJECT_ROOT / "data" / "interim"

    config = load_config(config_path)
    dataset_config = get_dataset_config(config)

    slice_df = build_slice_index(
        raw_dir=dataset_config["raw_dir"],
        healthy_folder=dataset_config["healthy_folder"],
        steatosis_folder=dataset_config["steatosis_folder"],
    )

    group_df = build_group_index(slice_df)
    conflicts = validate_group_label_consistency(slice_df)
    summary = summarize_index(slice_df, group_df)

    save_outputs(slice_df, group_df, summary, output_dir)
    print_summary(summary, conflicts)

    if not conflicts.empty:
        raise RuntimeError(
            "Foram encontrados inferred_group_id associados a mais de uma classe. "
            "Investigue antes de seguir para split ou modelagem."
        )

    print("\nArquivos gerados:")
    print(f"  - {output_dir / 'dataset_index.csv'}")
    print(f"  - {output_dir / 'group_index.csv'}")
    print(f"  - {output_dir / 'dataset_summary.json'}")


if __name__ == "__main__":
    main()
