from __future__ import annotations

import random
from pathlib import Path
from typing import Any

import pandas as pd


def split_groups(
    group_df: pd.DataFrame,
    train_size: float = 0.70,
    val_size: float = 0.15,
    test_size: float = 0.15,
    seed: int = 42,
) -> pd.DataFrame:
    """
    Cria splits estratificados por classe usando inferred_group_id como unidade.

    A regra central é:
    - todos os slices de um mesmo inferred_group_id devem permanecer no mesmo split;
    - o split nunca deve ser feito diretamente por imagem/slice.

    Retorna um dataframe com as colunas originais de group_df mais a coluna `split`.
    """
    _validate_split_sizes(train_size, val_size, test_size)

    required_columns = {"inferred_group_id", "class_name", "label", "n_slices"}
    missing = required_columns - set(group_df.columns)

    if missing:
        raise ValueError(f"Colunas ausentes em group_df: {sorted(missing)}")

    rng = random.Random(seed)
    split_parts: list[pd.DataFrame] = []

    for class_name, class_groups in group_df.groupby("class_name", sort=True):
        class_groups = class_groups.sort_values("inferred_group_id").reset_index(drop=True)

        records = class_groups.to_dict(orient="records")
        rng.shuffle(records)

        n_groups = len(records)
        n_train = int(n_groups * train_size)
        n_val = int(n_groups * val_size)

        train_records = records[:n_train]
        val_records = records[n_train : n_train + n_val]
        test_records = records[n_train + n_val :]

        split_parts.append(_records_to_split_df(train_records, "train"))
        split_parts.append(_records_to_split_df(val_records, "val"))
        split_parts.append(_records_to_split_df(test_records, "test"))

    split_df = pd.concat(split_parts, ignore_index=True)

    return split_df.sort_values(["split", "class_name", "inferred_group_id"]).reset_index(
        drop=True
    )


def expand_splits_to_slices(
    slice_df: pd.DataFrame,
    split_group_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Expande a atribuição de split dos grupos para as imagens/slices.

    Cada slice recebe o split do seu inferred_group_id.
    """
    required_slice_columns = {"inferred_group_id", "filename", "class_name", "label"}
    required_group_columns = {"inferred_group_id", "split"}

    missing_slice = required_slice_columns - set(slice_df.columns)
    missing_group = required_group_columns - set(split_group_df.columns)

    if missing_slice:
        raise ValueError(f"Colunas ausentes em slice_df: {sorted(missing_slice)}")

    if missing_group:
        raise ValueError(f"Colunas ausentes em split_group_df: {sorted(missing_group)}")

    group_to_split = dict(
        zip(split_group_df["inferred_group_id"], split_group_df["split"])
    )

    expanded = slice_df.copy()
    expanded["split"] = expanded["inferred_group_id"].map(group_to_split)

    missing_split = expanded[expanded["split"].isna()]

    if not missing_split.empty:
        n_missing_groups = missing_split["inferred_group_id"].nunique()
        raise ValueError(
            f"Há {n_missing_groups} inferred_group_id sem split atribuído."
        )

    return expanded.sort_values(
        ["split", "class_name", "inferred_group_id", "slice_id", "filename"],
        na_position="last",
    ).reset_index(drop=True)


def validate_no_split_leakage(split_group_df: pd.DataFrame) -> dict[str, int]:
    """
    Verifica se existe vazamento de inferred_group_id entre splits.

    Retorna a quantidade de interseções entre pares de splits.
    """
    split_sets = {
        split_name: set(
            split_group_df.loc[
                split_group_df["split"] == split_name, "inferred_group_id"
            ]
        )
        for split_name in ["train", "val", "test"]
    }

    return {
        "train_val_leakage": len(split_sets["train"] & split_sets["val"]),
        "train_test_leakage": len(split_sets["train"] & split_sets["test"]),
        "val_test_leakage": len(split_sets["val"] & split_sets["test"]),
    }


def summarize_splits(
    split_group_df: pd.DataFrame,
    split_slice_df: pd.DataFrame,
) -> dict[str, Any]:
    """
    Resume distribuição de grupos e slices por split e classe.
    """
    groups_by_split_class = (
        split_group_df.groupby(["split", "class_name"])
        .size()
        .reset_index(name="n_groups")
        .sort_values(["split", "class_name"])
        .to_dict(orient="records")
    )

    slices_by_split_class = (
        split_slice_df.groupby(["split", "class_name"])
        .size()
        .reset_index(name="n_slices")
        .sort_values(["split", "class_name"])
        .to_dict(orient="records")
    )

    total_groups_by_split = (
        split_group_df.groupby("split")
        .size()
        .reset_index(name="n_groups")
        .sort_values("split")
        .to_dict(orient="records")
    )

    total_slices_by_split = (
        split_slice_df.groupby("split")
        .size()
        .reset_index(name="n_slices")
        .sort_values("split")
        .to_dict(orient="records")
    )

    leakage = validate_no_split_leakage(split_group_df)

    return {
        "groups_by_split_class": groups_by_split_class,
        "slices_by_split_class": slices_by_split_class,
        "total_groups_by_split": total_groups_by_split,
        "total_slices_by_split": total_slices_by_split,
        "leakage": leakage,
    }


def save_split_outputs(
    split_group_df: pd.DataFrame,
    split_slice_df: pd.DataFrame,
    output_dir: str | Path,
) -> None:
    """
    Salva arquivos locais de split em data/interim.

    Esses arquivos são derivados do dataset e não devem ser versionados.
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    split_group_df.to_csv(
        output_path / "split_groups.csv", index=False, encoding="utf-8"
    )
    split_slice_df.to_csv(
        output_path / "split_slices.csv", index=False, encoding="utf-8"
    )

    for split_name in ["train", "val", "test"]:
        split_group_df.loc[split_group_df["split"] == split_name].to_csv(
            output_path / f"{split_name}_groups.csv",
            index=False,
            encoding="utf-8",
        )

        split_slice_df.loc[split_slice_df["split"] == split_name].to_csv(
            output_path / f"{split_name}_slices.csv",
            index=False,
            encoding="utf-8",
        )


def _validate_split_sizes(train_size: float, val_size: float, test_size: float) -> None:
    """
    Valida proporções de split.
    """
    total = train_size + val_size + test_size

    if abs(total - 1.0) > 1e-8:
        raise ValueError(
            "As proporções de split devem somar 1. "
            f"Recebido: {train_size} + {val_size} + {test_size} = {total}"
        )

    if min(train_size, val_size, test_size) <= 0:
        raise ValueError("Todas as proporções de split devem ser positivas.")


def _records_to_split_df(records: list[dict[str, Any]], split_name: str) -> pd.DataFrame:
    """
    Converte registros em dataframe e adiciona o nome do split.
    """
    df = pd.DataFrame(records)

    if df.empty:
        return df

    df["split"] = split_name
    return df
