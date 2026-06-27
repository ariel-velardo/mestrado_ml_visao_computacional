from __future__ import annotations

from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import pandas as pd
from PIL import Image


def prepare_audit_dataframe(audit_df: pd.DataFrame) -> pd.DataFrame:
    """
    Prepara o dataframe de auditoria para EDA visual e técnica.

    Garante tipos numéricos e remove registros sem arquivo legível.
    """
    df = audit_df.copy()

    numeric_columns = [
        "label",
        "slice_id",
        "file_size_bytes",
        "width_audit",
        "height_audit",
        "mean_intensity",
        "std_intensity",
        "min_intensity",
        "max_intensity",
    ]

    for column in numeric_columns:
        if column in df.columns:
            df[column] = pd.to_numeric(df[column], errors="coerce")

    if "file_exists" in df.columns:
        df = df[df["file_exists"] == True].copy()

    if "is_readable_audit" in df.columns:
        df = df[df["is_readable_audit"] == True].copy()

    return df.reset_index(drop=True)


def build_summary_by_class(audit_df: pd.DataFrame) -> pd.DataFrame:
    """
    Resume estatísticas técnicas por classe.
    """
    return (
        audit_df.groupby("class_name")
        .agg(
            n_images=("filename", "count"),
            n_groups=("inferred_group_id", "nunique"),
            mean_intensity_avg=("mean_intensity", "mean"),
            mean_intensity_std=("mean_intensity", "std"),
            std_intensity_avg=("std_intensity", "mean"),
            min_intensity_avg=("min_intensity", "mean"),
            max_intensity_avg=("max_intensity", "mean"),
            file_size_avg=("file_size_bytes", "mean"),
            file_size_std=("file_size_bytes", "std"),
        )
        .round(4)
        .reset_index()
    )


def build_summary_by_split_class(audit_df: pd.DataFrame) -> pd.DataFrame:
    """
    Resume estatísticas técnicas por split e classe.
    """
    if "split" not in audit_df.columns:
        return pd.DataFrame()

    return (
        audit_df.groupby(["split", "class_name"])
        .agg(
            n_images=("filename", "count"),
            n_groups=("inferred_group_id", "nunique"),
            mean_intensity_avg=("mean_intensity", "mean"),
            mean_intensity_std=("mean_intensity", "std"),
            std_intensity_avg=("std_intensity", "mean"),
            file_size_avg=("file_size_bytes", "mean"),
            file_size_std=("file_size_bytes", "std"),
        )
        .round(4)
        .reset_index()
        .sort_values(["split", "class_name"])
    )


def sample_one_slice_per_group(
    audit_df: pd.DataFrame,
    group_columns: Iterable[str],
    n_per_group: int = 6,
    seed: int = 42,
) -> pd.DataFrame:
    """
    Seleciona amostras visuais evitando pegar muitos slices do mesmo grupo.

    Para cada combinação de group_columns, seleciona no máximo um slice por
    inferred_group_id e depois amostra n_per_group grupos.
    """
    sampled_parts = []

    for idx, (key, group) in enumerate(audit_df.groupby(list(group_columns), sort=True)):
        one_per_group = (
            group.sort_values(["inferred_group_id", "slice_id", "filename"])
            .groupby("inferred_group_id", as_index=False)
            .first()
        )

        n = min(n_per_group, len(one_per_group))

        sampled = one_per_group.sample(n=n, random_state=seed + idx).copy()

        if not isinstance(key, tuple):
            key = (key,)

        sampled["display_group"] = " | ".join(map(str, key))
        sampled_parts.append(sampled)

    if not sampled_parts:
        return pd.DataFrame()

    return pd.concat(sampled_parts, ignore_index=True)


def plot_sample_grid(
    sampled_df: pd.DataFrame,
    output_path: str | Path,
    n_cols: int = 6,
    title: str = "Amostras de imagens",
) -> None:
    """
    Plota uma grade de amostras de imagens.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if sampled_df.empty:
        raise ValueError("sampled_df está vazio.")

    groups = list(sampled_df.groupby("display_group", sort=True))
    n_rows = len(groups)

    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(n_cols * 2.2, n_rows * 2.4),
        squeeze=False,
    )

    fig.suptitle(title, fontsize=14)

    for row_idx, (display_group, group) in enumerate(groups):
        records = group.to_dict(orient="records")

        for col_idx in range(n_cols):
            ax = axes[row_idx][col_idx]
            ax.axis("off")

            if col_idx >= len(records):
                continue

            record = records[col_idx]
            image_path = Path(record["file_path"])

            with Image.open(image_path) as image:
                ax.imshow(image.convert("L"), cmap="gray")

            ax.set_title(
                f"{record['inferred_group_id']}\nslice={record.get('slice_id', '')}",
                fontsize=8,
            )

            if col_idx == 0:
                ax.set_ylabel(display_group, fontsize=10)

    plt.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_histogram_by_class(
    audit_df: pd.DataFrame,
    value_column: str,
    output_path: str | Path,
    title: str,
    xlabel: str,
    bins: int = 30,
) -> None:
    """
    Plota histograma de uma variável técnica por classe.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(9, 5))

    for class_name, group in audit_df.groupby("class_name", sort=True):
        ax.hist(
            group[value_column].dropna(),
            bins=bins,
            alpha=0.5,
            label=class_name,
        )

    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Frequência")
    ax.legend()

    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_boxplot_by_class(
    audit_df: pd.DataFrame,
    value_column: str,
    output_path: str | Path,
    title: str,
    ylabel: str,
) -> None:
    """
    Plota boxplot de uma variável técnica por classe.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    labels = []
    values = []

    for class_name, group in audit_df.groupby("class_name", sort=True):
        labels.append(class_name)
        values.append(group[value_column].dropna())

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.boxplot(values, showfliers=False)
    ax.set_xticks(range(1, len(labels) + 1))
    ax.set_xticklabels(labels)
    ax.set_title(title)
    ax.set_ylabel(ylabel)

    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_boxplot_by_split_class(
    audit_df: pd.DataFrame,
    value_column: str,
    output_path: str | Path,
    title: str,
    ylabel: str,
) -> None:
    """
    Plota boxplot de uma variável técnica por split e classe.
    """
    if "split" not in audit_df.columns:
        return

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    labels = []
    values = []

    ordered = audit_df.sort_values(["split", "class_name"])

    for (split, class_name), group in ordered.groupby(["split", "class_name"], sort=True):
        labels.append(f"{split}\n{class_name}")
        values.append(group[value_column].dropna())

    fig, ax = plt.subplots(figsize=(11, 5))
    ax.boxplot(values, showfliers=False)
    ax.set_xticks(range(1, len(labels) + 1))
    ax.set_xticklabels(labels)
    ax.set_title(title)
    ax.set_ylabel(ylabel)

    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

