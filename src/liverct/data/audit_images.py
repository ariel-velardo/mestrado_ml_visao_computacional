from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

import pandas as pd
from PIL import Image, ImageStat, UnidentifiedImageError


def compute_file_md5(file_path: str | Path, chunk_size: int = 8192) -> str:
    """
    Calcula hash MD5 de um arquivo.

    Uso principal:
    - detectar duplicatas exatas de imagem.
    """
    path = Path(file_path)
    md5 = hashlib.md5()

    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(chunk_size), b""):
            md5.update(chunk)

    return md5.hexdigest()


def compute_image_statistics(file_path: str | Path) -> dict[str, Any]:
    """
    Calcula estatísticas simples de uma imagem.

    As imagens são convertidas para escala de cinza para obter:
    - largura;
    - altura;
    - média de intensidade;
    - desvio padrão de intensidade;
    - mínimo;
    - máximo.

    Observação:
    Como o dataset está em JPEG, essas estatísticas não devem ser interpretadas
    como valores HU de tomografia. Elas representam apenas intensidade dos pixels
    no arquivo disponível.
    """
    path = Path(file_path)

    try:
        with Image.open(path) as image:
            gray = image.convert("L")
            width, height = gray.size
            stat = ImageStat.Stat(gray)
            min_intensity, max_intensity = gray.getextrema()

            return {
                "width_audit": width,
                "height_audit": height,
                "mean_intensity": float(stat.mean[0]),
                "std_intensity": float(stat.stddev[0]),
                "min_intensity": int(min_intensity),
                "max_intensity": int(max_intensity),
                "is_readable_audit": True,
                "audit_error": "",
            }

    except (UnidentifiedImageError, OSError, ValueError) as error:
        return {
            "width_audit": None,
            "height_audit": None,
            "mean_intensity": None,
            "std_intensity": None,
            "min_intensity": None,
            "max_intensity": None,
            "is_readable_audit": False,
            "audit_error": str(error),
        }


def build_image_quality_audit(slice_df: pd.DataFrame) -> pd.DataFrame:
    """
    Cria auditoria de qualidade para todas as imagens listadas no índice de slices.

    Entrada esperada:
    - dataframe derivado de dataset_index.csv ou split_slices.csv.
    - deve conter file_path, class_name, label, inferred_group_id e filename.
    """
    required_columns = {
        "file_path",
        "filename",
        "class_name",
        "label",
        "inferred_group_id",
    }

    missing = required_columns - set(slice_df.columns)

    if missing:
        raise ValueError(f"Colunas ausentes em slice_df: {sorted(missing)}")

    rows: list[dict[str, Any]] = []

    for row in slice_df.to_dict(orient="records"):
        file_path = Path(row["file_path"])

        audit_row = dict(row)

        if not file_path.exists():
            audit_row.update(
                {
                    "file_exists": False,
                    "md5": None,
                    "width_audit": None,
                    "height_audit": None,
                    "mean_intensity": None,
                    "std_intensity": None,
                    "min_intensity": None,
                    "max_intensity": None,
                    "is_readable_audit": False,
                    "audit_error": "Arquivo não encontrado.",
                }
            )
        else:
            audit_row["file_exists"] = True
            audit_row["md5"] = compute_file_md5(file_path)
            audit_row.update(compute_image_statistics(file_path))

        rows.append(audit_row)

    audit_df = pd.DataFrame(rows)

    return audit_df.reset_index(drop=True)


def find_exact_duplicates(audit_df: pd.DataFrame) -> pd.DataFrame:
    """
    Identifica duplicatas exatas por hash MD5.

    Retorna apenas imagens cujo hash aparece mais de uma vez.
    """
    if "md5" not in audit_df.columns:
        raise ValueError("Coluna md5 não encontrada em audit_df.")

    valid_hashes = audit_df.dropna(subset=["md5"]).copy()

    duplicated_hashes = (
        valid_hashes.groupby("md5")
        .size()
        .reset_index(name="n_files")
        .query("n_files > 1")
    )

    if duplicated_hashes.empty:
        return pd.DataFrame(
            columns=[
                "md5",
                "n_files",
                "class_name",
                "split",
                "inferred_group_id",
                "filename",
                "file_path",
            ]
        )

    duplicates = valid_hashes.merge(duplicated_hashes, on="md5", how="inner")

    selected_columns = [
        "md5",
        "n_files",
        "class_name",
        "inferred_group_id",
        "filename",
        "file_path",
    ]

    if "split" in duplicates.columns:
        selected_columns.insert(3, "split")

    return duplicates[selected_columns].sort_values(
        ["md5", "class_name", "inferred_group_id", "filename"]
    ).reset_index(drop=True)


def summarize_image_audit(
    audit_df: pd.DataFrame,
    duplicates_df: pd.DataFrame,
) -> dict[str, Any]:
    """
    Resume a auditoria de qualidade das imagens.
    """
    summary: dict[str, Any] = {
        "total_images": int(len(audit_df)),
        "missing_files": int((~audit_df["file_exists"]).sum()),
        "unreadable_images": int((~audit_df["is_readable_audit"]).sum()),
        "duplicate_hash_groups": int(duplicates_df["md5"].nunique())
        if not duplicates_df.empty
        else 0,
        "duplicate_files": int(len(duplicates_df)),
    }

    summary["by_class"] = (
        audit_df.groupby("class_name")
        .agg(
            n_images=("filename", "count"),
            n_groups=("inferred_group_id", "nunique"),
            mean_intensity_avg=("mean_intensity", "mean"),
            mean_intensity_std=("mean_intensity", "std"),
            std_intensity_avg=("std_intensity", "mean"),
            file_size_avg=("file_size_bytes", "mean"),
        )
        .round(4)
        .reset_index()
        .to_dict(orient="records")
    )

    if "split" in audit_df.columns:
        summary["by_split_class"] = (
            audit_df.groupby(["split", "class_name"])
            .agg(
                n_images=("filename", "count"),
                n_groups=("inferred_group_id", "nunique"),
                mean_intensity_avg=("mean_intensity", "mean"),
                std_intensity_avg=("std_intensity", "mean"),
                file_size_avg=("file_size_bytes", "mean"),
            )
            .round(4)
            .reset_index()
            .to_dict(orient="records")
        )

    summary["dimensions"] = (
        audit_df.groupby(["width_audit", "height_audit"])
        .size()
        .reset_index(name="n_images")
        .sort_values("n_images", ascending=False)
        .to_dict(orient="records")
    )

    return summary
