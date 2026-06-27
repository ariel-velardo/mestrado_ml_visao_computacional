from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd
from PIL import Image, UnidentifiedImageError


CLASS_TO_LABEL = {
    "Healthy": 0,
    "Hepatic_Steatosis": 1,
}


def parse_filename(filename: str) -> dict[str, Any]:
    """
    Parseia nomes no padrão esperado do dataset.

    Exemplo:
        1-img-00004-00080.jpg

    Retorna:
        inferred_group_id = 1-img-00004
        slice_id = 80

    Observação:
        inferred_group_id é um agrupamento técnico inferido do nome do arquivo.
        Não deve ser tratado como patient_id clinicamente validado.
    """
    path = Path(filename)
    stem = path.stem
    parts = stem.split("-")

    if len(parts) >= 4:
        inferred_group_id = "-".join(parts[:3])
        slice_raw = parts[3]

        try:
            slice_id = int(slice_raw)
        except ValueError:
            slice_id = None
    else:
        inferred_group_id = stem
        slice_id = None

    return {
        "inferred_group_id": inferred_group_id,
        "slice_id": slice_id,
    }


def read_image_size(image_path: Path) -> tuple[int | None, int | None, bool]:
    """
    Lê largura e altura de uma imagem sem carregá-la integralmente para memória.

    Retorna:
        width, height, is_readable
    """
    try:
        with Image.open(image_path) as image:
            width, height = image.size
        return width, height, True
    except (UnidentifiedImageError, OSError):
        return None, None, False


def build_slice_index(
    raw_dir: str | Path,
    healthy_folder: str = "Healthy",
    steatosis_folder: str = "Hepatic_Steatosis",
    allowed_extensions: tuple[str, ...] = (".jpg", ".jpeg", ".png"),
) -> pd.DataFrame:
    """
    Cria um índice com uma linha por imagem/slice.

    O índice inclui classe, label, caminho do arquivo, agrupamento inferido,
    slice_id, extensão, tamanho do arquivo e dimensões da imagem.
    """
    raw_path = Path(raw_dir)

    if not raw_path.exists():
        raise FileNotFoundError(f"Dataset não encontrado em: {raw_path}")

    class_folders = {
        healthy_folder: CLASS_TO_LABEL["Healthy"],
        steatosis_folder: CLASS_TO_LABEL["Hepatic_Steatosis"],
    }

    rows: list[dict[str, Any]] = []

    for class_name, label in class_folders.items():
        class_path = raw_path / class_name

        if not class_path.exists():
            raise FileNotFoundError(f"Pasta da classe não encontrada: {class_path}")

        for file_path in sorted(class_path.rglob("*")):
            if not file_path.is_file():
                continue

            extension = file_path.suffix.lower()

            if extension not in allowed_extensions:
                continue

            parsed = parse_filename(file_path.name)
            width, height, is_readable = read_image_size(file_path)

            rows.append(
                {
                    "class_name": class_name,
                    "label": label,
                    "filename": file_path.name,
                    "inferred_group_id": parsed["inferred_group_id"],
                    "slice_id": parsed["slice_id"],
                    "extension": extension,
                    "file_size_bytes": file_path.stat().st_size,
                    "width": width,
                    "height": height,
                    "is_readable": is_readable,
                    "file_path": str(file_path),
                }
            )

    df = pd.DataFrame(rows)

    if df.empty:
        raise ValueError(f"Nenhuma imagem encontrada em: {raw_path}")

    return df.sort_values(
        ["class_name", "inferred_group_id", "slice_id", "filename"],
        na_position="last",
    ).reset_index(drop=True)


def build_group_index(slice_df: pd.DataFrame) -> pd.DataFrame:
    """
    Cria um índice agregado por inferred_group_id.

    Cada linha representa um agrupamento técnico inferido do nome dos arquivos.
    """
    required_columns = {
        "class_name",
        "label",
        "inferred_group_id",
        "slice_id",
        "file_size_bytes",
        "width",
        "height",
        "is_readable",
    }

    missing = required_columns - set(slice_df.columns)

    if missing:
        raise ValueError(f"Colunas ausentes em slice_df: {sorted(missing)}")

    grouped_rows: list[dict[str, Any]] = []

    for group_id, group in slice_df.groupby("inferred_group_id", sort=True):
        class_names = sorted(group["class_name"].dropna().unique().tolist())
        labels = sorted(group["label"].dropna().unique().tolist())

        slice_values = group["slice_id"].dropna()

        grouped_rows.append(
            {
                "inferred_group_id": group_id,
                "class_name": "|".join(map(str, class_names)),
                "label": "|".join(map(str, labels)),
                "n_slices": int(len(group)),
                "min_slice_id": int(slice_values.min()) if not slice_values.empty else None,
                "max_slice_id": int(slice_values.max()) if not slice_values.empty else None,
                "total_size_bytes": int(group["file_size_bytes"].sum()),
                "width_mode": _mode_or_none(group["width"]),
                "height_mode": _mode_or_none(group["height"]),
                "n_unreadable": int((~group["is_readable"]).sum()),
            }
        )

    return pd.DataFrame(grouped_rows).sort_values("inferred_group_id").reset_index(drop=True)


def validate_group_label_consistency(slice_df: pd.DataFrame) -> pd.DataFrame:
    """
    Verifica se algum inferred_group_id aparece em mais de uma classe.

    Retorna um dataframe apenas com grupos problemáticos.
    Se retornar vazio, não há conflito de classe por agrupamento.
    """
    validation = (
        slice_df.groupby("inferred_group_id")
        .agg(
            n_classes=("class_name", "nunique"),
            classes=("class_name", lambda x: "|".join(sorted(set(x)))),
            n_labels=("label", "nunique"),
            labels=("label", lambda x: "|".join(map(str, sorted(set(x))))),
            n_slices=("filename", "count"),
        )
        .reset_index()
    )

    return validation[
        (validation["n_classes"] > 1) | (validation["n_labels"] > 1)
    ].reset_index(drop=True)


def summarize_index(slice_df: pd.DataFrame, group_df: pd.DataFrame) -> dict[str, Any]:
    """
    Gera um resumo geral do dataset indexado.
    """
    total_images = int(len(slice_df))
    total_groups = int(len(group_df))

    images_by_class = (
        slice_df.groupby("class_name")
        .size()
        .sort_index()
        .to_dict()
    )

    groups_by_class = (
        group_df.groupby("class_name")
        .size()
        .sort_index()
        .to_dict()
    )

    slices_by_group_summary = (
        group_df.groupby("class_name")["n_slices"]
        .agg(["min", "mean", "max"])
        .round(2)
        .reset_index()
        .to_dict(orient="records")
    )

    dimensions = (
        slice_df.groupby(["width", "height"])
        .size()
        .sort_values(ascending=False)
        .head(10)
        .reset_index(name="count")
        .to_dict(orient="records")
    )

    return {
        "total_images": total_images,
        "total_groups": total_groups,
        "images_by_class": images_by_class,
        "groups_by_class": groups_by_class,
        "slices_by_group_summary": slices_by_group_summary,
        "top_dimensions": dimensions,
        "n_unreadable_images": int((~slice_df["is_readable"]).sum()),
    }


def _mode_or_none(series: pd.Series) -> int | None:
    """
    Retorna a moda de uma série, ignorando valores ausentes.
    """
    clean = series.dropna()

    if clean.empty:
        return None

    return int(clean.mode().iloc[0])
