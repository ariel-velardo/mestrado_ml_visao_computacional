from __future__ import annotations

from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset


REQUIRED_COLUMNS = {
    "file_path",
    "label",
    "split",
    "inferred_group_id",
}

VALID_SPLITS = {"train", "val", "test"}


def load_split_slices(split_csv_path: str | Path) -> pd.DataFrame:
    """
    Load and validate the existing split_slices.csv file.

    This function never creates or modifies splits.
    """
    split_csv_path = Path(split_csv_path)
    if not split_csv_path.exists():
        raise FileNotFoundError(f"Split file not found: {split_csv_path}")

    df = pd.read_csv(split_csv_path)

    missing = REQUIRED_COLUMNS - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns in split file: {sorted(missing)}")

    unknown_splits = set(df["split"].dropna().unique()) - VALID_SPLITS
    if unknown_splits:
        raise ValueError(f"Unknown split values: {sorted(unknown_splits)}")

    df = df.copy()
    df["label"] = pd.to_numeric(df["label"], errors="raise").astype(int)
    df["file_path"] = df["file_path"].astype(str)
    df["inferred_group_id"] = df["inferred_group_id"].astype(str)
    df["split"] = df["split"].astype(str)

    return df.reset_index(drop=True)


def validate_group_split_integrity(
    df: pd.DataFrame,
    group_col: str = "inferred_group_id",
    split_col: str = "split",
) -> None:
    """
    Validate that each inferred_group_id appears in one split only.
    """
    required_columns = {group_col, split_col}
    missing = required_columns - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns in dataframe: {sorted(missing)}")

    split_counts = df.groupby(group_col)[split_col].nunique()
    leaking_groups = split_counts[split_counts > 1]
    if not leaking_groups.empty:
        raise ValueError(
            "Potential leakage: inferred_group_id appears in multiple splits. "
            f"Examples: {leaking_groups.index.tolist()[:10]}"
        )


def select_split(
    df: pd.DataFrame,
    split: Literal["train", "val", "test"],
) -> pd.DataFrame:
    """
    Return a copy of one split from the split dataframe.
    """
    split_df = df[df["split"] == split].copy()
    if split_df.empty:
        raise ValueError(f"Split is empty: {split}")
    return split_df.reset_index(drop=True)


class LiverCTSliceDataset(Dataset):
    """
    PyTorch dataset for JPEG CT slices.

    Images are loaded from file_path, converted to grayscale, resized to
    image_size when needed, and normalized to [0, 1].
    """

    def __init__(
        self,
        df: pd.DataFrame,
        image_size: int = 256,
    ) -> None:
        missing = REQUIRED_COLUMNS - set(df.columns)
        if missing:
            raise ValueError(f"Missing columns in dataframe: {sorted(missing)}")

        self.df = df.reset_index(drop=True).copy()
        self.image_size = int(image_size)

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor | str]:
        row = self.df.iloc[index]
        image_path = Path(row["file_path"])

        with Image.open(image_path) as image:
            image = image.convert("L")
            if image.size != (self.image_size, self.image_size):
                image = image.resize(
                    (self.image_size, self.image_size),
                    resample=Image.BILINEAR,
                )
            image_array = np.asarray(image, dtype=np.float32) / 255.0

        image_tensor = torch.from_numpy(image_array).unsqueeze(0)
        label_tensor = torch.tensor(float(row["label"]), dtype=torch.float32)

        return {
            "image": image_tensor,
            "label": label_tensor,
            "inferred_group_id": str(row["inferred_group_id"]),
            "split": str(row["split"]),
            "file_path": str(row["file_path"]),
        }


def build_dataloader(
    df: pd.DataFrame,
    batch_size: int,
    image_size: int = 256,
    shuffle: bool = False,
    num_workers: int = 0,
) -> DataLoader:
    """
    Build a dataloader for a split dataframe.
    """
    dataset = LiverCTSliceDataset(df=df, image_size=image_size)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )
