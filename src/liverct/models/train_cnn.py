from __future__ import annotations

import json
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch import nn

from liverct.evaluation.classification_metrics import evaluate_predictions_by_split
from liverct.evaluation.group_aggregation import aggregate_probabilities_by_group
from liverct.models.cnn_dataset import (
    build_dataloader,
    load_split_slices,
    select_split,
    validate_group_split_integrity,
)
from liverct.models.simple_cnn import SimpleCNN2D, count_trainable_parameters


@dataclass
class CNNTrainingConfig:
    split_csv_path: Path
    reports_dir: Path
    checkpoint_dir: Path
    image_size: int = 256
    batch_size: int = 32
    epochs: int = 20
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    dropout: float = 0.25
    patience: int = 5
    seed: int = 42
    num_workers: int = 0
    threshold: float = 0.5
    checkpoint_name: str = "baseline_cnn_best.pt"


def set_global_seed(seed: int) -> None:
    """
    Set seeds for reproducible baseline runs.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def get_device() -> torch.device:
    """
    Select CUDA when available, otherwise CPU.
    """
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def compute_pos_weight(train_df: pd.DataFrame) -> torch.Tensor | None:
    """
    Compute BCE positive-class weight from the training split only.
    """
    positives = int((train_df["label"] == 1).sum())
    negatives = int((train_df["label"] == 0).sum())

    if positives == 0 or negatives == 0:
        return None

    return torch.tensor([negatives / positives], dtype=torch.float32)


def train_one_epoch(
    model: nn.Module,
    dataloader: torch.utils.data.DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> float:
    """
    Train one epoch and return mean loss.
    """
    model.train()
    total_loss = 0.0
    total_items = 0

    for batch in dataloader:
        images = batch["image"].to(device)
        labels = batch["label"].to(device)

        optimizer.zero_grad(set_to_none=True)
        logits = model(images)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        batch_size = int(labels.shape[0])
        total_loss += float(loss.item()) * batch_size
        total_items += batch_size

    return total_loss / max(total_items, 1)


@torch.no_grad()
def evaluate_loss(
    model: nn.Module,
    dataloader: torch.utils.data.DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> float:
    """
    Evaluate mean loss without updating weights.
    """
    model.eval()
    total_loss = 0.0
    total_items = 0

    for batch in dataloader:
        images = batch["image"].to(device)
        labels = batch["label"].to(device)
        logits = model(images)
        loss = criterion(logits, labels)

        batch_size = int(labels.shape[0])
        total_loss += float(loss.item()) * batch_size
        total_items += batch_size

    return total_loss / max(total_items, 1)


@torch.no_grad()
def predict_with_model(
    model: nn.Module,
    dataloader: torch.utils.data.DataLoader,
    device: torch.device,
    threshold: float = 0.5,
) -> pd.DataFrame:
    """
    Generate slice-level probabilities with a trained CNN.
    """
    model.eval()
    rows: list[dict[str, Any]] = []

    for batch in dataloader:
        images = batch["image"].to(device)
        logits = model(images)
        probabilities = torch.sigmoid(logits).detach().cpu().numpy()
        labels = batch["label"].detach().cpu().numpy().astype(int)

        for idx, prob_positive in enumerate(probabilities):
            rows.append(
                {
                    "split": batch["split"][idx],
                    "label": int(labels[idx]),
                    "prob_positive": float(prob_positive),
                    "pred_label": int(float(prob_positive) >= threshold),
                    "inferred_group_id": batch["inferred_group_id"][idx],
                    "file_path": batch["file_path"][idx],
                }
            )

    return pd.DataFrame(rows)


def evaluate_prediction_tables(
    slice_pred_df: pd.DataFrame,
    threshold: float = 0.5,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Build slice predictions, group predictions, and metrics tables.
    """
    group_pred_df = aggregate_probabilities_by_group(
        slice_pred_df,
        threshold=threshold,
    )

    slice_metrics = evaluate_predictions_by_split(
        slice_pred_df,
        level="slice",
        threshold=threshold,
    )
    group_metrics = evaluate_predictions_by_split(
        group_pred_df,
        level="group",
        threshold=threshold,
    )
    metrics_df = pd.concat([slice_metrics, group_metrics], ignore_index=True)
    metrics_df.insert(0, "model", "simple_cnn")

    return slice_pred_df, group_pred_df, metrics_df


def save_checkpoint(
    path: Path,
    model: nn.Module,
    config: CNNTrainingConfig,
    epoch: int,
    val_loss: float,
    train_loss: float,
) -> None:
    """
    Save a model checkpoint with minimal reproducibility metadata.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    checkpoint = {
        "model_state_dict": model.state_dict(),
        "config": {key: str(value) for key, value in asdict(config).items()},
        "epoch": int(epoch),
        "val_loss": float(val_loss),
        "train_loss": float(train_loss),
        "model_name": "SimpleCNN2D",
    }
    torch.save(checkpoint, path)


def load_cnn_checkpoint(
    checkpoint_path: str | Path,
    device: torch.device | None = None,
) -> tuple[SimpleCNN2D, dict[str, Any]]:
    """
    Load a SimpleCNN2D checkpoint for evaluation.
    """
    checkpoint_path = Path(checkpoint_path)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    device = device or get_device()
    try:
        checkpoint = torch.load(
            checkpoint_path,
            map_location=device,
            weights_only=False,
        )
    except TypeError:
        checkpoint = torch.load(checkpoint_path, map_location=device)

    config = checkpoint.get("config", {})
    dropout = float(config.get("dropout", 0.25))
    model = SimpleCNN2D(dropout=dropout)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()

    return model, checkpoint


def run_cnn_training(config: CNNTrainingConfig) -> dict[str, Any]:
    """
    Train a simple CNN using train split and early stopping on val split.

    Test split is intentionally not evaluated here.
    """
    set_global_seed(config.seed)

    config.reports_dir.mkdir(parents=True, exist_ok=True)
    config.checkpoint_dir.mkdir(parents=True, exist_ok=True)

    all_df = load_split_slices(config.split_csv_path)
    validate_group_split_integrity(all_df)
    train_df = select_split(all_df, "train")
    val_df = select_split(all_df, "val")

    train_loader = build_dataloader(
        train_df,
        batch_size=config.batch_size,
        image_size=config.image_size,
        shuffle=True,
        num_workers=config.num_workers,
    )
    val_loader = build_dataloader(
        val_df,
        batch_size=config.batch_size,
        image_size=config.image_size,
        shuffle=False,
        num_workers=config.num_workers,
    )

    device = get_device()
    model = SimpleCNN2D(dropout=config.dropout).to(device)
    pos_weight = compute_pos_weight(train_df)
    if pos_weight is not None:
        pos_weight = pos_weight.to(device)

    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )

    best_val_loss = float("inf")
    best_epoch = 0
    epochs_without_improvement = 0
    history: list[dict[str, float | int]] = []
    checkpoint_path = config.checkpoint_dir / config.checkpoint_name

    print(f"Device: {device}")
    print(f"Train slices: {len(train_df)} | Val slices: {len(val_df)}")
    print(f"Trainable parameters: {count_trainable_parameters(model)}")

    for epoch in range(1, config.epochs + 1):
        train_loss = train_one_epoch(
            model=model,
            dataloader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            device=device,
        )
        val_loss = evaluate_loss(
            model=model,
            dataloader=val_loader,
            criterion=criterion,
            device=device,
        )

        improved = val_loss < best_val_loss
        if improved:
            best_val_loss = val_loss
            best_epoch = epoch
            epochs_without_improvement = 0
            save_checkpoint(
                path=checkpoint_path,
                model=model,
                config=config,
                epoch=epoch,
                val_loss=val_loss,
                train_loss=train_loss,
            )
        else:
            epochs_without_improvement += 1

        history.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "val_loss": val_loss,
                "best_val_loss": best_val_loss,
                "improved": int(improved),
            }
        )

        print(
            f"Epoch {epoch:03d}/{config.epochs} | "
            f"train_loss={train_loss:.5f} | "
            f"val_loss={val_loss:.5f} | "
            f"best_epoch={best_epoch}"
        )

        if epochs_without_improvement >= config.patience:
            print(f"Early stopping at epoch {epoch}.")
            break

    history_df = pd.DataFrame(history)
    history_path = config.reports_dir / "baseline_cnn_training_history.csv"
    history_df.to_csv(history_path, index=False, encoding="utf-8")

    best_model, checkpoint = load_cnn_checkpoint(checkpoint_path, device=device)
    val_pred_df = predict_with_model(
        model=best_model,
        dataloader=val_loader,
        device=device,
        threshold=config.threshold,
    )
    val_slice_pred_df, val_group_pred_df, val_metrics_df = evaluate_prediction_tables(
        val_pred_df,
        threshold=config.threshold,
    )

    val_slice_pred_path = config.reports_dir / "baseline_cnn_val_slice_predictions.csv"
    val_group_pred_path = config.reports_dir / "baseline_cnn_val_group_predictions.csv"
    val_metrics_path = config.reports_dir / "baseline_cnn_validation_metrics.csv"
    summary_path = config.reports_dir / "baseline_cnn_training_summary.json"

    val_slice_pred_df.to_csv(val_slice_pred_path, index=False, encoding="utf-8")
    val_group_pred_df.to_csv(val_group_pred_path, index=False, encoding="utf-8")
    val_metrics_df.to_csv(val_metrics_path, index=False, encoding="utf-8")

    summary = {
        "model": "simple_cnn",
        "best_epoch": int(checkpoint["epoch"]),
        "best_val_loss": float(checkpoint["val_loss"]),
        "seed": int(config.seed),
        "checkpoint_path": str(checkpoint_path),
        "test_used_during_training": False,
        "statistical_baseline_reference": "docs/04_baseline_estatistico_controle.md",
    }

    with summary_path.open("w", encoding="utf-8") as file:
        json.dump(summary, file, ensure_ascii=False, indent=2)

    return {
        "history_df": history_df,
        "val_metrics_df": val_metrics_df,
        "summary": summary,
        "checkpoint_path": checkpoint_path,
        "output_paths": [
            history_path,
            val_slice_pred_path,
            val_group_pred_path,
            val_metrics_path,
            summary_path,
        ],
    }
