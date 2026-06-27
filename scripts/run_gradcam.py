"""
Executa Grad-CAM nos casos selecionados pela análise visual de erros.

Objetivo
--------
Aplicar Grad-CAM sobre slices representativos dos grupos selecionados na etapa
de análise visual de erros, permitindo investigar se a CNN está focando em
regiões plausíveis ou em artefatos visuais.

Entradas
--------
1. models/checkpoints/baseline_cnn_best.pt
   Checkpoint treinado do baseline CNN 2D.

2. reports/tables/error_analysis_slice_cases.csv
   Tabela de slices selecionados para análise, gerada por:
   scripts/run_error_analysis.py

3. Imagens originais
   Caminhos indicados na coluna file_path.

Saídas
------
1. reports/tables/gradcam_cases.csv
   Tabela com os slices processados e probabilidades recalculadas.

2. reports/figures/gradcam/*.png
   Figuras com imagem original, heatmap Grad-CAM e sobreposição.

Observações metodológicas
-------------------------
- O script não treina modelo.
- O script não altera splits.
- O script usa apenas casos já selecionados do conjunto de teste.
- Grad-CAM é uma técnica de interpretabilidade visual aproximada.
- Os mapas não provam causalidade nem validade clínica.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from PIL import Image

from liverct.explainability.gradcam import GradCAM
from liverct.models.simple_cnn import SimpleCNN2D


PROJECT_ROOT = Path(__file__).resolve().parents[1]

CHECKPOINT_PATH = PROJECT_ROOT / "models" / "checkpoints" / "baseline_cnn_best.pt"
INPUT_CASES_PATH = PROJECT_ROOT / "reports" / "tables" / "error_analysis_slice_cases.csv"

OUTPUT_TABLES_DIR = PROJECT_ROOT / "reports" / "tables"
OUTPUT_FIGURES_DIR = PROJECT_ROOT / "reports" / "figures" / "gradcam"

OUTPUT_CASES_PATH = OUTPUT_TABLES_DIR / "gradcam_cases.csv"

IMAGE_SIZE = 256

LABEL_NAME = {
    0: "Healthy",
    1: "Hepatic_Steatosis",
}


def validate_inputs() -> None:
    """Valida arquivos de entrada."""
    required = [
        CHECKPOINT_PATH,
        INPUT_CASES_PATH,
    ]

    missing = [path for path in required if not path.exists()]

    if missing:
        missing_text = "\n".join(str(path) for path in missing)
        raise FileNotFoundError(f"Arquivos de entrada não encontrados:\n{missing_text}")


def load_model() -> tuple[SimpleCNN2D, dict]:
    """Carrega modelo SimpleCNN2D a partir do checkpoint."""
    checkpoint = torch.load(CHECKPOINT_PATH, map_location="cpu")

    if not isinstance(checkpoint, dict):
        raise TypeError("Checkpoint esperado como dict.")

    config = checkpoint.get("config", {})
    dropout = float(config.get("dropout", 0.25)) if isinstance(config, dict) else 0.25

    model = SimpleCNN2D(dropout=dropout)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    return model, checkpoint


def load_image_as_tensor(image_path: Path) -> tuple[torch.Tensor, np.ndarray]:
    """
    Carrega imagem em escala de cinza e retorna tensor e array para visualização.

    Retorno
    -------
    tensor:
        Tensor [1, 1, 256, 256], normalizado para [0, 1].

    image_array:
        Array [256, 256], normalizado para [0, 1].
    """
    image = Image.open(image_path).convert("L")
    image = image.resize((IMAGE_SIZE, IMAGE_SIZE))

    image_array = np.asarray(image).astype(np.float32) / 255.0

    tensor = torch.from_numpy(image_array).unsqueeze(0).unsqueeze(0).float()

    return tensor, image_array


def resize_heatmap_to_image(heatmap: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    """Redimensiona heatmap para o tamanho da imagem usando PIL."""
    heatmap_uint8 = np.uint8(255 * heatmap)
    heatmap_image = Image.fromarray(heatmap_uint8).resize(size, resample=Image.BILINEAR)
    return np.asarray(heatmap_image).astype(np.float32) / 255.0


def plot_gradcam_case(
    image_array: np.ndarray,
    heatmap: np.ndarray,
    row: pd.Series,
    output_path: Path,
    target_layer_name: str,
    recalculated_probability: float,
) -> None:
    """Gera figura com imagem original, heatmap e sobreposição."""
    heatmap_resized = resize_heatmap_to_image(
        heatmap,
        size=(image_array.shape[1], image_array.shape[0]),
    )

    outcome = row["group_outcome"]
    group_id = row["inferred_group_id"]
    filename = row["filename"]
    true_name = LABEL_NAME[int(row["label"])]
    pred_name = LABEL_NAME[int(row["pred_label"])]
    slice_prob = float(row["prob_positive"])
    group_prob = float(row["group_prob_positive"])

    fig, axes = plt.subplots(1, 3, figsize=(12, 4))

    axes[0].imshow(image_array, cmap="gray")
    axes[0].set_title("Imagem original")
    axes[0].axis("off")

    axes[1].imshow(heatmap_resized, cmap="jet")
    axes[1].set_title("Grad-CAM")
    axes[1].axis("off")

    axes[2].imshow(image_array, cmap="gray")
    axes[2].imshow(heatmap_resized, cmap="jet", alpha=0.45)
    axes[2].set_title("Sobreposição")
    axes[2].axis("off")

    fig.suptitle(
        f"{outcome} | {group_id} | {filename}\n"
        f"real={true_name} | pred={pred_name} | "
        f"p_slice_csv={slice_prob:.3f} | p_slice_recalc={recalculated_probability:.3f} | "
        f"p_grupo={group_prob:.3f} | camada={target_layer_name}",
        fontsize=10,
    )

    fig.tight_layout()
    fig.savefig(output_path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def make_output_filename(row: pd.Series) -> str:
    """Cria nome de arquivo estável para a figura."""
    outcome = row["group_outcome"]
    group_id = str(row["inferred_group_id"])
    filename_stem = Path(str(row["filename"])).stem

    return f"{outcome}_{group_id}_{filename_stem}_gradcam.png"


def main() -> None:
    validate_inputs()

    OUTPUT_TABLES_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    print("=== ENTRADAS ===")
    print(f"Checkpoint: {CHECKPOINT_PATH}")
    print(f"Slice cases: {INPUT_CASES_PATH}")

    model, checkpoint = load_model()
    gradcam = GradCAM(model)

    cases = pd.read_csv(INPUT_CASES_PATH)

    output_rows = []

    print("\n=== CHECKPOINT ===")
    print(f"model_name: {checkpoint.get('model_name')}")
    print(f"epoch: {checkpoint.get('epoch')}")
    print(f"val_loss: {checkpoint.get('val_loss')}")
    print(f"target_layer: {gradcam.target_layer_name}")

    print("\n=== PROCESSANDO CASOS ===")

    for _, row in cases.iterrows():
        image_path = Path(row["file_path"])

        if not image_path.exists():
            print(f"[AVISO] imagem não encontrada: {image_path}")
            continue

        image_tensor, image_array = load_image_as_tensor(image_path)

        result = gradcam(image_tensor)

        output_filename = make_output_filename(row)
        output_path = OUTPUT_FIGURES_DIR / output_filename

        plot_gradcam_case(
            image_array=image_array,
            heatmap=result.heatmap,
            row=row,
            output_path=output_path,
            target_layer_name=result.target_layer_name,
            recalculated_probability=result.probability,
        )

        output_row = row.to_dict()
        output_row["gradcam_probability"] = result.probability
        output_row["gradcam_logit"] = result.logit
        output_row["target_layer_name"] = result.target_layer_name
        output_row["gradcam_figure_path"] = str(output_path)

        output_rows.append(output_row)

        print(
            f"{row['group_outcome']} | {row['inferred_group_id']} | "
            f"{row['filename']} | p={result.probability:.4f} | {output_path.name}"
        )

    gradcam.remove_hooks()

    output_df = pd.DataFrame(output_rows)
    output_df.to_csv(OUTPUT_CASES_PATH, index=False)

    print("\n=== SAÍDAS ===")
    print(f"Grad-CAM cases: {OUTPUT_CASES_PATH}")
    print(f"Figures dir:    {OUTPUT_FIGURES_DIR}")
    print(f"Figures generated: {len(output_rows)}")


if __name__ == "__main__":
    main()
