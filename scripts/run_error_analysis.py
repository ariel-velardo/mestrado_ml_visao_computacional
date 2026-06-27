"""
Análise visual de erros do baseline CNN 2D.

Objetivo
--------
Selecionar casos representativos do conjunto de teste para análise visual:
- falsos positivos;
- falsos negativos;
- verdadeiros positivos;
- verdadeiros negativos.

Esta etapa ainda NÃO aplica Grad-CAM. Ela prepara os casos que serão usados
na etapa seguinte de interpretabilidade.

Entradas
--------
1. reports/tables/baseline_cnn_test_slice_predictions.csv
   Predições da CNN no teste em nível de slice.

2. reports/tables/baseline_cnn_test_group_predictions.csv
   Predições da CNN no teste agregadas por inferred_group_id.

3. data/interim/split_slices.csv
   Índice completo dos slices, com filename, slice_id, class_name e file_path.

Saídas
------
1. reports/tables/error_analysis_group_cases.csv
   Tabela com grupos selecionados para análise visual.

2. reports/tables/error_analysis_slice_cases.csv
   Tabela com slices representativos de cada grupo selecionado.

3. reports/figures/error_analysis/*.png
   Figuras com painéis de slices por grupo.

Observações metodológicas
-------------------------
- O script usa apenas o split de teste.
- O script não altera splits.
- O script não treina modelo.
- O script não usa file_path, filename, slice_id ou inferred_group_id como variável preditora.
- O objetivo é análise pós-modelagem.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from PIL import Image


PROJECT_ROOT = Path(__file__).resolve().parents[1]

SLICE_PRED_PATH = PROJECT_ROOT / "reports" / "tables" / "baseline_cnn_test_slice_predictions.csv"
GROUP_PRED_PATH = PROJECT_ROOT / "reports" / "tables" / "baseline_cnn_test_group_predictions.csv"
SPLIT_SLICES_PATH = PROJECT_ROOT / "data" / "interim" / "split_slices.csv"

OUTPUT_TABLES_DIR = PROJECT_ROOT / "reports" / "tables"
OUTPUT_FIGURES_DIR = PROJECT_ROOT / "reports" / "figures" / "error_analysis"

GROUP_CASES_PATH = OUTPUT_TABLES_DIR / "error_analysis_group_cases.csv"
SLICE_CASES_PATH = OUTPUT_TABLES_DIR / "error_analysis_slice_cases.csv"


LABEL_NAME = {
    0: "Healthy",
    1: "Hepatic_Steatosis",
}


def classify_outcome(row: pd.Series) -> str:
    """Classifica a predição em TP, TN, FP ou FN."""
    label = int(row["label"])
    pred = int(row["pred_label"])

    if label == 1 and pred == 1:
        return "TP"
    if label == 0 and pred == 0:
        return "TN"
    if label == 0 and pred == 1:
        return "FP"
    if label == 1 and pred == 0:
        return "FN"

    raise ValueError(f"Combinação inválida: label={label}, pred_label={pred}")


def validate_inputs() -> None:
    """Valida existência dos arquivos de entrada."""
    required = [
        SLICE_PRED_PATH,
        GROUP_PRED_PATH,
        SPLIT_SLICES_PATH,
    ]

    missing = [path for path in required if not path.exists()]

    if missing:
        missing_text = "\n".join(str(path) for path in missing)
        raise FileNotFoundError(f"Arquivos de entrada não encontrados:\n{missing_text}")


def select_group_cases(group_df: pd.DataFrame, max_per_outcome: int = 3) -> pd.DataFrame:
    """
    Seleciona grupos representativos por tipo de resultado.

    Critérios:
    - FP: maiores probabilidades positivas entre negativos reais.
    - FN: menores probabilidades positivas entre positivos reais.
    - TP: maiores probabilidades positivas entre positivos reais.
    - TN: menores probabilidades positivas entre negativos reais.
    """
    selected = []

    sort_rules = {
        "FP": ("prob_positive", False),
        "FN": ("prob_positive", True),
        "TP": ("prob_positive", False),
        "TN": ("prob_positive", True),
    }

    for outcome, (sort_col, ascending) in sort_rules.items():
        subset = group_df[group_df["outcome"] == outcome].copy()

        if subset.empty:
            continue

        subset = subset.sort_values(sort_col, ascending=ascending).head(max_per_outcome)
        selected.append(subset)

    if not selected:
        return pd.DataFrame(columns=group_df.columns)

    result = pd.concat(selected, ignore_index=True)

    result["label_name"] = result["label"].map(LABEL_NAME)
    result["pred_label_name"] = result["pred_label"].map(LABEL_NAME)

    ordered_cols = [
        "outcome",
        "split",
        "inferred_group_id",
        "label",
        "label_name",
        "pred_label",
        "pred_label_name",
        "prob_positive",
        "prob_positive_std",
        "n_slices",
    ]

    return result[ordered_cols].sort_values(["outcome", "inferred_group_id"])


def select_slice_cases(
    slice_df: pd.DataFrame,
    group_cases: pd.DataFrame,
    max_slices_per_group: int = 6,
) -> pd.DataFrame:
    """
    Seleciona slices representativos de cada grupo.

    Para cada grupo, os slices são ordenados por probabilidade positiva.
    Selecionamos posições igualmente espaçadas na distribuição de probabilidades
    para obter uma visão do grupo inteiro, não apenas do slice mais extremo.
    """
    selected_rows = []

    for _, group_row in group_cases.iterrows():
        group_id = group_row["inferred_group_id"]
        group_slices = slice_df[slice_df["inferred_group_id"] == group_id].copy()

        if group_slices.empty:
            continue

        group_slices = group_slices.sort_values("prob_positive").reset_index(drop=True)

        n = len(group_slices)
        if n <= max_slices_per_group:
            positions = list(range(n))
        else:
            positions = sorted(
                set(
                    round(i * (n - 1) / (max_slices_per_group - 1))
                    for i in range(max_slices_per_group)
                )
            )

        selected = group_slices.iloc[positions].copy()
        selected["group_outcome"] = group_row["outcome"]
        selected["group_prob_positive"] = group_row["prob_positive"]
        selected["group_pred_label"] = group_row["pred_label"]
        selected["group_label"] = group_row["label"]

        selected_rows.append(selected)

    if not selected_rows:
        return pd.DataFrame()

    result = pd.concat(selected_rows, ignore_index=True)

    result["label_name"] = result["label"].map(LABEL_NAME)
    result["pred_label_name"] = result["pred_label"].map(LABEL_NAME)

    ordered_cols = [
        "group_outcome",
        "split",
        "inferred_group_id",
        "class_name",
        "filename",
        "slice_id",
        "label",
        "label_name",
        "pred_label",
        "pred_label_name",
        "prob_positive",
        "group_prob_positive",
        "group_label",
        "group_pred_label",
        "file_path",
    ]

    available_cols = [col for col in ordered_cols if col in result.columns]
    return result[available_cols].sort_values(
        ["group_outcome", "inferred_group_id", "prob_positive"]
    )


def plot_group_panel(group_id: str, slice_cases: pd.DataFrame, output_path: Path) -> None:
    """Gera painel visual para um grupo selecionado."""
    group_slices = slice_cases[slice_cases["inferred_group_id"] == group_id].copy()

    if group_slices.empty:
        return

    group_slices = group_slices.sort_values("prob_positive")

    outcome = group_slices["group_outcome"].iloc[0]
    true_name = group_slices["label_name"].iloc[0]
    pred_name = group_slices["pred_label_name"].iloc[0]
    group_prob = group_slices["group_prob_positive"].iloc[0]

    n = len(group_slices)
    fig, axes = plt.subplots(1, n, figsize=(3.2 * n, 3.8))

    if n == 1:
        axes = [axes]

    for ax, (_, row) in zip(axes, group_slices.iterrows()):
        image_path = Path(row["file_path"])

        image = Image.open(image_path).convert("L")

        ax.imshow(image, cmap="gray")
        ax.axis("off")
        ax.set_title(
            f"slice={row.get('slice_id', '')}\n"
            f"p={row['prob_positive']:.3f}",
            fontsize=9,
        )

    fig.suptitle(
        f"{outcome} | {group_id} | real={true_name} | pred={pred_name} | "
        f"p_grupo={group_prob:.3f}",
        fontsize=12,
    )

    fig.tight_layout()
    fig.savefig(output_path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    validate_inputs()

    OUTPUT_TABLES_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    print("=== ENTRADAS ===")
    print(f"Slice predictions: {SLICE_PRED_PATH}")
    print(f"Group predictions: {GROUP_PRED_PATH}")
    print(f"Split slices:       {SPLIT_SLICES_PATH}")

    slice_pred = pd.read_csv(SLICE_PRED_PATH)
    group_pred = pd.read_csv(GROUP_PRED_PATH)
    split_slices = pd.read_csv(SPLIT_SLICES_PATH)

    split_slices_test = split_slices[split_slices["split"] == "test"].copy()

    group_pred = group_pred[group_pred["split"] == "test"].copy()
    slice_pred = slice_pred[slice_pred["split"] == "test"].copy()

    group_pred["outcome"] = group_pred.apply(classify_outcome, axis=1)

    slice_pred = slice_pred.merge(
        split_slices_test[
            [
                "file_path",
                "filename",
                "class_name",
                "slice_id",
                "file_size_bytes",
                "width",
                "height",
            ]
        ],
        on="file_path",
        how="left",
    )

    group_cases = select_group_cases(group_pred, max_per_outcome=3)
    slice_cases = select_slice_cases(slice_pred, group_cases, max_slices_per_group=6)

    group_cases.to_csv(GROUP_CASES_PATH, index=False)
    slice_cases.to_csv(SLICE_CASES_PATH, index=False)

    generated_figures = []

    for group_id in group_cases["inferred_group_id"].unique():
        outcome = group_cases.loc[
            group_cases["inferred_group_id"] == group_id, "outcome"
        ].iloc[0]

        output_path = OUTPUT_FIGURES_DIR / f"{outcome}_{group_id}.png"
        plot_group_panel(group_id, slice_cases, output_path)
        generated_figures.append(output_path)

    print("\n=== RESUMO DOS GRUPOS NO TESTE ===")
    print(group_pred["outcome"].value_counts().sort_index().to_string())

    print("\n=== GRUPOS SELECIONADOS ===")
    print(
        group_cases[
            [
                "outcome",
                "inferred_group_id",
                "label_name",
                "pred_label_name",
                "prob_positive",
                "n_slices",
            ]
        ].to_string(index=False)
    )

    print("\n=== SAÍDAS ===")
    print(f"Group cases: {GROUP_CASES_PATH}")
    print(f"Slice cases: {SLICE_CASES_PATH}")
    print(f"Figures dir: {OUTPUT_FIGURES_DIR}")
    print(f"Figures generated: {len(generated_figures)}")

    for path in generated_figures:
        print(f" - {path}")


if __name__ == "__main__":
    main()
