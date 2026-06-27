from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


FEATURE_COLUMNS = [
    "mean_intensity",
    "std_intensity",
    "min_intensity",
    "max_intensity",
]

TARGET_COLUMN = "label"
GROUP_COLUMN = "inferred_group_id"
SPLIT_COLUMN = "split"


def prepare_statistical_baseline_data(
    audit_df: pd.DataFrame,
    feature_columns: list[str] | None = None,
) -> pd.DataFrame:
    """
    Prepara dataframe para o baseline estatístico.

    Este baseline usa apenas estatísticas simples de intensidade dos pixels.
    Não usa caminho, nome de arquivo, tamanho de arquivo ou identificador do grupo
    como variável preditora.
    """
    feature_columns = feature_columns or FEATURE_COLUMNS

    required_columns = set(feature_columns + [TARGET_COLUMN, GROUP_COLUMN, SPLIT_COLUMN])
    missing = required_columns - set(audit_df.columns)

    if missing:
        raise ValueError(f"Colunas ausentes no dataframe: {sorted(missing)}")

    df = audit_df.copy()

    for column in feature_columns + [TARGET_COLUMN]:
        df[column] = pd.to_numeric(df[column], errors="coerce")

    df = df.dropna(subset=feature_columns + [TARGET_COLUMN, GROUP_COLUMN, SPLIT_COLUMN])
    df[TARGET_COLUMN] = df[TARGET_COLUMN].astype(int)

    return df.reset_index(drop=True)


def build_logistic_regression_model() -> Pipeline:
    """
    Cria modelo simples de regressão logística com padronização.

    class_weight='balanced' é usado porque há desbalanceamento entre classes.
    """
    return Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            (
                "model",
                LogisticRegression(
                    max_iter=1000,
                    class_weight="balanced",
                    random_state=42,
                ),
            ),
        ]
    )


def build_dummy_model(strategy: str = "most_frequent") -> DummyClassifier:
    """
    Cria modelo dummy para comparação.
    """
    return DummyClassifier(strategy=strategy, random_state=42)


def train_model(
    model: Any,
    train_df: pd.DataFrame,
    feature_columns: list[str] | None = None,
) -> Any:
    """
    Treina modelo usando apenas o split de treino.
    """
    feature_columns = feature_columns or FEATURE_COLUMNS

    x_train = train_df[feature_columns]
    y_train = train_df[TARGET_COLUMN]

    model.fit(x_train, y_train)

    return model


def predict_probabilities(
    model: Any,
    df: pd.DataFrame,
    feature_columns: list[str] | None = None,
) -> pd.DataFrame:
    """
    Gera probabilidades por slice.
    """
    feature_columns = feature_columns or FEATURE_COLUMNS

    output = df.copy()
    x = output[feature_columns]

    if hasattr(model, "predict_proba"):
        output["prob_positive"] = model.predict_proba(x)[:, 1]
    else:
        output["prob_positive"] = model.predict(x)

    output["pred_label"] = (output["prob_positive"] >= 0.5).astype(int)

    return output


def aggregate_predictions_by_group(pred_df: pd.DataFrame) -> pd.DataFrame:
    """
    Agrega probabilidades por inferred_group_id.

    A predição final do grupo é a média das probabilidades dos slices.
    """
    required_columns = {
        GROUP_COLUMN,
        SPLIT_COLUMN,
        TARGET_COLUMN,
        "prob_positive",
        "pred_label",
    }

    missing = required_columns - set(pred_df.columns)

    if missing:
        raise ValueError(f"Colunas ausentes em pred_df: {sorted(missing)}")

    group_df = (
        pred_df.groupby([SPLIT_COLUMN, GROUP_COLUMN])
        .agg(
            label=(TARGET_COLUMN, "first"),
            prob_positive=("prob_positive", "mean"),
            n_slices=("prob_positive", "size"),
            slice_positive_rate=("pred_label", "mean"),
        )
        .reset_index()
    )

    group_df["pred_label"] = (group_df["prob_positive"] >= 0.5).astype(int)

    return group_df


def evaluate_binary_predictions(
    y_true: pd.Series,
    y_pred: pd.Series,
    y_score: pd.Series,
) -> dict[str, float | int]:
    """
    Calcula métricas binárias.

    label 1 = Hepatic_Steatosis
    label 0 = Healthy
    """
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()

    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0

    metrics: dict[str, float | int] = {
        "n": int(len(y_true)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall_sensitivity": float(recall_score(y_true, y_pred, zero_division=0)),
        "specificity": float(specificity),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
    }

    if len(set(y_true)) > 1:
        metrics["roc_auc"] = float(roc_auc_score(y_true, y_score))
        metrics["average_precision"] = float(average_precision_score(y_true, y_score))
    else:
        metrics["roc_auc"] = float("nan")
        metrics["average_precision"] = float("nan")

    return metrics


def evaluate_by_split(
    pred_df: pd.DataFrame,
    level: str,
) -> pd.DataFrame:
    """
    Avalia predições por split.

    level:
    - 'slice'
    - 'group'
    """
    rows: list[dict[str, Any]] = []

    for split_name, split_df in pred_df.groupby(SPLIT_COLUMN, sort=True):
        metrics = evaluate_binary_predictions(
            y_true=split_df[TARGET_COLUMN],
            y_pred=split_df["pred_label"],
            y_score=split_df["prob_positive"],
        )

        metrics["split"] = split_name
        metrics["level"] = level

        rows.append(metrics)

    return pd.DataFrame(rows)


def run_statistical_baseline(
    audit_df: pd.DataFrame,
    output_dir: str | Path,
    feature_columns: list[str] | None = None,
) -> dict[str, Any]:
    """
    Executa baseline estatístico completo.

    Modelos:
    - Dummy most_frequent
    - Regressão logística

    Avaliações:
    - por slice
    - por inferred_group_id
    """
    feature_columns = feature_columns or FEATURE_COLUMNS
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    df = prepare_statistical_baseline_data(audit_df, feature_columns=feature_columns)

    train_df = df[df[SPLIT_COLUMN] == "train"].copy()

    if train_df.empty:
        raise ValueError("Split de treino vazio.")

    models = {
        "dummy_most_frequent": build_dummy_model(strategy="most_frequent"),
        "logistic_regression": build_logistic_regression_model(),
    }

    all_metrics = []
    output_summary: dict[str, Any] = {
        "features": feature_columns,
        "models": {},
    }

    for model_name, model in models.items():
        fitted_model = train_model(model, train_df, feature_columns=feature_columns)

        pred_df = predict_probabilities(
            fitted_model,
            df,
            feature_columns=feature_columns,
        )

        group_pred_df = aggregate_predictions_by_group(pred_df)

        slice_metrics = evaluate_by_split(pred_df, level="slice")
        group_metrics = evaluate_by_split(group_pred_df, level="group")

        slice_metrics["model"] = model_name
        group_metrics["model"] = model_name

        model_metrics = pd.concat([slice_metrics, group_metrics], ignore_index=True)
        all_metrics.append(model_metrics)

        pred_df.to_csv(
            output_path / f"{model_name}_slice_predictions.csv",
            index=False,
            encoding="utf-8",
        )

        group_pred_df.to_csv(
            output_path / f"{model_name}_group_predictions.csv",
            index=False,
            encoding="utf-8",
        )

        output_summary["models"][model_name] = {
            "n_slice_predictions": int(len(pred_df)),
            "n_group_predictions": int(len(group_pred_df)),
        }

    metrics_df = pd.concat(all_metrics, ignore_index=True)

    ordered_columns = [
        "model",
        "level",
        "split",
        "n",
        "accuracy",
        "balanced_accuracy",
        "precision",
        "recall_sensitivity",
        "specificity",
        "f1",
        "roc_auc",
        "average_precision",
        "tn",
        "fp",
        "fn",
        "tp",
    ]

    metrics_df = metrics_df[ordered_columns]

    metrics_df.to_csv(
        output_path / "statistical_baseline_metrics.csv",
        index=False,
        encoding="utf-8",
    )

    with (output_path / "statistical_baseline_summary.json").open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(output_summary, file, ensure_ascii=False, indent=2)

    return {
        "metrics_df": metrics_df,
        "summary": output_summary,
    }
