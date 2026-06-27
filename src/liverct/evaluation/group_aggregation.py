from __future__ import annotations

import pandas as pd


def aggregate_probabilities_by_group(
    pred_df: pd.DataFrame,
    group_col: str = "inferred_group_id",
    label_col: str = "label",
    prob_col: str = "prob_positive",
    split_col: str = "split",
    threshold: float = 0.5,
) -> pd.DataFrame:
    """
    Aggregate slice probabilities by inferred technical group.

    The group probability is the mean positive-class probability across slices.
    The group identifier is technical and must not be interpreted as a
    clinically validated patient_id.
    """
    required_columns = {group_col, label_col, prob_col, split_col}
    missing = required_columns - set(pred_df.columns)
    if missing:
        raise ValueError(f"Missing columns in pred_df: {sorted(missing)}")

    label_counts = pred_df.groupby(group_col)[label_col].nunique()
    inconsistent_groups = label_counts[label_counts > 1]
    if not inconsistent_groups.empty:
        raise ValueError(
            "Groups with inconsistent labels found: "
            f"{inconsistent_groups.index.tolist()[:10]}"
        )

    group_df = (
        pred_df.groupby([split_col, group_col], sort=True)
        .agg(
            label=(label_col, "first"),
            prob_positive=(prob_col, "mean"),
            n_slices=(prob_col, "size"),
            prob_positive_std=(prob_col, "std"),
        )
        .reset_index()
    )

    group_df["prob_positive_std"] = group_df["prob_positive_std"].fillna(0.0)
    group_df["pred_label"] = (group_df["prob_positive"] >= threshold).astype(int)

    return group_df
