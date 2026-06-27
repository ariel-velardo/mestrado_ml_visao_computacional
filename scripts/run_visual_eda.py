from __future__ import annotations

from pathlib import Path

import pandas as pd

from liverct.visualization.eda_images import (
    build_summary_by_class,
    build_summary_by_split_class,
    plot_boxplot_by_class,
    plot_boxplot_by_split_class,
    plot_histogram_by_class,
    plot_sample_grid,
    prepare_audit_dataframe,
    sample_one_slice_per_group,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    interim_dir = PROJECT_ROOT / "data" / "interim"
    figures_dir = PROJECT_ROOT / "reports" / "figures"
    tables_dir = PROJECT_ROOT / "reports" / "tables"

    audit_path = interim_dir / "image_quality_audit.csv"

    if not audit_path.exists():
        raise FileNotFoundError(
            f"Arquivo não encontrado: {audit_path}\n"
            "Execute antes: python scripts/audit_images.py"
        )

    figures_dir.mkdir(parents=True, exist_ok=True)
    tables_dir.mkdir(parents=True, exist_ok=True)

    audit_df = pd.read_csv(audit_path)
    audit_df = prepare_audit_dataframe(audit_df)

    summary_by_class = build_summary_by_class(audit_df)
    summary_by_split_class = build_summary_by_split_class(audit_df)

    summary_by_class.to_csv(
        tables_dir / "eda_image_summary_by_class.csv",
        index=False,
        encoding="utf-8",
    )

    if not summary_by_split_class.empty:
        summary_by_split_class.to_csv(
            tables_dir / "eda_image_summary_by_split_class.csv",
            index=False,
            encoding="utf-8",
        )

    samples_by_class = sample_one_slice_per_group(
        audit_df,
        group_columns=["class_name"],
        n_per_group=6,
        seed=42,
    )

    plot_sample_grid(
        samples_by_class,
        output_path=figures_dir / "eda_samples_by_class.png",
        n_cols=6,
        title="Amostras por classe",
    )

    if "split" in audit_df.columns:
        samples_by_split_class = sample_one_slice_per_group(
            audit_df,
            group_columns=["split", "class_name"],
            n_per_group=4,
            seed=42,
        )

        plot_sample_grid(
            samples_by_split_class,
            output_path=figures_dir / "eda_samples_by_split_class.png",
            n_cols=4,
            title="Amostras por split e classe",
        )

    plot_histogram_by_class(
        audit_df,
        value_column="mean_intensity",
        output_path=figures_dir / "eda_mean_intensity_hist_by_class.png",
        title="Distribuição da intensidade média por classe",
        xlabel="Intensidade média do pixel",
    )

    plot_histogram_by_class(
        audit_df,
        value_column="file_size_bytes",
        output_path=figures_dir / "eda_file_size_hist_by_class.png",
        title="Distribuição do tamanho de arquivo por classe",
        xlabel="Tamanho do arquivo em bytes",
    )

    plot_boxplot_by_class(
        audit_df,
        value_column="mean_intensity",
        output_path=figures_dir / "eda_mean_intensity_boxplot_by_class.png",
        title="Intensidade média por classe",
        ylabel="Intensidade média do pixel",
    )

    plot_boxplot_by_class(
        audit_df,
        value_column="file_size_bytes",
        output_path=figures_dir / "eda_file_size_boxplot_by_class.png",
        title="Tamanho de arquivo por classe",
        ylabel="Tamanho do arquivo em bytes",
    )

    if "split" in audit_df.columns:
        plot_boxplot_by_split_class(
            audit_df,
            value_column="mean_intensity",
            output_path=figures_dir / "eda_mean_intensity_boxplot_by_split_class.png",
            title="Intensidade média por split e classe",
            ylabel="Intensidade média do pixel",
        )

        plot_boxplot_by_split_class(
            audit_df,
            value_column="file_size_bytes",
            output_path=figures_dir / "eda_file_size_boxplot_by_split_class.png",
            title="Tamanho de arquivo por split e classe",
            ylabel="Tamanho do arquivo em bytes",
        )

    print("\n=== VISUAL EDA GERADA ===")
    print(f"Tabelas em: {tables_dir}")
    print(f"Figuras em: {figures_dir}")

    print("\nArquivos principais:")
    print(f"  - {tables_dir / 'eda_image_summary_by_class.csv'}")
    print(f"  - {tables_dir / 'eda_image_summary_by_split_class.csv'}")
    print(f"  - {figures_dir / 'eda_samples_by_class.png'}")
    print(f"  - {figures_dir / 'eda_samples_by_split_class.png'}")
    print(f"  - {figures_dir / 'eda_mean_intensity_hist_by_class.png'}")
    print(f"  - {figures_dir / 'eda_file_size_hist_by_class.png'}")


if __name__ == "__main__":
    main()
