from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from mcsr.utils.test_expr import test_pickled_expressions


def clean_evaluation_data(results: list[dict], min_r2: float) -> pd.DataFrame:
    df = pd.DataFrame(results)
    if df.empty:
        return df

    df["test_r2_clipped"] = df["test_r2"].clip(lower=min_r2)
    df["structural_similarity"] = 1.0 - df["ned"]

    return df


def plot_r2_heatmap(df: pd.DataFrame, output_dir: Path) -> None:
    pivot_data = df.pivot(index="name", columns="method", values="test_r2_clipped")

    plt.figure(figsize=(10, 8))
    sns.heatmap(
        pivot_data,
        annot=True,
        cmap="viridis",
        fmt=".3f",
        linewidths=0.5,
        vmin=0.0,
        vmax=1.0,
    )
    plt.title("Test R² Heatmap (Equation vs Method)", pad=15)
    plt.ylabel("Equation")
    plt.xlabel("Method")
    plt.tight_layout()
    plt.savefig(output_dir / "heatmap.png", dpi=300)
    plt.close()


def plot_ned_heatmap(df: pd.DataFrame, output_dir: Path) -> None:
    pivot_data = df.pivot(index="name", columns="method", values="ned")

    plt.figure(figsize=(10, 8))
    sns.heatmap(
        pivot_data,
        annot=True,
        cmap="viridis",
        fmt=".3f",
        linewidths=0.5,
        vmin=0.0,
        vmax=1.0,
    )
    plt.title("NED Heatmap (Equation vs Method)", pad=15)
    plt.ylabel("Equation")
    plt.xlabel("Method")
    plt.tight_layout()
    plt.savefig(output_dir / "ned_heatmap.png", dpi=300)
    plt.close()


def plot_r2_bar_chart(df: pd.DataFrame, output_dir: Path) -> None:
    plt.figure(figsize=(14, 6))
    sns.barplot(data=df, x="name", y="test_r2_clipped", hue="method", palette="deep")
    plt.xticks(rotation=45, ha="right")
    plt.title("Test R² per Equation", pad=15)
    plt.ylabel("Test R² (Clipped)")
    plt.xlabel("Equation")
    plt.legend(title="Method", bbox_to_anchor=(1.05, 1), loc="upper left")
    plt.tight_layout()
    plt.savefig(output_dir / "barplot_grouped_r2.png", dpi=300)
    plt.close()


def plot_ned_bar_chart(df: pd.DataFrame, output_dir: Path) -> None:
    plt.figure(figsize=(14, 6))
    sns.barplot(data=df, x="name", y="ned", hue="method", palette="rocket")
    plt.xticks(rotation=45, ha="right")
    plt.title("Normalized Edit Distance (NED) per Equation", pad=15)
    plt.ylabel("NED")
    plt.xlabel("Equation")
    plt.legend(title="Method", bbox_to_anchor=(1.05, 1), loc="upper left")
    plt.tight_layout()
    plt.savefig(output_dir / "barplot_grouped_ned.png", dpi=300)
    plt.close()


def plot_metrics_box_plot(df: pd.DataFrame, output_dir: Path) -> None:
    df_melted = df.melt(
        id_vars=["method"],
        value_vars=["test_r2_clipped", "structural_similarity"],
        var_name="Metric",
        value_name="Score",
    )

    df_melted["Metric"] = df_melted["Metric"].replace(
        {
            "test_r2_clipped": "Test R²",
            "structural_similarity": "Structural Sim. (1 - NED)",
        }
    )

    plt.figure(figsize=(10, 6))
    sns.boxplot(data=df_melted, x="method", y="Score", hue="Metric", palette="Set2")
    plt.title("Distributions of Test R² and Structural Similarity per Method", pad=15)
    plt.ylabel("Score")
    plt.xlabel("Method")
    plt.legend(title="Metric", bbox_to_anchor=(1.05, 1), loc="upper left")
    plt.tight_layout()
    plt.savefig(output_dir / "boxplot_metrics.png", dpi=300)
    plt.close()


def generate_plots(results: list[dict], output_path: str | Path) -> None:
    output_dir = Path(output_path)
    output_dir.mkdir(parents=True, exist_ok=True)

    df = clean_evaluation_data(results, min_r2=-0.5)

    if df.empty:
        print("No data available to plot.")
        return

    sns.set_theme(style="whitegrid", context="notebook", font_scale=1.1)

    plot_r2_heatmap(df, output_dir)
    plot_r2_bar_chart(df, output_dir)
    plot_ned_heatmap(df, output_dir)
    plot_ned_bar_chart(df, output_dir)
    plot_metrics_box_plot(df, output_dir)


def main() -> None:
    base_pickles_path = Path("artifacts/pickles")
    figures_path = Path("results") / "figures"

    all_evaluation_results = []

    for model_dir in base_pickles_path.iterdir():
        print(f"Evaluating model: {model_dir.name}...")
        model_results = test_pickled_expressions(model_dir.name)
        all_evaluation_results.extend(model_results)

    print("Generating plots...")
    generate_plots(all_evaluation_results, figures_path)
    print(
        f"Plots saved at {figures_path}: heatmap.png, barplot_grouped_r2.png, barplot_grouped_ned.png, boxplot_metrics.png"
    )


if __name__ == "__main__":
    main()
