import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.metrics import RocCurveDisplay
from sklearn.preprocessing import LabelEncoder


def main():
    parser = argparse.ArgumentParser(description="根据定量和统计结果生成分析图表。")
    parser.add_argument("--input-csv", required=True, help="quantify_fat.py 输出的汇总 CSV。")
    parser.add_argument("--stats-dir", required=True, help="analyze_statistics.py 输出目录。")
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()

    df = pd.read_csv(args.input_csv)
    stats_dir = Path(args.stats_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    sns.set_theme(style="whitegrid", context="paper")

    draw_correlation_heatmap(df, output_dir / "fat_correlation_matrix.png")
    draw_boxplots(df, output_dir)
    draw_auc_barplot(stats_dir / "random_forest_auc.csv", output_dir / "random_forest_auc.png")
    draw_roc_curves(stats_dir / "random_forest_probabilities.csv", output_dir / "random_forest_roc.png")

    print(f"图表已保存到: {output_dir}")


def draw_correlation_heatmap(df, output_path):
    """各组内脂肪分数 Spearman 相关热图。"""
    metrics = ["total_fat_fraction", "head_fat_fraction", "body_fat_fraction", "tail_fat_fraction"]
    groups = list(df["group"].dropna().unique())
    fig, axes = plt.subplots(1, len(groups), figsize=(4 * len(groups), 3.5), squeeze=False)

    for ax, group in zip(axes[0], groups):
        corr = df.loc[df["group"] == group, metrics].corr(method="spearman")
        sns.heatmap(corr, vmin=-1, vmax=1, cmap="coolwarm", annot=True, fmt=".2f", ax=ax)
        ax.set_title(group)

    fig.tight_layout()
    fig.savefig(output_path, dpi=300)
    plt.close(fig)


def draw_boxplots(df, output_dir):
    """头/体/尾体积、脂肪分数的组间箱线图。"""
    volume_cols = ["head_volume_ml", "body_volume_ml", "tail_volume_ml"]
    fat_cols = ["head_fat_fraction", "body_fat_fraction", "tail_fat_fraction"]

    for cols, name, ylabel in [
        (volume_cols, "regional_volume_boxplots.png", "Volume (mL)"),
        (fat_cols, "regional_fat_boxplots.png", "Fat fraction"),
    ]:
        long_df = df.melt(id_vars=["patient_id", "group"], value_vars=cols, var_name="region", value_name="value")
        fig, ax = plt.subplots(figsize=(7, 4))
        sns.boxplot(data=long_df, x="region", y="value", hue="group", ax=ax)
        ax.set_xlabel("")
        ax.set_ylabel(ylabel)
        ax.tick_params(axis="x", rotation=20)
        fig.tight_layout()
        fig.savefig(output_dir / name, dpi=300)
        plt.close(fig)


def draw_auc_barplot(auc_csv, output_path):
    """随机森林各特征组合的 macro AUC 柱状图。"""
    auc_df = pd.read_csv(auc_csv)
    fig, ax = plt.subplots(figsize=(6, 4))
    sns.barplot(data=auc_df, x="feature_set", y="macro_ovr_auc", ax=ax, color="#4C78A8")
    ax.set_ylim(0.5, 1.0)
    ax.set_xlabel("")
    ax.set_ylabel("Macro one-vs-rest AUC")
    ax.tick_params(axis="x", rotation=25)
    fig.tight_layout()
    fig.savefig(output_path, dpi=300)
    plt.close(fig)


def draw_roc_curves(prob_csv, output_path):
    """用 CV 预测概率画多分类 one-vs-rest ROC。"""
    prob_df = pd.read_csv(prob_csv)
    encoder = LabelEncoder()
    encoder.fit(prob_df["group"])

    fig, ax = plt.subplots(figsize=(6, 5))
    for feature_set, sub in prob_df.groupby("feature_set"):
        # one-vs-rest：逐类当阳性画 ROC
        class_names = list(encoder.classes_)
        for class_name in class_names:
            score_col = f"prob_{class_name}"
            if score_col not in sub.columns:
                continue
            y_binary = (sub["group"] == class_name).astype(int)
            RocCurveDisplay.from_predictions(y_binary, sub[score_col], name=f"{feature_set}-{class_name}", ax=ax)

    fig.tight_layout()
    fig.savefig(output_path, dpi=300)
    plt.close(fig)


if __name__ == "__main__":
    main()
