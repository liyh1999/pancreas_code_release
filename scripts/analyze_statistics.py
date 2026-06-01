import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import kruskal, spearmanr
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.preprocessing import LabelEncoder


FEATURE_SETS = {
    "total_fat": ["total_fat_fraction"],
    "head_fat": ["head_fat_fraction"],
    "body_fat": ["body_fat_fraction"],
    "tail_fat": ["tail_fat_fraction"],
    "combined_fat": ["total_fat_fraction", "head_fat_fraction", "body_fat_fraction", "tail_fat_fraction"],
}


def main():
    parser = argparse.ArgumentParser(description="完成组间统计检验、相关分析和随机森林分类评估。")
    parser.add_argument("--input-csv", required=True, help="quantify_fat.py 输出的汇总 CSV。")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n-splits", type=int, default=5)
    args = parser.parse_args()

    df = pd.read_csv(args.input_csv)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    metric_cols = [
        "head_volume_ml",
        "body_volume_ml",
        "tail_volume_ml",
        "head_fat_fraction",
        "body_fat_fraction",
        "tail_fat_fraction",
    ]

    kw_rows = kruskal_by_group(df, metric_cols)
    pd.DataFrame(kw_rows).to_csv(output_dir / "kruskal_wallis_results.csv", index=False)

    corr_rows = spearman_by_group(df)
    pd.DataFrame(corr_rows).to_csv(output_dir / "spearman_correlations.csv", index=False)

    auc_rows, prob_df = random_forest_auc(df, args.seed, args.n_splits)
    pd.DataFrame(auc_rows).to_csv(output_dir / "random_forest_auc.csv", index=False)
    prob_df.to_csv(output_dir / "random_forest_probabilities.csv", index=False)

    print(f"统计分析结果已保存到: {output_dir}")


def kruskal_by_group(df, metric_cols):
    """各体积/脂肪指标做 Kruskal-Wallis H 检验。"""
    rows = []
    groups = [g for _, g in df.groupby("group")]
    for col in metric_cols:
        values = [g[col].dropna().to_numpy() for g in groups]
        stat, p_value = kruskal(*values)
        rows.append({"metric": col, "statistic": stat, "p_value": p_value})
    return rows


def spearman_by_group(df):
    """组内 total 与各区域脂肪分数的 Spearman 相关。"""
    rows = []
    for group, sub in df.groupby("group"):
        for region_col in ["head_fat_fraction", "body_fat_fraction", "tail_fat_fraction"]:
            valid = sub[["total_fat_fraction", region_col]].dropna()
            if len(valid) < 3:
                rho, p_value = np.nan, np.nan
            else:
                rho, p_value = spearmanr(valid["total_fat_fraction"], valid[region_col])
            rows.append(
                {
                    "group": group,
                    "region_metric": region_col,
                    "rho": rho,
                    "p_value": p_value,
                    "n": len(valid),
                }
            )
    return rows


def random_forest_auc(df, seed, n_splits):
    """CV 预测概率 + 多分类 one-vs-rest macro ROC-AUC。"""
    encoder = LabelEncoder()
    y = encoder.fit_transform(df["group"])
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)

    auc_rows = []
    prob_tables = []

    for name, features in FEATURE_SETS.items():
        clean = df[["patient_id", "group", *features]].dropna().reset_index(drop=True)
        y_clean = encoder.fit_transform(clean["group"])
        x = clean[features].to_numpy()

        model = RandomForestClassifier(
            n_estimators=500,
            random_state=seed,
            class_weight="balanced",
            min_samples_leaf=2,
        )
        probabilities = cross_val_predict(model, x, y_clean, cv=cv, method="predict_proba")
        auc = roc_auc_score(y_clean, probabilities, multi_class="ovr", average="macro")
        auc_rows.append({"feature_set": name, "features": ",".join(features), "macro_ovr_auc": auc, "n": len(clean)})

        prob = clean[["patient_id", "group"]].copy()
        prob["feature_set"] = name
        for class_index, class_name in enumerate(encoder.classes_):
            prob[f"prob_{class_name}"] = probabilities[:, class_index]
        prob_tables.append(prob)

    return auc_rows, pd.concat(prob_tables, ignore_index=True)


if __name__ == "__main__":
    main()
