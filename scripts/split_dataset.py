import argparse
from pathlib import Path

import pandas as pd
from sklearn.model_selection import StratifiedKFold


def main():
    parser = argparse.ArgumentParser(description="按患者级别生成分层五折交叉验证划分。")
    parser.add_argument("metadata_csv", help="包含 patient_id 和 group 的 CSV 文件。")
    parser.add_argument("output_csv", help="输出 fold 划分 CSV。")
    parser.add_argument("--n-splits", type=int, default=5, help="交叉验证折数。")
    parser.add_argument("--seed", type=int, default=42, help="随机种子。")
    args = parser.parse_args()

    df = pd.read_csv(args.metadata_csv)
    required = {"patient_id", "group"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"metadata 缺少必要列: {sorted(missing)}")

    # 患者级去重，别让同一人的多序列落到不同 fold
    patients = df[["patient_id", "group"]].drop_duplicates("patient_id").reset_index(drop=True)

    splitter = StratifiedKFold(n_splits=args.n_splits, shuffle=True, random_state=args.seed)
    patients["fold"] = -1

    for fold, (_, valid_idx) in enumerate(splitter.split(patients["patient_id"], patients["group"])):
        patients.loc[valid_idx, "fold"] = fold

    output_path = Path(args.output_csv)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    patients.to_csv(output_path, index=False)
    print(f"已保存患者级 fold 划分: {output_path}")


if __name__ == "__main__":
    main()
