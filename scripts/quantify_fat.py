import argparse
from pathlib import Path

import pandas as pd

from pancreas_region_analysis.imaging import load_nifti
from pancreas_region_analysis.quantification import pdff_from_dixon, quantify_regions


def main():
    parser = argparse.ArgumentParser(description="计算全胰腺及头、体、尾的体积和脂肪分数。")
    parser.add_argument("--region-mask", required=True, help="区域标签 mask，1=尾部，2=体部，3=头部。")
    parser.add_argument("--pdff", help="PDFF 图，取值可为 0-1 或 0-100。")
    parser.add_argument("--in-phase", help="Dixon 同相位图。")
    parser.add_argument("--opposed-phase", help="Dixon 反相位图。")
    parser.add_argument("--patient-id", required=True)
    parser.add_argument("--group", required=True, help="分组，例如 Healthy/Prediabetic/Diabetic。")
    parser.add_argument("--output-csv", required=True)
    args = parser.parse_args()

    if args.pdff is None and (args.in_phase is None or args.opposed_phase is None):
        raise ValueError("请提供 --pdff，或同时提供 --in-phase 与 --opposed-phase。")

    mask_img, region_mask = load_nifti(args.region_mask)

    if args.pdff:
        pdff_img, pdff_map = load_nifti(args.pdff)
        if pdff_map.max() > 1.5:
            # PDFF 若是 0-100 百分数，先归一化到 0-1
            pdff_map = pdff_map / 100.0
    else:
        _, in_phase = load_nifti(args.in_phase)
        _, opposed_phase = load_nifti(args.opposed_phase)
        pdff_map = pdff_from_dixon(in_phase, opposed_phase)

    row = quantify_regions(region_mask, mask_img, pdff_map)
    row = {"patient_id": args.patient_id, "group": args.group, **row}

    output_path = Path(args.output_csv)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if output_path.exists():
        old = pd.read_csv(output_path)
        out = pd.concat([old, pd.DataFrame([row])], ignore_index=True)
        # 同一 patient 重复跑时保留最后一次
        out = out.drop_duplicates("patient_id", keep="last")
    else:
        out = pd.DataFrame([row])

    out.to_csv(output_path, index=False)
    print(f"已更新定量结果: {output_path}")


if __name__ == "__main__":
    main()
