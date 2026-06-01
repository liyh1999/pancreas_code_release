import argparse
import json
from pathlib import Path

from pancreas_region_analysis.imaging import load_nifti, save_like
from pancreas_region_analysis.partition import partition_mask


def main():
    parser = argparse.ArgumentParser(description="将全胰腺 mask 半自动初始化划分为头、体、尾。")
    parser.add_argument("--mask", required=True, help="全胰腺二值/标签 mask，NIfTI 格式。")
    parser.add_argument("--output", required=True, help="输出区域标签 mask，1=尾部，2=体部，3=头部。")
    parser.add_argument("--head-side", choices=["max", "min"], default="max", help="主轴投影哪一端视为胰头。")
    parser.add_argument("--ratios", nargs=3, type=float, default=[13, 10, 5], metavar=("HEAD", "BODY", "TAIL"))
    args = parser.parse_args()

    img, mask = load_nifti(args.mask)
    region_mask, thresholds = partition_mask(mask, img.affine, head_side=args.head_side, ratios=tuple(args.ratios))
    save_like(img, region_mask, args.output)

    # 阈值和参数落盘，方便人工核查前对照初始划分
    meta_path = Path(args.output).with_suffix("").with_suffix(".partition.json")
    meta = {
        "input_mask": str(args.mask),
        "output_mask": str(args.output),
        "head_side": args.head_side,
        "ratios_head_body_tail": args.ratios,
        "projection_thresholds": list(map(float, thresholds)),
        "labels": {"tail": 1, "body": 2, "head": 3},
    }
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"已保存区域标签: {args.output}")
    print(f"已保存划分参数: {meta_path}")


if __name__ == "__main__":
    main()
