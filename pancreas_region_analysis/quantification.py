import numpy as np

from .imaging import voxel_volume_ml


REGION_NAMES = {
    1: "tail",
    2: "body",
    3: "head",
}


def pdff_from_dixon(in_phase, opposed_phase, eps=1e-6):
    """从 Dixon 同/反相位估 PDFF。

    简化模型：I_in = I_fat + I_water，I_opp = I_fat - I_water。
    有现成 PDFF 图的话直接用，不必走这条路径。
    """
    in_phase = in_phase.astype(float)
    opposed_phase = opposed_phase.astype(float)
    fat_signal = (in_phase + opposed_phase) / 2.0
    water_signal = (in_phase - opposed_phase) / 2.0
    denominator = np.abs(fat_signal) + np.abs(water_signal) + eps
    pdff = np.abs(fat_signal) / denominator
    return np.clip(pdff, 0.0, 1.0)


def quantify_regions(region_mask, reference_img, pdff_map):
    """头/体/尾及全胰的体积与平均脂肪分数。"""
    if region_mask.shape != pdff_map.shape:
        raise ValueError("region_mask 与 pdff_map 的维度不一致。")

    voxel_ml = voxel_volume_ml(reference_img)
    rows = {}

    pancreas_mask = region_mask > 0
    rows["total_volume_ml"] = float(pancreas_mask.sum() * voxel_ml)
    rows["total_fat_fraction"] = _safe_mean(pdff_map[pancreas_mask])

    for label, name in REGION_NAMES.items():
        mask = region_mask == label
        rows[f"{name}_volume_ml"] = float(mask.sum() * voxel_ml)
        rows[f"{name}_fat_fraction"] = _safe_mean(pdff_map[mask])

    return rows


def _safe_mean(values):
    """空区域返回 NaN，方便后面 QC 筛掉。"""
    if values.size == 0:
        return float("nan")
    return float(np.nanmean(values))
