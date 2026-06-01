import numpy as np
from scipy.spatial import ConvexHull, QhullError

from .imaging import voxel_coordinates, voxel_to_world


REGION_LABELS = {
    "tail": 1,
    "body": 2,
    "head": 3,
}


def _main_axis_by_convex_hull(world_coords):
    """凸包 + 旋转卡壳思路估胰腺主轴。

    在 XY/XZ/YZ 三个投影上算凸包，找最长直径方向，
    取跨度最大的那个作为 3D 主轴。
    """
    axis_candidates = []
    projection_axes = [(0, 1), (0, 2), (1, 2)]

    for axis_a, axis_b in projection_axes:
        points_2d = np.unique(world_coords[:, [axis_a, axis_b]], axis=0)
        if len(points_2d) < 3:
            continue

        try:
            hull = ConvexHull(points_2d)
            hull_points = points_2d[hull.vertices]
        except QhullError:
            continue

        start, end, length = _longest_hull_diameter(hull_points)
        if length <= 0:
            continue

        direction_3d = np.zeros(3, dtype=float)
        direction_3d[axis_a] = end[0] - start[0]
        direction_3d[axis_b] = end[1] - start[1]
        direction_3d = direction_3d / np.linalg.norm(direction_3d)
        axis_candidates.append((length, direction_3d))

    if not axis_candidates:
        raise ValueError("无法从凸包投影中估计胰腺主轴。")

    _, axis = max(axis_candidates, key=lambda item: item[0])
    return axis


def _longest_hull_diameter(hull_points):
    """凸包顶点两两距离，取最大当作直径。

    比纯旋转卡壳好写，结果也够稳。
    """
    diff = hull_points[:, None, :] - hull_points[None, :, :]
    dist_sq = np.sum(diff * diff, axis=2)
    i, j = np.unravel_index(np.argmax(dist_sq), dist_sq.shape)
    length = float(np.sqrt(dist_sq[i, j]))
    return hull_points[i], hull_points[j], length


def _orient_axis_from_head_side(axis, world_coords, head_side):
    """保留主轴方向；胰头在哪端由后面分段规则定。"""
    if head_side == "max":
        return axis
    if head_side == "min":
        return axis
    return axis


def _normalize_axis(axis):
    """单位化，投影长度别受向量模长影响。"""
    axis = axis / np.linalg.norm(axis)
    return axis


def partition_mask(mask, affine, head_side="max", ratios=(13, 10, 5)):
    """沿主轴按头/体/尾比例切分全胰 mask。

    Parameters
    ----------
    mask : ndarray
        非零区域视为胰腺。
    affine : ndarray
        NIfTI affine，体素转物理坐标，减轻各向异性影响。
    head_side : str
        "max" 或 "min"，主轴投影哪端算胰头；需对照扫描方向确认。
    ratios : tuple
        头/体/尾长度比，默认 13:10:5。

    Returns
    -------
    region_mask : ndarray
        1=尾，2=体，3=头。
    thresholds : tuple
        主轴投影上的两个切分阈值，核查用。
    """
    if head_side not in {"max", "min"}:
        raise ValueError("head_side 只能是 'max' 或 'min'。")

    head_ratio, body_ratio, tail_ratio = ratios
    total_ratio = float(head_ratio + body_ratio + tail_ratio)

    coords = voxel_coordinates(mask)
    world_coords = voxel_to_world(coords, affine)
    axis = _normalize_axis(_main_axis_by_convex_hull(world_coords))
    axis = _orient_axis_from_head_side(axis, world_coords, head_side)

    # 体素投影到主轴，按长度比例切段
    projections = world_coords @ axis
    p_min = float(projections.min())
    p_max = float(projections.max())
    p_range = p_max - p_min
    if p_range <= 0:
        raise ValueError("主轴投影范围为 0，无法进行区域划分。")

    region_mask = np.zeros(mask.shape, dtype=np.int16)

    if head_side == "max":
        tail_end = p_min + p_range * (tail_ratio / total_ratio)
        body_end = p_min + p_range * ((tail_ratio + body_ratio) / total_ratio)

        tail_idx = projections <= tail_end
        body_idx = (projections > tail_end) & (projections <= body_end)
        head_idx = projections > body_end
        thresholds = (tail_end, body_end)
    else:
        head_end = p_min + p_range * (head_ratio / total_ratio)
        body_end = p_min + p_range * ((head_ratio + body_ratio) / total_ratio)

        head_idx = projections <= head_end
        body_idx = (projections > head_end) & (projections <= body_end)
        tail_idx = projections > body_end
        thresholds = (head_end, body_end)

    # 1D 分段结果写回 3D 标签图
    region_mask[tuple(coords[tail_idx].T)] = REGION_LABELS["tail"]
    region_mask[tuple(coords[body_idx].T)] = REGION_LABELS["body"]
    region_mask[tuple(coords[head_idx].T)] = REGION_LABELS["head"]

    return region_mask, thresholds
