from pathlib import Path

import nibabel as nib
import numpy as np


def load_nifti(path):
    """读 NIfTI，返回 header 和体素数组。"""
    img = nib.load(str(path))
    data = np.asanyarray(img.dataobj)
    return img, data


def save_like(reference_img, data, output_path, dtype=np.int16):
    """沿用参考图的 affine/header 写 NIfTI。"""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    header = reference_img.header.copy()
    header.set_data_dtype(dtype)
    out_img = nib.Nifti1Image(data.astype(dtype), reference_img.affine, header)
    nib.save(out_img, str(output_path))


def voxel_volume_ml(img):
    """header 里 zooms 算单个体素体积 (mL)。"""
    zooms = img.header.get_zooms()[:3]
    voxel_volume_mm3 = float(np.prod(zooms))
    return voxel_volume_mm3 / 1000.0


def voxel_coordinates(mask):
    """非零体素坐标，N×3。"""
    coords = np.argwhere(mask > 0)
    if coords.size == 0:
        raise ValueError("输入 mask 中没有非零体素。")
    return coords


def voxel_to_world(coords, affine):
    """体素坐标 → 物理空间。"""
    coords_h = np.c_[coords, np.ones(len(coords))]
    world = coords_h @ affine.T
    return world[:, :3]
