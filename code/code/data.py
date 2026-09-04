"""纯 Jittor 版本的数据读取、归一化和训练 patch 数据集。"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Iterable, List, Optional, Tuple, Union

import jittor as jt
from jittor.dataset import Dataset
import numpy as np
from scipy.spatial import cKDTree

from model import resolve_path


def iter_noisy_files(data_root: Union[str, os.PathLike], limit: int = 0, files: Optional[Iterable[str]] = None) -> List[Path]:
    """列出比赛输入文件，目录固定为 shapenet/<类别>/<模型>/noisy.npy。"""
    if files:
        out = [resolve_path(f, must_exist=True) for f in files]
    else:
        root = resolve_path(data_root, must_exist=True)
        out = sorted(root.glob("shapenet/*/*/noisy.npy"))
        if not out:
            out = sorted(root.rglob("noisy.npy"))
    return out[:limit] if limit else out


def relative_shape_dir(noisy_path: Path, data_root: Union[str, os.PathLike]) -> Path:
    return noisy_path.parent.relative_to(resolve_path(data_root, must_exist=True))


def load_cloud(path: Union[str, os.PathLike]) -> np.ndarray:
    arr = np.load(path).astype(np.float32)
    if arr.ndim != 2 or arr.shape[1] != 3:
        raise ValueError(f"point cloud must be (N,3), got {arr.shape}: {path}")
    if not np.isfinite(arr).all():
        raise ValueError(f"point cloud contains NaN/Inf: {path}")
    return arr


def normalize_unit_sphere_np(pcl: np.ndarray) -> Tuple[np.ndarray, np.ndarray, float]:
    """单位球归一化：记录 center/scale，推理结束后必须用同一组参数反归一化。"""
    p_max = pcl.max(axis=0, keepdims=True)
    p_min = pcl.min(axis=0, keepdims=True)
    center = ((p_max + p_min) / 2).astype(np.float32)
    centered = pcl - center
    scale = float(np.sqrt((centered**2).sum(axis=1)).max())
    return (centered / scale).astype(np.float32), center, scale


def _rot_matrix(axis: int, deg: float) -> np.ndarray:
    a = np.pi * deg / 180.0
    s, c = np.sin(a), np.cos(a)
    if axis == 0:
        return np.array([[1, 0, 0], [0, c, s], [0, -s, c]], dtype=np.float32)
    if axis == 1:
        return np.array([[c, 0, -s], [0, 1, 0], [s, 0, c]], dtype=np.float32)
    return np.array([[c, s, 0], [-s, c, 0], [0, 0, 1]], dtype=np.float32)


class CleanPatchDataset(Dataset):
    """从外部 clean ``.npy`` 目录即时合成 Laplace noisy/clean patch。

    训练输入只包含 clean 点云，噪声在 ``__getitem__`` 内生成；noisy/clean
    始终使用同一组 KNN 索引，保持逐点监督关系。
    """

    def __init__(
        self,
        root: Optional[Union[str, os.PathLike]] = None,
        patch_size: int = 2048,
        num_patches: int = 1,
        noise_min: float = 0.004,
        noise_max: float = 0.017,
        aug_rotate: bool = False,
        batch_size: int = 8,
        num_workers: int = 4,
        shuffle: bool = True,
        base_seed: int = 2023,
        expected_clouds: int = 0,
        limit_clouds: int = 0,
    ):
        super().__init__()
        if root is None:
            raise ValueError("clean point-cloud root is required")
        root_p = resolve_path(root, must_exist=True)
        files = sorted(path for path in Path(root_p).rglob("*.npy") if path.is_file())
        if not files:
            raise FileNotFoundError(f"no .npy point clouds under {root_p}")
        if expected_clouds and len(files) != int(expected_clouds):
            raise ValueError(f"expected {expected_clouds} clouds under {root_p}, found {len(files)}")
        if limit_clouds:
            files = files[: int(limit_clouds)]
        self.files = [str(path) for path in files]
        self.base_seed = int(base_seed)
        self.patch_size = int(patch_size)
        self.num_patches = int(num_patches)
        self.noise_min = float(noise_min)
        self.noise_max = float(noise_max)
        self.aug_rotate = bool(aug_rotate)
        if self.patch_size <= 0 or self.num_patches <= 0:
            raise ValueError("patch_size and num_patches must be positive")
        if not 0.0 < self.noise_min <= self.noise_max:
            raise ValueError("noise bounds must satisfy 0 < min <= max")
        total = len(self.files) * self.num_patches
        print(
            f"[data] synthetic_laplace_only=true {len(self.files)} clouds x {self.num_patches} "
            f"patches = {total} items (lazy load, rotation={'full' if self.aug_rotate else 'none'})",
            flush=True,
        )
        self.set_attrs(total_len=total, batch_size=batch_size, num_workers=num_workers, shuffle=shuffle)

    @staticmethod
    def _rng(base_seed: int, idx: int, stream: int) -> np.random.RandomState:
        # 每类随机量使用独立流，worker 调度变化不会改变噪声和增强结果。
        seed = np.random.SeedSequence([base_seed, idx, stream]).generate_state(1, dtype=np.uint32)[0]
        return np.random.RandomState(int(seed))

    def __getitem__(self, idx):
        idx = int(idx)
        noise_rng = self._rng(self.base_seed, idx, 1)
        aug_rng = self._rng(self.base_seed, idx, 2)
        path = self.files[idx % len(self.files)]
        clean = np.load(path, allow_pickle=False)
        if clean.shape != (50000, 3) or clean.dtype != np.float32 or not np.isfinite(clean).all():
            raise ValueError(f"expected finite float32 (50000,3) clean cloud: {path}")
        clean, _center, _scale = normalize_unit_sphere_np(clean)
        noise_std = float(noise_rng.uniform(self.noise_min, self.noise_max))
        noisy = clean + noise_rng.laplace(0.0, noise_std, size=clean.shape).astype(np.float32)
        scale_aug = float(aug_rng.uniform(0.8, 1.2))
        clean = clean * scale_aug
        noisy = noisy * scale_aug
        if self.aug_rotate:
            for axis in (0, 1, 2):
                deg = float(aug_rng.uniform(-180.0, 180.0))
                rot = _rot_matrix(axis, deg)
                clean = clean @ rot
                noisy = noisy @ rot
        # 用 noisy 上的随机 seed 做 KNN patch，clean 取相同索引，保持逐点配对。
        seed_i = int(aug_rng.randint(noisy.shape[0]))
        _, knn = cKDTree(noisy).query(noisy[seed_i], k=self.patch_size)
        knn = np.asarray(knn, dtype=np.int64)
        residual = noisy[knn] - clean[knn]
        return (
            jt.array(noisy[knn]),
            jt.array(clean[knn]),
            jt.array(residual.astype(np.float32)),
            jt.array(noisy[seed_i : seed_i + 1]),
            np.float32(noise_std),
            np.float32(scale_aug),
        )

    def collate_batch(self, batch):
        noisy = jt.stack([b[0] for b in batch], dim=0)
        clean = jt.stack([b[1] for b in batch], dim=0)
        residual = jt.stack([b[2] for b in batch], dim=0)
        seed = jt.stack([b[3] for b in batch], dim=0)
        noise_std = jt.array(np.array([b[4] for b in batch], dtype=np.float32))
        scale = jt.array(np.array([b[5] for b in batch], dtype=np.float32))
        return {
            "pcl_noisy": noisy,
            "pcl_clean": clean,
            "pcl_noise": residual,
            "seed_pnts": seed,
            "pcl_std": noise_std,
            "scale": scale,
        }
