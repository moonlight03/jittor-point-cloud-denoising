"""纯 Jittor 推理与预测格式校验。

推理主链路为：整云单位球归一化 -> FPS 选择 patch 中心 -> KNN 切 patch ->
三阶段模型去噪 -> poly 加权拼回整云，并固定执行两轮该流程。
"""

from __future__ import annotations

import time
from typing import Dict, Iterable, List, Optional

import jittor as jt
import numpy as np

from data import iter_noisy_files, load_cloud, normalize_unit_sphere_np, relative_shape_dir
from model import load_heavy_model, resolve_path


_ROTATION_MATRICES = {
    "identity": np.eye(3, dtype=np.float32),
    "x90": np.array(((1, 0, 0), (0, 0, 1), (0, -1, 0)), dtype=np.float32),
    "x180": np.array(((1, 0, 0), (0, -1, 0), (0, 0, -1)), dtype=np.float32),
    "x270": np.array(((1, 0, 0), (0, 0, -1), (0, 1, 0)), dtype=np.float32),
    "y90": np.array(((0, 0, -1), (0, 1, 0), (1, 0, 0)), dtype=np.float32),
    "y180": np.array(((-1, 0, 0), (0, 1, 0), (0, 0, -1)), dtype=np.float32),
    "y270": np.array(((0, 0, 1), (0, 1, 0), (-1, 0, 0)), dtype=np.float32),
    "z90": np.array(((0, 1, 0), (-1, 0, 0), (0, 0, 1)), dtype=np.float32),
    "z180": np.array(((-1, 0, 0), (0, -1, 0), (0, 0, 1)), dtype=np.float32),
    "z270": np.array(((0, -1, 0), (1, 0, 0), (0, 0, 1)), dtype=np.float32),
}


def rotation_matrix(name: str) -> np.ndarray:
    """返回固定旋转矩阵；正式配置只使用 identity。"""
    key = str(name).lower()
    if key not in _ROTATION_MATRICES:
        raise ValueError(f"unknown rotation {name!r}; choose from {sorted(_ROTATION_MATRICES)}")
    return _ROTATION_MATRICES[key]


def fix_count(pred: np.ndarray, target: np.ndarray) -> np.ndarray:
    """兜底保证输出点数等于输入点数，提交格式永远稳定。"""
    n_target, n_pred = target.shape[0], pred.shape[0]
    if n_pred == n_target:
        return pred.astype(np.float32)
    rng = np.random.RandomState(0)
    if n_pred > n_target:
        return pred[rng.choice(n_pred, n_target, replace=False)].astype(np.float32)
    extra = rng.choice(n_pred, n_target - n_pred, replace=True)
    return np.concatenate([pred, pred[extra]], axis=0).astype(np.float32)


def normalize_unit_sphere_jt(pc):
    p_max = pc.max(dim=0, keepdims=True)
    p_min = pc.min(dim=0, keepdims=True)
    center = (p_max + p_min) / 2
    centered = pc - center
    scale = (centered * centered).sum(dim=1, keepdims=True).sqrt().max()
    return centered / scale, center, scale


def farthest_point_sampling(pts, num, start_idx=0):
    """FPS 选 patch 中心；固定起点保证每次切出的 patch 一致。"""
    n_pts = pts.shape[0]
    if n_pts <= 0:
        raise ValueError("FPS requires at least one point")
    selected, dist, far = [], jt.ones((n_pts,)) * 1e10, int(start_idx) % n_pts
    for _ in range(num):
        selected.append(far)
        d = ((pts - pts[far]) ** 2).sum(dim=1)
        dist = jt.minimum(dist, d)
        far = int(jt.argmax(dist, dim=0)[0].item())
    idx = jt.array(np.array(selected, dtype=np.int32))
    return pts[idx], idx


def knn_patch(seed, pcl, patch_size):
    """为每个 FPS 中心查找最近的 ``patch_size`` 个原始点。"""
    dists, idx = jt.misc.knn(seed.unsqueeze(0), pcl.unsqueeze(0), patch_size)
    dists, idx = dists[0], idx[0]
    nn_pts = pcl[idx.reshape(-1)].reshape(idx.shape[0], idx.shape[1], 3)
    return dists, idx, nn_pts


def patch_denoise(
    model,
    pcl_noisy,
    patch_size=2048,
    seed_k=3,
    seed_k_alpha=5.0,
    stitch="winner",
    stitch_power=2.0,
    patch_batch_size=0,
    fps_start=0,
):
    """把整云切成重叠 patch，分批去噪后再拼接回原始点序。"""
    n_pts = pcl_noisy.shape[0]
    num_patches = max(1, int(seed_k * n_pts / patch_size))
    seed_pts, _ = farthest_point_sampling(pcl_noisy, num_patches, start_idx=fps_start)
    patch_dists, point_idxs, patches = knn_patch(seed_pts, pcl_noisy, patch_size)
    seed_rep = seed_pts.unsqueeze(1).broadcast(patches.shape)
    patches = patches - seed_rep
    # 距离归一化后用 exp(-d) 选每个原始点归属哪个 patch。
    patch_dists = patch_dists / patch_dists[:, -1:].broadcast(patch_dists.shape)
    all_dists = np.full((num_patches, n_pts), np.inf, dtype=np.float32)
    pidx_np = point_idxs.numpy()
    pdist_np = patch_dists.numpy()
    for i in range(num_patches):
        all_dists[i, pidx_np[i]] = pdist_np[i]
    owner = np.argmax(np.exp(-all_dists), axis=0)

    default_patch_step = max(1, int(num_patches / seed_k_alpha))
    patch_step = (
        min(num_patches, int(patch_batch_size))
        if patch_batch_size and patch_batch_size > 0
        else default_patch_step
    )
    # patch 分成小批次前向，避免一次性占满显存。
    outs, pos = [], 0
    while pos < num_patches:
        cur = patches[pos : pos + patch_step]
        nb = cur.shape[0]
        if nb < patch_step:
            pad = patch_step - nb
            cur = jt.concat([cur, cur[:1].broadcast((pad, cur.shape[1], cur.shape[2]))], dim=0)
        with jt.no_grad():
            den = model.denoise(cur)
        outs.append(den[:nb])
        pos += patch_step
    patches_denoised = jt.concat(outs, dim=0) + seed_rep
    pden_np = patches_denoised.numpy()

    result = np.empty((n_pts, 3), dtype=np.float32)
    filled = np.zeros((n_pts,), dtype=bool)
    # 同一个点可能被多个 patch 预测；poly 模式按离中心的距离加权融合，
    # 减少 patch 边界处的接缝，同时严格保留原始点数和点序。
    if stitch in ("weighted", "poly"):
        result.fill(0.0)
        weight_sum = np.zeros((n_pts,), dtype=np.float32)
        for i in range(num_patches):
            gids = pidx_np[i]
            # 越靠近 patch 中心权重越高，可抑制重叠边界的接缝。
            weights = np.maximum(1.0 - pdist_np[i], 0.0) ** stitch_power + 1e-6
            np.add.at(result, gids, pden_np[i] * weights[:, None])
            np.add.at(weight_sum, gids, weights)
        filled = weight_sum > 0
        result[filled] /= weight_sum[filled, None]
    elif stitch == "winner":
        for i in range(num_patches):
            gids = pidx_np[i]
            mask = owner[gids] == i
            result[gids[mask]] = pden_np[i][mask]
            filled[gids[mask]] = True
    else:
        raise ValueError(f"unknown stitch mode: {stitch}")
    if not filled.all():
        miss = np.where(~filled)[0]
        result[miss] = pcl_noisy.numpy()[miss]
    return jt.array(result)


def denoise_loop(
    model,
    pcl_raw_np: np.ndarray,
    patch_size=2048,
    seed_k=3,
    niters=2,
    stitch="winner",
    stitch_power=2.0,
    patch_batch_size=0,
    stitch_schedule=None,
    fps_start=0,
    rotation="identity",
) -> np.ndarray:
    """整云推理：单位球归一化 -> niters 轮 patch 去噪 -> 反归一化。"""
    pcl_raw = jt.array(pcl_raw_np.astype(np.float32))
    pcl_noisy, center, scale = normalize_unit_sphere_jt(pcl_raw)
    seed_k_alpha = max(1.0, pcl_raw_np.shape[0] / 10000.0)
    rot = rotation_matrix(rotation)
    rot_jt = None if str(rotation).lower() == "identity" else jt.array(rot)
    cur = pcl_noisy if rot_jt is None else jt.matmul(pcl_noisy, rot_jt)
    schedule = list(stitch_schedule) if stitch_schedule else [stitch] * niters
    if len(schedule) != niters:
        raise ValueError(f"stitch schedule must contain exactly {niters} entries, got {len(schedule)}")
    for iteration_stitch in schedule:
        cur = patch_denoise(
            model,
            cur,
            patch_size,
            seed_k,
            seed_k_alpha,
            stitch=iteration_stitch,
            stitch_power=stitch_power,
            patch_batch_size=patch_batch_size,
            fps_start=fps_start,
        )
    # 先把预测旋回原方向，再撤销输入时的单位球归一化。
    if rot_jt is not None:
        cur = jt.matmul(cur, rot_jt.transpose(0, 1))
    out = cur * scale + center
    return out.numpy().astype(np.float32)


def infer_dataset(
    ckpt,
    data_root,
    out_root,
    patch_size=2048,
    seed_k=4,
    niters=2,
    limit=0,
    files: Optional[Iterable[str]] = None,
    device="cuda",
    overwrite=False,
    stitch="weighted",
    stitch_power=1.0,
    patch_batch_size=12,
    stitch_schedule=None,
    fps_start=0,
    rotation="identity",
    seed=2023,
    num_shards=1,
    shard_index=0,
):
    if num_shards < 1:
        raise ValueError("num_shards must be at least 1")
    if not 0 <= shard_index < num_shards:
        raise ValueError("shard_index must be in [0, num_shards)")
    jt.set_global_seed(seed)
    np.random.seed(seed)
    model = load_heavy_model(ckpt, device=device, strict=True)
    data_root_p = resolve_path(data_root, must_exist=True)
    out_root_p = resolve_path(out_root)
    all_noisy_files = iter_noisy_files(data_root_p, files=files)
    noisy_files = all_noisy_files[shard_index::num_shards]
    if limit:
        noisy_files = noisy_files[:limit]
    rows: List[Dict[str, object]] = []
    schedule_label = ",".join(stitch_schedule) if stitch_schedule else stitch
    print(
        f"[infer] selected={len(noisy_files)}/{len(all_noisy_files)} "
        f"shard={shard_index}/{num_shards} ps={patch_size} seed_k={seed_k} "
        f"niters={niters} stitch={schedule_label} "
        f"patch_batch={patch_batch_size or 'auto'} fps_start={fps_start} "
        f"rotation={rotation} seed={seed}",
        flush=True,
    )
    for idx, noisy_path in enumerate(noisy_files, 1):
        rel_dir = relative_shape_dir(noisy_path, data_root_p)
        out_path = out_root_p / rel_dir / "denoised.npy"
        if out_path.exists() and not overwrite:
            print(f"  [{idx}/{len(noisy_files)}] {rel_dir} exists, skip", flush=True)
            continue
        noisy = load_cloud(noisy_path)
        t0 = time.time()
        pred = denoise_loop(
            model,
            noisy,
            patch_size,
            seed_k,
            niters,
            stitch,
            stitch_power,
            patch_batch_size,
            stitch_schedule,
            fps_start,
            rotation,
        )
        pred = fix_count(pred, noisy)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        np.save(out_path, pred.astype(np.float32))
        elapsed = time.time() - t0
        row = {"input": str(noisy_path), "output": str(out_path), "elapsed": elapsed}
        rows.append(row)
        print(f"  [{idx}/{len(noisy_files)}] {rel_dir} saved={pred.shape} ({elapsed:.1f}s)", flush=True)
    return {
        "all_inputs": len(all_noisy_files),
        "selected_inputs": len(noisy_files),
        "written": len(rows),
        "out_root": str(out_root_p),
        "seed": seed,
        "num_shards": num_shards,
        "shard_index": shard_index,
        "rows": rows,
    }


def validate_predictions(pred_root, data_root) -> Dict[str, object]:
    """逐样本校验 ID、shape、float32、有限值以及多余/缺失输出。"""
    pred_root_p = resolve_path(pred_root, must_exist=True)
    data_root_p = resolve_path(data_root, must_exist=True)
    expected = iter_noisy_files(data_root_p)
    pred_files = sorted(pred_root_p.glob("shapenet/*/*/denoised.npy"))
    expected_rels = {
        relative_shape_dir(noisy_path, data_root_p) / "denoised.npy"
        for noisy_path in expected
    }
    actual_rels = {path.relative_to(pred_root_p) for path in pred_files}
    extra_rels = sorted(actual_rels - expected_rels)
    issues, matched = [f"extra {rel.parent}" for rel in extra_rels], 0
    for noisy_path in expected:
        rel_dir = relative_shape_dir(noisy_path, data_root_p)
        pred_path = pred_root_p / rel_dir / "denoised.npy"
        if not pred_path.exists():
            issues.append(f"missing {rel_dir}")
            continue
        matched += 1
        noisy, pred = np.load(noisy_path), np.load(pred_path)
        if pred.shape != noisy.shape:
            issues.append(f"shape {rel_dir}: {pred.shape} != {noisy.shape}")
        if pred.dtype != np.float32:
            issues.append(f"dtype {rel_dir}: {pred.dtype}")
        if not np.isfinite(pred).all():
            issues.append(f"nonfinite {rel_dir}")
    return {
        "ok": not issues,
        "count": matched,
        "expected": len(expected),
        "total_pred_files": len(pred_files),
        "extra_count": len(extra_rels),
        "issues": issues,
    }
