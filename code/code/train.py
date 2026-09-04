"""纯 Jittor 的 A+B 预训练与 Dataset B 精调流程。

阶段一固定训练 19 轮并选用第 19 轮权重；阶段二只加载该模型参数，
重新创建优化器后训练 9 轮并选用第 9 轮权重。
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import random
import time
import uuid
from pathlib import Path
from typing import Dict, Optional, Tuple

import jittor as jt
import numpy as np

from data import CleanPatchDataset
from emd import AuctionEMDLoss
from model import (
    HeavyDenoiseFlow,
    load_state,
    resolve_path,
    save_state,
    set_cuda,
    update_lipschitz,
)


STAGE_DEFAULTS = {
    "stage1_ab_fixed_broad": {
        "epochs": 19,
        "expected_clouds": 35632,
        "noise_min": 0.004,
        "noise_max": 0.017,
        "lr": 2e-4,
        "lr_schedule": "plateau",
        "min_lr": 1e-6,
        "plateau_patience": 2,
        "plateau_factor": 0.5,
    },
    "stage2_uniform_b0080_b0090": {
        "epochs": 9,
        "expected_clouds": 19699,
        "noise_min": 0.008,
        "noise_max": 0.009,
        "lr": 2e-5,
        "lr_schedule": "constant",
        "min_lr": 2e-5,
        "plateau_patience": 0,
        "plateau_factor": 1.0,
    },
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".tmp-{os.getpid()}")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(str(temporary), str(path))


def append_jsonl(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, sort_keys=True) + "\n")
        handle.flush()


def _pairwise_d2(first, second):
    difference = first.unsqueeze(2) - second.unsqueeze(1)
    return (difference * difference).sum(dim=-1)


def _normalize_like_reference(prediction, reference):
    maximum = reference.max(dim=1, keepdims=True)
    minimum = reference.min(dim=1, keepdims=True)
    center = (maximum + minimum) * 0.5
    centered = reference - center
    scale = jt.maximum(
        (centered * centered).sum(dim=-1, keepdims=True).sqrt().max(dim=1, keepdims=True),
        jt.array(1e-12),
    )
    return (prediction - center) / scale, centered / scale


def chamfer_distance_unit_sphere(prediction, reference):
    """在同一单位球坐标系中计算双向最近邻距离。"""
    prediction_norm, reference_norm = _normalize_like_reference(prediction, reference)
    distances = _pairwise_d2(prediction_norm, reference_norm)
    return distances.min(dim=2).mean() + distances.min(dim=1).mean()


def repulsion_loss(points, radius=0.05, k=8):
    """惩罚半径内过近邻居，避免去噪点在局部聚成团。"""
    if k <= 0 or k >= points.shape[1]:
        raise ValueError(f"repulsion k must be in [1,N), got k={k} N={points.shape[1]}")
    distances = _pairwise_d2(points, points)
    batch, count, _ = distances.shape
    identity = jt.array(np.eye(count, dtype=np.float32)).reshape(1, count, count)
    identity = identity.broadcast((batch, count, count))
    nearest, _ = jt.topk(distances + identity * 1e6, k=k, dim=-1, largest=False)
    nearest = jt.sqrt(jt.maximum(nearest, jt.array(1e-12)))
    return jt.maximum(jt.array(float(radius)) - nearest, jt.zeros_like(nearest)).sqr().mean()


def _prefix_for_emd(value, maximum_points=1024):
    """Auction EMD 固定处理前 1,024 点，保证子集选择可重复。"""
    if value.shape[1] < maximum_points:
        raise ValueError(f"Auction EMD requires at least {maximum_points} points")
    return value[:, :maximum_points, :]


def _forward_all(model, points):
    outputs = []
    current = points
    for flow in model.flows:
        current = flow(current)
        outputs.append(current)
    return outputs


def _model_finite(model) -> bool:
    return all(np.isfinite(np.asarray(value.numpy())).all() for value in model.state_dict().values())


def _model_state_numpy(model) -> Dict[str, np.ndarray]:
    return {key: np.ascontiguousarray(value.numpy()) for key, value in model.state_dict().items()}


def _save_training_state(
    model,
    optimizer,
    path: Path,
    *,
    stage: str,
    epoch: int,
    global_step: int,
    optimizer_instance_id: str,
    config: dict,
) -> Path:
    payload = {
        "format": "jittor-two-stage-training-state-v1",
        "model_state": _model_state_numpy(model),
        "optimizer_state": optimizer.state_dict(),
        "stage": stage,
        "epoch": int(epoch),
        "global_step": int(global_step),
        "optimizer_instance_id": optimizer_instance_id,
        "config": config,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.stem + f".tmp-{os.getpid()}.pkl")
    jt.save(payload, str(temporary))
    os.replace(str(temporary), str(path))
    return path


def _verify_saved_pair(model_path: Path, state_path: Path, optimizer_instance_id: str) -> dict:
    reloaded = HeavyDenoiseFlow()
    load_result = load_state(reloaded, model_path, strict=True)
    payload = jt.load(str(state_path))
    if not isinstance(payload, dict) or payload.get("format") != "jittor-two-stage-training-state-v1":
        raise ValueError(f"invalid training state: {state_path}")
    if payload.get("optimizer_instance_id") != optimizer_instance_id:
        raise ValueError("optimizer instance ID changed after state reload")
    if load_result != {"assigned": 1998, "missing": 0, "extra": 0}:
        raise ValueError(f"strict checkpoint reload failed: {load_result}")
    if not _model_finite(reloaded):
        raise ValueError("reloaded model contains non-finite tensors")
    return load_result


def _plateau_update(
    current_lr: float,
    epoch_loss: float,
    best_loss: float,
    bad_epochs: int,
    patience: int,
    factor: float,
    minimum_lr: float,
) -> Tuple[float, float, int]:
    if epoch_loss < best_loss:
        return current_lr, epoch_loss, 0
    bad_epochs += 1
    if bad_epochs > patience:
        return max(minimum_lr, current_lr * factor), best_loss, 0
    return current_lr, best_loss, bad_epochs


def train_stage(
    *,
    stage: str,
    init_checkpoint: str,
    data_root: str,
    output_root: str,
    device: str = "cuda",
    epochs: Optional[int] = None,
    expected_clouds: Optional[int] = None,
    noise_min: Optional[float] = None,
    noise_max: Optional[float] = None,
    lr: Optional[float] = None,
    lr_schedule: Optional[str] = None,
    min_lr: Optional[float] = None,
    plateau_patience: Optional[int] = None,
    plateau_factor: Optional[float] = None,
    batch_size: int = 8,
    patch_size: int = 2048,
    patches_per_cloud: int = 1,
    num_workers: int = 4,
    max_updates: int = 0,
    limit_clouds: int = 0,
    seed: int = 2023,
    save_every: int = 1,
    log_every: int = 50,
    cd_weight: float = 40.0,
    emd_weight: float = 0.00625,
    repulsion_weight: float = 40.0,
    repulsion_radius: float = 0.05,
    repulsion_k: int = 8,
    repulsion_spacing_multiplier: float = 10.0,
    stage_noise_decay: float = 5.3,
    grad_clip: float = 0.001,
    smoke: bool = False,
) -> dict:
    if stage not in STAGE_DEFAULTS:
        raise ValueError(f"unknown stage {stage!r}; expected one of {sorted(STAGE_DEFAULTS)}")
    defaults = STAGE_DEFAULTS[stage]
    epochs = int(defaults["epochs"] if epochs is None else epochs)
    expected_clouds = int(defaults["expected_clouds"] if expected_clouds is None else expected_clouds)
    noise_min = float(defaults["noise_min"] if noise_min is None else noise_min)
    noise_max = float(defaults["noise_max"] if noise_max is None else noise_max)
    lr = float(defaults["lr"] if lr is None else lr)
    lr_schedule = str(defaults["lr_schedule"] if lr_schedule is None else lr_schedule)
    min_lr = float(defaults["min_lr"] if min_lr is None else min_lr)
    plateau_patience = int(defaults["plateau_patience"] if plateau_patience is None else plateau_patience)
    plateau_factor = float(defaults["plateau_factor"] if plateau_factor is None else plateau_factor)
    if lr_schedule not in ("constant", "plateau"):
        raise ValueError("lr_schedule must be constant or plateau")
    if epochs <= 0 or batch_size <= 0 or patch_size < 1024 or max_updates < 0:
        raise ValueError("invalid epochs/batch/patch/max_updates")
    if smoke:
        max_updates = 1
        limit_clouds = max(batch_size, limit_clouds or batch_size)

    set_cuda(device)
    jt.set_global_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    output = resolve_path(output_root)
    checkpoints = output / "checkpoints"
    logs = output / "logs"
    checkpoints.mkdir(parents=True, exist_ok=True)
    logs.mkdir(parents=True, exist_ok=True)

    initial = resolve_path(init_checkpoint, must_exist=True)
    model = HeavyDenoiseFlow()
    initial_load = load_state(model, initial, strict=True)
    if initial_load != {"assigned": 1998, "missing": 0, "extra": 0}:
        raise ValueError(f"strict initial load failed: {initial_load}")
    model.train()
    optimizer = jt.optim.Adam(model.parameters(), lr=lr)
    optimizer_instance_id = f"{stage}-{uuid.uuid4().hex}"
    loader = CleanPatchDataset(
        root=data_root,
        patch_size=patch_size,
        num_patches=patches_per_cloud,
        noise_min=noise_min,
        noise_max=noise_max,
        aug_rotate=False,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=True,
        base_seed=seed,
        expected_clouds=expected_clouds,
        limit_clouds=limit_clouds,
    )
    emd = AuctionEMDLoss(eps=0.005, iters=50)
    config = {
        "stage": stage,
        "initial_checkpoint": str(initial),
        "initial_checkpoint_sha256": sha256(initial),
        "optimizer_instance_id": optimizer_instance_id,
        "optimizer_created_fresh": True,
        "optimizer": "adam",
        "epochs": epochs,
        "expected_clouds": expected_clouds,
        "batch_size": batch_size,
        "patch_size": patch_size,
        "patches_per_cloud": patches_per_cloud,
        "noise_distribution": "laplace",
        "noise_min": noise_min,
        "noise_max": noise_max,
        "synthetic_noise_only": True,
        "scale_min": 0.8,
        "scale_max": 1.2,
        "rotation": "none",
        "lr": lr,
        "lr_schedule": lr_schedule,
        "min_lr": min_lr,
        "plateau_patience": plateau_patience,
        "plateau_factor": plateau_factor,
        "emd_weight": emd_weight,
        "emd_backend": "jittor_auction_cuda",
        "emd_points": 1024,
        "emd_subset": "deterministic_prefix",
        "cd_weight": cd_weight,
        "repulsion_weight": repulsion_weight,
        "repulsion_mode": "fixed",
        "repulsion_radius": repulsion_radius,
        "repulsion_k": repulsion_k,
        "repulsion_spacing_multiplier": repulsion_spacing_multiplier,
        "repulsion_spacing_multiplier_effect": "metadata_only_for_fixed_radius",
        "target_noise_mode": "residual",
        "stage_noise_decay": stage_noise_decay,
        "grad_clip": grad_clip,
        "seed": seed,
        "max_updates": max_updates,
        "smoke": smoke,
    }
    atomic_json(output / "train_config.json", config)
    print(
        f"[train] stage={stage} init={initial.name} epochs={epochs} batch={batch_size} "
        f"patch={patch_size} noise=[{noise_min},{noise_max}] lr={lr:.3e} "
        f"schedule={lr_schedule} synthetic_only=true optimizer={optimizer_instance_id}",
        flush=True,
    )

    global_step = 0
    current_lr = lr
    best_loss = float("inf")
    bad_epochs = 0
    final_epoch = 0
    stop = False
    for epoch in range(1, epochs + 1):
        epoch_values = []
        started = time.time()
        optimizer.lr = current_lr
        for batch in loader:
            noisy = batch["pcl_noisy"]
            clean = batch["pcl_clean"]
            residual = batch["pcl_noise"]
            seed_points = batch["seed_pnts"]
            seed_repeated = seed_points.broadcast(noisy.shape)
            noisy = noisy - seed_repeated
            clean = clean - seed_repeated
            targets = (
                clean + residual / stage_noise_decay,
                clean + residual / (stage_noise_decay * stage_noise_decay),
                clean,
            )
            outputs = _forward_all(model, noisy)
            # 三段输出分别监督逐步去噪目标，最后一段额外计算 CD 和排斥项。
            emd_terms = []
            for prediction, target in zip(outputs, targets):
                per_sample = emd(_prefix_for_emd(prediction), _prefix_for_emd(target))
                emd_terms.append(per_sample.sum())
            emd_total = sum(emd_terms)
            cd = chamfer_distance_unit_sphere(outputs[-1], clean)
            repulsion = repulsion_loss(outputs[-1], radius=repulsion_radius, k=repulsion_k)
            weighted = (
                emd_weight * emd_total,
                cd_weight * cd,
                repulsion_weight * repulsion,
            )
            loss = sum(weighted)
            optimizer.zero_grad()
            optimizer.backward(loss)
            if grad_clip > 0:
                optimizer.clip_grad_norm(grad_clip)
            optimizer.step()
            for flow in model.flows:
                update_lipschitz(flow)
            global_step += 1
            loss_value = float(loss.item())
            if not math.isfinite(loss_value) or not _model_finite(model):
                raise FloatingPointError(f"non-finite training state at update {global_step}")
            epoch_values.append(loss_value)
            row = {
                "stage": stage,
                "epoch": epoch,
                "global_step": global_step,
                "loss": loss_value,
                "emd": float(weighted[0].item()),
                "cd": float(weighted[1].item()),
                "repulsion": float(weighted[2].item()),
                "lr": float(current_lr),
                "optimizer_instance_id": optimizer_instance_id,
                "batch_kind": "synthetic_laplace",
            }
            append_jsonl(logs / "train.jsonl", row)
            if global_step <= 5 or global_step % log_every == 0:
                print(
                    f"[update] stage={stage} epoch={epoch} step={global_step} "
                    f"loss={loss_value:.6f} lr={current_lr:.3e}",
                    flush=True,
                )
            if max_updates and global_step >= max_updates:
                stop = True
                break
        if not epoch_values:
            raise RuntimeError(f"stage {stage} epoch {epoch} produced no updates")
        final_epoch = epoch
        epoch_loss = float(np.mean(epoch_values))
        if lr_schedule == "plateau":
            current_lr, best_loss, bad_epochs = _plateau_update(
                current_lr,
                epoch_loss,
                best_loss,
                bad_epochs,
                plateau_patience,
                plateau_factor,
                min_lr,
            )
        append_jsonl(
            logs / "epochs.jsonl",
            {
                "stage": stage,
                "epoch": epoch,
                "global_step": global_step,
                "mean_loss": epoch_loss,
                "seconds": time.time() - started,
                "next_lr": current_lr,
                "stopped_by_max_updates": stop,
            },
        )
        if save_every > 0 and epoch % save_every == 0 and not stop:
            model_path = save_state(model, checkpoints / f"epoch_{epoch:04d}_model.pkl")
            _save_training_state(
                model,
                optimizer,
                checkpoints / f"epoch_{epoch:04d}_state.pkl",
                stage=stage,
                epoch=epoch,
                global_step=global_step,
                optimizer_instance_id=optimizer_instance_id,
                config=config,
            )
        if stop:
            break

    # 每个阶段最后都使用固定轮次文件；正式运行对应第 19 轮和第 9 轮。
    model_path = save_state(model, checkpoints / f"epoch_{final_epoch:04d}_model.pkl")
    state_path = _save_training_state(
        model,
        optimizer,
        checkpoints / f"epoch_{final_epoch:04d}_state.pkl",
        stage=stage,
        epoch=final_epoch,
        global_step=global_step,
        optimizer_instance_id=optimizer_instance_id,
        config=config,
    )
    strict_reload = _verify_saved_pair(model_path, state_path, optimizer_instance_id)
    summary = {
        "status": "smoke_complete" if smoke else ("max_updates" if stop else "complete"),
        "stage": stage,
        "epoch": final_epoch,
        "global_step": global_step,
        "optimizer_instance_id": optimizer_instance_id,
        "optimizer_created_fresh": True,
        "initial_checkpoint": str(initial),
        "initial_checkpoint_sha256": sha256(initial),
        "model_checkpoint": str(model_path),
        "model_checkpoint_sha256": sha256(model_path),
        "training_state": str(state_path),
        "training_state_sha256": sha256(state_path),
        "strict_reload": strict_reload,
        "finite_after_update": True,
        "synthetic_noise_only": True,
        "selected_epoch": final_epoch,
    }
    atomic_json(output / "summary.json", summary)
    return summary


def train_two_stage(
    *,
    initial_checkpoint: str,
    stage1_data_root: str,
    stage2_data_root: str,
    output_root: str,
    device: str = "cuda",
    smoke: bool = False,
    num_workers: int = 4,
    limit_clouds: int = 0,
) -> dict:
    root = resolve_path(output_root)
    stage1 = train_stage(
        stage="stage1_ab_fixed_broad",
        init_checkpoint=initial_checkpoint,
        data_root=stage1_data_root,
        output_root=str(root / "stage1_epoch19"),
        device=device,
        smoke=smoke,
        num_workers=num_workers,
        limit_clouds=limit_clouds,
    )
    # train_stage 每次都会新建 Adam，因此阶段二不会继承阶段一的优化器状态。
    stage2 = train_stage(
        stage="stage2_uniform_b0080_b0090",
        init_checkpoint=stage1["model_checkpoint"],
        data_root=stage2_data_root,
        output_root=str(root / "stage2_epoch9"),
        device=device,
        smoke=smoke,
        num_workers=num_workers,
        limit_clouds=limit_clouds,
    )
    if stage1["optimizer_instance_id"] == stage2["optimizer_instance_id"]:
        raise RuntimeError("stage2 must rebuild Adam with a fresh optimizer instance")
    if stage2["initial_checkpoint_sha256"] != stage1["model_checkpoint_sha256"]:
        raise RuntimeError("stage2 did not initialize from the stage1 model-only checkpoint")
    summary = {
        "status": "complete",
        "stage1": stage1,
        "stage2": stage2,
        "stage2_optimizer_rebuilt": True,
        "stage2_model_only_initialization": True,
        "selected_stage1_epoch": 19 if not smoke else stage1["selected_epoch"],
        "selected_stage2_epoch": 9 if not smoke else stage2["selected_epoch"],
    }
    atomic_json(root / "two_stage_summary.json", summary)
    return summary
