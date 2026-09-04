#!/usr/bin/env python3
"""A+B 两阶段训练与 Dataset B 推理命令入口。"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
from pathlib import Path
from typing import Union

from denoise import infer_dataset, validate_predictions
from model import DEFAULT_CKPT, HeavyDenoiseFlow, load_state, resolve_path, set_cuda


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Union[str, os.PathLike], payload: object) -> None:
    output = resolve_path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(output.name + f".tmp-{os.getpid()}")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(str(temporary), str(output))


def command_infer(args) -> None:
    if args.patch_radius_mode != "none":
        raise ValueError("this reproduction fixes patch_radius_mode=none")
    result = infer_dataset(
        ckpt=args.ckpt,
        data_root=args.data_root,
        out_root=args.out_root,
        patch_size=args.patch_size,
        seed_k=args.seed_k,
        niters=args.niters,
        limit=args.limit,
        files=args.files,
        device=args.device,
        overwrite=args.overwrite,
        stitch=args.stitch,
        stitch_power=args.stitch_power,
        patch_batch_size=args.patch_batch_size,
        fps_start=0,
        rotation=args.rotation,
        seed=args.seed,
        num_shards=args.num_shards,
        shard_index=args.shard_index,
    )
    result["contract"] = {
        "model": "heavy",
        "patch_size": args.patch_size,
        "patch_batch": args.patch_batch_size,
        "niters": args.niters,
        "seed_k": args.seed_k,
        "stitch": args.stitch,
        "stitch_power": args.stitch_power,
        "rotation": args.rotation,
        "seed": args.seed,
        "patch_radius_mode": args.patch_radius_mode,
    }
    if args.out_json:
        write_json(args.out_json, result)
    print(json.dumps(result, indent=2, sort_keys=True))


def command_validate(args) -> None:
    result = validate_predictions(args.pred_root, args.data_root)
    if args.expected_count and result["expected"] != args.expected_count:
        result["ok"] = False
        result["issues"].append(
            f"expected_count argument is {args.expected_count}, dataset inventory is {result['expected']}"
        )
    if args.out_json:
        write_json(args.out_json, result)
    print(json.dumps(result, indent=2, sort_keys=True))
    raise SystemExit(0 if result["ok"] else 1)


def command_merge(args) -> None:
    output = resolve_path(args.out_root)
    if output.exists() and any(output.rglob("denoised.npy")):
        raise FileExistsError(f"refusing to merge into non-empty prediction root: {output}")
    output.mkdir(parents=True, exist_ok=True)
    copied = {}
    for source_text in args.part_roots:
        source = resolve_path(source_text, must_exist=True)
        for path in sorted(source.rglob("denoised.npy")):
            relative = path.relative_to(source)
            if relative in copied:
                raise ValueError(f"duplicate prediction ID {relative.parent}: {copied[relative]} and {path}")
            target = output / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(str(path), str(target))
            copied[relative] = str(path)
    result = {
        "ok": len(copied) == args.expected_count,
        "count": len(copied),
        "expected": args.expected_count,
        "part_roots": [str(resolve_path(path, must_exist=True)) for path in args.part_roots],
        "out_root": str(output),
    }
    if args.out_json:
        write_json(args.out_json, result)
    print(json.dumps(result, indent=2, sort_keys=True))
    raise SystemExit(0 if result["ok"] else 1)


def command_checkpoint_check(args) -> None:
    set_cuda(args.device)
    rows = []
    accepted = True
    for checkpoint_text in args.checkpoints:
        checkpoint = resolve_path(checkpoint_text, must_exist=True)
        model = HeavyDenoiseFlow()
        load_result = load_state(model, checkpoint, strict=True)
        row = {"checkpoint": str(checkpoint), "sha256": sha256(checkpoint), "load": load_result}
        rows.append(row)
        accepted = accepted and load_result == {"assigned": 1998, "missing": 0, "extra": 0}
    result = {"accepted": accepted, "checkpoints": rows}
    if args.out_json:
        write_json(args.out_json, result)
    print(json.dumps(result, indent=2, sort_keys=True))
    raise SystemExit(0 if accepted else 1)


def command_train_stage(args) -> None:
    from train import train_stage

    result = train_stage(
        stage=args.stage,
        init_checkpoint=args.init_checkpoint,
        data_root=args.data_root,
        output_root=args.output_root,
        device=args.device,
        num_workers=args.num_workers,
        max_updates=args.max_updates,
        limit_clouds=args.limit_clouds,
        seed=args.seed,
        save_every=args.save_every,
        log_every=args.log_every,
        smoke=args.smoke,
    )
    print(json.dumps(result, indent=2, sort_keys=True))


def command_train_two_stage(args) -> None:
    from train import train_two_stage

    result = train_two_stage(
        initial_checkpoint=args.initial_checkpoint,
        stage1_data_root=args.stage1_data_root,
        stage2_data_root=args.stage2_data_root,
        output_root=args.output_root,
        device=args.device,
        smoke=args.smoke,
        num_workers=args.num_workers,
        limit_clouds=args.limit_clouds,
    )
    print(json.dumps(result, indent=2, sort_keys=True))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command")

    infer = commands.add_parser("infer", help="使用固定参数执行第 9 轮模型推理")
    infer.add_argument("--ckpt", default=str(DEFAULT_CKPT))
    infer.add_argument("--data_root", required=True)
    infer.add_argument("--out_root", required=True)
    infer.add_argument("--patch_size", type=int, default=2048)
    infer.add_argument("--patch_batch_size", type=int, default=12)
    infer.add_argument("--niters", type=int, default=2)
    infer.add_argument("--seed_k", type=int, default=8)
    infer.add_argument("--stitch", choices=("poly",), default="poly")
    infer.add_argument("--stitch_power", type=float, default=1.0)
    infer.add_argument("--rotation", choices=("identity",), default="identity")
    infer.add_argument("--seed", type=int, default=2023)
    infer.add_argument("--patch_radius_mode", choices=("none",), default="none")
    infer.add_argument("--device", default="cuda")
    infer.add_argument("--limit", type=int, default=0)
    infer.add_argument("--files", nargs="*", default=None)
    infer.add_argument("--num_shards", type=int, default=1)
    infer.add_argument("--shard_index", type=int, default=0)
    infer.add_argument("--overwrite", action="store_true")
    infer.add_argument("--out_json", default="")
    infer.set_defaults(func=command_infer)

    validate = commands.add_parser("validate", help="检查输出 ID、shape、dtype 和有限值")
    validate.add_argument("--pred_root", required=True)
    validate.add_argument("--data_root", required=True)
    validate.add_argument("--expected_count", type=int, default=0)
    validate.add_argument("--out_json", default="")
    validate.set_defaults(func=command_validate)

    merge = commands.add_parser("merge", help="strictly merge disjoint GPU shards")
    merge.add_argument("--part_roots", nargs="+", required=True)
    merge.add_argument("--out_root", required=True)
    merge.add_argument("--expected_count", type=int, required=True)
    merge.add_argument("--out_json", default="")
    merge.set_defaults(func=command_merge)

    check = commands.add_parser("checkpoint-check", help="严格检查模型权重")
    check.add_argument("--checkpoints", nargs="+", required=True)
    check.add_argument("--device", default="cpu")
    check.add_argument("--out_json", default="")
    check.set_defaults(func=command_checkpoint_check)

    stage = commands.add_parser("train-stage", help="run one model-only initialized training stage")
    stage.add_argument("--stage", choices=tuple(sorted((
        "stage1_ab_fixed_broad", "stage2_uniform_b0080_b0090"
    ))), required=True)
    stage.add_argument("--init_checkpoint", required=True)
    stage.add_argument("--data_root", required=True)
    stage.add_argument("--output_root", required=True)
    stage.add_argument("--device", default="cuda")
    stage.add_argument("--num_workers", type=int, default=4)
    stage.add_argument("--max_updates", type=int, default=0)
    stage.add_argument("--limit_clouds", type=int, default=0)
    stage.add_argument("--seed", type=int, default=2023)
    stage.add_argument("--save_every", type=int, default=1)
    stage.add_argument("--log_every", type=int, default=50)
    stage.add_argument("--smoke", action="store_true")
    stage.set_defaults(func=command_train_stage)

    two_stage = commands.add_parser("train-two-stage", help="A+B 训练 19 轮后继续训练 9 轮")
    two_stage.add_argument("--initial_checkpoint", required=True)
    two_stage.add_argument("--stage1_data_root", required=True)
    two_stage.add_argument("--stage2_data_root", required=True)
    two_stage.add_argument("--output_root", required=True)
    two_stage.add_argument("--device", default="cuda")
    two_stage.add_argument("--num_workers", type=int, default=4)
    two_stage.add_argument("--limit_clouds", type=int, default=0)
    two_stage.add_argument("--smoke", action="store_true")
    two_stage.set_defaults(func=command_train_two_stage)
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    if not hasattr(args, "func"):
        parser.print_help()
        raise SystemExit(2)
    args.func(args)


if __name__ == "__main__":
    main()
