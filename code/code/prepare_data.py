#!/usr/bin/env python3
"""把 A/B 榜 OBJ 确定性转换为 NPY，并构建联合训练目录。"""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
from typing import Dict, List, Sequence, Tuple

import numpy as np
import trimesh


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".tmp-{os.getpid()}")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(str(temporary), str(path))


def parse_shape_id(value: str) -> Tuple[str, str, str]:
    """严格接受 shapenet/<类别>/<模型>，防止清单路径越界。"""
    normalized = value.strip().strip("/")
    parts = Path(normalized).parts
    if len(parts) != 3 or parts[0] != "shapenet" or any(part in ("", ".", "..") for part in parts):
        raise ValueError(f"invalid shape ID: {value!r}")
    return str(parts[0]), str(parts[1]), str(parts[2])


def read_split_files(paths: Sequence[Path]) -> List[str]:
    if not paths:
        raise ValueError("at least one split file is required")
    rows: List[str] = []
    for path in paths:
        if not path.is_file():
            raise FileNotFoundError(path)
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                rows.append("/".join(parse_shape_id(line)))
    if not rows:
        raise ValueError("split files contain no shape IDs")
    if len(rows) != len(set(rows)):
        seen, duplicates = set(), []
        for row in rows:
            if row in seen:
                duplicates.append(row)
            seen.add(row)
        raise ValueError(f"duplicate shape IDs: {sorted(set(duplicates))[:10]}")
    return rows


def stable_seed(base_seed: int, dataset_tag: str, shape_id: str) -> int:
    """使用内容哈希派生 seed，避免 Python 随机盐和 worker 顺序造成漂移。"""
    payload = f"{int(base_seed)}\0{dataset_tag}\0{shape_id}".encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:4], "little", signed=False)


def output_name(dataset_tag: str, shape_id: str) -> str:
    return dataset_tag + "__" + shape_id.replace("/", "__") + ".npy"


def load_triangle_mesh(path: Path) -> trimesh.Trimesh:
    loaded = trimesh.load(str(path), force="mesh", process=False)
    if not isinstance(loaded, trimesh.Trimesh):
        raise ValueError(f"not a triangle mesh: {path}")
    vertices = np.asarray(loaded.vertices)
    faces = np.asarray(loaded.faces)
    if vertices.ndim != 2 or vertices.shape[1] != 3 or faces.ndim != 2 or faces.shape[1] != 3:
        raise ValueError(f"invalid triangle arrays: {path}")
    if not np.isfinite(vertices).all() or len(faces) == 0:
        raise ValueError(f"invalid mesh values: {path}")
    return loaded


def sample_surface(mesh: trimesh.Trimesh, count: int, rng: np.random.RandomState) -> np.ndarray:
    """按三角形面积选面，再用重心坐标在面内均匀采样。"""
    vertices = np.asarray(mesh.vertices, dtype=np.float64)
    faces = np.asarray(mesh.faces, dtype=np.int64)
    triangles = vertices[faces]
    cross = np.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0])
    areas = np.linalg.norm(cross, axis=1)
    total_area = float(areas.sum())
    if not np.isfinite(total_area) or total_area <= 0:
        raise ValueError("mesh has no positive-area triangle")
    face_ids = rng.choice(len(faces), size=int(count), replace=True, p=areas / total_area)
    chosen = triangles[face_ids]
    uv = rng.random_sample((int(count), 2))
    root_u = np.sqrt(uv[:, :1])
    barycentric = np.concatenate(
        (1.0 - root_u, root_u * (1.0 - uv[:, 1:]), root_u * uv[:, 1:]),
        axis=1,
    )
    return np.ascontiguousarray(np.einsum("ni,nij->nj", barycentric, chosen), dtype=np.float32)


def normalize_unit_sphere(points: np.ndarray) -> np.ndarray:
    """以包围盒中心平移，并用最远点半径缩放到单位球。"""
    points = np.asarray(points, dtype=np.float32)
    center = ((points.max(axis=0) + points.min(axis=0)) * 0.5).astype(np.float32)
    centered = points - center
    scale = float(np.sqrt(np.square(centered).sum(axis=1)).max())
    if not np.isfinite(scale) or scale <= 1e-12:
        raise ValueError("degenerate point cloud")
    return np.ascontiguousarray(centered / scale, dtype=np.float32)


def atomic_save_npy(path: Path, points: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".tmp-{os.getpid()}")
    with temporary.open("wb") as handle:
        np.save(handle, np.ascontiguousarray(points, dtype=np.float32), allow_pickle=False)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(str(temporary), str(path))


def validate_cloud(path: Path, count: int) -> Dict[str, object]:
    value = np.load(path, mmap_mode="r", allow_pickle=False)
    if value.shape != (int(count), 3) or value.dtype != np.float32:
        raise ValueError(f"invalid cloud {path}: shape={value.shape} dtype={value.dtype}")
    if not np.isfinite(value).all():
        raise FloatingPointError(f"non-finite cloud: {path}")
    return {
        "path": str(path.resolve()),
        "shape": [int(count), 3],
        "dtype": "float32",
        "sha256": sha256_file(path),
    }


def prepare_one(task: Dict[str, object]) -> Dict[str, object]:
    shape_id = str(task["shape_id"])
    mesh_path = Path(str(task["mesh_root"])) / shape_id / "models" / "model_normalized.obj"
    if not mesh_path.is_file():
        raise FileNotFoundError(mesh_path)
    output = Path(str(task["output_root"])) / output_name(str(task["dataset_tag"]), shape_id)
    count = int(task["points"])
    reused = output.is_file() and not bool(task["overwrite"])
    if reused:
        record = validate_cloud(output, count)
    else:
        rng = np.random.RandomState(int(task["sample_seed"]) & 0xFFFFFFFF)
        points = normalize_unit_sphere(sample_surface(load_triangle_mesh(mesh_path), count, rng))
        atomic_save_npy(output, points)
        record = validate_cloud(output, count)
    return {
        "dataset_tag": str(task["dataset_tag"]),
        "shape_id": shape_id,
        "source_obj": str(mesh_path.resolve()),
        "sample_seed": int(task["sample_seed"]),
        "reused": reused,
        "output": record,
    }


def run_prepare(args: argparse.Namespace) -> Dict[str, object]:
    mesh_root = args.mesh_root.expanduser().resolve()
    output_root = args.output_root.expanduser().resolve()
    if not mesh_root.is_dir():
        raise FileNotFoundError(mesh_root)
    rows = read_split_files([path.expanduser().resolve() for path in args.split_file])
    if args.limit:
        rows = rows[: int(args.limit)]
    output_root.mkdir(parents=True, exist_ok=True)
    tasks = [
        {
            "shape_id": shape_id,
            "mesh_root": str(mesh_root),
            "output_root": str(output_root),
            "dataset_tag": args.dataset_tag,
            "points": int(args.points),
            "sample_seed": stable_seed(args.seed, args.dataset_tag, shape_id),
            "overwrite": bool(args.overwrite),
        }
        for shape_id in rows
    ]
    workers = max(1, int(args.workers))
    if workers == 1:
        records = [prepare_one(task) for task in tasks]
    else:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            records = list(pool.map(prepare_one, tasks))
    records.sort(key=lambda item: str(item["shape_id"]))
    summary = {
        "command": "meshes",
        "dataset_tag": args.dataset_tag,
        "mesh_root": str(mesh_root),
        "split_files": [str(path.expanduser().resolve()) for path in args.split_file],
        "output_root": str(output_root),
        "points": int(args.points),
        "seed": int(args.seed),
        "count": len(records),
        "written": sum(not bool(record["reused"]) for record in records),
        "reused": sum(bool(record["reused"]) for record in records),
        "created_at": datetime.now(timezone.utc).isoformat(),
        "records": records,
    }
    atomic_json(output_root / "manifest.json", summary)
    return summary


def parse_source(value: str) -> Tuple[str, Path]:
    if "=" not in value:
        raise ValueError("source must use TAG=PATH")
    tag, raw_path = value.split("=", 1)
    if tag not in ("a", "b"):
        raise ValueError("source tag must be a or b")
    path = Path(raw_path).expanduser().resolve()
    if not path.is_dir():
        raise FileNotFoundError(path)
    return tag, path


def link_or_copy(source: Path, destination: Path, mode: str) -> None:
    if mode == "copy":
        shutil.copy2(str(source), str(destination))
        return
    relative = os.path.relpath(str(source), str(destination.parent))
    os.symlink(relative, str(destination))


def run_joint(args: argparse.Namespace) -> Dict[str, object]:
    sources = [parse_source(value) for value in args.source]
    if len({tag for tag, _path in sources}) != len(sources):
        raise ValueError("each source tag may appear only once")
    output = args.output_root.expanduser().resolve()
    if output.exists():
        raise FileExistsError(f"refusing to replace existing output: {output}")
    staging = output.with_name(output.name + ".building")
    if staging.exists():
        raise FileExistsError(f"stale staging directory exists: {staging}")
    inventory: List[Tuple[str, Path]] = []
    for tag, root in sources:
        files = sorted(root.glob(f"{tag}__shapenet__*__*.npy"))
        if not files:
            raise FileNotFoundError(f"no prepared NPY files for source {tag}: {root}")
        inventory.extend((tag, path) for path in files)
    if len(inventory) != int(args.expected_count):
        raise ValueError(f"expected {args.expected_count} clouds, found {len(inventory)}")
    names = [path.name for _tag, path in inventory]
    if len(names) != len(set(names)):
        raise ValueError("joint sources contain duplicate output names")
    staging.parent.mkdir(parents=True, exist_ok=True)
    staging.mkdir()
    counts = {"a": 0, "b": 0}
    for tag, source in inventory:
        validate_cloud(source, int(args.points))
        link_or_copy(source, staging / source.name, args.mode)
        counts[tag] += 1
    manifest = {
        "command": "joint",
        "sources": {tag: str(path) for tag, path in sources},
        "output_root": str(output),
        "mode": args.mode,
        "points": int(args.points),
        "expected_count": int(args.expected_count),
        "counts": {**counts, "total": len(inventory)},
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    atomic_json(staging / "manifest.json", manifest)
    os.replace(str(staging), str(output))
    return manifest


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)

    meshes = commands.add_parser("meshes", help="按外部清单把 OBJ 转为 NPY")
    meshes.add_argument("--mesh-root", type=Path, required=True)
    meshes.add_argument("--split-file", type=Path, action="append", required=True)
    meshes.add_argument("--dataset-tag", choices=("a", "b"), required=True)
    meshes.add_argument("--output-root", type=Path, required=True)
    meshes.add_argument("--points", type=int, default=50000)
    meshes.add_argument("--seed", type=int, default=2023)
    meshes.add_argument("--workers", type=int, default=8)
    meshes.add_argument("--limit", type=int, default=0)
    meshes.add_argument("--overwrite", action="store_true")
    meshes.set_defaults(func=run_prepare)

    joint = commands.add_parser("joint", help="合并 A/B NPY 为平铺训练目录")
    joint.add_argument("--source", action="append", required=True, help="TAG=PATH，可分别传入 a 和 b")
    joint.add_argument("--output-root", type=Path, required=True)
    joint.add_argument("--expected-count", type=int, default=35632)
    joint.add_argument("--points", type=int, default=50000)
    joint.add_argument("--mode", choices=("symlink", "copy"), default="symlink")
    joint.set_defaults(func=run_joint)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if getattr(args, "points", 0) <= 0:
        raise ValueError("points must be positive")
    result = args.func(args)
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
