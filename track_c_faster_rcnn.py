"""Reproducible Track C runner for Faster R-CNN on YOLO-format data.

Commands:
  download   Download the public Google Drive folder (small folders only).
  extract    Extract browser-downloaded Google Drive ZIP files.
  roboflow   Download the exact public Roboflow v5 export with a private API key.
  subset     Select 3,000 train images deterministically and preserve val/test.
  audit      Validate split structure, labels, and bounding boxes.
  smoke      Run dataset/model/checkpoint smoke tests.
  train      Fine-tune for the configured number of epochs with resume.
  evaluate   Compute COCO mAP, FPS, and qualitative examples.
  benchmark  Re-evaluate YOLO and create the same-GPU comparison table.
  all        Audit, train, evaluate, and benchmark when YOLO weights exist.
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import json
import math
import os
import platform
import random
import shutil
import statistics
import sys
import time
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

REPO_ROOT = Path(__file__).resolve().parent
os.environ.setdefault("MPLCONFIGDIR", str(REPO_ROOT / "artifacts" / ".matplotlib"))
os.environ.setdefault("TORCH_HOME", str(REPO_ROOT / "artifacts" / ".torch"))
os.environ.setdefault("YOLO_CONFIG_DIR", str(REPO_ROOT / "artifacts" / ".ultralytics"))

import numpy as np
import pandas as pd
import torch
import torchvision
import yaml
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision.models.detection import (
    FasterRCNN_ResNet50_FPN_Weights,
    fasterrcnn_resnet50_fpn,
)
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.transforms import functional as TF


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
LOG_FIELDS = [
    "timestamp", "model", "run_name", "imgsz", "epochs", "batch",
    "seed", "conf", "iou", "precision", "recall", "map50",
    "map50_95", "fps", "num_params", "train_time_min", "notes",
]


def log(message: str) -> None:
    print(f"[{dt.datetime.now().isoformat(timespec='seconds')}] {message}", flush=True)


def replace_with_retry(tmp: Path, path: Path, attempts: int = 100) -> None:
    """Atomically replace a file while tolerating short Windows reader/AV locks."""
    for attempt in range(attempts):
        try:
            os.replace(tmp, path)
            return
        except PermissionError:
            if attempt + 1 == attempts:
                raise
            time.sleep(0.1)


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + f".{os.getpid()}.tmp")
    tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    replace_with_retry(tmp, path)


def atomic_torch_save(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + f".{os.getpid()}.tmp")
    torch.save(payload, tmp)
    replace_with_retry(tmp, path)


def tensor_to_json(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        if value.numel() == 1:
            return value.item()
        return value.detach().cpu().tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(k): tensor_to_json(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [tensor_to_json(v) for v in value]
    return value


@dataclass(frozen=True)
class RuntimeChoice:
    batch_size: int
    amp: bool
    accumulation_steps: int

    @property
    def effective_batch_size(self) -> int:
        return self.batch_size * self.accumulation_steps

    def as_dict(self) -> dict[str, Any]:
        return {
            "batch_size": self.batch_size,
            "amp": self.amp,
            "accumulation_steps": self.accumulation_steps,
            "effective_batch_size": self.effective_batch_size,
        }


def runtime_choices() -> list[RuntimeChoice]:
    return [
        RuntimeChoice(4, False, 1),
        RuntimeChoice(4, True, 1),
        RuntimeChoice(2, True, 2),
    ]


class YoloDetectionDataset(Dataset):
    def __init__(self, root: Path, split: str) -> None:
        self.root = root
        self.split = split
        self.image_dir = root / split / "images"
        self.label_dir = root / split / "labels"
        self.images = sorted(
            path for path in self.image_dir.iterdir()
            if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
        )
        if not self.images:
            raise RuntimeError(f"No images found in {self.image_dir}")

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        image_path = self.images[index]
        image = Image.open(image_path).convert("RGB")
        width, height = image.size
        boxes: list[list[float]] = []
        labels: list[int] = []
        label_path = self.label_dir / f"{image_path.stem}.txt"
        if label_path.exists():
            for line_number, raw in enumerate(label_path.read_text(encoding="utf-8").splitlines(), 1):
                if not raw.strip():
                    continue
                parts = raw.split()
                if len(parts) != 5:
                    raise ValueError(f"{label_path}:{line_number}: expected 5 values")
                cls, xc, yc, bw, bh = map(float, parts)
                x1 = max(0.0, (xc - bw / 2) * width)
                y1 = max(0.0, (yc - bh / 2) * height)
                x2 = min(float(width), (xc + bw / 2) * width)
                y2 = min(float(height), (yc + bh / 2) * height)
                if not (0 <= x1 < x2 <= width and 0 <= y1 < y2 <= height):
                    raise ValueError(f"{label_path}:{line_number}: invalid pixel box {(x1, y1, x2, y2)}")
                boxes.append([x1, y1, x2, y2])
                labels.append(int(cls) + 1)  # torchvision reserves 0 for background

        box_tensor = torch.tensor(boxes, dtype=torch.float32).reshape(-1, 4)
        label_tensor = torch.tensor(labels, dtype=torch.int64)
        area = ((box_tensor[:, 2] - box_tensor[:, 0]) *
                (box_tensor[:, 3] - box_tensor[:, 1])) if boxes else torch.zeros(0)
        target = {
            "boxes": box_tensor,
            "labels": label_tensor,
            "image_id": torch.tensor(index, dtype=torch.int64),
            "area": area.to(torch.float32),
            "iscrowd": torch.zeros(len(labels), dtype=torch.uint8),
        }
        return TF.pil_to_tensor(image).to(torch.float32).div(255), target


def collate_fn(batch: Sequence[Any]) -> tuple[tuple[Any, ...], tuple[Any, ...]]:
    return tuple(zip(*batch))


def load_config(path: Path) -> dict[str, Any]:
    config = json.loads(path.read_text(encoding="utf-8"))
    repo = path.resolve().parent
    for key in (
        "dataset_root", "roboflow_root", "subset_root", "download_root",
        "incoming_dir", "roboflow_api_key_file", "artifact_dir", "yolo_weights",
    ):
        candidate = Path(config[key])
        config[key] = candidate if candidate.is_absolute() else repo / candidate
    config["raw_dataset_root"] = (
        config["roboflow_root"]
        if (Path(config["roboflow_root"]) / "data.yaml").exists()
        else config["dataset_root"]
    )
    config["dataset_root"] = (
        config["subset_root"]
        if (Path(config["subset_root"]) / "data.yaml").exists()
        else config["raw_dataset_root"]
    )
    config["config_path"] = path.resolve()
    return config


def set_seed(seed: int) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def worker_seed(worker_id: int) -> None:
    seed = torch.initial_seed() % (2**32)
    random.seed(seed)
    np.random.seed(seed)


def make_loader(
    dataset: Dataset,
    batch_size: int,
    shuffle: bool,
    config: dict[str, Any],
) -> DataLoader:
    generator = torch.Generator().manual_seed(int(config["seed"]))
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=int(config["num_workers"]),
        pin_memory=torch.cuda.is_available(),
        collate_fn=collate_fn,
        worker_init_fn=worker_seed,
        generator=generator,
        persistent_workers=int(config["num_workers"]) > 0,
    )


def class_names(dataset_root: Path) -> list[str]:
    data_yaml = dataset_root / "data.yaml"
    if not data_yaml.exists():
        return []
    content = yaml.safe_load(data_yaml.read_text(encoding="utf-8")) or {}
    names = content.get("names", [])
    if isinstance(names, dict):
        return [str(names[key]) for key in sorted(names, key=lambda x: int(x))]
    return [str(name) for name in names]


def source_image_key(path: Path) -> str:
    """Collapse Roboflow augmentation hashes back to one original-image key."""
    return path.stem.split(".rf.", 1)[0]


def link_or_copy(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        return
    try:
        os.link(source, destination)
    except OSError:
        shutil.copy2(source, destination)


def create_reproducible_subset(config: dict[str, Any]) -> dict[str, Any]:
    """Select 3,000 seeded train images and preserve the official val/test."""
    raw_root = Path(config["raw_dataset_root"])
    subset_root = Path(config["subset_root"])
    max_images = int(config["max_dataset_images"])
    seed = int(config["seed"])
    if not raw_root.exists():
        raise FileNotFoundError(raw_root)

    selected_by_split: dict[str, list[Path]] = {}
    raw_counts: dict[str, int] = {}
    for split in ("train", "valid", "test"):
        image_dir = raw_root / split / "images"
        images = sorted(
            path for path in image_dir.iterdir()
            if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
        )
        raw_counts[split] = len(images)
        selected_by_split[split] = images

    train_pool = selected_by_split["train"]
    train_sample_size = min(int(config["train_sample_size"]), len(train_pool))
    selected_by_split["train"] = sorted(random.Random(seed).sample(train_pool, train_sample_size))
    selected_total = sum(len(paths) for paths in selected_by_split.values())
    if selected_total > max_images:
        raise ValueError(
            f"Seeded train sample plus full val/test has {selected_total} images, "
            f"above max_dataset_images={max_images}"
        )

    manifest_rows: list[dict[str, Any]] = []
    for split, images in selected_by_split.items():
        for image in sorted(images):
            label = raw_root / split / "labels" / f"{image.stem}.txt"
            image_destination = subset_root / split / "images" / image.name
            label_destination = subset_root / split / "labels" / label.name
            link_or_copy(image, image_destination)
            if label.exists():
                link_or_copy(label, label_destination)
            else:
                label_destination.parent.mkdir(parents=True, exist_ok=True)
            manifest_rows.append({
                "split": split, "source_key": source_image_key(image),
                "image": image.name, "label_exists": label.exists(),
            })

    source_yaml = raw_root / "data.yaml"
    yaml_payload = yaml.safe_load(source_yaml.read_text(encoding="utf-8")) if source_yaml.exists() else {}
    yaml_payload.update({
        "path": ".", "train": "train/images",
        "val": "valid/images", "test": "test/images",
        "nc": int(config["num_classes"]),
    })
    subset_root.mkdir(parents=True, exist_ok=True)
    (subset_root / "data.yaml").write_text(
        yaml.safe_dump(yaml_payload, allow_unicode=True, sort_keys=False), encoding="utf-8"
    )
    manifest = pd.DataFrame(manifest_rows)
    manifest.to_csv(subset_root / "subset_manifest.csv", index=False)
    artifact_dir = Path(config["artifact_dir"])
    artifact_dir.mkdir(parents=True, exist_ok=True)
    manifest.to_csv(artifact_dir / "subset_manifest.csv", index=False)
    report = {
        "raw_counts": raw_counts,
        "selected_counts": {split: len(paths) for split, paths in selected_by_split.items()},
        "raw_total": sum(raw_counts.values()), "selected_total": len(manifest_rows),
        "max_images": max_images, "train_sample_size": train_sample_size, "seed": seed,
        "selection": "random.Random(seed).sample(sorted(train_images), train_sample_size); full val/test",
    }
    atomic_json(artifact_dir / "subset_report.json", report)
    log(f"Subset ready: {report}")
    return report


def audit_dataset(config: dict[str, Any]) -> dict[str, Any]:
    root = Path(config["dataset_root"])
    expected_classes = int(config["num_classes"])
    if not root.exists():
        raise FileNotFoundError(f"Dataset root not found: {root}")
    names = class_names(root)
    if names and len(names) != expected_classes:
        raise ValueError(f"data.yaml has {len(names)} names, expected {expected_classes}")

    report: dict[str, Any] = {"dataset_root": str(root), "classes": len(names) or expected_classes, "splits": {}}
    errors: list[str] = []
    for split in ("train", "valid", "test"):
        image_dir = root / split / "images"
        label_dir = root / split / "labels"
        if not image_dir.is_dir() or not label_dir.is_dir():
            errors.append(f"missing split directories: {split}")
            continue
        images = sorted(p for p in image_dir.iterdir() if p.suffix.lower() in IMAGE_EXTENSIONS)
        labels = {p.stem: p for p in label_dir.glob("*.txt")}
        instances = 0
        empty_images = 0
        class_hist = [0] * expected_classes
        for image_path in images:
            label_path = labels.get(image_path.stem)
            if label_path is None:
                empty_images += 1
                continue
            with Image.open(image_path) as image:
                width, height = image.size
            rows = [row for row in label_path.read_text(encoding="utf-8").splitlines() if row.strip()]
            if not rows:
                empty_images += 1
            for line_number, row in enumerate(rows, 1):
                try:
                    parts = row.split()
                    if len(parts) != 5:
                        raise ValueError("expected 5 columns")
                    cls_f, xc, yc, bw, bh = map(float, parts)
                    cls = int(cls_f)
                    if cls_f != cls or not 0 <= cls < expected_classes:
                        raise ValueError(f"class {cls_f} outside 0..{expected_classes - 1}")
                    if not all(math.isfinite(v) for v in (xc, yc, bw, bh)):
                        raise ValueError("non-finite coordinate")
                    if not (0 <= xc <= 1 and 0 <= yc <= 1 and 0 < bw <= 1 and 0 < bh <= 1):
                        raise ValueError("normalized values outside valid range")
                    x1, y1 = (xc - bw / 2) * width, (yc - bh / 2) * height
                    x2, y2 = (xc + bw / 2) * width, (yc + bh / 2) * height
                    tolerance = 1e-3
                    if not (-tolerance <= x1 < x2 <= width + tolerance and
                            -tolerance <= y1 < y2 <= height + tolerance):
                        raise ValueError("box extends outside image")
                    class_hist[cls] += 1
                    instances += 1
                except Exception as exc:
                    errors.append(f"{label_path}:{line_number}: {exc}")
        orphans = sorted(stem for stem in labels if not any((image_dir / f"{stem}{ext}").exists() for ext in IMAGE_EXTENSIONS))
        report["splits"][split] = {
            "images": len(images), "label_files": len(labels), "instances": instances,
            "empty_or_missing_labels": empty_images, "orphan_labels": len(orphans),
            "class_histogram": class_hist,
        }
    report["errors"] = errors[:100]
    report["error_count"] = len(errors)
    artifact_dir = Path(config["artifact_dir"])
    atomic_json(artifact_dir / "dataset_audit.json", report)
    if errors:
        raise ValueError(f"Dataset audit found {len(errors)} errors; see dataset_audit.json")
    log(f"Dataset audit passed: {json.dumps(report['splits'], ensure_ascii=False)}")
    return report


def build_model(num_classes: int, pretrained: bool = True) -> torch.nn.Module:
    weights = FasterRCNN_ResNet50_FPN_Weights.COCO_V1 if pretrained else None
    model = fasterrcnn_resnet50_fpn(weights=weights, weights_backbone=None if not pretrained else None)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes + 1)
    return model


def environment_info(device: torch.device) -> dict[str, Any]:
    info: dict[str, Any] = {
        "timestamp": dt.datetime.now().isoformat(timespec="seconds"),
        "platform": platform.platform(),
        "python": sys.version,
        "torch": torch.__version__,
        "torchvision": torchvision.__version__,
        "cuda_available": torch.cuda.is_available(),
        "torch_cuda": torch.version.cuda,
        "device": str(device),
    }
    if torch.cuda.is_available():
        props = torch.cuda.get_device_properties(device)
        info.update({"gpu": props.name, "vram_gb": props.total_memory / 1024**3})
    return info


def require_cuda() -> torch.device:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable. Install the cu132 PyTorch wheel and retry.")
    return torch.device("cuda:0")


def optimizer_for(model: torch.nn.Module, config: dict[str, Any]) -> torch.optim.Optimizer:
    return torch.optim.SGD(
        [parameter for parameter in model.parameters() if parameter.requires_grad],
        lr=float(config["learning_rate"]), momentum=float(config["momentum"]),
        weight_decay=float(config["weight_decay"]),
    )


def move_targets(targets: Iterable[dict[str, torch.Tensor]], device: torch.device) -> list[dict[str, torch.Tensor]]:
    return [{key: value.to(device, non_blocking=True) for key, value in target.items()} for target in targets]


def rng_state() -> dict[str, Any]:
    return {
        "python": random.getstate(), "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
        "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
    }


def restore_rng(state: dict[str, Any] | None) -> None:
    if not state:
        return
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"].cpu())
    if torch.cuda.is_available() and state.get("cuda") is not None:
        torch.cuda.set_rng_state_all([item.cpu() for item in state["cuda"]])


def write_history(path: Path, history: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(history).to_csv(path, index=False)


def write_heartbeat(config: dict[str, Any], **fields: Any) -> None:
    payload = {"timestamp": dt.datetime.now().isoformat(timespec="seconds"), **fields}
    atomic_json(Path(config["artifact_dir"]) / "heartbeat.json", tensor_to_json(payload))


def preflight_choice(config: dict[str, Any], dataset: Dataset, device: torch.device) -> RuntimeChoice:
    choices = runtime_choices()
    if not torch.cuda.is_available():
        return RuntimeChoice(1, False, 1)
    for choice in choices:
        model: torch.nn.Module | None = None
        optimizer: torch.optim.Optimizer | None = None
        try:
            torch.cuda.empty_cache()
            model = build_model(int(config["num_classes"]), pretrained=False).to(device)
            optimizer = optimizer_for(model, config)
            scaler = torch.amp.GradScaler("cuda", enabled=choice.amp)
            loader = make_loader(dataset, choice.batch_size, True, config)
            images, targets = next(iter(loader))
            images = [image.to(device, non_blocking=True) for image in images]
            targets_gpu = move_targets(targets, device)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=choice.amp):
                losses = model(images, targets_gpu)
                loss = sum(losses.values()) / choice.accumulation_steps
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            log(f"VRAM preflight passed: {choice.as_dict()}")
            return choice
        except torch.OutOfMemoryError:
            log(f"VRAM preflight OOM: {choice.as_dict()}")
        finally:
            del model, optimizer
            torch.cuda.empty_cache()
    raise RuntimeError("All VRAM fallback configurations failed")


def checkpoint_payload(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    history: list[dict[str, Any]],
    elapsed_seconds: float,
    choice: RuntimeChoice,
    config: dict[str, Any],
    scaler: torch.amp.GradScaler,
) -> dict[str, Any]:
    return {
        "model": model.state_dict(), "optimizer": optimizer.state_dict(), "epoch": epoch,
        "history": history, "elapsed_seconds": elapsed_seconds, "runtime": choice.as_dict(),
        "rng": rng_state(), "scaler": scaler.state_dict(),
        "config": {k: tensor_to_json(v) for k, v in config.items()},
    }


def train_once(
    config: dict[str, Any],
    choice: RuntimeChoice,
    device: torch.device,
    force_runtime: bool = False,
) -> Path:
    artifact_dir = Path(config["artifact_dir"])
    artifact_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = artifact_dir / "fasterrcnn_ckpt.pt"
    final_path = artifact_dir / "fasterrcnn_baseline.pt"
    history_path = artifact_dir / "train_history.csv"
    dataset = YoloDetectionDataset(Path(config["dataset_root"]), "train")
    loader = make_loader(dataset, choice.batch_size, True, config)
    model = build_model(int(config["num_classes"]), pretrained=True).to(device)
    optimizer = optimizer_for(model, config)
    scaler = torch.amp.GradScaler("cuda", enabled=choice.amp)
    start_epoch, history, elapsed_before = 0, [], 0.0

    if checkpoint_path.exists():
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        start_epoch = int(checkpoint["epoch"]) + 1
        history = list(checkpoint.get("history", []))
        elapsed_before = float(checkpoint.get("elapsed_seconds", 0.0))
        saved_runtime = checkpoint.get("runtime", {})
        if saved_runtime and not force_runtime:
            choice = RuntimeChoice(
                int(saved_runtime["batch_size"]), bool(saved_runtime["amp"]),
                int(saved_runtime["accumulation_steps"]),
            )
            loader = make_loader(dataset, choice.batch_size, True, config)
            scaler = torch.amp.GradScaler("cuda", enabled=choice.amp)
        if checkpoint.get("scaler") and not force_runtime:
            scaler.load_state_dict(checkpoint["scaler"])
        restore_rng(checkpoint.get("rng"))
        mode = "forced fallback" if force_runtime else "saved runtime"
        log(f"Resuming from epoch {start_epoch} with {choice.as_dict()} ({mode})")

    if start_epoch >= int(config["epochs"]):
        log("Training already complete; preserving existing checkpoint")
        atomic_torch_save(final_path, model.state_dict())
        return final_path

    run_start = time.perf_counter()
    heartbeat_every = max(1, int(config["heartbeat_batches"]))
    for epoch in range(start_epoch, int(config["epochs"])):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        epoch_loss = 0.0
        component_totals: dict[str, float] = {}
        epoch_start = time.perf_counter()
        for batch_index, (images, targets) in enumerate(loader):
            images_gpu = [image.to(device, non_blocking=True) for image in images]
            targets_gpu = move_targets(targets, device)
            try:
                with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=choice.amp):
                    loss_dict = model(images_gpu, targets_gpu)
                    raw_loss = sum(loss_dict.values())
                    loss = raw_loss / choice.accumulation_steps
                if not torch.isfinite(raw_loss):
                    raise FloatingPointError(f"Non-finite loss: {raw_loss.item()}")
                scaler.scale(loss).backward()
                should_step = ((batch_index + 1) % choice.accumulation_steps == 0 or
                               batch_index + 1 == len(loader))
                if should_step:
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad(set_to_none=True)
            except torch.OutOfMemoryError:
                optimizer.zero_grad(set_to_none=True)
                torch.cuda.empty_cache()
                raise
            raw_value = float(raw_loss.detach().item())
            epoch_loss += raw_value
            for name, value in loss_dict.items():
                component_totals[name] = component_totals.get(name, 0.0) + float(value.detach().item())
            if batch_index % heartbeat_every == 0:
                done = epoch * len(loader) + batch_index + 1
                total = int(config["epochs"]) * len(loader)
                elapsed = elapsed_before + time.perf_counter() - run_start
                eta = elapsed / done * max(0, total - done) if done else None
                write_heartbeat(
                    config, status="training", epoch=epoch + 1, epochs=int(config["epochs"]),
                    batch=batch_index + 1, batches=len(loader), loss=raw_value,
                    elapsed_seconds=elapsed, eta_seconds=eta, runtime=choice.as_dict(), pid=os.getpid(),
                )
                log(f"epoch={epoch + 1}/{config['epochs']} batch={batch_index + 1}/{len(loader)} loss={raw_value:.4f}")

        epoch_seconds = time.perf_counter() - epoch_start
        row = {
            "epoch": epoch + 1, "loss": epoch_loss / len(loader), "seconds": epoch_seconds,
            "runtime_batch": choice.batch_size, "effective_batch": choice.effective_batch_size,
            "amp": choice.amp,
        }
        for name, total in component_totals.items():
            row[name] = total / len(loader)
        history.append(row)
        elapsed_total = elapsed_before + time.perf_counter() - run_start
        write_history(history_path, history)
        atomic_torch_save(
            checkpoint_path,
            checkpoint_payload(model, optimizer, epoch, history, elapsed_total, choice, config, scaler),
        )
        write_heartbeat(
            config, status="checkpointed", epoch=epoch + 1, epochs=int(config["epochs"]),
            loss=row["loss"], elapsed_seconds=elapsed_total, runtime=choice.as_dict(), pid=os.getpid(),
        )
        log(f"Epoch {epoch + 1} complete: mean_loss={row['loss']:.4f}, seconds={epoch_seconds:.1f}")

    atomic_torch_save(final_path, model.state_dict())
    write_heartbeat(config, status="training_complete", epoch=int(config["epochs"]), pid=os.getpid())
    return final_path


def train(config: dict[str, Any]) -> Path:
    device = require_cuda()
    set_seed(int(config["seed"]))
    checkpoint = Path(config["artifact_dir"]) / "fasterrcnn_ckpt.pt"
    choices = runtime_choices()
    if checkpoint.exists():
        saved = torch.load(checkpoint, map_location="cpu", weights_only=False)
        runtime = saved.get("runtime", {})
        choice = RuntimeChoice(
            int(runtime.get("batch_size", config["train_batch_size"])),
            bool(runtime.get("amp", False)), int(runtime.get("accumulation_steps", 1)),
        )
    else:
        choice = preflight_choice(
            config, YoloDetectionDataset(Path(config["dataset_root"]), "train"), device,
        )
    atomic_json(Path(config["artifact_dir"]) / "environment.json", environment_info(device))
    try:
        start_index = choices.index(choice)
    except ValueError:
        start_index = 0
    last_error: torch.OutOfMemoryError | None = None
    for index in range(start_index, len(choices)):
        candidate = choices[index]
        try:
            return train_once(
                config, candidate, device,
                force_runtime=index > start_index,
            )
        except torch.OutOfMemoryError as exc:
            last_error = exc
            torch.cuda.empty_cache()
            write_heartbeat(
                config, status="cuda_oom_fallback", failed_runtime=candidate.as_dict(),
                next_runtime=choices[index + 1].as_dict() if index + 1 < len(choices) else None,
                pid=os.getpid(),
            )
            log(f"CUDA OOM during training with {candidate.as_dict()}; trying next fallback")
    raise RuntimeError("All runtime configurations failed during training") from last_error


def append_experiment(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    exists = path.exists()
    with path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=LOG_FIELDS)
        if not exists:
            writer.writeheader()
        writer.writerow({key: row.get(key, "") for key in LOG_FIELDS})


def load_final_model(config: dict[str, Any], device: torch.device, weights_path: Path | None = None) -> torch.nn.Module:
    weights_path = weights_path or Path(config["artifact_dir"]) / "fasterrcnn_baseline.pt"
    if not weights_path.exists():
        raise FileNotFoundError(weights_path)
    model = build_model(int(config["num_classes"]), pretrained=False)
    state = torch.load(weights_path, map_location=device, weights_only=True)
    model.load_state_dict(state)
    return model.to(device).eval()


def evaluate(config: dict[str, Any], weights_path: Path | None = None) -> dict[str, Any]:
    from torchmetrics.detection.mean_ap import MeanAveragePrecision

    device = require_cuda()
    model = load_final_model(config, device, weights_path)
    dataset = YoloDetectionDataset(Path(config["dataset_root"]), "valid")
    loader = make_loader(dataset, int(config["val_batch_size"]), False, config)
    metric = MeanAveragePrecision(box_format="xyxy", iou_type="bbox", class_metrics=True)
    write_heartbeat(config, status="evaluating_map", pid=os.getpid())
    with torch.inference_mode():
        for batch_index, (images, targets) in enumerate(loader):
            images_gpu = [image.to(device, non_blocking=True) for image in images]
            predictions = model(images_gpu)
            preds_cpu = [
                {key: value.detach().cpu() for key, value in pred.items() if key in {"boxes", "labels", "scores"}}
                for pred in predictions
            ]
            targets_cpu = [
                {key: value.detach().cpu() for key, value in target.items() if key in {"boxes", "labels"}}
                for target in targets
            ]
            metric.update(preds_cpu, targets_cpu)
            if batch_index % 20 == 0:
                write_heartbeat(config, status="evaluating_map", batch=batch_index + 1, batches=len(loader), pid=os.getpid())
    result = metric.compute()

    test_dataset = YoloDetectionDataset(Path(config["dataset_root"]), "test")
    image_paths = test_dataset.images[: int(config["benchmark_images"])]
    times: list[float] = []
    with torch.inference_mode():
        for image_path in image_paths:
            image = Image.open(image_path).convert("RGB")
            tensor = TF.pil_to_tensor(image).to(torch.float32).div(255).to(device)
            torch.cuda.synchronize()
            start = time.time()
            model([tensor])
            torch.cuda.synchronize()
            times.append(time.time() - start)
    fps = 1 / (sum(times) / len(times))
    history_path = Path(config["artifact_dir"]) / "train_history.csv"
    train_time = float(pd.read_csv(history_path)["seconds"].sum() / 60) if history_path.exists() else None
    metrics = {
        "model": "FasterRCNN-ResNet50FPN", "map50": result["map_50"].item(),
        "map50_95": result["map"].item(), "map_small": result["map_small"].item(),
        "map_medium": result["map_medium"].item(), "map_large": result["map_large"].item(),
        "map_per_class": tensor_to_json(result["map_per_class"]),
        "classes": tensor_to_json(result["classes"]), "fps": fps,
        "benchmark_images": len(image_paths), "num_params": sum(p.numel() for p in model.parameters()),
        "train_time_min": train_time, "transform_min_size": list(model.transform.min_size),
        "transform_max_size": model.transform.max_size,
    }
    artifact_dir = Path(config["artifact_dir"])
    atomic_json(artifact_dir / "fasterrcnn_metrics.json", metrics)
    checkpoint = torch.load(artifact_dir / "fasterrcnn_ckpt.pt", map_location="cpu", weights_only=False)
    runtime = checkpoint.get("runtime", {})
    append_experiment(artifact_dir / "experiment_log.csv", {
        "timestamp": dt.datetime.now().isoformat(timespec="seconds"),
        "model": metrics["model"], "run_name": "baseline_640", "imgsz": 640,
        "epochs": int(config["epochs"]), "batch": runtime.get("batch_size", ""),
        "seed": int(config["seed"]), "map50": metrics["map50"],
        "map50_95": metrics["map50_95"], "fps": round(fps, 2),
        "num_params": metrics["num_params"], "train_time_min": train_time,
        "notes": ("torchvision default resize; 640 is comparison label; "
                  f"amp={runtime.get('amp')}; effective_batch={runtime.get('effective_batch_size')}")
    })
    save_examples(config, model, device, image_paths[:4])
    write_heartbeat(config, status="evaluation_complete", metrics=metrics, pid=os.getpid())
    log(f"Evaluation complete: mAP50={metrics['map50']:.4f}, mAP50-95={metrics['map50_95']:.4f}, FPS={fps:.2f}")
    return metrics


def save_examples(config: dict[str, Any], model: torch.nn.Module, device: torch.device, paths: Sequence[Path]) -> None:
    from torchvision.utils import draw_bounding_boxes

    output_dir = Path(config["artifact_dir"]) / "inference_examples"
    output_dir.mkdir(parents=True, exist_ok=True)
    names = class_names(Path(config["dataset_root"]))
    with torch.inference_mode():
        for path in paths:
            image_uint8 = torchvision.io.read_image(str(path), mode=torchvision.io.ImageReadMode.RGB)
            pred = model([image_uint8.to(device).float().div(255)])[0]
            keep = pred["scores"] >= 0.25
            labels = []
            for cls, score in zip(pred["labels"][keep].tolist(), pred["scores"][keep].tolist()):
                name = names[cls - 1] if 0 < cls <= len(names) else str(cls)
                labels.append(f"{name} {score:.2f}")
            drawn = draw_bounding_boxes(
                image_uint8, pred["boxes"][keep].detach().cpu(), labels=labels,
                colors="red", width=2, font_size=14,
            )
            torchvision.io.write_png(drawn, str(output_dir / f"{path.stem}.png"))


def runtime_data_yaml(config: dict[str, Any]) -> Path:
    root = Path(config["dataset_root"]).resolve()
    names = class_names(root)
    path = Path(config["artifact_dir"]) / "data_runtime.yaml"
    payload = {
        "path": str(root), "train": "train/images", "val": "valid/images",
        "test": "test/images", "nc": int(config["num_classes"]), "names": names,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(payload, allow_unicode=True, sort_keys=False), encoding="utf-8")
    return path


def benchmark_yolo(config: dict[str, Any], yolo_weights: Path | None = None) -> dict[str, Any] | None:
    yolo_weights = yolo_weights or Path(config["yolo_weights"])
    if not yolo_weights.exists():
        log(f"YOLO weights not found yet: {yolo_weights}; benchmark deferred")
        write_heartbeat(config, status="waiting_for_yolo_weights", expected_path=str(yolo_weights), pid=os.getpid())
        return None
    from ultralytics import YOLO

    device = require_cuda()
    model = YOLO(str(yolo_weights))
    data_yaml = runtime_data_yaml(config)
    validation = model.val(data=str(data_yaml), split="val", imgsz=640, device=0, verbose=False)
    test_images = YoloDetectionDataset(Path(config["dataset_root"]), "test").images[: int(config["benchmark_images"])]
    # Preload the exact same images so disk/cache speed cannot favor either model.
    # Both measurements below include in-memory preprocessing, host-to-device copy,
    # model forward and postprocessing, at batch size 1, after ten warm-up images.
    loaded_images = []
    for path in test_images:
        with Image.open(path) as image:
            loaded_images.append(image.convert("RGB").copy())
    warmup_images = loaded_images[: min(10, len(loaded_images))]

    for image in warmup_images:
        model.predict(image, imgsz=640, device=0, verbose=False)
    benchmark_repetitions = 3
    yolo_repeat_fps: list[float] = []
    for _ in range(benchmark_repetitions):
        torch.cuda.synchronize()
        start = time.perf_counter()
        for image in loaded_images:
            model.predict(image, imgsz=640, device=0, verbose=False)
        torch.cuda.synchronize()
        yolo_repeat_fps.append(len(loaded_images) / (time.perf_counter() - start))
    fps = statistics.mean(yolo_repeat_fps)

    frcnn_model = load_final_model(config, device)
    with torch.inference_mode():
        for image in warmup_images:
            tensor = TF.pil_to_tensor(image).to(torch.float32).div(255).to(device)
            frcnn_model([tensor])
        frcnn_repeat_fps: list[float] = []
        for _ in range(benchmark_repetitions):
            torch.cuda.synchronize()
            start = time.perf_counter()
            for image in loaded_images:
                tensor = TF.pil_to_tensor(image).to(torch.float32).div(255).to(device)
                frcnn_model([tensor])
            torch.cuda.synchronize()
            frcnn_repeat_fps.append(len(loaded_images) / (time.perf_counter() - start))
    frcnn_fps = statistics.mean(frcnn_repeat_fps)
    benchmark_protocol = (
        "100 identical preloaded test images; batch=1; 10 warm-up images; "
        "includes RAM preprocessing, H2D, forward and postprocessing; excludes disk I/O"
    )
    yolo_metrics = {
        "model": "YOLOv8n", "resolution": 640, "map50": float(validation.box.map50),
        "map50_95": float(validation.box.map), "fps": fps,
        "num_params": sum(p.numel() for p in model.model.parameters()),
        "benchmark_images": len(loaded_images), "benchmark_repetitions": benchmark_repetitions,
        "fps_std": statistics.pstdev(yolo_repeat_fps), "benchmark_protocol": benchmark_protocol,
    }
    artifact_dir = Path(config["artifact_dir"])
    atomic_json(artifact_dir / "yolo_metrics_rtx4060.json", yolo_metrics)
    frcnn_path = artifact_dir / "fasterrcnn_metrics.json"
    if not frcnn_path.exists():
        raise FileNotFoundError("Run evaluate before benchmark")
    frcnn = json.loads(frcnn_path.read_text(encoding="utf-8"))
    comparison = pd.DataFrame([
        {"model": yolo_metrics["model"], "resolution": 640, "train_images": 3000,
         "train_set": "Track A canonical subset", "map50": yolo_metrics["map50"],
         "map50_95": yolo_metrics["map50_95"], "fps": fps, "num_params": yolo_metrics["num_params"],
         "fps_std": statistics.pstdev(yolo_repeat_fps), "evaluation_gpu": "RTX 4060",
         "benchmark_protocol": benchmark_protocol},
        {"model": frcnn["model"], "resolution": "torchvision-default (640 label)",
         "train_images": 3000, "train_set": "Track A canonical subset",
         "map50": frcnn["map50"], "map50_95": frcnn["map50_95"], "fps": frcnn_fps,
         "num_params": frcnn["num_params"], "fps_std": statistics.pstdev(frcnn_repeat_fps),
         "evaluation_gpu": "RTX 4060",
         "benchmark_protocol": benchmark_protocol},
    ])
    comparison.to_csv(artifact_dir / "comparison_rtx4060.csv", index=False)
    write_heartbeat(config, status="all_complete", comparison=comparison.to_dict(orient="records"), pid=os.getpid())
    log(f"Same-GPU comparison complete: {comparison.to_dict(orient='records')}")
    return yolo_metrics


def download_roboflow_dataset(config: dict[str, Any]) -> None:
    from roboflow import Roboflow

    key_path = Path(config["roboflow_api_key_file"])
    if not key_path.exists():
        raise FileNotFoundError(
            f"Place the Roboflow private API key in {key_path}; it will not be logged"
        )
    api_key = key_path.read_text(encoding="utf-8").strip()
    if not api_key or any(character.isspace() for character in api_key):
        raise ValueError(f"Invalid Roboflow API key file: {key_path}")

    destination = Path(config["roboflow_root"])
    destination.parent.mkdir(parents=True, exist_ok=True)
    log(
        "Requesting Roboflow export "
        f"{config['roboflow_workspace']}/{config['roboflow_project']} "
        f"version {config['roboflow_version']} as YOLOv8"
    )
    client = Roboflow(api_key=api_key)
    project = client.workspace(str(config["roboflow_workspace"])).project(
        str(config["roboflow_project"])
    )
    dataset = project.version(int(config["roboflow_version"])).download(
        "yolov8", location=str(destination), overwrite=False
    )
    actual = Path(dataset.location).resolve()
    if not (actual / "data.yaml").exists():
        raise FileNotFoundError(f"Roboflow export completed without data.yaml at {actual}")
    if actual != destination.resolve():
        raise RuntimeError(
            f"Roboflow SDK used unexpected location {actual}; expected {destination.resolve()}"
        )
    log(f"Roboflow dataset ready at {actual}")


def download_dataset(config: dict[str, Any]) -> None:
    import gdown

    output = Path(config["download_root"])
    output.mkdir(parents=True, exist_ok=True)
    log(f"Downloading public Drive folder into {output}")
    files = gdown.download_folder(
        url=str(config["drive_folder_url"]), output=str(output), quiet=False,
        use_cookies=False, remaining_ok=True,
    )
    if not files:
        raise RuntimeError("Google Drive folder download returned no files")
    log(f"Downloaded {len(files)} files")


def extract_downloads(config: dict[str, Any]) -> None:
    """Extract one or more ZIP parts produced by Google Drive's browser UI."""
    incoming = Path(config["incoming_dir"])
    archives = sorted(incoming.glob("*.zip")) if incoming.exists() else []
    if not archives:
        raise FileNotFoundError(f"No ZIP files found in {incoming}")

    repo_data_root = Path(config["download_root"]).parent
    download_root = Path(config["download_root"])
    for archive in archives:
        log(f"Inspecting archive {archive.name}")
        with zipfile.ZipFile(archive) as bundle:
            members = [member for member in bundle.infolist() if not member.is_dir()]
            if not members:
                log(f"Skipping empty archive {archive.name}")
                continue
            top_levels = {
                Path(member.filename.replace("\\", "/")).parts[0]
                for member in members
            }
            destination = repo_data_root if top_levels == {"BTL_DeTai4"} else download_root
            destination_resolved = destination.resolve()
            for member in members:
                target = (destination / member.filename).resolve()
                if destination_resolved not in target.parents and target != destination_resolved:
                    raise ValueError(f"Unsafe ZIP member: {member.filename}")
            bundle.extractall(destination)
            log(f"Extracted {len(members)} files from {archive.name} into {destination}")

    root = Path(config["dataset_root"])
    if not root.exists():
        candidates = list(download_root.rglob("data.yaml"))
        raise FileNotFoundError(
            f"Expected dataset at {root}; found data.yaml candidates: {candidates}"
        )
    preferred = download_root / "runs_yolo" / "baseline_640" / "weights" / "best.pt"
    weight_candidates = [preferred] if preferred.exists() else sorted(download_root.rglob("best.pt"))
    if weight_candidates:
        yolo_destination = Path(config["yolo_weights"])
        link_or_copy(weight_candidates[0], yolo_destination)
        log(f"Prepared YOLO checkpoint at {yolo_destination}")


def smoke_test(config: dict[str, Any]) -> None:
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    dataset = YoloDetectionDataset(Path(config["dataset_root"]), "train")
    image, target = dataset[0]
    assert image.ndim == 3 and image.shape[0] == 3
    assert target["boxes"].shape[-1] == 4
    if len(target["labels"]):
        assert int(target["labels"].min()) >= 1
        assert int(target["labels"].max()) <= int(config["num_classes"])
    model = build_model(int(config["num_classes"]), pretrained=False).to(device)
    model.train()
    losses = model([image.to(device)], [move_targets([target], device)[0]])
    loss = sum(losses.values())
    if not torch.isfinite(loss):
        raise FloatingPointError("Smoke loss is not finite")
    loss.backward()
    model.eval()
    with torch.inference_mode():
        prediction = model([image.to(device)])
    assert len(prediction) == 1 and {"boxes", "labels", "scores"} <= prediction[0].keys()
    smoke_path = Path(config["artifact_dir"]) / "smoke_checkpoint.pt"
    atomic_torch_save(smoke_path, model.state_dict())
    clone = build_model(int(config["num_classes"]), pretrained=False)
    clone.load_state_dict(torch.load(smoke_path, map_location="cpu", weights_only=True))
    smoke_path.unlink(missing_ok=True)
    atomic_json(Path(config["artifact_dir"]) / "smoke_test.json", {
        "status": "passed", "device": str(device), "loss": float(loss.detach().cpu()),
        "image_shape": list(image.shape), "boxes": len(target["boxes"]),
    })
    log(f"Smoke test passed on {device}: loss={loss.item():.4f}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "command",
        choices=[
            "download", "roboflow", "extract", "subset", "audit", "smoke",
            "train", "evaluate", "benchmark", "all",
        ],
    )
    parser.add_argument("--config", type=Path, default=Path(__file__).with_name("track_c_config.json"))
    parser.add_argument("--weights", type=Path)
    parser.add_argument("--yolo-weights", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    Path(config["artifact_dir"]).mkdir(parents=True, exist_ok=True)
    write_heartbeat(config, status=f"starting_{args.command}", pid=os.getpid())
    if args.command == "download":
        download_dataset(config)
    elif args.command == "roboflow":
        download_roboflow_dataset(config)
    elif args.command == "extract":
        extract_downloads(config)
    elif args.command == "subset":
        create_reproducible_subset(config)
    elif args.command == "audit":
        audit_dataset(config)
    elif args.command == "smoke":
        smoke_test(config)
    elif args.command == "train":
        train(config)
    elif args.command == "evaluate":
        evaluate(config, args.weights)
    elif args.command == "benchmark":
        benchmark_yolo(config, args.yolo_weights)
    elif args.command == "all":
        if not (Path(config["subset_root"]) / "data.yaml").exists():
            create_reproducible_subset(config)
            config["dataset_root"] = config["subset_root"]
        audit_dataset(config)
        smoke_test(config)
        train(config)
        evaluate(config, args.weights)
        benchmark_yolo(config, args.yolo_weights)


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        try:
            args = parse_args()
            config = load_config(args.config)
            write_heartbeat(config, status="failed", error=repr(exc), pid=os.getpid())
        except Exception:
            pass
        log(f"FAILED: {exc!r}")
        raise
