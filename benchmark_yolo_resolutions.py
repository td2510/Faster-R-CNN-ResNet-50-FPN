"""Benchmark the three Track B YOLO checkpoints with one controlled protocol."""

from __future__ import annotations

import argparse
import csv
import os
import statistics
import time
from pathlib import Path

os.environ.setdefault("YOLO_CONFIG_DIR", str(Path(__file__).resolve().parent / "artifacts" / ".ultralytics"))

import torch
from PIL import Image
from ultralytics import YOLO


METRICS = {
    416: (0.8134942816618811, 0.6491390811092780),
    640: (0.8684651747599308, 0.6965199987502475),
    960: (0.9037716068996364, 0.7340860981447221),
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs-root", type=Path, required=True)
    parser.add_argument("--test-images", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--repetitions", type=int, default=3)
    args = parser.parse_args()

    paths = sorted(path for path in args.test_images.iterdir() if path.is_file())[:100]
    loaded = []
    for path in paths:
        with Image.open(path) as image:
            loaded.append(image.convert("RGB").copy())

    rows = []
    for resolution in (416, 640, 960):
        model = YOLO(str(args.runs_root / f"baseline_{resolution}" / "weights" / "best.pt"))
        for image in loaded[:10]:
            model.predict(image, imgsz=resolution, device=0, verbose=False)
        repeat_fps = []
        for _ in range(args.repetitions):
            torch.cuda.synchronize()
            started = time.perf_counter()
            for image in loaded:
                model.predict(image, imgsz=resolution, device=0, verbose=False)
            torch.cuda.synchronize()
            repeat_fps.append(len(loaded) / (time.perf_counter() - started))
        map50, map50_95 = METRICS[resolution]
        rows.append({
            "model": "YOLOv8n",
            "resolution": resolution,
            "training_images": 9536,
            "map50": map50,
            "map50_95": map50_95,
            "fps_mean": statistics.mean(repeat_fps),
            "fps_std": statistics.pstdev(repeat_fps),
            "repetitions": args.repetitions,
            "images_per_repetition": len(loaded),
            "gpu": "RTX 4060",
            "protocol": "preloaded RGB; batch=1; 10 warm-up; includes RAM preprocessing, H2D, forward and postprocessing; excludes disk I/O",
        })

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    for row in rows:
        print(row)


if __name__ == "__main__":
    main()
