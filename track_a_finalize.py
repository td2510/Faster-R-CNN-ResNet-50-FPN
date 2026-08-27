"""Finalize Track A dataset statistics and canonical YOLO error analysis.

This script intentionally keeps the main 3,000-image training subset separate
from the 9,536-image Roboflow-augmented training split.  It can also run the
canonical YOLOv8n checkpoint on the shared test set and produce mutually
exclusive error categories plus annotated examples.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import shutil
from collections import Counter, defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
from PIL import Image, ImageDraw, ImageFont


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
SIZE_ORDER = ("small", "medium", "large")
ERROR_COLORS = {
    "correct": "#28a745",
    "misclass": "#ff8c00",
    "mislocalized": "#9c27b0",
    "extra": "#dc3545",
    "missed": "#007bff",
}


def load_class_names(data_yaml: Path) -> list[str]:
    names: list[str] = []
    in_names = False
    for raw in data_yaml.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if line == "names:":
            in_names = True
            continue
        if in_names and line.startswith("-"):
            names.append(line[1:].strip().strip("'\""))
        elif in_names and line and not line.startswith("#"):
            break
    if not names:
        raise ValueError(f"Cannot read class names from {data_yaml}")
    return names


def image_paths(directory: Path) -> list[Path]:
    return sorted(p for p in directory.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES)


def parse_yolo_label(path: Path, predicted: bool = False) -> list[dict]:
    rows: list[dict] = []
    if not path.exists():
        return rows
    for line_number, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        parts = raw.split()
        if not parts:
            continue
        if len(parts) < 5:
            raise ValueError(f"Malformed label at {path}:{line_number}")
        class_id = int(float(parts[0]))
        xc, yc, width, height = map(float, parts[1:5])
        rows.append(
            {
                "class_id": class_id,
                "xywh": [xc, yc, width, height],
                "box": [xc - width / 2, yc - height / 2, xc + width / 2, yc + height / 2],
                "confidence": float(parts[5]) if predicted and len(parts) > 5 else 1.0,
            }
        )
    return rows


def size_group(box: list[float]) -> str:
    area = max(0.0, box[2] - box[0]) * max(0.0, box[3] - box[1])
    if area < (32 / 640) ** 2:
        return "small"
    if area < (96 / 640) ** 2:
        return "medium"
    return "large"


def iou(a: list[float], b: list[float]) -> float:
    left, top = max(a[0], b[0]), max(a[1], b[1])
    right, bottom = min(a[2], b[2]), min(a[3], b[3])
    intersection = max(0.0, right - left) * max(0.0, bottom - top)
    area_a = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])
    area_b = max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
    return intersection / (area_a + area_b - intersection + 1e-12)


def audit_split(dataset_root: Path, split: str, class_names: list[str]) -> dict:
    split_root = dataset_root / split
    images = image_paths(split_root / "images")
    labels_dir = split_root / "labels"
    class_counts: Counter[int] = Counter()
    size_counts: Counter[str] = Counter()
    empty_images = 0
    invalid_rows: list[str] = []
    for image_path in images:
        rows = parse_yolo_label(labels_dir / f"{image_path.stem}.txt")
        if not rows:
            empty_images += 1
        for row in rows:
            class_id = row["class_id"]
            box = row["box"]
            if not 0 <= class_id < len(class_names):
                invalid_rows.append(f"{image_path.name}: class_id={class_id}")
            if any(not math.isfinite(v) for v in box) or box[0] < 0 or box[1] < 0 or box[2] > 1 or box[3] > 1:
                invalid_rows.append(f"{image_path.name}: box={box}")
            class_counts[class_id] += 1
            size_counts[size_group(box)] += 1
    orphan_labels = sorted(p.name for p in labels_dir.glob("*.txt") if not any((split_root / "images" / f"{p.stem}{suffix}").exists() for suffix in IMAGE_SUFFIXES))
    return {
        "split": split,
        "images": len(images),
        "label_files": len(list(labels_dir.glob("*.txt"))),
        "boxes": sum(class_counts.values()),
        "empty_images": empty_images,
        "classes_present": len(class_counts),
        "class_counts": class_counts,
        "size_counts": size_counts,
        "invalid_rows": invalid_rows,
        "orphan_labels": orphan_labels,
    }


def write_dataset_outputs(dataset_root: Path, output_dir: Path, class_names: list[str]) -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)
    audits = [audit_split(dataset_root, split, class_names) for split in ("train_subset_3000", "valid", "test")]

    with (output_dir / "split_summary.csv").open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=["split", "images", "label_files", "boxes", "empty_images", "classes_present"])
        writer.writeheader()
        for audit in audits:
            writer.writerow({key: audit[key] for key in writer.fieldnames})

    train = audits[0]
    with (output_dir / "class_distribution_subset3000.csv").open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.writer(handle)
        writer.writerow(["class_id", "class_name", "box_count"])
        for class_id, class_name in enumerate(class_names):
            writer.writerow([class_id, class_name, train["class_counts"].get(class_id, 0)])

    with (output_dir / "size_distribution_subset3000.csv").open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.writer(handle)
        writer.writerow(["size_group", "box_count", "percentage"])
        total_boxes = train["boxes"]
        for group in SIZE_ORDER:
            count = train["size_counts"].get(group, 0)
            writer.writerow([group, count, count / total_boxes * 100 if total_boxes else 0])

    plt.figure(figsize=(15, 7))
    counts = [train["class_counts"].get(i, 0) for i in range(len(class_names))]
    colors = ["#e45756" if count == 0 else "#2463a7" for count in counts]
    plt.bar(range(len(class_names)), counts, color=colors)
    plt.xticks(range(len(class_names)), class_names, rotation=75, ha="right", fontsize=8)
    plt.ylabel("Số bounding box")
    plt.title("Phân bố lớp — tập train canonical 3.000 ảnh")
    plt.grid(axis="y", alpha=0.2)
    plt.tight_layout()
    plt.savefig(output_dir / "class_distribution_subset3000.png", dpi=180)
    plt.close()

    size_values = [train["size_counts"].get(group, 0) for group in SIZE_ORDER]
    plt.figure(figsize=(8, 5))
    bars = plt.bar(SIZE_ORDER, size_values, color=["#4c78a8", "#f2a541", "#59a14f"])
    for bar, value in zip(bars, size_values):
        plt.text(bar.get_x() + bar.get_width() / 2, value, f"{value}\n({value / train['boxes'] * 100:.1f}%)", ha="center", va="bottom")
    plt.ylabel("Số bounding box")
    plt.title("Phân bố kích thước box — tập train canonical 3.000 ảnh")
    plt.tight_layout()
    plt.savefig(output_dir / "size_distribution_subset3000.png", dpi=180)
    plt.close()

    class_by_split = {audit["split"]: audit["class_counts"] for audit in audits}
    missing_train = [
        {
            "class_id": i,
            "class_name": class_names[i],
            "train_boxes": class_by_split["train_subset_3000"].get(i, 0),
            "valid_boxes": class_by_split["valid"].get(i, 0),
            "test_boxes": class_by_split["test"].get(i, 0),
        }
        for i in range(len(class_names))
        if class_by_split["train_subset_3000"].get(i, 0) == 0
    ]
    summary = {
        "class_count_declared": len(class_names),
        "size_definition": "normalized box area; small < (32/640)^2, medium < (96/640)^2, otherwise large",
        "splits": [
            {key: value for key, value in audit.items() if key not in {"class_counts", "size_counts"}}
            | {"class_counts": dict(audit["class_counts"]), "size_counts": dict(audit["size_counts"])}
            for audit in audits
        ],
        "classes_missing_from_train": missing_train,
    }
    (output_dir / "dataset_subset3000_audit.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return summary


def run_yolo_prediction(dataset_root: Path, weights: Path, prediction_root: Path, force: bool) -> Path:
    labels_dir = prediction_root / "labels"
    expected = len(image_paths(dataset_root / "test" / "images"))
    if labels_dir.exists() and len(list(labels_dir.glob("*.txt"))) > 0 and not force:
        return labels_dir
    if prediction_root.exists() and force:
        shutil.rmtree(prediction_root)
    from ultralytics import YOLO
    import torch

    model = YOLO(str(weights))
    sources = [str(path) for path in image_paths(dataset_root / "test" / "images")]
    device = 0 if torch.cuda.is_available() else "cpu"
    model.predict(
        source=sources,
        imgsz=640,
        conf=0.25,
        iou=0.5,
        device=device,
        batch=16 if device != "cpu" else 8,
        save=False,
        save_txt=True,
        save_conf=True,
        project=str(prediction_root.parent),
        name=prediction_root.name,
        exist_ok=True,
        verbose=False,
    )
    return labels_dir


def match_image(gt: list[dict], predictions: list[dict]) -> list[dict]:
    events: list[dict] = []
    matched_gt: set[int] = set()
    for pred in sorted(predictions, key=lambda row: row["confidence"], reverse=True):
        candidates = [(iou(pred["box"], row["box"]), index) for index, row in enumerate(gt) if index not in matched_gt]
        best_iou, best_index = max(candidates, default=(0.0, -1))
        if best_iou >= 0.5:
            matched_gt.add(best_index)
            truth = gt[best_index]
            error_type = "correct" if pred["class_id"] == truth["class_id"] else "misclass"
            reference = truth
        elif best_iou >= 0.1:
            matched_gt.add(best_index)
            truth = gt[best_index]
            error_type = "mislocalized"
            reference = truth
        else:
            truth = None
            error_type = "extra"
            reference = pred
        events.append({"type": error_type, "gt": truth, "pred": pred, "iou": best_iou, "size": size_group(reference["box"])})
    for index, truth in enumerate(gt):
        if index not in matched_gt:
            events.append({"type": "missed", "gt": truth, "pred": None, "iou": 0.0, "size": size_group(truth["box"])})
    return events


def px_box(box: list[float], width: int, height: int) -> tuple[int, int, int, int]:
    return tuple(int(round(value * scale)) for value, scale in zip(box, (width, height, width, height)))


def draw_error_image(image_path: Path, events: list[dict], class_names: list[str], output_path: Path) -> None:
    image = Image.open(image_path).convert("RGB")
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default(size=16)
    width, height = image.size
    for event in events:
        if event["type"] == "correct":
            continue
        color = ERROR_COLORS[event["type"]]
        reference = event["gt"] or event["pred"]
        box = px_box(reference["box"], width, height)
        draw.rectangle(box, outline=color, width=max(2, int(min(width, height) / 250)))
        gt_name = class_names[event["gt"]["class_id"]] if event["gt"] else "-"
        pred_name = class_names[event["pred"]["class_id"]] if event["pred"] else "-"
        confidence = event["pred"]["confidence"] if event["pred"] else 0.0
        label = f"{event['type']} | GT:{gt_name} | Pred:{pred_name} | c={confidence:.2f} | IoU={event['iou']:.2f}"
        label_box = draw.textbbox((0, 0), label, font=font)
        label_width = label_box[2] - label_box[0]
        label_height = label_box[3] - label_box[1]
        x = max(0, min(box[0], width - label_width - 8))
        y = max(0, box[1] - label_height - 6)
        draw.rectangle((x, y, x + label_width + 8, y + label_height + 6), fill=color)
        draw.text((x + 4, y + 3), label, fill="white", font=font)
    image.save(output_path, quality=92)


def analyze_errors(dataset_root: Path, labels_dir: Path, output_dir: Path, class_names: list[str]) -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)
    examples_dir = output_dir / "error_examples_subset3000"
    examples_dir.mkdir(parents=True, exist_ok=True)
    totals: Counter[str] = Counter()
    by_size: Counter[tuple[str, str]] = Counter()
    by_class: Counter[tuple[str, int]] = Counter()
    pairs: Counter[tuple[int, int]] = Counter()
    candidates: list[tuple[Path, list[dict], set[str], set[str]]] = []
    gt_dir = dataset_root / "test" / "labels"
    images = image_paths(dataset_root / "test" / "images")
    prediction_instances = 0
    ground_truth_instances = 0

    for image_path in images:
        gt = parse_yolo_label(gt_dir / f"{image_path.stem}.txt")
        pred = parse_yolo_label(labels_dir / f"{image_path.stem}.txt", predicted=True)
        prediction_instances += len(pred)
        ground_truth_instances += len(gt)
        events = match_image(gt, pred)
        error_types = {event["type"] for event in events if event["type"] != "correct"}
        sizes = {event["size"] for event in events if event["type"] != "correct"}
        for event in events:
            totals[event["type"]] += 1
            by_size[(event["type"], event["size"])] += 1
            reference = event["gt"] or event["pred"]
            by_class[(event["type"], reference["class_id"])] += 1
            if event["type"] == "misclass":
                pairs[(event["gt"]["class_id"], event["pred"]["class_id"])] += 1
        if error_types:
            candidates.append((image_path, events, error_types, sizes))

    def write_counter(path: Path, header: list[str], rows: list[list]) -> None:
        with path.open("w", newline="", encoding="utf-8-sig") as handle:
            writer = csv.writer(handle)
            writer.writerow(header)
            writer.writerows(rows)

    write_counter(output_dir / "error_analysis_subset3000.csv", ["error_type", "count"], [[key, totals.get(key, 0)] for key in ("correct", "extra", "missed", "misclass", "mislocalized")])
    write_counter(
        output_dir / "error_analysis_subset3000_by_size.csv",
        ["error_type", "size_group", "count"],
        [[error, size, by_size.get((error, size), 0)] for error in ("correct", "extra", "missed", "misclass", "mislocalized") for size in SIZE_ORDER],
    )
    write_counter(
        output_dir / "error_analysis_subset3000_by_class.csv",
        ["error_type", "class_id", "class_name", "count"],
        [[error, class_id, class_names[class_id], count] for (error, class_id), count in sorted(by_class.items())],
    )
    write_counter(
        output_dir / "error_analysis_subset3000_misclass_pairs.csv",
        ["ground_truth_class_id", "ground_truth_class", "predicted_class_id", "predicted_class", "count"],
        [[gt, class_names[gt], pred, class_names[pred], count] for (gt, pred), count in pairs.most_common()],
    )

    error_order = ("extra", "missed", "misclass", "mislocalized")
    plt.figure(figsize=(8, 5))
    bars = plt.bar(error_order, [totals.get(key, 0) for key in error_order], color=[ERROR_COLORS[key] for key in error_order])
    for bar, key in zip(bars, error_order):
        plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height(), str(totals.get(key, 0)), ha="center", va="bottom")
    plt.ylabel("Số sự kiện")
    plt.title("Phân tích lỗi YOLOv8n canonical — 613 ảnh test")
    plt.tight_layout()
    plt.savefig(output_dir / "error_counts_subset3000.png", dpi=180)
    plt.close()

    plt.figure(figsize=(9, 5))
    bottoms = [0] * len(error_order)
    for group, color in zip(SIZE_ORDER, ("#4c78a8", "#f2a541", "#59a14f")):
        values = [by_size.get((key, group), 0) for key in error_order]
        plt.bar(error_order, values, bottom=bottoms, label=group, color=color)
        bottoms = [left + right for left, right in zip(bottoms, values)]
    plt.ylabel("Số sự kiện lỗi")
    plt.title("Lỗi theo kích thước vật thể")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / "error_by_size_subset3000.png", dpi=180)
    plt.close()

    selected: list[tuple[Path, list[dict], set[str], set[str]]] = []
    uncovered_types = {"extra", "missed", "misclass", "mislocalized"}
    uncovered_sizes = set(SIZE_ORDER)
    remaining = candidates.copy()
    while remaining and len(selected) < 12:
        best = max(remaining, key=lambda item: 4 * len(item[2] & uncovered_types) + 2 * len(item[3] & uncovered_sizes) + len(item[2]))
        selected.append(best)
        remaining.remove(best)
        uncovered_types -= best[2]
        uncovered_sizes -= best[3]
        if len(selected) >= 8 and not uncovered_types and not uncovered_sizes:
            break
    for index, (image_path, events, error_types, sizes) in enumerate(selected, 1):
        type_slug = "-".join(sorted(error_types))
        size_slug = "-".join(sorted(sizes))
        output_path = examples_dir / f"{index:02d}_{type_slug}_{size_slug}_{image_path.stem}.jpg"
        draw_error_image(image_path, events, class_names, output_path)

    error_events = sum(totals[k] for k in ("extra", "missed", "misclass", "mislocalized"))
    small_errors = sum(by_size[(k, "small")] for k in ("extra", "missed", "misclass", "mislocalized"))
    summary = {
        "model": "YOLOv8n baseline_640_subset3000",
        "weights": str((dataset_root.parent / "runs_yolo" / "baseline_640_subset3000" / "weights" / "best.pt").resolve()),
        "prediction_config": {"imgsz": 640, "conf": 0.25, "nms_iou": 0.5},
        "matching": "greedy by descending confidence; IoU>=0.5 match, 0.1<=IoU<0.5 mislocalized; mutually exclusive",
        "test_images": len(images),
        "ground_truth_instances": ground_truth_instances,
        "prediction_instances": prediction_instances,
        "counts": dict(totals),
        "errors_total": error_events,
        "small_errors": small_errors,
        "small_error_share": small_errors / error_events if error_events else 0,
        "misclass_pairs": [
            {"ground_truth_class": class_names[gt], "predicted_class": class_names[pred], "count": count}
            for (gt, pred), count in pairs.most_common()
        ],
        "example_images": [path.name for path in sorted(examples_dir.glob("*.jpg"))],
    }
    (output_dir / "error_analysis_subset3000_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return summary


def write_comparison_plots(root: Path, output_dir: Path) -> None:
    comparison_path = root / "runs_rcnn" / "comparison_rtx4060.csv"
    resolution_path = root / "runs_rcnn" / "track_b_analysis" / "yolo_resolution_benchmark_rtx4060.csv"
    with comparison_path.open(encoding="utf-8-sig") as handle:
        comparison = list(csv.DictReader(handle))
    labels = ["YOLOv8n\n640", "Faster R-CNN"]
    map50 = [float(row["map50"]) for row in comparison]
    map95 = [float(row["map50_95"]) for row in comparison]
    x = [0, 1]
    width = 0.35
    plt.figure(figsize=(8, 5))
    plt.bar([value - width / 2 for value in x], map50, width, label="mAP@0.5", color="#2463a7")
    plt.bar([value + width / 2 for value in x], map95, width, label="mAP@0.5:0.95", color="#f2a541")
    plt.xticks(x, labels)
    plt.ylim(0, 1)
    plt.ylabel("mAP")
    plt.title("So sánh có kiểm soát trên subset 3.000 ảnh")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / "comparison_map_subset3000.png", dpi=180)
    plt.close()

    with resolution_path.open(encoding="utf-8-sig") as handle:
        resolutions = list(csv.DictReader(handle))
    res_x = [int(row["resolution"]) for row in resolutions]
    res_map = [float(row["map50_95"]) for row in resolutions]
    res_fps = [float(row["fps_mean"]) for row in resolutions]
    figure, axis_map = plt.subplots(figsize=(8, 5))
    axis_fps = axis_map.twinx()
    line_map = axis_map.plot(res_x, res_map, marker="o", linewidth=2.5, color="#2463a7", label="mAP@0.5:0.95")
    line_fps = axis_fps.plot(res_x, res_fps, marker="s", linewidth=2.5, color="#e45756", label="FPS")
    axis_map.set_xlabel("Độ phân giải train/inference")
    axis_map.set_ylabel("mAP@0.5:0.95", color="#2463a7")
    axis_fps.set_ylabel("FPS RTX 4060", color="#e45756")
    axis_map.set_xticks(res_x)
    axis_map.grid(alpha=0.2)
    axis_map.legend(line_map + line_fps, [line.get_label() for line in line_map + line_fps], loc="center right")
    plt.title("Đánh đổi độ chính xác–tốc độ theo resolution (full-train)")
    figure.tight_layout()
    figure.savefig(output_dir / "resolution_tradeoff_rtx4060.png", dpi=180)
    plt.close(figure)

    history_path = root / "runs_rcnn" / "track_a_subset" / "train_history.csv"
    with history_path.open(encoding="utf-8-sig") as handle:
        history = list(csv.DictReader(handle))
    epochs = [int(float(row["epoch"])) for row in history]
    losses = [float(row["loss"]) for row in history]
    plt.figure(figsize=(8, 5))
    plt.plot(epochs, losses, marker="o", color="#2463a7", linewidth=2)
    plt.xlabel("Epoch")
    plt.ylabel("Loss trung bình")
    plt.title("Faster R-CNN — quá trình huấn luyện 18 epoch")
    plt.grid(alpha=0.25)
    plt.tight_layout()
    plt.savefig(output_dir / "fasterrcnn_train_loss.png", dpi=180)
    plt.close()

    metrics = json.loads((root / "runs_rcnn" / "track_a_subset" / "fasterrcnn_metrics.json").read_text(encoding="utf-8"))
    ap_values = [float(metrics[key]) for key in ("map_small", "map_medium", "map_large")]
    plt.figure(figsize=(8, 5))
    bars = plt.bar(SIZE_ORDER, ap_values, color=["#4c78a8", "#f2a541", "#59a14f"])
    for bar, value in zip(bars, ap_values):
        plt.text(bar.get_x() + bar.get_width() / 2, value, f"{value:.4f}", ha="center", va="bottom")
    plt.ylim(0, 0.8)
    plt.ylabel("AP@0.5:0.95")
    plt.title("Faster R-CNN — AP theo kích thước vật thể")
    plt.tight_layout()
    plt.savefig(output_dir / "fasterrcnn_ap_by_size.png", dpi=180)
    plt.close()

    full_error_path = root / "runs_rcnn" / "track_b_analysis" / "error_analysis.csv"
    with full_error_path.open(encoding="utf-8-sig") as handle:
        full_error_rows = {row["error_type"]: int(row["count"]) for row in csv.DictReader(handle)}
    full_error_order = ("misclass", "mislocalized", "missed", "extra")
    plt.figure(figsize=(8, 5))
    bars = plt.bar(
        full_error_order,
        [full_error_rows[key] for key in full_error_order],
        color=[ERROR_COLORS[key] for key in full_error_order],
    )
    for bar, key in zip(bars, full_error_order):
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height(),
            str(full_error_rows[key]),
            ha="center",
            va="bottom",
        )
    plt.ylabel("Số sự kiện lỗi")
    plt.title("Phân tích lỗi YOLOv8n full-train — 613 ảnh test")
    plt.tight_layout()
    plt.savefig(output_dir / "error_counts_full9536.png", dpi=180)
    plt.close()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("data/BTL_DeTai4_9000"))
    parser.add_argument("--output", type=Path, default=Path("data/BTL_DeTai4_9000/track_a_final/analysis"))
    parser.add_argument("--predict", action="store_true", help="Run canonical YOLO prediction before error analysis")
    parser.add_argument("--force-predict", action="store_true")
    args = parser.parse_args()

    root = args.root.resolve()
    dataset_root = root / "dataset"
    output_dir = args.output.resolve()
    class_names = load_class_names(dataset_root / "data.yaml")
    dataset_summary = write_dataset_outputs(dataset_root, output_dir, class_names)
    write_comparison_plots(root, output_dir)
    print(json.dumps({"dataset": dataset_summary}, ensure_ascii=False, indent=2))

    prediction_root = output_dir / "yolo_subset_predictions"
    if args.predict or (prediction_root / "labels").exists():
        weights = root / "runs_yolo" / "baseline_640_subset3000" / "weights" / "best.pt"
        labels_dir = run_yolo_prediction(dataset_root, weights, prediction_root, args.force_predict)
        error_summary = analyze_errors(dataset_root, labels_dir, output_dir, class_names)
        print(json.dumps({"errors": error_summary}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
