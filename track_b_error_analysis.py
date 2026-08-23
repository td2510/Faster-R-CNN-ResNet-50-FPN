"""Reproduce mutually-exclusive YOLO error counts from saved YOLO labels.

The matching policy is greedy by descending confidence. A prediction can match
one unmatched ground-truth box and every ground-truth box can be matched once.
The output categories are mutually exclusive: correct, misclass, mislocalized,
extra, and missed.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path


def read_yolo(path: Path, predicted: bool) -> list[tuple[int, list[float], float]]:
    rows: list[tuple[int, list[float], float]] = []
    if not path.exists():
        return rows
    for line in path.read_text(encoding="utf-8").splitlines():
        parts = line.split()
        if not parts:
            continue
        class_id = int(float(parts[0]))
        xc, yc, width, height = map(float, parts[1:5])
        box = [xc - width / 2, yc - height / 2, xc + width / 2, yc + height / 2]
        confidence = float(parts[5]) if predicted and len(parts) > 5 else 1.0
        rows.append((class_id, box, confidence))
    return rows


def iou(a: list[float], b: list[float]) -> float:
    left, top = max(a[0], b[0]), max(a[1], b[1])
    right, bottom = min(a[2], b[2]), min(a[3], b[3])
    intersection = max(0.0, right - left) * max(0.0, bottom - top)
    area_a = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])
    area_b = max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
    return intersection / (area_a + area_b - intersection + 1e-12)


def size_group(box: list[float]) -> str:
    area = max(0.0, box[2] - box[0]) * max(0.0, box[3] - box[1])
    if area < (32 / 640) ** 2:
        return "small"
    if area < (96 / 640) ** 2:
        return "medium"
    return "large"


def write_counter(path: Path, fieldnames: list[str], counter: Counter) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for key, count in sorted(counter.items()):
            values = key if isinstance(key, tuple) else (key,)
            writer.writerow(dict(zip(fieldnames[:-1], values)) | {"count": count})


def analyze(dataset_root: Path, prediction_dir: Path, output_dir: Path) -> dict:
    image_dir = dataset_root / "test" / "images"
    gt_dir = dataset_root / "test" / "labels"
    image_paths = sorted(path for path in image_dir.iterdir() if path.is_file())
    totals: Counter = Counter()
    by_size: Counter = Counter()
    by_class: Counter = Counter()
    misclass_pairs: Counter = Counter()
    images_with_error: Counter = Counter()

    for image_path in image_paths:
        stem = image_path.stem
        ground_truth = read_yolo(gt_dir / f"{stem}.txt", predicted=False)
        predictions = sorted(
            read_yolo(prediction_dir / f"{stem}.txt", predicted=True),
            key=lambda row: row[2],
            reverse=True,
        )
        matched_gt: set[int] = set()
        image_error_types: set[str] = set()

        for predicted_class, predicted_box, _ in predictions:
            candidates = [
                (iou(predicted_box, gt_box), index)
                for index, (_, gt_box, _) in enumerate(ground_truth)
                if index not in matched_gt
            ]
            best_iou, best_index = max(candidates, default=(0.0, -1))
            if best_iou >= 0.5:
                matched_gt.add(best_index)
                gt_class, gt_box, _ = ground_truth[best_index]
                error_type = "correct" if predicted_class == gt_class else "misclass"
                reference_box, reference_class = gt_box, gt_class
                if error_type == "misclass":
                    misclass_pairs[(gt_class, predicted_class)] += 1
            elif best_iou >= 0.1:
                matched_gt.add(best_index)
                gt_class, gt_box, _ = ground_truth[best_index]
                error_type = "mislocalized"
                reference_box, reference_class = gt_box, gt_class
            else:
                error_type = "extra"
                reference_box, reference_class = predicted_box, predicted_class

            group = size_group(reference_box)
            totals[error_type] += 1
            by_size[(error_type, group)] += 1
            by_class[(error_type, reference_class)] += 1
            if error_type != "correct":
                image_error_types.add(error_type)

        for index, (gt_class, gt_box, _) in enumerate(ground_truth):
            if index in matched_gt:
                continue
            error_type = "missed"
            group = size_group(gt_box)
            totals[error_type] += 1
            by_size[(error_type, group)] += 1
            by_class[(error_type, gt_class)] += 1
            image_error_types.add(error_type)

        for error_type in image_error_types:
            images_with_error[error_type] += 1

    output_dir.mkdir(parents=True, exist_ok=True)
    write_counter(output_dir / "error_analysis.csv", ["error_type", "count"], totals)
    write_counter(output_dir / "error_analysis_by_size.csv", ["error_type", "size_group", "count"], by_size)
    write_counter(output_dir / "error_analysis_by_class.csv", ["error_type", "class_id", "count"], by_class)
    write_counter(
        output_dir / "error_analysis_misclass_pairs.csv",
        ["ground_truth_class_id", "predicted_class_id", "count"],
        misclass_pairs,
    )
    summary = {
        "matching": "greedy confidence; IoU>=0.5 matched, 0.1<=IoU<0.5 mislocalized",
        "categories_mutually_exclusive": True,
        "test_images": len(image_paths),
        "ground_truth_instances": sum(1 for path in gt_dir.glob("*.txt") for line in path.read_text(encoding="utf-8").splitlines() if line.strip()),
        "prediction_instances": sum(1 for path in prediction_dir.glob("*.txt") for line in path.read_text(encoding="utf-8").splitlines() if line.strip()),
        "counts": dict(totals),
        "misclass_pairs": [
            {"ground_truth_class_id": gt, "predicted_class_id": pred, "count": count}
            for (gt, pred), count in misclass_pairs.most_common()
        ],
        "images_with_error_type": dict(images_with_error),
    }
    (output_dir / "error_analysis_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--prediction-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(analyze(args.dataset_root, args.prediction_dir, args.output_dir), indent=2))


if __name__ == "__main__":
    main()
