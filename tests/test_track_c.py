import json
import tempfile
import unittest
from pathlib import Path

import torch
from PIL import Image

import track_c_faster_rcnn as runner


class TrackCPipelineTests(unittest.TestCase):
    def make_export(self, root: Path) -> None:
        root.mkdir(parents=True, exist_ok=True)
        names = [f"class_{index}" for index in range(58)]
        (root / "data.yaml").write_text(
            "nc: 58\nnames:\n" + "".join(f"  - {name}\n" for name in names),
            encoding="utf-8",
        )
        for split in ("train", "valid", "test"):
            image_dir = root / split / "images"
            label_dir = root / split / "labels"
            image_dir.mkdir(parents=True)
            label_dir.mkdir(parents=True)
            for source_index in range(2):
                for export_index in range(3):
                    stem = f"{split}_{source_index}.rf.{export_index:032x}"
                    Image.new("RGB", (64, 48), (source_index * 50, 20, 30)).save(image_dir / f"{stem}.jpg")
                    (label_dir / f"{stem}.txt").write_text(
                        f"{source_index} 0.5 0.5 0.25 0.25\n", encoding="utf-8"
                    )

    def test_subset_audit_and_dataset_conversion(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            base = Path(temp)
            raw = base / "dataset"
            subset = base / "subset"
            artifacts = base / "artifacts"
            self.make_export(raw)
            config = {
                "raw_dataset_root": raw,
                "dataset_root": raw,
                "subset_root": subset,
                "max_dataset_images": 5000,
                "train_sample_size": 2,
                "seed": 42,
                "num_classes": 58,
                "artifact_dir": artifacts,
            }
            report = runner.create_reproducible_subset(config)
            self.assertEqual(report["raw_total"], 18)
            self.assertEqual(report["selected_total"], 14)
            config["dataset_root"] = subset
            audit = runner.audit_dataset(config)
            self.assertEqual(audit["error_count"], 0)
            dataset = runner.YoloDetectionDataset(subset, "train")
            image, target = dataset[0]
            self.assertEqual(tuple(image.shape), (3, 48, 64))
            self.assertEqual(tuple(target["boxes"].shape), (1, 4))
            self.assertGreaterEqual(int(target["labels"].min()), 1)
            saved = json.loads((artifacts / "subset_report.json").read_text(encoding="utf-8"))
            self.assertEqual(saved["selected_counts"], {"train": 2, "valid": 6, "test": 6})

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA smoke test")
    def test_model_forward_backward_cuda(self) -> None:
        device = torch.device("cuda:0")
        model = runner.build_model(58, pretrained=False).to(device).train()
        image = torch.rand(3, 96, 128, device=device)
        target = {
            "boxes": torch.tensor([[20.0, 15.0, 70.0, 60.0]], device=device),
            "labels": torch.tensor([1], dtype=torch.int64, device=device),
        }
        optimizer = torch.optim.SGD(model.parameters(), lr=0.005)
        scaler = torch.amp.GradScaler("cuda", enabled=True)
        with torch.autocast("cuda", dtype=torch.float16):
            loss = sum(model([image], [target]).values())
        self.assertTrue(torch.isfinite(loss))
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()


if __name__ == "__main__":
    unittest.main()
