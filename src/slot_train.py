"""Fine-tune DINOv2 to classify normalized tray slot crops."""
from __future__ import annotations

import argparse
import random
from pathlib import Path

import albumentations as A
import cv2
import numpy as np
import torch
import yaml
from PIL import Image
from torch.utils.data import Dataset
from transformers import AutoImageProcessor, AutoModelForImageClassification, Trainer, TrainingArguments


LABELS = {"Empty": 0, "TLens": 1}


class SlotDataset(Dataset):
    def __init__(self, root: Path, processor: AutoImageProcessor, augment: bool):
        self.processor, self.augment = processor, augment
        self.records = [(path, LABELS[label]) for label in LABELS for path in sorted((root / label).glob("*.jpg"))]
        self.transforms = A.Compose([A.Rotate(10, p=.4), A.RandomBrightnessContrast(.12, .12, p=.4), A.GaussianBlur((3, 5), p=.1), A.GaussNoise(std_range=(.01, .03), p=.1)]) if augment else None

    def __len__(self) -> int: return len(self.records)

    def __getitem__(self, index: int) -> dict:
        path, label = self.records[index]; image = np.asarray(Image.open(path).convert("RGB"))
        if self.transforms: image = self.transforms(image=image)["image"]
        encoded = self.processor(images=[image], return_tensors="pt")
        return {"pixel_values": encoded["pixel_values"].squeeze(0), "labels": label}


def main() -> None:
    parser = argparse.ArgumentParser(); parser.add_argument("--config", type=Path, default=Path("configs/dinov2.yaml")); args = parser.parse_args()
    config = yaml.safe_load(args.config.read_text(encoding="utf-8")); random.seed(config["seed"]); np.random.seed(config["seed"]); torch.manual_seed(config["seed"])
    checkpoint, settings = config["model"]["checkpoint"], config["training"]
    processor = AutoImageProcessor.from_pretrained(checkpoint)
    model = AutoModelForImageClassification.from_pretrained(checkpoint, num_labels=2, id2label={0: "Empty", 1: "TLens"}, label2id=LABELS, ignore_mismatched_sizes=True)
    root = Path(config["data"]["dataset_dir"]); train_set, val_set = SlotDataset(root / "train", processor, True), SlotDataset(root / "val", processor, False)
    training_args = TrainingArguments(output_dir=config["model"]["output_dir"], num_train_epochs=settings["epochs"], learning_rate=settings["learning_rate"], per_device_train_batch_size=settings["batch_size"], per_device_eval_batch_size=settings["batch_size"], dataloader_num_workers=settings["num_workers"], eval_strategy="epoch", save_strategy="epoch", save_total_limit=settings["save_total_limit"], remove_unused_columns=False, report_to="none", load_best_model_at_end=True, metric_for_best_model="eval_loss", greater_is_better=False)
    trainer = Trainer(model=model, args=training_args, train_dataset=train_set, eval_dataset=val_set, data_collator=lambda batch: {"pixel_values": torch.stack([item["pixel_values"] for item in batch]), "labels": torch.tensor([item["labels"] for item in batch])})
    trainer.train(resume_from_checkpoint=settings.get("resume_from_checkpoint")); trainer.save_model(config["model"]["output_dir"]); processor.save_pretrained(config["model"]["output_dir"])


if __name__ == "__main__": main()
