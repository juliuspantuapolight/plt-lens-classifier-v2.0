"""Fine-tune MobileNetV3-Small on existing session-split slot datasets."""
import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader
from src.mobile_classifier import MobileProcessor, build_model
from src.slot_train import SlotDataset


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/mobilenet-bare.yaml"))
    parser.add_argument("--epochs", type=int, help="Override the configured training duration")
    args = parser.parse_args()
    config = yaml.safe_load(args.config.read_text())
    seed = config.get("seed", 42)
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    settings = config["training"]
    if args.epochs is not None:
        settings["epochs"] = args.epochs
    if settings["epochs"] < 1 or config["model"]["image_size"] < 32:
        raise ValueError("epochs must be positive and image_size must be at least 32")
    torch.set_num_threads(settings.get("cpu_threads", 4))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    processor = MobileProcessor(config["model"]["image_size"])
    root = Path(config["data"]["dataset_dir"])
    loaders = {}
    for split in ("train", "val"):
        dataset = SlotDataset(root / split, processor, split == "train")
        if not dataset.records or len({label for _, label in dataset.records}) != 2:
            raise ValueError(f"{split} must contain both Empty and TLens crops")
        loaders[split] = DataLoader(dataset, batch_size=settings["batch_size"], shuffle=split == "train",
                                    num_workers=settings.get("num_workers", 0))
    model = build_model(pretrained=True).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=settings["learning_rate"])
    output = Path(config["model"]["output_dir"])
    output.mkdir(parents=True, exist_ok=True)
    (output / "training_config.json").write_text(json.dumps(config, indent=2))
    best_loss = float("inf"); history = []
    for epoch in range(settings["epochs"]):
        record = {"epoch": epoch + 1}
        for split, loader in loaders.items():
            model.train(split == "train")
            loss_sum = correct = count = 0
            for batch in loader:
                images, labels = batch["pixel_values"].to(device), batch["labels"].to(device)
                with torch.set_grad_enabled(split == "train"):
                    logits = model(images)
                    loss = torch.nn.functional.cross_entropy(logits, labels)
                    if split == "train":
                        optimizer.zero_grad(); loss.backward(); optimizer.step()
                count += len(labels); loss_sum += loss.item() * len(labels)
                correct += (logits.argmax(1) == labels).sum().item()
            record[split + "_loss"] = loss_sum / count
            record[split + "_accuracy"] = correct / count
        if record["val_loss"] < best_loss:
            best_loss = record["val_loss"]
            torch.save({key: value.cpu() for key, value in model.state_dict().items()}, output / "model.pt")
            (output / "slot_model.json").write_text(json.dumps({"architecture": "mobilenet_v3_small",
                "image_size": processor.image_size, "labels": ["Empty", "TLens"]}, indent=2))
        history.append(record)
        (output / "training_history.json").write_text(json.dumps(history, indent=2))
        print(json.dumps(record), flush=True)


if __name__ == "__main__":
    main()
