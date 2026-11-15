"""Final Kaggle runner: Morphology-Aware Dual-View ConvNeXt (MADV).

No external plant data are used.  The backbone uses torchvision's official
ImageNet-1K pretrained ConvNeXt-Tiny weights (it is not trained from scratch).
The script first materializes a verified historical fallback when available,
then restores a complete five-checkpoint set or trains resumable MADV folds.
It never submits to Kaggle and never overwrites ``submission.csv``.
"""

from __future__ import annotations

import gc
import hashlib
import json
import os
import random
import shutil
import warnings
import zipfile
from contextlib import nullcontext
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd
from PIL import Image

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.model_selection import StratifiedKFold

warnings.filterwarnings("ignore")


# ============================================================
# CONFIG — validated ConvNeXt-Tiny settings; do not tune here
# ============================================================


@dataclass(frozen=True)
class Config:
    model_name: str = "convnext_tiny"
    seed: int = 42
    data_root: Path = Path("/kaggle/input/datasets/zijin233/neu-plant-seedlings-5class")
    output_dir: Path = Path("/kaggle/working")
    expected_train_images: int = 500
    expected_test_images: int = 378
    expected_classes: tuple[str, ...] = (
        "Black-grass",
        "Common wheat",
        "Loose Silky-bent",
        "Scentless Mayweed",
        "Sugar beet",
    )

    n_folds: int = 5
    img_size: int = 288
    batch_size: int = 16
    num_workers: int = 2

    head_epochs: int = 2
    finetune_epochs: int = 18
    head_lr: float = 3e-4
    backbone_lr: float = 4e-5
    classifier_lr: float = 2e-4
    weight_decay: float = 1e-4
    eta_min: float = 1e-6
    label_smoothing: float = 0.05
    gradient_clip_norm: float = 1.0
    patience: int = 5
    use_amp: bool = True
    use_tta: bool = False
    foreground_probability: float = 0.50
    foreground_margin: float = 0.20
    lambdas: tuple[float, ...] = (0.0, 0.15, 0.30)
    expected_reproduction_oof: float = 0.925738
    historical_public_score: float = 0.9470
    minimum_clear_delta: float = 0.003


CONFIG = Config()
IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg"}
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def checkpoint_path(config: Config, fold: int) -> Path:
    return config.output_dir / f"madv_convnext_tiny_fold{fold}.pth"


def last_checkpoint_path(config: Config, fold: int) -> Path:
    return config.output_dir / f"madv_convnext_tiny_fold{fold}_last.pth"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


# ============================================================
# REPRODUCIBILITY / DEVICE
# ============================================================


def seed_everything(seed: int) -> None:
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True


def worker_init_fn(worker_id: int) -> None:
    worker_seed = torch.initial_seed() % (2**32)
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def print_environment(device: torch.device) -> None:
    print("=" * 72)
    print("MADV-CONVNEXT FINAL RUNNER")
    print("=" * 72)
    print("PyTorch version :", torch.__version__)
    print("CUDA available  :", torch.cuda.is_available())
    print("Device          :", device)
    if device.type == "cuda":
        print("GPU             :", torch.cuda.get_device_name(0))
    print("Model           : ConvNeXt-Tiny")
    print("Image size      : 288")
    print("TTA             : False")
    print("Pretraining     : torchvision ImageNet-1K weights")
    print("External plants : none")
    print()


# ============================================================
# DATA INDEXING
# ============================================================


def list_images(directory: Path) -> list[Path]:
    return sorted(
        path
        for path in directory.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )


def build_dataframes(config: Config) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    train_dir = config.data_root / "train"
    test_dir = config.data_root / "test"
    assert train_dir.exists(), f"Train directory not found: {train_dir}"
    assert test_dir.exists(), f"Test directory not found: {test_dir}"

    class_names = sorted(path.name for path in train_dir.iterdir() if path.is_dir())
    assert len(class_names) == 5, f"Expected 5 classes, found {len(class_names)}"
    assert set(class_names) == set(config.expected_classes), (
        f"Expected {sorted(config.expected_classes)}, found {class_names}"
    )
    class_to_idx = {name: index for index, name in enumerate(class_names)}

    records = []
    for class_name in class_names:
        class_files = list_images(train_dir / class_name)
        assert len(class_files) == 100, (
            f"Expected 100 images for {class_name}, found {len(class_files)}"
        )
        for path in class_files:
            records.append(
                {
                    "path": str(path),
                    "filename": path.name,
                    "class_name": class_name,
                    "label": class_to_idx[class_name],
                }
            )
    train_df = pd.DataFrame(records)
    assert len(train_df) == config.expected_train_images

    test_paths = list_images(test_dir)
    assert len(test_paths) == config.expected_test_images, (
        f"Expected {config.expected_test_images} test images, found {len(test_paths)}"
    )
    test_df = pd.DataFrame(
        {"path": [str(path) for path in test_paths], "filename": [p.name for p in test_paths]}
    )
    return train_df, test_df, class_names


def assign_folds(train_df: pd.DataFrame, config: Config) -> pd.DataFrame:
    result = train_df.copy()
    result["fold"] = -1
    splitter = StratifiedKFold(
        n_splits=5,
        shuffle=True,
        random_state=42,
    )
    for fold, (_, validation_indices) in enumerate(
        splitter.split(result, result["label"])
    ):
        result.loc[validation_indices, "fold"] = fold
    assert (result["fold"] >= 0).all()
    return result


def find_submission_template(config: Config, test_filenames: Sequence[str]) -> pd.DataFrame:
    expected_ids = set(test_filenames)
    roots = [config.data_root, Path("/kaggle/input")]
    seen: set[Path] = set()
    for root in roots:
        if not root.exists():
            continue
        for path in root.rglob("submission-for-task2.csv"):
            if path in seen:
                continue
            seen.add(path)
            try:
                frame = pd.read_csv(path)
            except Exception:
                continue
            if {"ID", "Category"}.issubset(frame.columns) and set(frame["ID"]) == expected_ids:
                print("Using submission template:", path)
                return frame[["ID", "Category"]].copy()
    print("Matching submission template not found; using sorted test filenames.")
    return pd.DataFrame({"ID": sorted(test_filenames), "Category": ""})


# ============================================================
# TRANSFORMS / DATASET
# Exact augmentation from the validated ConvNeXt experiment
# ============================================================


def extract_foreground(
    image: Image.Image,
    margin: float = 0.20,
    return_info: bool = False,
):
    """Conservative class-agnostic HSV/Excess-Green crop with safe fallback."""
    rgb = image.convert("RGB")
    working = rgb.copy()
    working.thumbnail((512, 512), Image.Resampling.BILINEAR)
    array = np.asarray(working).astype(np.int16)
    red, green, blue = array[..., 0], array[..., 1], array[..., 2]
    hsv = np.asarray(working.convert("HSV"))
    hue, saturation, value = hsv[..., 0], hsv[..., 1], hsv[..., 2]
    excess_green = 2 * green - red - blue
    hsv_green = (hue >= 25) & (hue <= 105) & (saturation >= 45) & (value >= 25)
    mask = hsv_green & (excess_green >= 12) & (green >= 35)
    # Work at <=512 px, remove soil speckles, and retain the dominant connected plant.
    from scipy import ndimage
    mask = ndimage.binary_opening(mask, structure=np.ones((3, 3), dtype=bool))
    mask = ndimage.binary_closing(mask, structure=np.ones((7, 7), dtype=bool))
    connected = ndimage.binary_dilation(mask, iterations=2)
    labels, count = ndimage.label(connected)
    if count:
        sizes = np.bincount(labels.ravel()); sizes[0] = 0
        largest_size = int(sizes.max())
        # Keep substantial disconnected leaves, while rejecting tiny green soil speckles.
        minimum_component = max(12, int(round(0.08 * largest_size)))
        keep = np.flatnonzero(sizes >= minimum_component)
        mask = np.isin(labels, keep)
    else:
        mask = connected
    ys, xs = np.where(mask)
    width, height = rgb.size
    work_width, work_height = working.size
    coverage = float(mask.mean())
    info = {
        "used_foreground": False,
        "reason": "no_reliable_foreground",
        "coverage": coverage,
        "bbox": [0, 0, width, height],
    }
    if len(xs) < 16 or coverage < 0.003 or coverage > 0.75:
        return (rgb, info) if return_info else rgb
    scale_x, scale_y = width / work_width, height / work_height
    x0, x1 = int(np.floor(xs.min() * scale_x)), int(np.ceil((xs.max() + 1) * scale_x))
    y0, y1 = int(np.floor(ys.min() * scale_y)), int(np.ceil((ys.max() + 1) * scale_y))
    box_width, box_height = x1 - x0, y1 - y0
    if box_width < max(6, int(0.05 * width)) or box_height < max(6, int(0.05 * height)):
        return (rgb, info) if return_info else rgb
    pad_x, pad_y = int(round(box_width * margin)), int(round(box_height * margin))
    box = (max(0, x0 - pad_x), max(0, y0 - pad_y), min(width, x1 + pad_x), min(height, y1 + pad_y))
    crop = rgb.crop(box)
    # Near-full crops add no useful second view and may reflect background noise.
    if crop.width * crop.height >= 0.96 * width * height:
        info["reason"] = "near_full_frame"
        return (rgb, info) if return_info else rgb
    info.update({"used_foreground": True, "reason": "ok", "bbox": list(box)})
    return (crop, info) if return_info else crop


def build_transforms(config: Config):
    from torchvision import transforms

    train_transform = transforms.Compose(
        [
            transforms.RandomResizedCrop(
                config.img_size,
                scale=(0.80, 1.0),
                ratio=(0.90, 1.10),
                antialias=True,
            ),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.15),
            transforms.RandomRotation(degrees=15),
            transforms.ColorJitter(
                brightness=0.15,
                contrast=0.15,
                saturation=0.12,
                hue=0.02,
            ),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
            transforms.RandomErasing(p=0.10, scale=(0.02, 0.08), ratio=(0.5, 2.0)),
        ]
    )
    eval_transform = transforms.Compose(
        [
            transforms.Resize((config.img_size, config.img_size), antialias=True),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )
    return train_transform, eval_transform


class PlantDataset(Dataset):
    def __init__(self, frame: pd.DataFrame, transform, labeled: bool, view: str = "global", foreground_probability: float = 0.5):
        self.frame = frame.reset_index(drop=True)
        self.transform = transform
        self.labeled = labeled
        self.view = view
        self.foreground_probability = foreground_probability
        self.foreground_cache = {}

    def __len__(self) -> int:
        return len(self.frame)

    def __getitem__(self, index: int):
        row = self.frame.iloc[index]
        with Image.open(row["path"]) as source:
            image = source.convert("RGB")
        use_foreground = self.view == "foreground" or (
            self.view == "random" and random.random() < self.foreground_probability
        )
        if use_foreground:
            cache_key = str(row["path"])
            if cache_key not in self.foreground_cache:
                _, info = extract_foreground(image, return_info=True)
                self.foreground_cache[cache_key] = tuple(info["bbox"]) if info["used_foreground"] else None
            box = self.foreground_cache[cache_key]
            if box is not None:
                image = image.crop(box)
        image = self.transform(image)
        if self.labeled:
            return image, int(row["label"])
        return image


def make_loader(
    dataset: Dataset,
    shuffle: bool,
    config: Config,
    device: torch.device,
) -> DataLoader:
    generator = torch.Generator()
    generator.manual_seed(config.seed)
    return DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=shuffle,
        num_workers=config.num_workers,
        pin_memory=device.type == "cuda",
        drop_last=False,
        persistent_workers=config.num_workers > 0,
        worker_init_fn=worker_init_fn,
        generator=generator,
    )


# ============================================================
# MODEL — official pretrained weights only
# ============================================================


def create_model(num_classes: int, pretrained: bool = True) -> nn.Module:
    try:
        from torchvision.models import ConvNeXt_Tiny_Weights, convnext_tiny

        model = convnext_tiny(weights=ConvNeXt_Tiny_Weights.DEFAULT if pretrained else None)
    except Exception as exc:
        raise RuntimeError(
            "Failed to load official ImageNet pretrained ConvNeXt-Tiny weights. "
            "Enable Kaggle internet or make sure the torchvision weight cache exists. "
            "The script will not fall back to weights=None."
        ) from exc
    features = model.classifier[2].in_features
    model.classifier[2] = nn.Linear(features, num_classes)
    return model


def set_backbone_trainable(model: nn.Module, trainable: bool) -> None:
    for parameter in model.parameters():
        parameter.requires_grad = trainable
    if not trainable:
        for parameter in model.classifier.parameters():
            parameter.requires_grad = True


def create_finetune_optimizer(model: nn.Module, config: Config) -> torch.optim.Optimizer:
    classifier = list(model.classifier.parameters())
    classifier_ids = {id(parameter) for parameter in classifier}
    backbone = [parameter for parameter in model.parameters() if id(parameter) not in classifier_ids]
    return torch.optim.AdamW(
        [
            {"params": backbone, "lr": config.backbone_lr},
            {"params": classifier, "lr": config.classifier_lr},
        ],
        weight_decay=config.weight_decay,
    )


# ============================================================
# TRAIN / VALIDATE / PREDICT — no TTA
# ============================================================


def autocast_context(device: torch.device, enabled: bool):
    if device.type == "cuda":
        return torch.cuda.amp.autocast(enabled=enabled)
    return nullcontext()


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    scaler: torch.cuda.amp.GradScaler,
    config: Config,
    device: torch.device,
) -> tuple[float, float, float]:
    model.train()
    total_loss = 0.0
    labels_all: list[int] = []
    predictions_all: list[int] = []
    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        with autocast_context(device, config.use_amp):
            logits = model(images)
            loss = criterion(logits, labels)
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), config.gradient_clip_norm)
        scaler.step(optimizer)
        scaler.update()
        total_loss += loss.item() * images.size(0)
        labels_all.extend(labels.detach().cpu().numpy().tolist())
        predictions_all.extend(logits.argmax(dim=1).detach().cpu().numpy().tolist())
    return (
        total_loss / len(loader.dataset),
        accuracy_score(labels_all, predictions_all),
        f1_score(labels_all, predictions_all, average="macro"),
    )


@torch.no_grad()
def validate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> tuple[float, float, float]:
    model.eval()
    total_loss = 0.0
    labels_all: list[int] = []
    predictions_all: list[int] = []
    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        logits = model(images)
        loss = criterion(logits, labels)
        total_loss += loss.item() * images.size(0)
        labels_all.extend(labels.cpu().numpy().tolist())
        predictions_all.extend(logits.argmax(dim=1).cpu().numpy().tolist())
    return (
        total_loss / len(loader.dataset),
        accuracy_score(labels_all, predictions_all),
        f1_score(labels_all, predictions_all, average="macro"),
    )


@torch.no_grad()
def predict_probabilities(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    labeled: bool,
) -> np.ndarray:
    model.eval()
    batches = []
    for batch in loader:
        images = batch[0] if labeled else batch
        images = images.to(device, non_blocking=True)
        probabilities = torch.softmax(model(images), dim=1)
        batches.append(probabilities.cpu().numpy())
    return np.concatenate(batches)


def save_checkpoint(
    path: Path,
    model: nn.Module,
    fold: int,
    stage: str,
    epoch: int,
    val_macro_f1: float,
    class_names: Sequence[str],
    config: Config,
) -> None:
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "model_name": config.model_name,
            "fold": fold,
            "best_stage": stage,
            "best_epoch": epoch,
            "best_val_macro_f1": val_macro_f1,
            "class_names": list(class_names),
            "img_size": config.img_size,
            "use_tta": False,
        },
        path,
    )


def train_fold(
    fold: int,
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    class_names: Sequence[str],
    transforms_pair,
    config: Config,
    device: torch.device,
) -> dict:
    train_transform, eval_transform = transforms_pair
    fold_train = train_df[train_df["fold"] != fold].reset_index(drop=True)
    fold_val = train_df[train_df["fold"] == fold].reset_index()
    train_loader = make_loader(
        PlantDataset(fold_train, train_transform, True), True, config, device
    )
    val_loader = make_loader(
        PlantDataset(fold_val, eval_transform, True), False, config, device
    )
    test_loader = make_loader(
        PlantDataset(test_df, eval_transform, False), False, config, device
    )

    model = create_model(len(class_names)).to(device)
    criterion = nn.CrossEntropyLoss(label_smoothing=config.label_smoothing)
    scaler = torch.cuda.amp.GradScaler(enabled=config.use_amp and device.type == "cuda")
    path = checkpoint_path(config, fold)
    best_score = -1.0
    best_stage = ""
    best_epoch = -1

    print(f"\nFold {fold + 1}/{config.n_folds}: classifier warm-up")
    set_backbone_trainable(model, False)
    head_optimizer = torch.optim.AdamW(
        model.classifier.parameters(),
        lr=config.head_lr,
        weight_decay=config.weight_decay,
    )
    for epoch in range(1, config.head_epochs + 1):
        train_loss, train_acc, train_f1 = train_one_epoch(
            model, train_loader, criterion, head_optimizer, scaler, config, device
        )
        val_loss, val_acc, val_f1 = validate(model, val_loader, criterion, device)
        print(
            f"  head {epoch:02d}/{config.head_epochs} | train loss {train_loss:.4f} "
            f"acc {train_acc:.4f} f1 {train_f1:.4f} | val loss {val_loss:.4f} "
            f"acc {val_acc:.4f} f1 {val_f1:.4f}"
        )
        if val_f1 > best_score:
            best_score, best_stage, best_epoch = val_f1, "head", epoch
            save_checkpoint(
                path,
                model,
                fold,
                best_stage,
                best_epoch,
                best_score,
                class_names,
                config,
            )

    print(f"Fold {fold + 1}/{config.n_folds}: full fine-tuning")
    set_backbone_trainable(model, True)
    optimizer = create_finetune_optimizer(model, config)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=config.finetune_epochs,
        eta_min=config.eta_min,
    )
    best_finetune_score = -1.0
    epochs_without_improvement = 0
    for epoch in range(1, config.finetune_epochs + 1):
        train_loss, train_acc, train_f1 = train_one_epoch(
            model, train_loader, criterion, optimizer, scaler, config, device
        )
        val_loss, val_acc, val_f1 = validate(model, val_loader, criterion, device)
        print(
            f"  tune {epoch:02d}/{config.finetune_epochs} | lr "
            f"{optimizer.param_groups[0]['lr']:.2e}/{optimizer.param_groups[1]['lr']:.2e} | "
            f"train loss {train_loss:.4f} acc {train_acc:.4f} f1 {train_f1:.4f} | "
            f"val loss {val_loss:.4f} acc {val_acc:.4f} f1 {val_f1:.4f}"
        )
        if val_f1 > best_finetune_score:
            best_finetune_score = val_f1
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
        if val_f1 > best_score:
            best_score, best_stage, best_epoch = val_f1, "finetune", epoch
            save_checkpoint(
                path,
                model,
                fold,
                best_stage,
                best_epoch,
                best_score,
                class_names,
                config,
            )
        scheduler.step()
        if epochs_without_improvement >= config.patience:
            print(f"  early stopping after fine-tune epoch {epoch}")
            break

    checkpoint = torch.load(path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    val_probabilities = predict_probabilities(model, val_loader, device, labeled=True)
    test_probabilities = predict_probabilities(model, test_loader, device, labeled=False)
    labels = fold_val["label"].to_numpy()
    predictions = val_probabilities.argmax(axis=1)
    result = {
        "indices": fold_val["index"].to_numpy(),
        "val_probabilities": val_probabilities,
        "test_probabilities": test_probabilities,
        "metrics": {
            "fold": fold,
            "best_stage": checkpoint["best_stage"],
            "best_epoch": checkpoint["best_epoch"],
            "selection_val_macro_f1": checkpoint["best_val_macro_f1"],
            "oof_macro_f1": f1_score(labels, predictions, average="macro"),
            "oof_accuracy": accuracy_score(labels, predictions),
            "checkpoint": str(path),
        },
    }

    del model, train_loader, val_loader, test_loader
    del head_optimizer, optimizer, scheduler, scaler
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return result


# ============================================================
# FINAL RESUMABLE MADV PIPELINE
# ============================================================


def validate_submission(submission: pd.DataFrame, expected_ids, class_names) -> None:
    expected_ids = list(expected_ids)
    assert list(submission.columns) == ["ID", "Category"]
    assert len(submission) == len(expected_ids)
    assert submission["ID"].notna().all() and submission["Category"].notna().all()
    assert submission["ID"].nunique() == len(expected_ids)
    assert set(submission["ID"]) == set(expected_ids)
    assert set(submission["Category"]).issubset(set(class_names))


def probability_frame(ids, probabilities, class_names, id_column="filename"):
    frame = pd.DataFrame({id_column: list(ids)})
    for index, name in enumerate(class_names):
        frame[f"prob_{name}"] = probabilities[:, index]
    return frame


def prediction_metrics(labels, probabilities, class_names) -> dict:
    pred = probabilities.argmax(1)
    matrix = confusion_matrix(labels, pred, labels=np.arange(len(class_names)))
    black, silky = class_names.index("Black-grass"), class_names.index("Loose Silky-bent")
    per_class = f1_score(labels, pred, labels=np.arange(len(class_names)), average=None, zero_division=0)
    return {
        "macro_f1": float(f1_score(labels, pred, average="macro")),
        "accuracy": float(accuracy_score(labels, pred)),
        "black_grass_f1": float(per_class[black]),
        "loose_silky_bent_f1": float(per_class[silky]),
        "black_to_silky": int(matrix[black, silky]),
        "silky_to_black": int(matrix[silky, black]),
        "grass_confusion": int(matrix[black, silky] + matrix[silky, black]),
        "errors": int((pred != labels).sum()),
    }


def crossfit_lambda_selection(labels, global_prob, foreground_prob, folds, lambdas):
    folds = np.asarray(folds)
    crossfit = np.empty_like(global_prob)
    selected = {}
    search_rows = []
    for held_out in sorted(np.unique(folds)):
        tune = folds != held_out
        best_lambda, best_score = None, -1.0
        for value in lambdas:
            mixed = (1.0 - value) * global_prob[tune] + value * foreground_prob[tune]
            score = float(f1_score(labels[tune], mixed.argmax(1), average="macro"))
            search_rows.append({"held_out_fold": int(held_out), "lambda": value, "selection_macro_f1": score})
            if score > best_score + 1e-12:
                best_lambda, best_score = float(value), score
        selected[int(held_out)] = best_lambda
        mask = folds == held_out
        crossfit[mask] = (1.0 - best_lambda) * global_prob[mask] + best_lambda * foreground_prob[mask]
    values = list(selected.values())
    final_lambda = min(lambdas, key=lambda x: (-values.count(x), x))
    return {
        "crossfit_probabilities": crossfit,
        "selected_lambda_by_fold": selected,
        "final_lambda": float(final_lambda),
        "search": pd.DataFrame(search_rows),
    }


def madv_adoption_gate(global_metrics, crossfit_metrics, fold_deltas, corrected, harmed, minimum_delta=0.003):
    checks = {
        "clear_macro_gain": crossfit_metrics["macro_f1"] - global_metrics["macro_f1"] >= minimum_delta,
        "more_corrected_than_harmed": corrected > harmed,
        "grass_not_worse": crossfit_metrics["grass_confusion"] <= global_metrics["grass_confusion"],
        "fold_stability": sum(delta >= 0 for delta in fold_deltas) >= 3 and min(fold_deltas) >= -0.015,
    }
    return {"adopt_madv": bool(all(checks.values())), "checks": checks}


def checkpoint_metadata_ok(payload, fold, class_names, config, require_completed=False):
    ok = (
        payload.get("model_name") == "convnext_tiny"
        and int(payload.get("fold", -1)) == fold
        and list(payload.get("class_names", [])) == list(class_names)
        and int(payload.get("img_size", -1)) == config.img_size
    )
    return ok and (not require_completed or payload.get("completed") is True)


def discover_legacy_checkpoints(config, class_names):
    roots = [Path("/kaggle/input"), config.output_dir]
    prefixes = ("main_convnext_tiny_fold", "convnext_tiny_fold")
    for root in roots:
        if not root.exists():
            continue
        for prefix in prefixes:
            for first in root.rglob(f"{prefix}0.pth"):
                paths = [first.parent / f"{prefix}{fold}.pth" for fold in range(config.n_folds)]
                if not all(path.exists() for path in paths):
                    continue
                try:
                    if all(checkpoint_metadata_ok(torch.load(path, map_location="cpu"), fold, class_names, config) for fold, path in enumerate(paths)):
                        return paths
                except Exception as exc:
                    print("Rejected checkpoint set:", first.parent, exc)
    return None


def _verified_oof_score(path: Path, class_names) -> float | None:
    try:
        frame = pd.read_csv(path)
        label_col = "label"
        prefix = "main_prob_" if all(f"main_prob_{name}" in frame for name in class_names) else "prob_"
        columns = [f"{prefix}{name}" for name in class_names]
        if label_col not in frame or not all(column in frame for column in columns):
            return None
        return float(f1_score(frame[label_col].to_numpy(), frame[columns].to_numpy().argmax(1), average="macro"))
    except Exception:
        return None


def discover_trusted_baseline(test_ids, class_names, config):
    known_historical_sha256 = "1cce7c975e7785552f1d0ba68bb908ef651eba75c0495136aafe602defa67638"
    explicit = os.environ.get("TRUSTED_BASELINE_SUBMISSION", "").strip()
    if explicit:
        path = Path(explicit)
        frame = pd.read_csv(path)
        validate_submission(frame, test_ids, class_names)
        return frame, {"source": str(path), "proof": "explicit user-designated 0.9470 file", "sha256": sha256_file(path)}
    root = Path("/kaggle/input")
    if not root.exists():
        return None, None
    # Canonical CSV identity tolerates line-ending changes while preserving every ID/label pair.
    for candidate in root.rglob("*.csv"):
        try:
            if candidate.stat().st_size < 1_000_000:
                frame = pd.read_csv(candidate)
                if list(frame.columns) != ["ID", "Category"]:
                    continue
                canonical = frame.sort_values("ID").reset_index(drop=True).to_csv(index=False, lineterminator="\n").encode("utf-8")
                if hashlib.sha256(canonical).hexdigest() != known_historical_sha256:
                    continue
                validate_submission(frame, test_ids, class_names)
                return frame, {"source": str(candidate), "proof": "canonical SHA-256 match to verified 0.9470 submission", "sha256": known_historical_sha256}
        except Exception:
            pass
    for oof_path in root.rglob("oof_predictions.csv"):
        score = _verified_oof_score(oof_path, class_names)
        if score is None or abs(score - config.expected_reproduction_oof) > 0.002:
            continue
        parent = oof_path.parent
        for name in ("submission.csv", "submission (1).csv", "submission10.7.csv"):
            candidate = parent / name
            if candidate.exists():
                try:
                    frame = pd.read_csv(candidate)
                    validate_submission(frame, test_ids, class_names)
                    return frame, {"source": str(candidate), "proof": f"co-located OOF={score:.6f}", "sha256": sha256_file(candidate)}
                except Exception:
                    pass
        probability_path = parent / "test_probabilities.csv"
        if probability_path.exists():
            probs = pd.read_csv(probability_path)
            columns = [f"prob_{name}" for name in class_names]
            if "ID" in probs and all(column in probs for column in columns) and set(probs["ID"]) == set(test_ids):
                probs = probs.set_index("ID").loc[list(test_ids)].reset_index()
                frame = pd.DataFrame({"ID": list(test_ids), "Category": np.asarray(class_names)[probs[columns].to_numpy().argmax(1)]})
                validate_submission(frame, test_ids, class_names)
                return frame, {"source": str(probability_path), "proof": f"regenerated beside OOF={score:.6f}", "sha256": sha256_file(probability_path)}
    return None, None


def save_best_checkpoint(path, model, fold, stage, epoch, score, class_names, config, completed=False):
    torch.save({
        "model_state_dict": model.state_dict(), "model_name": "convnext_tiny", "fold": fold,
        "best_stage": stage, "best_epoch": epoch, "best_val_macro_f1": float(score),
        "class_names": list(class_names), "img_size": config.img_size, "use_tta": False,
        "method": "MADV random global/foreground training", "completed": completed,
        "pretrained_weights": "torchvision ConvNeXt_Tiny_Weights.DEFAULT (ImageNet-1K)",
    }, path)


def train_or_restore_fold(fold, train_df, class_names, transforms_pair, config, device):
    best_path, last_path = checkpoint_path(config, fold), last_checkpoint_path(config, fold)
    if best_path.exists():
        saved = torch.load(best_path, map_location="cpu")
        if checkpoint_metadata_ok(saved, fold, class_names, config, require_completed=True):
            print(f"Fold {fold}: completed checkpoint restored")
            return best_path, []
    train_transform, eval_transform = transforms_pair
    fold_train = train_df[train_df.fold != fold].reset_index(drop=True)
    fold_val = train_df[train_df.fold == fold].reset_index(drop=True)
    train_loader = make_loader(PlantDataset(fold_train, train_transform, True, "random", config.foreground_probability), True, config, device)
    val_loader = make_loader(PlantDataset(fold_val, eval_transform, True, "global"), False, config, device)
    resume = None
    if last_path.exists():
        candidate = torch.load(last_path, map_location=device)
        if checkpoint_metadata_ok(candidate, fold, class_names, config):
            resume = candidate
            print(f"Fold {fold}: resuming {candidate['stage']} epoch {candidate['next_epoch']}")
    model = create_model(len(class_names), pretrained=resume is None).to(device)
    if resume:
        model.load_state_dict(resume["model_state_dict"])
    criterion = nn.CrossEntropyLoss(label_smoothing=config.label_smoothing)
    scaler = torch.cuda.amp.GradScaler(enabled=config.use_amp and device.type == "cuda")
    best_score = float(resume.get("best_score", -1.0)) if resume else -1.0
    best_stage = str(resume.get("best_stage", "")) if resume else ""
    best_epoch = int(resume.get("best_epoch", -1)) if resume else -1
    records = []
    history_path = config.output_dir / f"fold{fold}_training_history.csv"
    stages = (("head", config.head_epochs), ("finetune", config.finetune_epochs))
    resume_stage = resume.get("stage") if resume else None
    for stage, total_epochs in stages:
        if resume_stage and stages.index((stage, total_epochs)) < [x[0] for x in stages].index(resume_stage):
            continue
        set_backbone_trainable(model, stage == "finetune")
        optimizer = (torch.optim.AdamW(model.classifier.parameters(), lr=config.head_lr, weight_decay=config.weight_decay)
                     if stage == "head" else create_finetune_optimizer(model, config))
        scheduler = (torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config.finetune_epochs, eta_min=config.eta_min)
                     if stage == "finetune" else None)
        start_epoch, no_improvement, best_stage_score = 1, 0, -1.0
        if resume and resume_stage == stage:
            start_epoch = int(resume["next_epoch"])
            optimizer.load_state_dict(resume["optimizer_state_dict"])
            if scheduler and resume.get("scheduler_state_dict"):
                scheduler.load_state_dict(resume["scheduler_state_dict"])
            if resume.get("scaler_state_dict"):
                scaler.load_state_dict(resume["scaler_state_dict"])
            no_improvement = int(resume.get("no_improvement", 0))
            best_stage_score = float(resume.get("best_stage_score", -1.0))
        for epoch in range(start_epoch, total_epochs + 1):
            train_loss, train_acc, train_f1 = train_one_epoch(model, train_loader, criterion, optimizer, scaler, config, device)
            val_loss, val_acc, val_f1 = validate(model, val_loader, criterion, device)
            record = {"fold": fold, "stage": stage, "epoch": epoch, "train_loss": train_loss, "train_accuracy": train_acc,
                      "train_macro_f1": train_f1, "val_loss": val_loss, "val_accuracy": val_acc, "val_macro_f1": val_f1}
            records.append(record)
            pd.DataFrame([record]).to_csv(history_path, mode="a", header=not history_path.exists(), index=False)
            print(f"fold {fold} {stage} {epoch:02d}/{total_epochs}: val F1={val_f1:.5f}")
            if val_f1 > best_score:
                best_score, best_stage, best_epoch = val_f1, stage, epoch
                save_best_checkpoint(best_path, model, fold, stage, epoch, val_f1, class_names, config)
            if val_f1 > best_stage_score:
                best_stage_score, no_improvement = val_f1, 0
            else:
                no_improvement += 1
            if scheduler:
                scheduler.step()
            torch.save({"model_state_dict": model.state_dict(), "model_name": "convnext_tiny", "fold": fold,
                        "class_names": list(class_names), "img_size": config.img_size, "stage": stage,
                        "next_epoch": epoch + 1, "optimizer_state_dict": optimizer.state_dict(),
                        "scheduler_state_dict": scheduler.state_dict() if scheduler else None,
                        "scaler_state_dict": scaler.state_dict(), "best_score": best_score, "best_stage": best_stage,
                        "best_epoch": best_epoch, "best_stage_score": best_stage_score, "no_improvement": no_improvement}, last_path)
            if stage == "finetune" and no_improvement >= config.patience:
                print(f"fold {fold}: early stopping")
                break
        resume = None
    saved = torch.load(best_path, map_location=device)
    model.load_state_dict(saved["model_state_dict"])
    save_best_checkpoint(best_path, model, fold, saved["best_stage"], saved["best_epoch"], saved["best_val_macro_f1"], class_names, config, completed=True)
    del model, train_loader, val_loader, optimizer, scaler
    gc.collect()
    if device.type == "cuda": torch.cuda.empty_cache()
    return best_path, records


def infer_fold_views(fold, checkpoint, train_df, test_df, class_names, eval_transform, config, device):
    val = train_df[train_df.fold == fold].copy()
    indices = val.index.to_numpy()
    model = create_model(len(class_names), pretrained=False).to(device)
    payload = torch.load(checkpoint, map_location=device)
    if not checkpoint_metadata_ok(payload, fold, class_names, config):
        raise RuntimeError(f"Incompatible checkpoint: {checkpoint}")
    model.load_state_dict(payload["model_state_dict"])
    arrays = {}
    for split, frame, labeled in (("val", val, True), ("test", test_df, False)):
        for view in ("global", "foreground"):
            loader = make_loader(PlantDataset(frame, eval_transform, labeled, view), False, config, device)
            arrays[f"{split}_{view}"] = predict_probabilities(model, loader, device, labeled)
    checkpoint_hash = sha256_file(checkpoint)
    np.savez_compressed(config.output_dir / f"fold{fold}_view_probabilities.npz", indices=indices,
                        checkpoint_sha256=np.asarray(checkpoint_hash), **arrays)
    probability_frame(val.filename, arrays["val_global"], class_names).assign(fold=fold).to_csv(
        config.output_dir / f"fold{fold}_oof_global.csv", index=False)
    probability_frame(val.filename, arrays["val_foreground"], class_names).assign(fold=fold).to_csv(
        config.output_dir / f"fold{fold}_oof_foreground.csv", index=False)
    probability_frame(test_df.filename, arrays["test_global"], class_names, "ID").to_csv(
        config.output_dir / f"fold{fold}_test_global.csv", index=False)
    probability_frame(test_df.filename, arrays["test_foreground"], class_names, "ID").to_csv(
        config.output_dir / f"fold{fold}_test_foreground.csv", index=False)
    pd.DataFrame({"fold": fold, "checkpoint": str(checkpoint), "checkpoint_sha256": sha256_file(checkpoint),
                  "best_stage": payload.get("best_stage"), "best_epoch": payload.get("best_epoch"),
                  "selection_val_macro_f1": payload.get("best_val_macro_f1")}, index=[0]).to_csv(config.output_dir / f"fold{fold}_checkpoint_record.csv", index=False)
    del model
    gc.collect()
    if device.type == "cuda": torch.cuda.empty_cache()
    return indices, arrays


def probability_cache_matches(cache: Path, checkpoint: Path) -> bool:
    if not cache.exists():
        return False
    try:
        saved = np.load(cache)
        return "checkpoint_sha256" in saved and saved["checkpoint_sha256"].item() == sha256_file(checkpoint)
    except Exception:
        return False


def make_foreground_sheet(train_df, output_path, count=12):
    from PIL import ImageDraw
    sample = train_df.sample(min(count, len(train_df)), random_state=42)
    tile_w, tile_h = 220, 190
    canvas = Image.new("RGB", (tile_w * 2, tile_h * len(sample)), "white")
    draw = ImageDraw.Draw(canvas)
    rows = []
    for row_number, (_, row) in enumerate(sample.iterrows()):
        with Image.open(row.path) as source: original = source.convert("RGB")
        foreground, info = extract_foreground(original, return_info=True)
        for column, image in enumerate((original, foreground)):
            preview = image.copy(); preview.thumbnail((tile_w - 8, tile_h - 35))
            canvas.paste(preview, (column * tile_w + (tile_w-preview.width)//2, row_number*tile_h + 22))
        draw.text((4, row_number*tile_h + 3), f"{row.filename} | {row.class_name} | {info['reason']}", fill="black")
        rows.append({"filename": row.filename, "class_name": row.class_name, **info})
    canvas.save(output_path)
    pd.DataFrame(rows).to_csv(output_path.with_suffix(".csv"), index=False)


def write_report(path, source_mode, baseline_info, global_metrics, foreground_metrics, crossfit_metrics, direct_rows, selection, gate, fold_deltas, corrected, harmed, chosen_method, chosen_lambda, config):
    report = f"""# 机器学习课程：五分类植物幼苗识别最终报告

## 1. 任务与数据
任务为 500 张训练图、378 张测试图的单标签五分类。主要困难是 Black-grass 与 Loose Silky-bent 形态接近。

## 2. 方法
主干为 torchvision ConvNeXt-Tiny，使用官方 ImageNet-1K 预训练权重，并非从零训练；未加入任何外部植物图片或标签。采用 seed 42 的五折 StratifiedKFold、288 输入、AdamW、Cosine 调度、分类头预热 2 epoch、全量微调和 early stopping，无 TTA。

MADV-ConvNeXt 是面向细粒度植物形态混淆的双视图输入与保守融合策略，不是全新网络架构。Global View 保留原图；Foreground View 以统一的 HSV/Excess Green 规则定位植被，边界外扩约 20%，检测不可靠则回退原图。新训练在两个视图间随机选择，因此它同时改变了训练分布和推理融合，不是单变量推理增强。

融合为 `P=(1-lambda)P_global+lambda P_foreground`，lambda 仅取 0、0.15、0.30。每个验证折的 lambda 只由其余四折选择，避免在同一 OOF 上选权重又报告该增益。

## 3. 运行来源与结果
- 模型来源模式：{source_mode}
- 历史 Public：{config.historical_public_score:.4f}
- 历史 OOF Macro-F1：{config.expected_reproduction_oof:.6f}
- 新 Global OOF：{global_metrics['macro_f1']:.6f}，Accuracy={global_metrics['accuracy']:.6f}
- Foreground-only OOF：{foreground_metrics['macro_f1']:.6f}
- 外层折交叉选择融合 OOF：{crossfit_metrics['macro_f1']:.6f}，Accuracy={crossfit_metrics['accuracy']:.6f}
- Black-grass F1：{crossfit_metrics['black_grass_f1']:.6f}
- Loose Silky-bent F1：{crossfit_metrics['loose_silky_bent_f1']:.6f}
- Black→Loose / Loose→Black：{crossfit_metrics['black_to_silky']} / {crossfit_metrics['silky_to_black']}
- corrected / harmed：{corrected} / {harmed}
- 各折融合相对 Global 的 F1 差：{fold_deltas}
- 外层折选择 lambda：{selection['selected_lambda_by_fold']}
- 最终固定 lambda：{chosen_lambda:.2f}
- 采用方法：{chosen_method}
- MADV 安全门：{gate}
- 历史回退来源：{baseline_info}

预设 lambda 的全 OOF结果（仅作有偏诊断，不作为无偏增益声明）：

```text
{pd.DataFrame(direct_rows).to_csv(index=False)}
```

## 4. 消融与选择
lambda=0 是同一新训练模型的 Global-only 消融；foreground-only 检查裁剪视图本身。最终采用与否依据外层折选择结果、草类混淆、corrected/harmed 和五折稳定性。若安全门未通过，保留来源确认的 0.9470 原版 submission。

## 5. 局限
训练集仅 500 张，新旧训练存在 GPU 非确定性和随机视图差异，小幅 OOF 差异不能视为确定收益。绿色阈值也可能漏检枯黄叶片或把绿色背景纳入前景。Public 分数不是模型选择依据，脚本不会自动提交。
"""
    path.write_text(report, encoding="utf-8")


def create_zip(config):
    zip_path = config.output_dir / "final_results.zip"
    allowed = {"submission_final.csv", "submission_madv.csv", "FINAL_REPORT.md", "final_method.py",
               "config.json",
               "oof_global_probabilities.csv", "oof_foreground_probabilities.csv", "oof_crossfit_probabilities.csv",
               "lambda_search_outer_fold.csv", "fold_metrics.csv", "classification_report.csv", "confusion_matrix.csv",
               "foreground_visualization.jpg", "foreground_visualization.csv", "run_manifest.json"}
    allowed.update(path.name for path in config.output_dir.glob("fold*_training_history.csv"))
    allowed.update(path.name for path in config.output_dir.glob("fold*_checkpoint_record.csv"))
    allowed.update(path.name for path in config.output_dir.glob("fold*_oof_*.csv"))
    allowed.update(path.name for path in config.output_dir.glob("fold*_test_*.csv"))
    allowed.update(path.name for path in config.output_dir.glob("fold*_view_probabilities.npz"))
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as archive:
        for name in sorted(allowed):
            path = config.output_dir / name
            if path.exists(): archive.write(path, arcname=name)
    return zip_path


def main(config: Config = CONFIG) -> None:
    if config.use_tta: raise ValueError("TTA must remain disabled")
    config.output_dir.mkdir(parents=True, exist_ok=True)
    config_payload = {key: (str(value) if isinstance(value, Path) else value) for key, value in asdict(config).items()}
    (config.output_dir / "config.json").write_text(json.dumps(config_payload, indent=2, ensure_ascii=False), encoding="utf-8")
    seed_everything(config.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print_environment(device)
    train_df, test_df, class_names = build_dataframes(config)
    train_df = assign_folds(train_df, config)
    template = find_submission_template(config, test_df.filename.tolist())
    test_df = test_df.set_index("filename", drop=False).loc[template.ID.tolist()].reset_index(drop=True)
    expected_ids = test_df.filename.tolist()
    baseline, baseline_info = discover_trusted_baseline(expected_ids, class_names, config)
    final_path = config.output_dir / "submission_final.csv"
    if baseline is not None:
        baseline = baseline.set_index("ID").loc[expected_ids].reset_index()
        validate_submission(baseline, expected_ids, class_names)
        baseline.to_csv(final_path, index=False)
        print("P0 fallback materialized:", final_path, baseline_info)
    else:
        print("WARNING: verified 0.9470 fallback was not found. Upload it and set TRUSTED_BASELINE_SUBMISSION.")
    source = Path(globals().get("__file__", ""))
    target_source = config.output_dir / "final_method.py"
    if source.exists() and source.resolve() != target_source.resolve(): shutil.copy2(source, target_source)
    initial_report = config.output_dir / "FINAL_REPORT.md"
    initial_report.write_text(
        "# 机器学习课程最终运行状态\n\n脚本已启动。历史 0.9470 回退："
        + (f"已验证并保存至 submission_final.csv。来源：{baseline_info}" if baseline is not None
           else "未找到；必须上传可信 submission 并设置 TRUSTED_BASELINE_SUBMISSION。")
        + "\n\nMADV 五折尚未完成，不得把部分折结果作为最终结果。\n",
        encoding="utf-8",
    )
    create_zip(config)
    make_foreground_sheet(train_df, config.output_dir / "foreground_visualization.jpg")
    legacy = discover_legacy_checkpoints(config, class_names)
    checkpoints, source_mode = [], "retrained MADV random dual-view"
    if legacy:
        transforms_pair = build_transforms(config)
        checkpoints, source_mode = legacy, "recovered five historical ConvNeXt checkpoints"
        print("Recovered complete checkpoint set:", [str(path) for path in checkpoints])
    else:
        if device.type != "cuda":
            raise RuntimeError("No complete checkpoint set and CUDA is unavailable. Safe fallback remains intact; five-fold training was not started.")
        transforms_pair = build_transforms(config)
        for fold in range(config.n_folds):
            path, _ = train_or_restore_fold(fold, train_df, class_names, transforms_pair, config, device)
            checkpoints.append(path)
            # Persist both validation/test views immediately after each fold.
            cache = config.output_dir / f"fold{fold}_view_probabilities.npz"
            if not probability_cache_matches(cache, path):
                infer_fold_views(fold, path, train_df, test_df, class_names, transforms_pair[1], config, device)
    n_classes = len(class_names)
    oof_global = np.full((len(train_df), n_classes), np.nan, np.float32)
    oof_foreground = np.full_like(oof_global, np.nan)
    test_global_folds, test_foreground_folds, checkpoint_rows = [], [], []
    for fold, checkpoint in enumerate(checkpoints):
        cache = config.output_dir / f"fold{fold}_view_probabilities.npz"
        cache_valid = probability_cache_matches(cache, checkpoint)
        if cache_valid:
            saved = np.load(cache)
        if cache_valid:
            indices = saved["indices"]
            arrays = {name: saved[name] for name in ("val_global", "val_foreground", "test_global", "test_foreground")}
            print(f"Fold {fold}: restored verified cached probabilities")
        else:
            indices, arrays = infer_fold_views(fold, checkpoint, train_df, test_df, class_names, transforms_pair[1], config, device)
        oof_global[indices], oof_foreground[indices] = arrays["val_global"], arrays["val_foreground"]
        test_global_folds.append(arrays["test_global"]); test_foreground_folds.append(arrays["test_foreground"])
        checkpoint_rows.append({"fold": fold, "checkpoint": str(checkpoint), "sha256": sha256_file(checkpoint)})
        probability_frame(train_df.loc[indices, "filename"], arrays["val_global"], class_names).assign(fold=fold).to_csv(config.output_dir / f"fold{fold}_oof_global.csv", index=False)
        probability_frame(train_df.loc[indices, "filename"], arrays["val_foreground"], class_names).assign(fold=fold).to_csv(config.output_dir / f"fold{fold}_oof_foreground.csv", index=False)
        probability_frame(test_df.filename, arrays["test_global"], class_names, "ID").to_csv(config.output_dir / f"fold{fold}_test_global.csv", index=False)
        probability_frame(test_df.filename, arrays["test_foreground"], class_names, "ID").to_csv(config.output_dir / f"fold{fold}_test_foreground.csv", index=False)
    if np.isnan(oof_global).any() or np.isnan(oof_foreground).any() or len(checkpoints) != 5:
        raise RuntimeError("Five-fold probabilities are incomplete; fallback remains selected.")
    labels, folds = train_df.label.to_numpy(), train_df.fold.to_numpy()
    selection = crossfit_lambda_selection(labels, oof_global, oof_foreground, folds, config.lambdas)
    crossfit = selection["crossfit_probabilities"]
    gm, fm, cm = prediction_metrics(labels, oof_global, class_names), prediction_metrics(labels, oof_foreground, class_names), prediction_metrics(labels, crossfit, class_names)
    global_pred, cross_pred = oof_global.argmax(1), crossfit.argmax(1)
    corrected = int(((global_pred != labels) & (cross_pred == labels)).sum())
    harmed = int(((global_pred == labels) & (cross_pred != labels)).sum())
    fold_deltas = []
    for fold in range(config.n_folds):
        mask = folds == fold
        fold_deltas.append(round(float(f1_score(labels[mask], cross_pred[mask], average="macro") - f1_score(labels[mask], global_pred[mask], average="macro")), 6))
    direct_rows = []
    for value in config.lambdas:
        metrics = prediction_metrics(labels, (1-value)*oof_global + value*oof_foreground, class_names)
        direct_rows.append({"lambda": value, **metrics})
    chosen_lambda = selection["final_lambda"]
    fixed_metrics = prediction_metrics(labels, (1-chosen_lambda)*oof_global + chosen_lambda*oof_foreground, class_names)
    gate = madv_adoption_gate(gm, cm, fold_deltas, corrected, harmed, config.minimum_clear_delta)
    gate["checks"]["fixed_lambda_consistent"] = (
        fixed_metrics["macro_f1"] - gm["macro_f1"] >= config.minimum_clear_delta
        and fixed_metrics["grass_confusion"] <= gm["grass_confusion"]
        and chosen_lambda > 0
    )
    gate["checks"]["recovered_global_reproduced"] = (
        source_mode != "recovered five historical ConvNeXt checkpoints"
        or abs(gm["macro_f1"] - config.expected_reproduction_oof) <= 0.005
    )
    gate["adopt_madv"] = bool(all(gate["checks"].values()))
    test_global = np.mean(test_global_folds, axis=0); test_foreground = np.mean(test_foreground_folds, axis=0)
    madv_test = (1-chosen_lambda)*test_global + chosen_lambda*test_foreground
    madv_submission = pd.DataFrame({"ID": expected_ids, "Category": np.asarray(class_names)[madv_test.argmax(1)]})
    validate_submission(madv_submission, expected_ids, class_names)
    madv_submission.to_csv(config.output_dir / "submission_madv.csv", index=False)
    chosen_method = "MADV" if gate["adopt_madv"] else "Original ConvNeXt"
    if gate["adopt_madv"]:
        madv_submission.to_csv(final_path, index=False)
    elif baseline is None:
        raise RuntimeError("MADV safety gate failed and no verified historical submission is available. submission_madv.csv is diagnostic only and was not promoted.")
    probability_frame(train_df.filename, oof_global, class_names).to_csv(config.output_dir / "oof_global_probabilities.csv", index=False)
    probability_frame(train_df.filename, oof_foreground, class_names).to_csv(config.output_dir / "oof_foreground_probabilities.csv", index=False)
    probability_frame(train_df.filename, crossfit, class_names).to_csv(config.output_dir / "oof_crossfit_probabilities.csv", index=False)
    selection["search"].to_csv(config.output_dir / "lambda_search_outer_fold.csv", index=False)
    pd.DataFrame([{**{"fold": fold}, **prediction_metrics(labels[folds == fold], crossfit[folds == fold], class_names)} for fold in range(5)]).to_csv(config.output_dir / "fold_metrics.csv", index=False)
    pd.DataFrame(classification_report(labels, cross_pred, target_names=class_names, output_dict=True, zero_division=0)).T.to_csv(config.output_dir / "classification_report.csv")
    pd.DataFrame(confusion_matrix(labels, cross_pred), index=class_names, columns=class_names).to_csv(config.output_dir / "confusion_matrix.csv")
    write_report(config.output_dir / "FINAL_REPORT.md", source_mode, baseline_info, gm, fm, cm, direct_rows, selection, gate, fold_deltas, corrected, harmed, chosen_method, chosen_lambda, config)
    manifest = {"selected_method": chosen_method, "selected_lambda": chosen_lambda, "historical_public": config.historical_public_score,
                "historical_oof": config.expected_reproduction_oof, "global_metrics": gm, "crossfit_metrics": cm,
                "gate": gate, "baseline": baseline_info, "checkpoints": checkpoint_rows, "fully_executed": True}
    (config.output_dir / "run_manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    zip_path = create_zip(config)
    final = pd.read_csv(final_path); validate_submission(final, expected_ids, class_names)
    print("\nFINAL SUBMISSION READY")
    print("Selected method:", chosen_method)
    print("Historical Public Score:", f"{config.historical_public_score:.4f}")
    print("Historical OOF:", f"{config.expected_reproduction_oof:.6f}")
    print("Final validated OOF:", f"{cm['macro_f1']:.6f}" if chosen_method == "MADV" else f"historical {config.expected_reproduction_oof:.6f}")
    print("Delta:", f"{cm['macro_f1']-gm['macro_f1']:+.6f}")
    print("Grass confusion:", f"{cm['black_to_silky']} / {cm['silky_to_black']}")
    print("Selected lambda:", f"{chosen_lambda:.2f}" if chosen_method == "MADV" else "0.00 (historical fallback)")
    print("Submission rows:", len(final))
    print("Submission:", final_path)
    print("Backup:", zip_path)
    print("Run status: five folds genuinely completed/restored and validated")


if __name__ == "__main__":
    main()

