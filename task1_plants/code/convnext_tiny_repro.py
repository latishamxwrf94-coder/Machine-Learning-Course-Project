"""Strict ConvNeXt-Tiny reproduction for NEU ML Task1.

This script intentionally trains one model only:
ConvNeXt-Tiny, 288x288, five shared Stratified folds, no TTA.
It preserves the validated augmentation and two-stage optimization settings
from the preceding four-model experiment and creates a downloadable result ZIP.
"""

from __future__ import annotations

import gc
import os
import random
import warnings
import zipfile
from contextlib import nullcontext
from dataclasses import dataclass
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

    expected_reproduction_oof: float = 0.9177
    expected_reproduction_accuracy: float = 0.9180


CONFIG = Config()
IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg"}
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def checkpoint_path(config: Config, fold: int) -> Path:
    return config.output_dir / f"convnext_tiny_fold{fold}.pth"


def result_files_for_zip(config: Config) -> list[Path]:
    names = [
        "submission.csv",
        "test_probabilities.csv",
        "oof_predictions.csv",
        "fold_metrics.csv",
        "classification_report.csv",
        "confusion_matrix.csv",
        "misclassified_oof.csv",
        "blackgrass_vs_silky_errors.csv",
        "convnext_tiny_oof_probabilities.csv",
        "convnext_tiny_test_probabilities.csv",
    ]
    files = [config.output_dir / name for name in names]
    files.extend(checkpoint_path(config, fold) for fold in range(config.n_folds))
    return files


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
    print("STRICT CONVNEXT-TINY REPRODUCTION")
    print("=" * 72)
    print("PyTorch version :", torch.__version__)
    print("CUDA available  :", torch.cuda.is_available())
    print("Device          :", device)
    if device.type == "cuda":
        print("GPU             :", torch.cuda.get_device_name(0))
    print("Model           : ConvNeXt-Tiny")
    print("Image size      : 288")
    print("TTA             : False")
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
    def __init__(self, frame: pd.DataFrame, transform, labeled: bool):
        self.frame = frame.reset_index(drop=True)
        self.transform = transform
        self.labeled = labeled

    def __len__(self) -> int:
        return len(self.frame)

    def __getitem__(self, index: int):
        row = self.frame.iloc[index]
        with Image.open(row["path"]) as source:
            image = source.convert("RGB")
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


def create_model(num_classes: int) -> nn.Module:
    try:
        from torchvision.models import ConvNeXt_Tiny_Weights, convnext_tiny

        model = convnext_tiny(weights=ConvNeXt_Tiny_Weights.DEFAULT)
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
# 5-FOLD TRAINING
# ============================================================


def run_cross_validation(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    class_names: Sequence[str],
    config: Config,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    transforms_pair = build_transforms(config)
    oof = np.full((len(train_df), len(class_names)), np.nan, dtype=np.float32)
    test = np.zeros((len(test_df), len(class_names)), dtype=np.float32)
    fold_metrics = []
    for fold in range(config.n_folds):
        try:
            result = train_fold(
                fold,
                train_df,
                test_df,
                class_names,
                transforms_pair,
                config,
                device,
            )
        except RuntimeError as exc:
            if "out of memory" in str(exc).lower():
                if device.type == "cuda":
                    torch.cuda.empty_cache()
                raise RuntimeError(
                    f"CUDA OOM for ConvNeXt-Tiny at {config.img_size}px with "
                    f"batch_size={config.batch_size}. Reduce batch_size manually and restart; "
                    "the script will not silently alter the validated configuration."
                ) from exc
            raise
        oof[result["indices"]] = result["val_probabilities"]
        test += result["test_probabilities"] / config.n_folds
        fold_metrics.append(result["metrics"])
    assert not np.isnan(oof).any(), "OOF probabilities are incomplete"
    return oof, test, pd.DataFrame(fold_metrics)


# ============================================================
# ERROR ANALYSIS / OUTPUT
# ============================================================


def probability_frame(
    identifiers: Sequence[str],
    probabilities: np.ndarray,
    class_names: Sequence[str],
    id_column: str,
) -> pd.DataFrame:
    frame = pd.DataFrame({id_column: list(identifiers)})
    for index, class_name in enumerate(class_names):
        frame[f"prob_{class_name}"] = probabilities[:, index]
    return frame


def build_oof_details(
    train_df: pd.DataFrame,
    probabilities: np.ndarray,
    class_names: Sequence[str],
) -> pd.DataFrame:
    frame = train_df[["filename", "path", "fold", "class_name", "label"]].copy()
    order = np.argsort(probabilities, axis=1)[:, ::-1]
    top1, top2 = order[:, 0], order[:, 1]
    names = np.asarray(class_names)
    frame["true_class"] = frame["class_name"]
    frame["predicted_class"] = names[top1]
    frame["prediction_confidence"] = probabilities[np.arange(len(frame)), top1]
    frame["top2_class"] = names[top2]
    frame["top2_probability"] = probabilities[np.arange(len(frame)), top2]
    frame["is_correct"] = frame["label"].to_numpy() == top1
    for index, class_name in enumerate(class_names):
        frame[f"prob_{class_name}"] = probabilities[:, index]
    return frame.drop(columns=["class_name"])


def validate_submission(submission: pd.DataFrame, expected_rows: int = 378) -> None:
    assert list(submission.columns) == ["ID", "Category"]
    assert len(submission) == expected_rows
    assert submission["ID"].nunique() == expected_rows
    assert submission["ID"].isna().sum() == 0
    assert submission["Category"].isna().sum() == 0
    assert (submission["Category"].astype(str).str.len() > 0).all()


def create_result_zip(config: Config) -> Path:
    zip_path = config.output_dir / "convnext_final_results.zip"
    files = result_files_for_zip(config)
    missing = [path for path in files if not path.exists()]
    if missing:
        raise RuntimeError(f"Cannot create result ZIP; missing files: {missing}")
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for path in files:
            archive.write(path, arcname=path.name)
    assert zip_path.exists()
    return zip_path


def save_outputs(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    class_names: Sequence[str],
    oof_probabilities: np.ndarray,
    test_probabilities: np.ndarray,
    fold_metrics: pd.DataFrame,
    template: pd.DataFrame,
    config: Config,
) -> tuple[Path, Path]:
    output = config.output_dir
    labels = train_df["label"].to_numpy()
    predictions = oof_probabilities.argmax(axis=1)

    model_oof = probability_frame(
        train_df["filename"], oof_probabilities, class_names, "filename"
    )
    model_test = probability_frame(
        test_df["filename"], test_probabilities, class_names, "ID"
    )
    model_oof.to_csv(output / "convnext_tiny_oof_probabilities.csv", index=False)
    model_test.to_csv(output / "convnext_tiny_test_probabilities.csv", index=False)
    model_test.to_csv(output / "test_probabilities.csv", index=False)
    fold_metrics.to_csv(output / "fold_metrics.csv", index=False)

    pd.DataFrame(
        classification_report(
            labels,
            predictions,
            labels=np.arange(len(class_names)),
            target_names=class_names,
            output_dict=True,
            zero_division=0,
        )
    ).transpose().to_csv(output / "classification_report.csv")
    pd.DataFrame(
        confusion_matrix(labels, predictions, labels=np.arange(len(class_names))),
        index=class_names,
        columns=class_names,
    ).to_csv(output / "confusion_matrix.csv")

    details = build_oof_details(train_df, oof_probabilities, class_names)
    details.to_csv(output / "oof_predictions.csv", index=False)
    errors = details[~details["is_correct"]].copy()
    errors.to_csv(output / "misclassified_oof.csv", index=False)
    grass_pair = {"Black-grass", "Loose Silky-bent"}
    grass_errors = errors[
        errors.apply(
            lambda row: {row["true_class"], row["predicted_class"]} == grass_pair,
            axis=1,
        )
    ]
    grass_errors.to_csv(output / "blackgrass_vs_silky_errors.csv", index=False)

    submission = template[["ID", "Category"]].copy()
    submission["Category"] = np.asarray(class_names)[test_probabilities.argmax(axis=1)]
    validate_submission(submission, config.expected_test_images)
    submission_path = output / "submission.csv"
    submission.to_csv(submission_path, index=False)
    zip_path = create_result_zip(config)
    return submission_path, zip_path


def final_file_safety_check(
    submission_path: Path, zip_path: Path, config: Config
) -> None:
    print("\nsubmission exists:", submission_path.exists())
    if not submission_path.exists():
        raise RuntimeError(f"Submission file was not created: {submission_path}")
    print("submission size:", submission_path.stat().st_size)

    check_df = pd.read_csv(submission_path)
    print("reloaded submission shape:", check_df.shape)
    print(check_df.head())
    assert check_df.shape == (config.expected_test_images, 2)
    validate_submission(check_df, config.expected_test_images)

    assert zip_path.exists()
    print("result zip exists:", zip_path.exists())
    print("result zip size:", zip_path.stat().st_size)
    if zip_path.stat().st_size <= 0:
        raise RuntimeError("Result ZIP is empty")

    print()
    print("=" * 60)
    print("TRAINING COMPLETE")
    print("SUBMISSION CREATED SUCCESSFULLY")
    print("DOWNLOAD THIS FILE NOW:")
    print("/kaggle/working/submission.csv")
    print("DO NOT RESTART SESSION BEFORE DOWNLOADING")
    print("=" * 60)


# ============================================================
# MAIN
# ============================================================


def main(config: Config = CONFIG) -> None:
    if config.use_tta:
        raise ValueError("Strict reproduction requires use_tta=False")
    config.output_dir.mkdir(parents=True, exist_ok=True)
    seed_everything(config.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print_environment(device)

    train_df, test_df, class_names = build_dataframes(config)
    train_df = assign_folds(train_df, config)
    print("Fold distribution:")
    print(pd.crosstab(train_df["fold"], train_df["class_name"]))

    template = find_submission_template(config, test_df["filename"].tolist())
    test_lookup = test_df.set_index("filename", drop=False)
    assert set(template["ID"]) == set(test_lookup.index)
    ordered_test_df = test_lookup.loc[template["ID"].tolist()].reset_index(drop=True)

    oof_probabilities, test_probabilities, fold_metrics = run_cross_validation(
        train_df, ordered_test_df, class_names, config, device
    )
    labels = train_df["label"].to_numpy()
    predictions = oof_probabilities.argmax(axis=1)
    overall_f1 = f1_score(labels, predictions, average="macro")
    overall_accuracy = accuracy_score(labels, predictions)

    print("\n" + "=" * 72)
    print("5-FOLD CONVNEXT-TINY RESULTS — NO TTA")
    print("=" * 72)
    for row in fold_metrics.to_dict("records"):
        print(
            f"Fold {int(row['fold']) + 1}: F1={row['oof_macro_f1']:.4f}, "
            f"Accuracy={row['oof_accuracy']:.4f}, best={row['best_stage']} "
            f"epoch {int(row['best_epoch'])}"
        )
    print(f"Fold mean / std : {fold_metrics['oof_macro_f1'].mean():.4f} / "
          f"{fold_metrics['oof_macro_f1'].std(ddof=0):.4f}")
    print(f"Worst fold F1   : {fold_metrics['oof_macro_f1'].min():.4f}")
    print(f"OOF Macro-F1    : {overall_f1:.4f}")
    print(f"OOF Accuracy    : {overall_accuracy:.4f}")
    print(f"Reference OOF   : {config.expected_reproduction_oof:.4f}")
    print("\nClassification report:")
    print(
        classification_report(
            labels,
            predictions,
            labels=np.arange(len(class_names)),
            target_names=class_names,
            digits=4,
            zero_division=0,
        )
    )
    print("Confusion matrix:")
    print(
        pd.DataFrame(
            confusion_matrix(labels, predictions, labels=np.arange(len(class_names))),
            index=class_names,
            columns=class_names,
        )
    )

    submission_path, zip_path = save_outputs(
        train_df,
        ordered_test_df,
        class_names,
        oof_probabilities,
        test_probabilities,
        fold_metrics,
        template,
        config,
    )
    submission = pd.read_csv(submission_path)
    assert len(submission) == 378
    assert submission["ID"].nunique() == 378
    assert submission["Category"].isna().sum() == 0
    print("\nsubmission.shape:", submission.shape)
    print("submission.head(10):")
    print(submission.head(10))
    print("submission Category counts:")
    print(submission["Category"].value_counts())
    final_file_safety_check(submission_path, zip_path, config)


if __name__ == "__main__":
    main()
