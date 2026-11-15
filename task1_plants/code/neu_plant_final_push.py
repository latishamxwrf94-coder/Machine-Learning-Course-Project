"""NEU ML Task1 final OOF-driven Kaggle pipeline.

Experiment structure (kept intentionally narrow):
1. One shared StratifiedKFold assignment for every model.
2. EfficientNet-B0@224 preserves the validated baseline configuration.
3. ConvNeXt-Tiny@288 remains the primary architecture.
4. EfficientNet-B2@260 and ResNet50@256 test useful capacity/diversity.
5. Every fold uses head warm-up followed by discriminative-LR fine-tuning.
6. Mild geometry/color augmentation protects complete leaf structure.
7. No MixUp, CutMix, focal loss, RandAugment, or size-model fusion.
8. Original and horizontal-flip TTA are compared using OOF predictions.
9. Grass-pair errors, confidence, fold stability, and diversity are reported.
10. Singles, pairs, and triples are searched using OOF probabilities only.
11. Near-tied ensembles prefer fewer models and simpler weights.
12. The submission uses exactly the OOF-selected model(s) and weights.
"""

from __future__ import annotations

import gc
import itertools
import json
import os
import random
import warnings
from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path
from typing import NamedTuple, Sequence

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
# CONFIG
# ============================================================


@dataclass(frozen=True)
class ModelSpec:
    name: str
    img_size: int
    batch_size: int
    backbone_lr: float
    classifier_lr: float


def default_model_specs() -> dict[str, ModelSpec]:
    return {
        "efficientnet_b0": ModelSpec("efficientnet_b0", 224, 32, 5e-5, 2e-4),
        "convnext_tiny": ModelSpec("convnext_tiny", 288, 16, 4e-5, 2e-4),
        "efficientnet_b2": ModelSpec("efficientnet_b2", 260, 16, 4e-5, 1.5e-4),
        "resnet50": ModelSpec("resnet50", 256, 32, 4e-5, 1.5e-4),
    }


@dataclass(frozen=True)
class Config:
    seed: int = 42
    data_root: Path = Path("/kaggle/input/datasets/zijin233/neu-plant-seedlings-5class")
    output_root: Path = Path("/kaggle/working")
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
    head_epochs: int = 2
    finetune_epochs: int = 18
    head_lr: float = 3e-4
    weight_decay: float = 1e-4
    eta_min: float = 1e-6
    label_smoothing: float = 0.05
    gradient_clip_norm: float = 1.0
    patience: int = 5
    num_workers: int = 2
    use_amp: bool = True
    use_horizontal_tta: bool = True
    use_vertical_tta: bool = False
    mixup: bool = False
    cutmix: bool = False
    focal_loss: bool = False
    randaugment: bool = False
    autoaugment: bool = False
    fast_dev_run: bool = False
    old_baseline_oof: float = 0.9023
    old_baseline_public: float = 0.9126
    previous_best_oof: float = 0.9137
    previous_best_public: float = 0.9232


CONFIG = Config()


@dataclass(frozen=True)
class RunSettings:
    fold_ids: list[int]
    head_epochs: int
    finetune_epochs: int
    output_dir: Path
    submission_filename: str


class TtaSelection(NamedTuple):
    name: str
    macro_f1: float
    probabilities: np.ndarray


@dataclass
class EnsembleCandidate:
    models: tuple[str, ...]
    weights: tuple[float, ...]
    macro_f1: float
    fold_mean: float
    fold_std: float
    worst_fold: float
    probabilities: np.ndarray


@dataclass
class EnsembleSearchResult:
    selected_models: tuple[str, ...]
    weights: dict[str, float]
    macro_f1: float
    probabilities: np.ndarray
    fold_mean: float
    fold_std: float
    worst_fold: float
    table: pd.DataFrame
    triple_eligible_models: tuple[str, ...]


def resolve_run_settings(config: Config) -> RunSettings:
    if config.fast_dev_run:
        return RunSettings(
            fold_ids=[0],
            head_epochs=1,
            finetune_epochs=1,
            output_dir=config.output_root / "fast_dev",
            submission_filename="submission_fast_dev.csv",
        )
    return RunSettings(
        fold_ids=list(range(config.n_folds)),
        head_epochs=config.head_epochs,
        finetune_epochs=config.finetune_epochs,
        output_dir=config.output_root,
        submission_filename="submission.csv",
    )


def validate_config(config: Config) -> None:
    forbidden = {
        "MixUp": config.mixup,
        "CutMix": config.cutmix,
        "FocalLoss": config.focal_loss,
        "RandAugment": config.randaugment,
        "AutoAugment": config.autoaugment,
    }
    enabled = [name for name, value in forbidden.items() if value]
    if enabled:
        raise NotImplementedError(
            "The controlled final experiment keeps these ablations disabled: "
            + ", ".join(enabled)
        )


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
    print("=" * 76)
    print("ENVIRONMENT")
    print("=" * 76)
    print("PyTorch version :", torch.__version__)
    print("CUDA available  :", torch.cuda.is_available())
    print("Device          :", device)
    if device.type == "cuda":
        print("GPU             :", torch.cuda.get_device_name(0))
    print()


# ============================================================
# DATA
# ============================================================


IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg"}


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
        for path in list_images(train_dir / class_name):
            with Image.open(path) as image:
                width, height = image.size
            records.append(
                {
                    "path": str(path),
                    "filename": path.name,
                    "class_name": class_name,
                    "label": class_to_idx[class_name],
                    "width": width,
                    "height": height,
                }
            )
    train_df = pd.DataFrame(records)
    assert len(train_df) == config.expected_train_images, (
        f"Expected {config.expected_train_images} training images, found {len(train_df)}"
    )

    test_records = []
    for path in list_images(test_dir):
        with Image.open(path) as image:
            width, height = image.size
        test_records.append(
            {"path": str(path), "filename": path.name, "width": width, "height": height}
        )
    test_df = pd.DataFrame(test_records)
    assert len(test_df) == config.expected_test_images, (
        f"Expected {config.expected_test_images} test images, found {len(test_df)}"
    )
    return train_df, test_df, class_names


def assign_folds(train_df: pd.DataFrame, config: Config) -> pd.DataFrame:
    result = train_df.copy()
    result["fold"] = -1
    splitter = StratifiedKFold(
        n_splits=config.n_folds,
        shuffle=True,
        random_state=config.seed,
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
# ============================================================


IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def build_transforms(spec: ModelSpec):
    from torchvision import transforms

    train_transform = transforms.Compose(
        [
            transforms.RandomResizedCrop(
                spec.img_size,
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
            transforms.Resize((spec.img_size, spec.img_size), antialias=True),
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
    batch_size: int,
    shuffle: bool,
    config: Config,
    device: torch.device,
) -> DataLoader:
    generator = torch.Generator()
    generator.manual_seed(config.seed)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=config.num_workers,
        pin_memory=device.type == "cuda",
        drop_last=False,
        persistent_workers=config.num_workers > 0,
        worker_init_fn=worker_init_fn,
        generator=generator,
    )


# ============================================================
# MODEL
# ============================================================


def create_model(model_name: str, num_classes: int) -> nn.Module:
    try:
        if model_name == "efficientnet_b0":
            from torchvision.models import EfficientNet_B0_Weights, efficientnet_b0

            model = efficientnet_b0(weights=EfficientNet_B0_Weights.DEFAULT)
            features = model.classifier[1].in_features
            model.classifier[1] = nn.Linear(features, num_classes)
        elif model_name == "convnext_tiny":
            from torchvision.models import ConvNeXt_Tiny_Weights, convnext_tiny

            model = convnext_tiny(weights=ConvNeXt_Tiny_Weights.DEFAULT)
            features = model.classifier[2].in_features
            model.classifier[2] = nn.Linear(features, num_classes)
        elif model_name == "efficientnet_b2":
            from torchvision.models import EfficientNet_B2_Weights, efficientnet_b2

            model = efficientnet_b2(weights=EfficientNet_B2_Weights.DEFAULT)
            features = model.classifier[1].in_features
            model.classifier[1] = nn.Linear(features, num_classes)
        elif model_name == "resnet50":
            from torchvision.models import ResNet50_Weights, resnet50

            model = resnet50(weights=ResNet50_Weights.DEFAULT)
            features = model.fc.in_features
            model.fc = nn.Linear(features, num_classes)
        else:
            raise ValueError(f"Unsupported model: {model_name}")
    except Exception as exc:
        raise RuntimeError(
            f"Failed to load official ImageNet weights for {model_name}. "
            "Use Kaggle internet or an existing torchvision cache. "
            "Random initialization is intentionally forbidden for this dataset."
        ) from exc
    return model


def classifier_parameters(model: nn.Module, model_name: str) -> list[nn.Parameter]:
    if model_name == "resnet50":
        return list(model.fc.parameters())
    return list(model.classifier.parameters())


def set_backbone_trainable(model: nn.Module, model_name: str, trainable: bool) -> None:
    for parameter in model.parameters():
        parameter.requires_grad = trainable
    if not trainable:
        for parameter in classifier_parameters(model, model_name):
            parameter.requires_grad = True


def create_finetune_optimizer(
    model: nn.Module, model_name: str, spec: ModelSpec, config: Config
) -> torch.optim.Optimizer:
    classifier = classifier_parameters(model, model_name)
    classifier_ids = {id(parameter) for parameter in classifier}
    backbone = [parameter for parameter in model.parameters() if id(parameter) not in classifier_ids]
    return torch.optim.AdamW(
        [
            {"params": backbone, "lr": spec.backbone_lr},
            {"params": classifier, "lr": spec.classifier_lr},
        ],
        weight_decay=config.weight_decay,
    )


# ============================================================
# TRAIN / VALIDATE
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
def predict_no_tta_and_tta(
    model: nn.Module,
    loader: DataLoader,
    config: Config,
    device: torch.device,
    labeled: bool,
) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    no_tta_batches = []
    tta_batches = []
    for batch in loader:
        images = batch[0] if labeled else batch
        images = images.to(device, non_blocking=True)
        original = torch.softmax(model(images), dim=1)
        views = [original]
        if config.use_horizontal_tta:
            views.append(torch.softmax(model(torch.flip(images, dims=[3])), dim=1))
        if config.use_vertical_tta:
            views.append(torch.softmax(model(torch.flip(images, dims=[2])), dim=1))
        no_tta_batches.append(original.cpu().numpy())
        tta_batches.append((sum(views) / len(views)).cpu().numpy())
    return np.concatenate(no_tta_batches), np.concatenate(tta_batches)


def save_checkpoint(
    path: Path,
    model: nn.Module,
    model_name: str,
    fold: int,
    stage: str,
    epoch: int,
    score: float,
    class_names: Sequence[str],
    spec: ModelSpec,
) -> None:
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "model_name": model_name,
            "fold": fold,
            "best_stage": stage,
            "best_epoch": epoch,
            "best_val_macro_f1": score,
            "class_names": list(class_names),
            "img_size": spec.img_size,
        },
        path,
    )


def train_fold(
    spec: ModelSpec,
    fold: int,
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    class_names: Sequence[str],
    config: Config,
    run: RunSettings,
    device: torch.device,
) -> dict:
    train_transform, eval_transform = build_transforms(spec)
    fold_train = train_df[train_df["fold"] != fold].reset_index(drop=True)
    fold_val = train_df[train_df["fold"] == fold].reset_index()
    train_loader = make_loader(
        PlantDataset(fold_train, train_transform, True),
        spec.batch_size,
        True,
        config,
        device,
    )
    val_loader = make_loader(
        PlantDataset(fold_val, eval_transform, True),
        spec.batch_size,
        False,
        config,
        device,
    )
    test_loader = make_loader(
        PlantDataset(test_df, eval_transform, False),
        spec.batch_size,
        False,
        config,
        device,
    )

    model = create_model(spec.name, len(class_names)).to(device)
    criterion = nn.CrossEntropyLoss(label_smoothing=config.label_smoothing)
    scaler = torch.cuda.amp.GradScaler(enabled=config.use_amp and device.type == "cuda")
    checkpoint_path = run.output_dir / "checkpoints" / f"{spec.name}_fold{fold}.pth"
    best_score = -1.0
    best_stage = ""
    best_epoch = -1

    print(f"\n[{spec.name}] fold {fold + 1}/{config.n_folds}: head warm-up")
    set_backbone_trainable(model, spec.name, False)
    head_optimizer = torch.optim.AdamW(
        classifier_parameters(model, spec.name),
        lr=config.head_lr,
        weight_decay=config.weight_decay,
    )
    for epoch in range(1, run.head_epochs + 1):
        train_loss, train_acc, train_f1 = train_one_epoch(
            model, train_loader, criterion, head_optimizer, scaler, config, device
        )
        val_loss, val_acc, val_f1 = validate(model, val_loader, criterion, device)
        print(
            f"  head {epoch:02d}/{run.head_epochs} | train {train_loss:.4f} "
            f"acc {train_acc:.4f} f1 {train_f1:.4f} | val {val_loss:.4f} "
            f"acc {val_acc:.4f} f1 {val_f1:.4f}"
        )
        if val_f1 > best_score:
            best_score, best_stage, best_epoch = val_f1, "head", epoch
            save_checkpoint(
                checkpoint_path,
                model,
                spec.name,
                fold,
                best_stage,
                best_epoch,
                best_score,
                class_names,
                spec,
            )

    print(f"[{spec.name}] fold {fold + 1}/{config.n_folds}: full fine-tuning")
    set_backbone_trainable(model, spec.name, True)
    optimizer = create_finetune_optimizer(model, spec.name, spec, config)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=max(1, run.finetune_epochs),
        eta_min=config.eta_min,
    )
    best_finetune_score = -1.0
    epochs_without_improvement = 0
    for epoch in range(1, run.finetune_epochs + 1):
        train_loss, train_acc, train_f1 = train_one_epoch(
            model, train_loader, criterion, optimizer, scaler, config, device
        )
        val_loss, val_acc, val_f1 = validate(model, val_loader, criterion, device)
        print(
            f"  tune {epoch:02d}/{run.finetune_epochs} | lr "
            f"{optimizer.param_groups[0]['lr']:.2e}/{optimizer.param_groups[1]['lr']:.2e} | "
            f"train {train_loss:.4f} acc {train_acc:.4f} f1 {train_f1:.4f} | "
            f"val {val_loss:.4f} acc {val_acc:.4f} f1 {val_f1:.4f}"
        )
        if val_f1 > best_finetune_score:
            best_finetune_score = val_f1
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
        if val_f1 > best_score:
            best_score, best_stage, best_epoch = val_f1, "finetune", epoch
            save_checkpoint(
                checkpoint_path,
                model,
                spec.name,
                fold,
                best_stage,
                best_epoch,
                best_score,
                class_names,
                spec,
            )
        scheduler.step()
        if epochs_without_improvement >= config.patience:
            print(f"  early stopping after fine-tune epoch {epoch}")
            break

    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    val_no_tta, val_tta = predict_no_tta_and_tta(model, val_loader, config, device, True)
    test_no_tta, test_tta = predict_no_tta_and_tta(model, test_loader, config, device, False)
    result = {
        "indices": fold_val["index"].to_numpy(),
        "val_no_tta": val_no_tta,
        "val_tta": val_tta,
        "test_no_tta": test_no_tta,
        "test_tta": test_tta,
        "checkpoint": str(checkpoint_path),
        "best_stage": checkpoint["best_stage"],
        "best_epoch": checkpoint["best_epoch"],
        "selection_val_macro_f1": checkpoint["best_val_macro_f1"],
    }

    del model, train_loader, val_loader, test_loader
    del head_optimizer, optimizer, scheduler, scaler
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return result


def run_model_cross_validation(
    spec: ModelSpec,
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    class_names: Sequence[str],
    config: Config,
    run: RunSettings,
    device: torch.device,
) -> dict:
    shape_oof = (len(train_df), len(class_names))
    shape_test = (len(test_df), len(class_names))
    oof_no_tta = np.full(shape_oof, np.nan, dtype=np.float32)
    oof_tta = np.full(shape_oof, np.nan, dtype=np.float32)
    test_no_tta = np.zeros(shape_test, dtype=np.float32)
    test_tta = np.zeros(shape_test, dtype=np.float32)
    fold_records = []

    for fold in run.fold_ids:
        try:
            result = train_fold(
                spec, fold, train_df, test_df, class_names, config, run, device
            )
        except RuntimeError as exc:
            if "out of memory" in str(exc).lower():
                if device.type == "cuda":
                    torch.cuda.empty_cache()
                raise RuntimeError(
                    f"CUDA OOM for {spec.name}: img_size={spec.img_size}, "
                    f"batch_size={spec.batch_size}. Reduce that model's configured batch size "
                    "and restart; the script will not change it automatically."
                ) from exc
            raise
        indices = result["indices"]
        oof_no_tta[indices] = result["val_no_tta"]
        oof_tta[indices] = result["val_tta"]
        test_no_tta += result["test_no_tta"] / len(run.fold_ids)
        test_tta += result["test_tta"] / len(run.fold_ids)
        fold_records.append(
            {
                "model": spec.name,
                "fold": fold,
                "best_stage": result["best_stage"],
                "best_epoch": result["best_epoch"],
                "selection_val_macro_f1": result["selection_val_macro_f1"],
                "checkpoint": result["checkpoint"],
            }
        )
    return {
        "oof_no_tta": oof_no_tta,
        "oof_tta": oof_tta,
        "test_no_tta": test_no_tta,
        "test_tta": test_tta,
        "fold_records": fold_records,
    }


# ============================================================
# METRICS / TTA / GRASS PAIR
# ============================================================


def macro_f1(labels: np.ndarray, probabilities: np.ndarray) -> float:
    return float(f1_score(labels, probabilities.argmax(axis=1), average="macro"))


def select_tta_variant(
    no_tta: np.ndarray, tta: np.ndarray, labels: np.ndarray
) -> TtaSelection:
    no_tta_score = macro_f1(labels, no_tta)
    tta_score = macro_f1(labels, tta)
    if tta_score > no_tta_score:
        return TtaSelection("tta", tta_score, tta)
    return TtaSelection("no_tta", no_tta_score, no_tta)


def fold_scores(
    labels: np.ndarray,
    probabilities: np.ndarray,
    folds: np.ndarray,
    fold_ids: Sequence[int],
) -> list[float]:
    return [
        macro_f1(labels[folds == fold], probabilities[folds == fold])
        for fold in fold_ids
    ]


def compute_grass_pair_metrics(
    labels: np.ndarray, predictions: np.ndarray, class_names: Sequence[str]
) -> dict[str, float | int]:
    black = class_names.index("Black-grass")
    silky = class_names.index("Loose Silky-bent")
    black_mask = labels == black
    silky_mask = labels == silky
    grass_mask = black_mask | silky_mask
    return {
        "blackgrass_recall": float(np.mean(predictions[black_mask] == black)),
        "silky_recall": float(np.mean(predictions[silky_mask] == silky)),
        "blackgrass_to_silky": int(np.sum(black_mask & (predictions == silky))),
        "silky_to_blackgrass": int(np.sum(silky_mask & (predictions == black))),
        "grass_pair_accuracy": float(np.mean(predictions[grass_mask] == labels[grass_mask])),
        "black_true_black_pred": int(np.sum(black_mask & (predictions == black))),
        "black_true_silky_pred": int(np.sum(black_mask & (predictions == silky))),
        "black_true_other_pred": int(np.sum(black_mask & ~np.isin(predictions, [black, silky]))),
        "silky_true_black_pred": int(np.sum(silky_mask & (predictions == black))),
        "silky_true_silky_pred": int(np.sum(silky_mask & (predictions == silky))),
        "silky_true_other_pred": int(np.sum(silky_mask & ~np.isin(predictions, [black, silky]))),
    }


def confidence_metrics(labels: np.ndarray, probabilities: np.ndarray) -> dict[str, float]:
    predictions = probabilities.argmax(axis=1)
    confidence = probabilities.max(axis=1)
    correct = predictions == labels
    return {
        "mean_max_probability": float(confidence.mean()),
        "correct_prediction_confidence": float(confidence[correct].mean()) if correct.any() else np.nan,
        "wrong_prediction_confidence": float(confidence[~correct].mean()) if (~correct).any() else np.nan,
    }


# ============================================================
# MODEL DIVERSITY
# ============================================================


def build_model_diversity(
    selected_oof: dict[str, np.ndarray],
    labels: np.ndarray,
    class_names: Sequence[str],
) -> pd.DataFrame:
    black = class_names.index("Black-grass")
    silky = class_names.index("Loose Silky-bent")
    grass_mask = np.isin(labels, [black, silky])
    rows = []
    for model_a, model_b in itertools.combinations(selected_oof, 2):
        pred_a = selected_oof[model_a].argmax(axis=1)
        pred_b = selected_oof[model_b].argmax(axis=1)
        wrong_a = pred_a != labels
        wrong_b = pred_b != labels
        error_union = wrong_a | wrong_b
        rows.append(
            {
                "model_a": model_a,
                "model_b": model_b,
                "model_a_macro_f1": f1_score(labels, pred_a, average="macro"),
                "model_b_macro_f1": f1_score(labels, pred_b, average="macro"),
                "prediction_disagreement_rate": float(np.mean(pred_a != pred_b)),
                "error_overlap_rate": float(
                    np.sum(wrong_a & wrong_b) / max(1, np.sum(error_union))
                ),
                "both_wrong_count": int(np.sum(wrong_a & wrong_b)),
                "a_wrong_b_correct_count": int(np.sum(wrong_a & ~wrong_b)),
                "a_correct_b_wrong_count": int(np.sum(~wrong_a & wrong_b)),
                "grass_subset_disagreement": float(np.mean(pred_a[grass_mask] != pred_b[grass_mask])),
            }
        )
    return pd.DataFrame(rows)


# ============================================================
# ENSEMBLE SEARCH
# ============================================================


def make_candidate(
    models: tuple[str, ...],
    weights: tuple[float, ...],
    probabilities: dict[str, np.ndarray],
    labels: np.ndarray,
    folds: np.ndarray,
    fold_ids: Sequence[int],
) -> EnsembleCandidate:
    blended = sum(weight * probabilities[name] for name, weight in zip(models, weights))
    scores = fold_scores(labels, blended, folds, fold_ids)
    return EnsembleCandidate(
        models=models,
        weights=weights,
        macro_f1=macro_f1(labels, blended),
        fold_mean=float(np.mean(scores)),
        fold_std=float(np.std(scores)),
        worst_fold=float(np.min(scores)),
        probabilities=blended,
    )


def weight_complexity(weights: Sequence[float]) -> int:
    return sum(abs(weight * 10 - round(weight * 10)) > 1e-8 for weight in weights)


def choose_best_candidate(
    candidates: Sequence[EnsembleCandidate], tolerance: float = 0.001
) -> EnsembleCandidate:
    maximum = max(candidate.macro_f1 for candidate in candidates)
    near_best = [candidate for candidate in candidates if candidate.macro_f1 >= maximum - tolerance]
    return min(
        near_best,
        key=lambda candidate: (
            len(candidate.models),
            weight_complexity(candidate.weights),
            candidate.fold_std,
            -candidate.worst_fold,
            -candidate.macro_f1,
        ),
    )


def search_ensembles(
    probabilities: dict[str, np.ndarray],
    labels: np.ndarray,
    folds: np.ndarray,
    fold_ids: Sequence[int],
) -> EnsembleSearchResult:
    model_names = tuple(probabilities)
    single_candidates = [
        make_candidate((name,), (1.0,), probabilities, labels, folds, fold_ids)
        for name in model_names
    ]
    pair_candidates = []
    for model_a, model_b in itertools.combinations(model_names, 2):
        for units_a in range(1, 20):
            weight_a = units_a / 20.0
            pair_candidates.append(
                make_candidate(
                    (model_a, model_b),
                    (weight_a, 1.0 - weight_a),
                    probabilities,
                    labels,
                    folds,
                    fold_ids,
                )
            )

    best_single = max(candidate.macro_f1 for candidate in single_candidates)
    single_scores = {candidate.models[0]: candidate.macro_f1 for candidate in single_candidates}
    best_pair_by_model = {
        name: max(
            (candidate.macro_f1 for candidate in pair_candidates if name in candidate.models),
            default=-1.0,
        )
        for name in model_names
    }
    eligible = tuple(
        name
        for name in model_names
        if single_scores[name] >= best_single - 0.03
        or best_pair_by_model[name] >= best_single + 0.0005
    )

    triple_candidates = []
    if len(eligible) >= 3:
        for triple in itertools.combinations(eligible, 3):
            for units_a in range(1, 19):
                for units_b in range(1, 20 - units_a):
                    units_c = 20 - units_a - units_b
                    if units_c < 1:
                        continue
                    weights = (units_a / 20.0, units_b / 20.0, units_c / 20.0)
                    triple_candidates.append(
                        make_candidate(triple, weights, probabilities, labels, folds, fold_ids)
                    )

    all_candidates = single_candidates + pair_candidates + triple_candidates
    selected = choose_best_candidate(all_candidates)
    rows = []
    for candidate in all_candidates:
        row = {
            "stage": {1: "single", 2: "pair", 3: "triple"}[len(candidate.models)],
            "models": " | ".join(candidate.models),
            "weights_json": json.dumps(dict(zip(candidate.models, candidate.weights))),
            "n_models": len(candidate.models),
            "macro_f1": candidate.macro_f1,
            "fold_mean": candidate.fold_mean,
            "fold_std": candidate.fold_std,
            "worst_fold_f1": candidate.worst_fold,
            "selected": candidate is selected,
        }
        for name in model_names:
            row[f"weight_{name}"] = (
                candidate.weights[candidate.models.index(name)] if name in candidate.models else 0.0
            )
        rows.append(row)
    return EnsembleSearchResult(
        selected_models=selected.models,
        weights=dict(zip(selected.models, selected.weights)),
        macro_f1=selected.macro_f1,
        probabilities=selected.probabilities,
        fold_mean=selected.fold_mean,
        fold_std=selected.fold_std,
        worst_fold=selected.worst_fold,
        table=pd.DataFrame(rows),
        triple_eligible_models=eligible,
    )


# ============================================================
# OUTPUT HELPERS
# ============================================================


def probability_frame(
    identifiers: Sequence[str],
    probabilities: np.ndarray,
    class_names: Sequence[str],
    id_column: str,
    evaluated: np.ndarray | None = None,
) -> pd.DataFrame:
    frame = pd.DataFrame({id_column: list(identifiers)})
    if evaluated is not None:
        frame["is_evaluated"] = evaluated
    for index, class_name in enumerate(class_names):
        frame[f"prob_{class_name}"] = probabilities[:, index]
    return frame


def build_oof_details(
    train_df: pd.DataFrame,
    probabilities: np.ndarray,
    class_names: Sequence[str],
    active_mask: np.ndarray,
) -> pd.DataFrame:
    frame = train_df.loc[
        active_mask, ["filename", "path", "fold", "class_name", "label"]
    ].copy()
    probs = probabilities[active_mask]
    order = np.argsort(probs, axis=1)[:, ::-1]
    top1, top2 = order[:, 0], order[:, 1]
    names = np.asarray(class_names)
    frame["true_class"] = frame["class_name"]
    frame["predicted_class"] = names[top1]
    frame["prediction_confidence"] = probs[np.arange(len(probs)), top1]
    frame["top2_class"] = names[top2]
    frame["top2_probability"] = probs[np.arange(len(probs)), top2]
    frame["is_correct"] = frame["label"].to_numpy() == top1
    for index, class_name in enumerate(class_names):
        frame[f"prob_{class_name}"] = probs[:, index]
    return frame.drop(columns=["class_name"])


def validate_submission(submission: pd.DataFrame, expected_rows: int) -> None:
    assert list(submission.columns) == ["ID", "Category"]
    assert len(submission) == expected_rows
    assert submission["ID"].nunique() == expected_rows
    assert submission["ID"].isna().sum() == 0
    assert submission["Category"].isna().sum() == 0
    assert (submission["Category"].astype(str).str.len() > 0).all()


def save_outputs(
    config: Config,
    run: RunSettings,
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    class_names: Sequence[str],
    active_mask: np.ndarray,
    model_runs: dict[str, dict],
    selected_oof_full: dict[str, np.ndarray],
    selected_test: dict[str, np.ndarray],
    tta_choices: dict[str, str],
    fold_metrics: pd.DataFrame,
    model_summary: pd.DataFrame,
    diversity: pd.DataFrame,
    ensemble: EnsembleSearchResult,
    final_test: np.ndarray,
    grass_metrics: pd.DataFrame,
    submission_template: pd.DataFrame,
) -> None:
    out = run.output_dir
    train_df.to_csv(out / "fold_assignments.csv", index=False)
    fold_metrics.to_csv(out / "fold_metrics.csv", index=False)
    model_summary.to_csv(out / "model_summary.csv", index=False)
    diversity.to_csv(out / "model_diversity.csv", index=False)
    ensemble.table.to_csv(out / "ensemble_search.csv", index=False)
    grass_metrics.to_csv(out / "grass_pair_metrics.csv", index=False)

    for model_name, model_run in model_runs.items():
        variants = {
            "no_tta": (model_run["oof_no_tta"], model_run["test_no_tta"]),
            "tta": (model_run["oof_tta"], model_run["test_tta"]),
            "selected": (selected_oof_full[model_name], selected_test[model_name]),
        }
        for variant, (oof_probs, test_probs) in variants.items():
            suffix = "" if variant == "selected" else f"_{variant}"
            probability_frame(
                train_df["filename"], oof_probs, class_names, "filename", active_mask
            ).to_csv(out / f"{model_name}_oof_probabilities{suffix}.csv", index=False)
            probability_frame(
                test_df["filename"], test_probs, class_names, "ID"
            ).to_csv(out / f"{model_name}_test_probabilities{suffix}.csv", index=False)

    final_oof_full = np.full((len(train_df), len(class_names)), np.nan, dtype=np.float32)
    final_oof_full[active_mask] = ensemble.probabilities
    details = build_oof_details(train_df, final_oof_full, class_names, active_mask)
    details.to_csv(out / "oof_predictions.csv", index=False)
    errors = details[~details["is_correct"]].copy()
    errors.to_csv(out / "misclassified_oof.csv", index=False)
    grass_set = {"Black-grass", "Loose Silky-bent"}
    grass_errors = errors[
        errors.apply(
            lambda row: {row["true_class"], row["predicted_class"]} == grass_set,
            axis=1,
        )
    ]
    grass_errors.to_csv(out / "blackgrass_vs_silky_errors.csv", index=False)

    labels = train_df.loc[active_mask, "label"].to_numpy()
    predictions = ensemble.probabilities.argmax(axis=1)
    pd.DataFrame(
        confusion_matrix(labels, predictions, labels=np.arange(len(class_names))),
        index=class_names,
        columns=class_names,
    ).to_csv(out / "confusion_matrix.csv")
    pd.DataFrame(
        classification_report(
            labels,
            predictions,
            labels=np.arange(len(class_names)),
            target_names=class_names,
            output_dict=True,
            zero_division=0,
        )
    ).transpose().to_csv(out / "classification_report.csv")

    probability_frame(test_df["filename"], final_test, class_names, "ID").to_csv(
        out / "test_probabilities.csv", index=False
    )
    submission = submission_template[["ID", "Category"]].copy()
    submission["Category"] = np.asarray(class_names)[final_test.argmax(axis=1)]
    validate_submission(submission, config.expected_test_images)
    submission_path = out / run.submission_filename
    submission.to_csv(submission_path, index=False)

    selection = {
        "tta_choices": tta_choices,
        "selected_models": list(ensemble.selected_models),
        "weights": ensemble.weights,
        "oof_macro_f1": ensemble.macro_f1,
        "fast_dev_run": config.fast_dev_run,
    }
    (out / "selected_ensemble.json").write_text(
        json.dumps(selection, indent=2), encoding="utf-8"
    )
    print("\nSubmission shape:", submission.shape)
    print(submission.head())
    print("\nCategory distribution:")
    print(submission["Category"].value_counts())
    print("Saved path:", submission_path)


# ============================================================
# MAIN
# ============================================================


def main(config: Config = CONFIG) -> None:
    validate_config(config)
    run = resolve_run_settings(config)
    run.output_dir.mkdir(parents=True, exist_ok=True)
    (run.output_dir / "checkpoints").mkdir(parents=True, exist_ok=True)
    seed_everything(config.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print_environment(device)
    print("FAST_DEV_RUN      :", config.fast_dev_run)
    print("Output directory  :", run.output_dir)
    print("Folds             :", run.fold_ids)
    print("Head/fine-tune    :", run.head_epochs, "/", run.finetune_epochs)
    print("Horizontal TTA    :", config.use_horizontal_tta)
    print("Vertical TTA      :", config.use_vertical_tta)

    train_df, test_df, class_names = build_dataframes(config)
    train_df = assign_folds(train_df, config)
    print("\nShared fold distribution:")
    print(pd.crosstab(train_df["fold"], train_df["class_name"]))
    template = find_submission_template(config, test_df["filename"].tolist())
    test_lookup = test_df.set_index("filename", drop=False)
    assert set(template["ID"]) == set(test_lookup.index)
    ordered_test_df = test_lookup.loc[template["ID"].tolist()].reset_index(drop=True)

    specs = default_model_specs()
    model_runs = {}
    for model_name, spec in specs.items():
        print("\n" + "#" * 76)
        print(f"TRAINING {model_name}: {spec.img_size}px, batch {spec.batch_size}")
        print("#" * 76)
        model_runs[model_name] = run_model_cross_validation(
            spec,
            train_df,
            ordered_test_df,
            class_names,
            config,
            run,
            device,
        )

    active_mask = train_df["fold"].isin(run.fold_ids).to_numpy()
    labels = train_df.loc[active_mask, "label"].to_numpy()
    active_folds = train_df.loc[active_mask, "fold"].to_numpy()
    selected_oof_active = {}
    selected_oof_full = {}
    selected_test = {}
    tta_choices = {}
    fold_rows = []
    summary_rows = []
    grass_rows = []

    for model_name, run_data in model_runs.items():
        no_tta = run_data["oof_no_tta"][active_mask]
        tta = run_data["oof_tta"][active_mask]
        selection = select_tta_variant(no_tta, tta, labels)
        tta_choices[model_name] = selection.name
        selected_oof_active[model_name] = selection.probabilities
        selected_oof_full[model_name] = (
            run_data["oof_tta"] if selection.name == "tta" else run_data["oof_no_tta"]
        )
        selected_test[model_name] = (
            run_data["test_tta"] if selection.name == "tta" else run_data["test_no_tta"]
        )

        no_tta_scores = fold_scores(labels, no_tta, active_folds, run.fold_ids)
        tta_scores = fold_scores(labels, tta, active_folds, run.fold_ids)
        selected_scores = fold_scores(
            labels, selection.probabilities, active_folds, run.fold_ids
        )
        records_by_fold = {record["fold"]: record for record in run_data["fold_records"]}
        for position, fold in enumerate(run.fold_ids):
            record = records_by_fold[fold]
            fold_rows.append(
                {
                    **record,
                    "no_tta_macro_f1": no_tta_scores[position],
                    "tta_macro_f1": tta_scores[position],
                    "selected_variant": selection.name,
                    "selected_macro_f1": selected_scores[position],
                }
            )

        predictions = selection.probabilities.argmax(axis=1)
        grass = compute_grass_pair_metrics(labels, predictions, class_names)
        confidence = confidence_metrics(labels, selection.probabilities)
        summary_rows.append(
            {
                "model": model_name,
                "img_size": specs[model_name].img_size,
                "batch_size": specs[model_name].batch_size,
                "selected_tta": selection.name,
                "oof_macro_f1_no_tta": macro_f1(labels, no_tta),
                "oof_macro_f1_tta": macro_f1(labels, tta),
                "oof_macro_f1": selection.macro_f1,
                "oof_accuracy": accuracy_score(labels, predictions),
                "fold_mean": float(np.mean(selected_scores)),
                "fold_std": float(np.std(selected_scores)),
                "worst_fold_f1": float(np.min(selected_scores)),
                **confidence,
                **grass,
                "temperature": 1.0,
                "calibration_used": False,
            }
        )
        grass_rows.append({"model": model_name, **grass})

    diversity = build_model_diversity(selected_oof_active, labels, class_names)
    ensemble = search_ensembles(
        selected_oof_active, labels, active_folds, run.fold_ids
    )
    final_test = sum(
        weight * selected_test[model]
        for model, weight in ensemble.weights.items()
    )
    final_predictions = ensemble.probabilities.argmax(axis=1)
    final_grass = compute_grass_pair_metrics(labels, final_predictions, class_names)
    grass_rows.append({"model": "final_ensemble", **final_grass})

    model_summary = pd.DataFrame(summary_rows)
    model_summary["eligible_for_triple"] = model_summary["model"].isin(
        ensemble.triple_eligible_models
    )
    fold_metrics = pd.DataFrame(fold_rows)
    grass_metrics = pd.DataFrame(grass_rows)

    print("\n" + "=" * 76)
    print("MODEL RESULTS")
    print("=" * 76)
    for row in model_summary.to_dict("records"):
        print(f"\n{row['model']}")
        print(f"OOF Macro-F1       : {row['oof_macro_f1']:.4f}")
        print(f"Without / with TTA : {row['oof_macro_f1_no_tta']:.4f} / {row['oof_macro_f1_tta']:.4f}")
        print(f"Selected TTA       : {row['selected_tta']}")
        print(f"Fold mean          : {row['fold_mean']:.4f}")
        print(f"Fold std           : {row['fold_std']:.4f}")
        print(f"Worst fold F1      : {row['worst_fold_f1']:.4f}")
        print(f"Grass pair accuracy: {row['grass_pair_accuracy']:.4f}")
        print(
            "Black→Silky / Silky→Black: "
            f"{row['blackgrass_to_silky']} / {row['silky_to_blackgrass']}"
        )

    highest_mean_row = model_summary.loc[model_summary["fold_mean"].idxmax()]
    best_single_oof = model_summary["oof_macro_f1"].max()
    competitive = model_summary[
        model_summary["oof_macro_f1"] >= best_single_oof - 0.005
    ]
    stable_row = competitive.loc[competitive["fold_std"].idxmin()]
    if stable_row["model"] != highest_mean_row["model"]:
        print(
            "\nSTABILITY NOTE: "
            f"{highest_mean_row['model']} has the highest fold mean "
            f"({highest_mean_row['fold_mean']:.4f}) but "
            f"{stable_row['model']} is more stable among models within 0.005 OOF "
            f"(std {stable_row['fold_std']:.4f} vs "
            f"{highest_mean_row['fold_std']:.4f})."
        )
    else:
        print(
            "\nSTABILITY NOTE: "
            f"{highest_mean_row['model']} has both the highest fold mean and the "
            "lowest fold std among models within 0.005 OOF of the best single model."
        )

    final_report = classification_report(
        labels,
        final_predictions,
        labels=np.arange(len(class_names)),
        target_names=class_names,
        output_dict=True,
        zero_division=0,
    )
    print("\n" + "=" * 76)
    print("BEST ENSEMBLE")
    print("=" * 76)
    print("Models              :", list(ensemble.selected_models))
    print("Weights             :", ensemble.weights)
    print(f"OOF Macro-F1        : {ensemble.macro_f1:.4f}")
    print(f"OOF Accuracy        : {accuracy_score(labels, final_predictions):.4f}")
    print(f"Fold mean / std     : {ensemble.fold_mean:.4f} / {ensemble.fold_std:.4f}")
    print(f"Worst fold F1       : {ensemble.worst_fold:.4f}")
    print(f"Old baseline        : {config.old_baseline_oof:.4f}")
    print(f"Previous best       : {config.previous_best_oof:.4f}")
    print(f"Improvement baseline: {ensemble.macro_f1 - config.old_baseline_oof:+.4f}")
    print(f"Improvement previous: {ensemble.macro_f1 - config.previous_best_oof:+.4f}")
    print(f"Black-grass F1      : {final_report['Black-grass']['f1-score']:.4f}")
    print(f"Loose Silky-bent F1 : {final_report['Loose Silky-bent']['f1-score']:.4f}")
    print(f"Grass pair accuracy : {final_grass['grass_pair_accuracy']:.4f}")
    print(
        "Calibration        : not applied; temperature scaling preserves single-model "
        "argmax and was not added as another search variable. Confidence diagnostics are saved."
    )

    save_outputs(
        config,
        run,
        train_df,
        ordered_test_df,
        class_names,
        active_mask,
        model_runs,
        selected_oof_full,
        selected_test,
        tta_choices,
        fold_metrics,
        model_summary,
        diversity,
        ensemble,
        final_test,
        grass_metrics,
        template,
    )


if __name__ == "__main__":
    main()
