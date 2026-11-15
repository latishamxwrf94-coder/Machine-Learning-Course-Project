"""2026task2 high-performance ensemble: B2 anchor + regularized B2 + ConvNeXt-Tiny.

Kaggle: add original competition and the 2026task2-folds-v3 dataset to Input,
       enable GPU and Internet (or cache both ImageNet pretrained weights),
       then execute this file with %run /kaggle/working/fer_strong_ensemble.py
       or paste its contents into one notebook cell.

Outputs: /kaggle/working/fer_strong_ensemble/ and /kaggle/working/submission_best.csv
Only official ImageNet pretrained weights are used, NOT FER-trained external weights.
"""

import copy
import gc
import hashlib
import json
import math
import os
import random
import shutil
import time
import zipfile
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms
from torchvision.transforms import InterpolationMode
from sklearn.metrics import classification_report, confusion_matrix


# ============================================================
# Configuration. Keep all paths unchanged if mounting matches prior notebooks.
# ============================================================
DATA_ROOT = Path('/kaggle/input/competitions/2026task2/fer_data/fer_data')
FOLDS_PATH = Path('/kaggle/input/datasets/zijin233/2026task2-folds-v3/2026task2_folds.csv')
GROUPS_PATH = Path('/kaggle/input/datasets/zijin233/2026task2-folds-v3/2026task2_groups.csv')
TEMPLATE_PATH = Path('/kaggle/input/competitions/2026task2/submission.csv')
OUT = Path('/kaggle/working/fer_strong_ensemble')
OUT.mkdir(parents=True, exist_ok=True)

LABEL_MAP = {'Angry': 0, 'Fear': 1, 'Happy': 2, 'Sad': 3, 'Surprise': 4, 'Neutral': 5}
CLASSES = list(LABEL_MAP)
IMAGE_SIZE = 192
WORKERS = 2
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# A minimum validation gain is demanded before replacing the best single
# with a 3-model ensemble; it limits selection noise but does not eliminate it.
ENSEMBLE_MIN_NET_CORRECT = 6

# Optional previously trained D checkpoint from your exp-d Version 1.
# If not mounted, it is skipped automatically (no retraining).
EXTRA_D_CHECKPOINT = Path('/kaggle/input/notebooks/zijin233/exp-d/fer_experiment_D/best_efficientnet_b2_D.pt')


@dataclass
class Recipe:
    name: str
    arch: str
    seed: int
    epochs: int
    patience: int
    batch_size: int
    backbone_lr: float
    head_lr: float
    weight_decay: float
    dropout: float
    mixup_alpha: float
    mixup_prob: float
    ema_decay: float
    aug: str
    label_smoothing: float = 0.05


RECIPES = [
    Recipe('b2_anchor', 'b2', 2026, 25, 7, 64, 2e-4, 2e-4, 1e-4, 0.30, 0.0, 0.0, 0.0, 'original'),
    Recipe('b2_robust', 'b2', 2077, 25, 8, 64, 1.0e-4, 3.0e-4, 1e-2, 0.35, 0.15, 0.35, 0.995, 'smooth'),
    Recipe('convnext_tiny', 'convnext', 2101, 24, 7, 32, 6.0e-5, 2.0e-4, 1.5e-2, 0.25, 0.12, 0.30, 0.995, 'smooth'),
]


# ============================================================
# General utilities
# ============================================================
def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed % (2 ** 32))
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def ensure(condition, message):
    if not condition:
        raise RuntimeError(message)


def save_json(obj, path):
    def convert(v):
        if isinstance(v, (np.integer,)):
            return int(v)
        if isinstance(v, (np.floating,)):
            return float(v)
        if isinstance(v, Path):
            return str(v)
        raise TypeError(type(v).__name__)
    Path(path).write_text(json.dumps(obj, indent=2, ensure_ascii=False, default=convert), encoding='utf-8')


def atomic_torch_save(payload, path):
    path = Path(path)
    tmp = path.with_suffix('.tmp')
    torch.save(payload, tmp)
    os.replace(tmp, path)


def cpu_state_dict(model):
    return {k: t.detach().cpu().clone() for k, t in model.state_dict().items()}


def num_correct(probs, labels):
    return int((probs.argmax(1) == labels).sum())


# ============================================================
# Fold 0 and B-equivalent SHA256 cleaning
# ============================================================
def load_manifests():
    for p in (DATA_ROOT, FOLDS_PATH, GROUPS_PATH, TEMPLATE_PATH):
        ensure(p.exists(), f'Missing Kaggle Input: {p}')
    folds = pd.read_csv(FOLDS_PATH)
    groups = pd.read_csv(GROUPS_PATH)
    assert len(folds) == len(groups) == 28275, 'Wrong train audit version'
    assert folds.sample_id.is_unique and groups.sample_id.is_unique
    assert set(folds.sample_id) == set(groups.sample_id)
    assert folds.class_name.map(LABEL_MAP).eq(folds.label).all(), 'Label mapping mismatch'
    assert folds.groupby('group_id').fold.nunique().max() == 1, 'Group leakage'
    full = folds.merge(groups[['sample_id', 'group_id', 'sha256']], on='sample_id',
                       how='inner', suffixes=('', '_audit'), validate='one_to_one')
    assert full.group_id.eq(full.group_id_audit).all(), 'Group audit mismatch'
    assert full.groupby('sha256').fold.nunique().max() == 1, 'Exact duplicate leakage'
    orig = full.loc[full.fold != 0].copy()
    val = full.loc[full.fold == 0].sort_values('sample_id').reset_index(drop=True)
    conflicts = set(orig.groupby('sha256').label.nunique().pipe(lambda x: x[x > 1]).index)
    clean = orig.loc[~orig.sha256.isin(conflicts)].sort_values('sample_id').drop_duplicates('sha256')
    train = pd.concat((clean, orig.loc[orig.sha256.isin(conflicts)]), ignore_index=True)
    train = train.sort_values('sample_id').reset_index(drop=True)
    removed = orig.loc[~orig.sample_id.isin(train.sample_id)].copy()
    assert (len(orig), len(train), len(val), len(removed)) == (22619, 21712, 5656, 907), (
        'Wrong final_v3 fold or B data cleaning; refusing to train')
    assert set(train.sample_id).isdisjoint(set(val.sample_id))
    assert set(train.label) == set(val.label) == set(range(6))
    for name, frame in [('train_manifest', train), ('val_manifest', val),
                        ('removed_duplicate_manifest', removed)]:
        frame.to_csv(OUT / f'{name}.csv', index=False)
    print('Data OK | B-style train:', len(train), '| validation:', len(val),
          '| removed same-label copies:', len(removed), '| cross-label SHA groups preserved:', len(conflicts), flush=True)
    return train, val


def preload_images(sample_ids, test=False):
    """48x48 grayscale originals fit in RAM (~80 MB train+val+test)."""
    images = []
    for i, sid in enumerate(sample_ids):
        path = DATA_ROOT / ('test/' + str(sid) if test else str(sid))
        if not path.is_file():
            raise FileNotFoundError(path)
        with Image.open(path) as im:
            arr = np.array(im.convert('L'), dtype=np.uint8, copy=True)
        ensure(arr.shape == (48, 48), f'Unexpected image size: {path} {arr.shape}')
        images.append(arr)
        if (i + 1) % 15000 == 0:
            print('  Cached', i + 1, 'images', flush=True)
    return np.stack(images)


class FaceDataset(Dataset):
    def __init__(self, arrays, labels, transform):
        self.arrays = arrays
        self.labels = None if labels is None else np.asarray(labels, dtype=np.int64)
        self.transform = transform

    def __len__(self):
        return len(self.arrays)

    def __getitem__(self, idx):
        im = Image.fromarray(self.arrays[idx], mode='L').convert('RGB')
        x = self.transform(im)
        if self.labels is None:
            return x
        return x, int(self.labels[idx])


NORMALIZE = transforms.Normalize((.485, .456, .406), (.229, .224, .225))
VAL_TRANSFORM = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE), interpolation=InterpolationMode.BILINEAR),
    transforms.ToTensor(), NORMALIZE,
])


def make_train_transform(recipe):
    if recipe.aug == 'original':
        # Deliberately retain historical B transforms for the anchor.
        return transforms.Compose([
            transforms.Resize((192, 192), interpolation=InterpolationMode.BILINEAR),
            transforms.RandomHorizontalFlip(.5),
            transforms.RandomAffine(degrees=10, translate=(.05, .05), scale=(.95, 1.05)),
            transforms.ColorJitter(brightness=.12, contrast=.12),
            transforms.ToTensor(), NORMALIZE,
        ])
    return transforms.Compose([
        transforms.Resize((192, 192), interpolation=InterpolationMode.BICUBIC),
        transforms.RandomHorizontalFlip(.5),
        transforms.RandomAffine(degrees=8, translate=(.04, .04), scale=(.96, 1.04),
                                interpolation=InterpolationMode.BILINEAR, fill=(0, 0, 0)),
        transforms.ColorJitter(brightness=.10, contrast=.10),
        transforms.ToTensor(), NORMALIZE,
    ])


def worker_seed(worker_id):
    s = torch.initial_seed() % (2 ** 32)
    np.random.seed(s)
    random.seed(s)


def get_loader(ds, bs, shuffle, seed, num_workers=WORKERS):
    gen = torch.Generator().manual_seed(seed)
    return DataLoader(ds, batch_size=bs, shuffle=shuffle, num_workers=num_workers,
                      pin_memory=(DEVICE.type == 'cuda'), drop_last=False,
                      persistent_workers=(num_workers > 0),
                      generator=gen, worker_init_fn=worker_seed if num_workers > 0 else None)


# ============================================================
# Models, optimizer, EMA
# ============================================================
def build_model(recipe, pretrained):
    if recipe.arch == 'b2':
        weights = models.EfficientNet_B2_Weights.IMAGENET1K_V1 if pretrained else None
        model = models.efficientnet_b2(weights=weights)
        model.classifier[0].p = recipe.dropout
        model.classifier[1] = nn.Linear(model.classifier[1].in_features, 6)
    elif recipe.arch == 'convnext':
        weights = models.ConvNeXt_Tiny_Weights.IMAGENET1K_V1 if pretrained else None
        model = models.convnext_tiny(weights=weights)
        old = model.classifier
        model.classifier = nn.Sequential(old[0], old[1], nn.Dropout(recipe.dropout),
                                         nn.Linear(old[2].in_features, 6))
    else:
        raise ValueError(recipe.arch)
    return model


def build_optimizer(model, recipe):
    if recipe.name == 'b2_anchor':
        return torch.optim.AdamW(model.parameters(), lr=recipe.backbone_lr,
                                 weight_decay=recipe.weight_decay)
    # No weight decay on biases/norm scale, different learning rates for pretrained trunk vs head.
    buckets = {}
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        is_head = name.startswith('classifier.')
        lr = recipe.head_lr if is_head else recipe.backbone_lr
        wd = 0.0 if parameter.ndim == 1 else recipe.weight_decay
        buckets.setdefault((lr, wd), []).append(parameter)
    groups = [{'params': params, 'lr': lr, 'weight_decay': wd}
              for (lr, wd), params in buckets.items()]
    return torch.optim.AdamW(groups)


class EMA:
    def __init__(self, model, decay):
        self.model = copy.deepcopy(model).eval()
        for p in self.model.parameters():
            p.requires_grad_(False)
        self.decay = decay
        self.updates = 0

    @torch.no_grad()
    def update(self, source):
        self.updates += 1
        for target, current in zip(self.model.parameters(), source.parameters()):
            target.mul_(self.decay).add_(current.detach(), alpha=1. - self.decay)
        self.sync_buffers(source)

    @torch.no_grad()
    def sync_buffers(self, source):
        for target, current in zip(self.model.buffers(), source.buffers()):
            target.copy_(current)



def train_epoch(model, ema, train_loader, optimizer, scaler, criterion, recipe, epoch):
    model.train()
    loss_sum, correct, seen, skipped = 0.0, 0, 0, 0
    for step, (x, y) in enumerate(train_loader):
        x = x.to(DEVICE, non_blocking=True)
        y = y.to(DEVICE, non_blocking=True)
        mix = None
        if recipe.mixup_alpha > 0 and random.random() < recipe.mixup_prob and len(x) > 1:
            alpha = recipe.mixup_alpha
            lam = np.random.beta(alpha, alpha)
            idx = torch.randperm(len(x), device=DEVICE)
            x = x * lam + x[idx] * (1.0 - lam)
            mix = (y[idx], lam)
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type='cuda', dtype=torch.float16, enabled=DEVICE.type == 'cuda'):
            logits = model(x)
            if mix is None:
                loss = criterion(logits, y)
            else:
                other_y, lam = mix
                loss = lam * criterion(logits, y) + (1.0 - lam) * criterion(logits, other_y)
        if not torch.isfinite(loss).item():
            raise RuntimeError('Non-finite loss; stop and inspect')
        scale_before = scaler.get_scale()
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(optimizer)
        scaler.update()
        step_skipped = scaler.get_scale() < scale_before
        skipped += int(step_skipped)
        if ema is not None and not step_skipped and (step + 1) % 4 == 0:
            ema.update(model)
        bs = len(y)
        loss_sum += float(loss.detach()) * bs
        correct += int((logits.argmax(1) == y).sum()) if mix is None else 0
        seen += bs
    ensure(skipped < len(train_loader), 'All AMP optimizer updates were skipped')
    if ema is not None:
        ema.sync_buffers(model)
    return loss_sum / seen, correct / seen if recipe.mixup_alpha == 0 else float('nan'), skipped


@torch.inference_mode()
def predict_validation(model, loader, tta=False):
    model.eval()
    probs_all = []
    for x, _ in loader:
        x = x.to(DEVICE, non_blocking=True)
        with torch.autocast(device_type='cuda', dtype=torch.float16, enabled=DEVICE.type == 'cuda'):
            logits = model(x)
            flipped = model(x.flip(-1)) if tta else None
        p = logits.float().softmax(1)
        if tta:
            p = (p + flipped.float().softmax(1)) * 0.5
        probs_all.append(p.cpu().numpy())
    return np.concatenate(probs_all)


@torch.inference_mode()
def predict_test(model, loader, tta=False):
    model.eval()
    probs_all = []
    for x in loader:
        x = x.to(DEVICE, non_blocking=True)
        with torch.autocast(device_type='cuda', dtype=torch.float16, enabled=DEVICE.type == 'cuda'):
            logits = model(x)
            flipped = model(x.flip(-1)) if tta else None
        p = logits.float().softmax(1)
        if tta:
            p = (p + flipped.float().softmax(1)) * 0.5
        probs_all.append(p.cpu().numpy())
    return np.concatenate(probs_all)


def save_prediction_table(path, ids, labels, probs, test=False):
    col = 'ID' if test else 'sample_id'
    frame = pd.DataFrame({col: ids, 'prediction': probs.argmax(1)})
    if labels is not None:
        frame.insert(1, 'label', labels)
    for j, emotion in enumerate(CLASSES):
        frame['p_' + emotion] = probs[:, j]
    frame.to_csv(path, index=False)


def eval_details(model, loader, labels):
    probs = predict_validation(model, loader)
    nll = -np.log(np.clip(probs[np.arange(len(labels)), labels], 1e-12, 1)).mean()
    return probs, float(nll)


# ============================================================
# Train 3 independent models. Save best student OR EMA (by validation).
# ============================================================
def train_recipe(recipe, train_arrays, val_arrays, train_labels, val_labels, val_ids):
    seed_all(recipe.seed)
    run = OUT / recipe.name
    run.mkdir(parents=True, exist_ok=True)
    best_path = run / 'best.pt'
    save_json(asdict(recipe), run / 'recipe.json')

    train_ds = FaceDataset(train_arrays, train_labels, make_train_transform(recipe))
    val_ds = FaceDataset(val_arrays, val_labels, VAL_TRANSFORM)
    train_loader = get_loader(train_ds, recipe.batch_size, True, recipe.seed)
    val_loader = get_loader(val_ds, max(64, recipe.batch_size), False, recipe.seed + 11)

    print('\n' + '=' * 70, '\nSTART:', recipe.name, '|', recipe.arch,
          '| weight_decay:', recipe.weight_decay, '| mixup:', recipe.mixup_alpha,
          '\n' + '=' * 70, flush=True)
    try:
        model = build_model(recipe, pretrained=True).to(DEVICE)
    except Exception as exc:
        raise RuntimeError('Cannot load ImageNet pretrained weights; turn Kaggle Internet ON or cache the weights.') from exc
    optimizer = build_optimizer(model, recipe)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=recipe.epochs, eta_min=1e-6)
    scaler = torch.amp.GradScaler('cuda', enabled=(DEVICE.type == 'cuda'))
    criterion = nn.CrossEntropyLoss(label_smoothing=recipe.label_smoothing)
    ema = EMA(model, recipe.ema_decay) if recipe.ema_decay > 0 else None

    best_correct = -1
    best_nll = float('inf')
    best_epoch = 0
    bad_epochs = 0
    best_variant = None
    history = []
    started = time.time()
    for epoch in range(1, recipe.epochs + 1):
        t0 = time.time()
        current_lr = optimizer.param_groups[0]['lr']
        train_loss, online_accuracy, skipped = train_epoch(model, ema, train_loader,
                                                           optimizer, scaler, criterion, recipe, epoch)
        student_probs, student_nll = eval_details(model, val_loader, val_labels)
        student_correct = num_correct(student_probs, val_labels)
        selected_model = model
        epoch_variant = 'student'
        epoch_correct = student_correct
        epoch_nll = student_nll
        ema_correct = None
        ema_nll = None
        # EMA has meaningful history after several epochs.
        if ema is not None and epoch >= 3 and ema.updates > 0:
            ema_probs, ema_nll = eval_details(ema.model, val_loader, val_labels)
            ema_correct = num_correct(ema_probs, val_labels)
            if (ema_correct > epoch_correct or
                    (ema_correct == epoch_correct and ema_nll < epoch_nll)):
                selected_model = ema.model
                epoch_variant = 'ema'
                epoch_correct = ema_correct
                epoch_nll = ema_nll
        improved = epoch_correct > best_correct or (
            epoch_correct == best_correct and epoch_nll < best_nll - 1e-6)
        if improved:
            best_correct, best_nll = epoch_correct, epoch_nll
            best_epoch, best_variant = epoch, epoch_variant
            bad_epochs = 0
            atomic_torch_save({'model': cpu_state_dict(selected_model), 'recipe': asdict(recipe),
                               'epoch': epoch, 'variant': epoch_variant,
                               'val_correct': epoch_correct, 'val_accuracy': epoch_correct / len(val_labels)},
                              best_path)
        else:
            bad_epochs += 1
        scheduler.step()
        row = dict(epoch=epoch, train_loss=float(train_loss),
                   train_acc_online=float(online_accuracy),
                   val_student_acc=student_correct / len(val_labels),
                   val_student_nll=float(student_nll),
                   val_ema_acc=None if ema_correct is None else ema_correct / len(val_labels),
                   val_ema_nll=ema_nll,
                   best_val_acc=best_correct / len(val_labels),
                   learning_rate=current_lr, amp_skipped_steps=skipped,
                   elapsed_seconds=round(time.time() - t0, 1))
        history.append(row)
        pd.DataFrame(history).to_csv(run / 'history.csv', index=False)
        print(f"{recipe.name} epoch {epoch:02d}/{recipe.epochs} "
              f"train_loss={train_loss:.4f} student={student_correct/len(val_labels):.4f} "
              f"ema={'n/a' if ema_correct is None else f'{ema_correct/len(val_labels):.4f}'} "
              f"best={best_correct/len(val_labels):.4f} [{best_variant}] "
              f"lr={current_lr:.7f} AMPskip={skipped} time={time.time()-t0:.0f}s"
              + (' NEW BEST' if improved else ''), flush=True)
        if bad_epochs >= recipe.patience:
            print('Early stop:', recipe.name, 'epoch', epoch, flush=True)
            break

    # Release two heavyweight training states BEFORE inference/next model.
    del optimizer, scheduler, scaler, ema, model, train_loader
    gc.collect()
    if DEVICE.type == 'cuda':
        torch.cuda.empty_cache()

    checkpoint = torch.load(best_path, map_location='cpu', weights_only=False)
    best_model = build_model(recipe, pretrained=False)
    best_model.load_state_dict(checkpoint['model'], strict=True)
    best_model.to(DEVICE).eval()
    single_probs, single_nll = eval_details(best_model, val_loader, val_labels)
    ensure(num_correct(single_probs, val_labels) == best_correct,
           f'Best checkpoint accuracy mismatch in {recipe.name}; refusing submission')
    flip_probs = predict_validation(best_model, val_loader, tta=True)
    flip_correct = num_correct(flip_probs, val_labels)
    single_correct = num_correct(single_probs, val_labels)
    use_tta = flip_correct > single_correct
    chosen_probs = flip_probs if use_tta else single_probs
    print(recipe.name, 'best epoch:', best_epoch, 'variant:', best_variant,
          'single:', round(single_correct / len(val_labels), 5),
          'flip TTA:', round(flip_correct / len(val_labels), 5),
          'selected:', 'flip TTA' if use_tta else 'single', flush=True)

    for mode, p in [('single', single_probs), ('tta', flip_probs)]:
        save_prediction_table(run / f'val_{mode}_probabilities.csv', val_ids, val_labels, p)
    chosen_pred = chosen_probs.argmax(1)
    pd.DataFrame(classification_report(val_labels, chosen_pred, labels=range(6),
                                       target_names=CLASSES, output_dict=True,
                                       zero_division=0)).T.to_csv(run / 'classification_report.csv')
    pd.DataFrame(confusion_matrix(val_labels, chosen_pred, labels=range(6)),
                 index=CLASSES, columns=CLASSES).to_csv(run / 'confusion_matrix.csv')
    save_json({'best_epoch': best_epoch, 'best_variant': best_variant, 'use_tta': use_tta,
               'single_val_acc': single_correct / len(val_labels),
               'tta_val_acc': flip_correct / len(val_labels),
               'elapsed_minutes': round((time.time() - started) / 60, 2)}, run / 'summary.json')
    del val_loader
    gc.collect()
    return chosen_probs, best_model, use_tta


# ============================================================
# Ensemble weight choices: few prespecified candidates to limit overfitting.
# ============================================================
def ensemble_candidates(n):
    """Small predeclared probability-average set, works with 1-4 models."""
    import itertools
    if n < 2:
        return {}
    candidates = {}
    for i, j in itertools.combinations(range(n), 2):
        w = [0.0] * n
        w[i] = w[j] = 0.5
        candidates[f'pair_{i}_{j}'] = w
    if n >= 3:
        for indices in itertools.combinations(range(n), 3):
            w = [0.0] * n
            for i in indices:
                w[i] = 1.0 / 3
            candidates['triple_' + '_'.join(map(str, indices))] = w
    if n >= 4:
        candidates['equal_all'] = [1.0 / n] * n
    return candidates


def evaluate_existing_D(val_imgs, val_labels, val_ids, test_imgs, test_ids):
    if not EXTRA_D_CHECKPOINT.is_file():
        print('Optional historic D checkpoint not mounted; skipping it.', flush=True)
        return None
    print('Optional historic D found; evaluating without training:', EXTRA_D_CHECKPOINT, flush=True)
    ckpt = torch.load(EXTRA_D_CHECKPOINT, map_location='cpu', weights_only=False)
    dropout = float(ckpt.get('dropout', .5))
    recipe = Recipe('legacy_D', 'b2', 2026, 1, 1, 64, 0, 0, 0, dropout, 0, 0, 0, 'original')
    model = build_model(recipe, pretrained=False)
    model.load_state_dict(ckpt['model'], strict=True)
    model.to(DEVICE).eval()
    val_loader = get_loader(FaceDataset(val_imgs, val_labels, VAL_TRANSFORM), 64, False, 7020)
    single = predict_validation(model, val_loader, tta=False)
    tta = predict_validation(model, val_loader, tta=True)
    use_tta = num_correct(tta, val_labels) > num_correct(single, val_labels)
    selected_val = tta if use_tta else single
    test_loader = get_loader(FaceDataset(test_imgs, None, VAL_TRANSFORM), 64, False, 7021)
    selected_test = predict_test(model, test_loader, tta=use_tta)
    del model, val_loader, test_loader
    gc.collect()
    torch.cuda.empty_cache()
    run = OUT / 'legacy_D'
    run.mkdir(parents=True, exist_ok=True)
    save_prediction_table(run / 'val_probabilities.csv', val_ids, val_labels, selected_val)
    save_prediction_table(run / 'test_probabilities.csv', test_ids, None, selected_test, test=True)
    score = num_correct(selected_val, val_labels)
    print('Historic D validation:', score / len(val_labels), 'TTA:', use_tta, flush=True)
    return selected_val, selected_test, {'model': 'legacy_D', 'val_correct': score,
                                        'val_accuracy': score / len(val_labels), 'tta': use_tta}


def main():
    print('Torch:', torch.__version__, '| GPU:',
          torch.cuda.get_device_name(0) if DEVICE.type == 'cuda' else 'NOT ENABLED', flush=True)
    ensure(DEVICE.type == 'cuda', 'Please enable GPU in Kaggle settings before starting.')
    seed_all(2026)
    save_json({'created': datetime.now().isoformat(), 'torch': torch.__version__,
               'torchvision': __import__('torchvision').__version__,
               'recipes': [asdict(r) for r in RECIPES],
               'data_root': str(DATA_ROOT), 'folds': str(FOLDS_PATH)}, OUT / 'configuration.json')
    train, val = load_manifests()
    template = pd.read_csv(TEMPLATE_PATH, dtype={'ID': str})
    ensure(list(template.columns) == ['ID', 'Emotion'] and len(template) == 7065 and template.ID.is_unique,
           'Bad submission template')
    ensure(template.ID.notna().all(), 'Missing test ID')
    ensure(template.ID.map(lambda s: Path(s).name == s and '\\' not in s and s not in {'.', '..', ''}).all(),
           'Unsafe test IDs')

    print('Caching small grayscale images in RAM...', flush=True)
    train_imgs = preload_images(train.sample_id)
    val_imgs = preload_images(val.sample_id)
    test_imgs = preload_images(template.ID, test=True)
    train_labels = train.label.to_numpy(np.int64)
    val_labels = val.label.to_numpy(np.int64)
    all_val_probs = []
    all_test_probs = []
    report = []
    active_names = []
    for recipe in RECIPES:
        try:
            val_probs, best_model, uses_tta = train_recipe(recipe, train_imgs, val_imgs,
                                                           train_labels, val_labels, val.sample_id.to_numpy())
        except RuntimeError as exc:
            # Third model is optional: if the ImageNet weights cannot download,
            # finalize the already trained B2 models instead of failing the version.
            if recipe.arch == 'convnext' and 'Cannot load ImageNet pretrained weights' in str(exc):
                print('WARNING: Skipping ConvNeXt; unavailable ImageNet weights:', exc, flush=True)
                continue
            raise
        test_ds = FaceDataset(test_imgs, None, VAL_TRANSFORM)
        test_loader = get_loader(test_ds, max(64, recipe.batch_size), False, recipe.seed + 100)
        test_probs = predict_test(best_model, test_loader, tta=uses_tta)
        assert test_probs.shape == (7065, 6) and np.isfinite(test_probs).all()
        assert val_probs.shape == (5656, 6) and np.isfinite(val_probs).all()
        model_dir = OUT / recipe.name
        save_prediction_table(model_dir / 'test_probabilities.csv', template.ID, None, test_probs, test=True)
        submission = template.copy()
        submission['Emotion'] = test_probs.argmax(1)
        submission.to_csv(OUT / f'submission_{recipe.name}.csv', index=False)
        all_val_probs.append(val_probs)
        all_test_probs.append(test_probs)
        active_names.append(recipe.name)
        score = num_correct(val_probs, val_labels)
        report.append({'model': recipe.name, 'val_correct': score,
                       'val_accuracy': score / len(val_labels), 'tta': uses_tta})
        pd.DataFrame(report).to_csv(OUT / 'model_scores.csv', index=False)
        print('Finished', recipe.name, '| val:', round(score / len(val_labels), 5),
              '| submission saved', flush=True)
        del best_model, test_loader, test_ds
        gc.collect()
        torch.cuda.empty_cache()

    # Existing D from your prior version is a free additional candidate.
    extra = evaluate_existing_D(val_imgs, val_labels, val.sample_id.to_numpy(),
                                test_imgs, template.ID.to_numpy())
    if extra is not None:
        val_p, test_p, extra_report = extra
        all_val_probs.append(val_p)
        all_test_probs.append(test_p)
        report.append(extra_report)
        active_names.append('legacy_D')
        individual_submission = template.copy()
        individual_submission['Emotion'] = test_p.argmax(1)
        individual_submission.to_csv(OUT / 'submission_legacy_D.csv', index=False)

    scores = [r['val_correct'] for r in report]
    solo_idx = int(np.argmax(scores))
    best_solo_name = active_names[solo_idx]
    best_solo_val = all_val_probs[solo_idx]
    best_solo_test = all_test_probs[solo_idx]
    best_solo_correct = scores[solo_idx]

    combos = ensemble_candidates(len(active_names))
    ensemble_scores = []
    for name, weights in combos.items():
        probs = sum(w * p for w, p in zip(weights, all_val_probs))
        hits = num_correct(probs, val_labels)
        ensemble_scores.append({'name': name, 'val_correct': hits,
                                'val_accuracy': hits / len(val_labels), 'weights': weights})
    ensemble_scores.sort(key=lambda x: (-x['val_correct'], x['name']))
    pd.DataFrame(ensemble_scores).to_csv(OUT / 'ensemble_scores.csv', index=False)
    if ensemble_scores:
        best_ens = ensemble_scores[0]
        ens_weights = best_ens['weights']
        ens_val = sum(w * p for w, p in zip(ens_weights, all_val_probs))
        ens_test = sum(w * p for w, p in zip(ens_weights, all_test_probs))
    else:
        best_ens = {'name': best_solo_name, 'val_correct': best_solo_correct,
                    'val_accuracy': best_solo_correct / len(val_labels), 'weights': [1.0]}
        ens_weights = [1.0]
        ens_val, ens_test = best_solo_val, best_solo_test

    # Guard against choosing an ensemble due to a negligible (~one-picture) val gain.
    selected_ensemble = bool(ensemble_scores) and best_ens['val_correct'] >= best_solo_correct + ENSEMBLE_MIN_NET_CORRECT
    chosen_val = ens_val if selected_ensemble else best_solo_val
    chosen_test = ens_test if selected_ensemble else best_solo_test
    chosen_name = best_ens['name'] if selected_ensemble else best_solo_name

    def submission_file(path, probs):
        result = template.copy()
        result['Emotion'] = probs.argmax(1).astype(int)
        assert len(result) == 7065 and result['Emotion'].between(0, 5).all()
        result.to_csv(path, index=False)

    submission_file(OUT / 'submission_best_single.csv', best_solo_test)
    submission_file(OUT / 'submission_best_ensemble.csv', ens_test)
    submission_file(OUT / 'submission_SELECTED.csv', chosen_test)
    submission_file('/kaggle/working/submission_best.csv', chosen_test)

    save_prediction_table(OUT / 'selected_validation_probabilities.csv',
                          val.sample_id, val_labels, chosen_val)
    save_prediction_table(OUT / 'selected_test_probabilities.csv',
                          template.ID, None, chosen_test, test=True)
    y_hat = chosen_val.argmax(1)
    pd.DataFrame(classification_report(val_labels, y_hat, labels=range(6),
                                       target_names=CLASSES, output_dict=True,
                                       zero_division=0)).T.to_csv(OUT / 'selected_classification_report.csv')
    pd.DataFrame(confusion_matrix(val_labels, y_hat, labels=range(6)),
                 index=CLASSES, columns=CLASSES).to_csv(OUT / 'selected_confusion_matrix.csv')
    summary = {
        'individual': report,
        'best_single': {'name': best_solo_name, 'accuracy': best_solo_correct / len(val_labels)},
        'best_ensemble': best_ens,
        'selected_name': chosen_name,
        'selected_val_accuracy': num_correct(chosen_val, val_labels) / len(val_labels),
        'selected_is_ensemble': selected_ensemble,
        'historical_B_val_reference': 0.6715,
        'historical_B_kaggle_reference': 0.682,
        'submission': '/kaggle/working/submission_best.csv',
        'note': 'Validation-selected candidates only; no test labels used, no Kaggle submission performed.'
    }
    save_json(summary, OUT / 'FINAL_SUMMARY.json')
    print('\n' + '=' * 65)
    print('ALL MODELS DONE | Results', flush=True)
    for item in report:
        print(f"{item['model']}: {item['val_accuracy']:.5f}, TTA={item['tta']}")
    print('Best ensemble candidate:', best_ens['name'], round(best_ens['val_accuracy'], 6), ens_weights)
    print('SELECTED:', chosen_name, '| Val accuracy:', summary['selected_val_accuracy'])
    print('Kaggle submission:', summary['submission'])
    print('Checkpoints:', OUT / '<model_name>' / 'best.pt')
    print('No automatic Kaggle submission.', flush=True)

    # Download-friendly ZIP: exclude optimizer states; include all best models.
    zip_path = Path('/kaggle/working/fer_strong_ensemble_results.zip')
    with zipfile.ZipFile(zip_path, 'w', compression=zipfile.ZIP_DEFLATED,
                         compresslevel=1, allowZip64=True) as z:
        for p in sorted(OUT.rglob('*')):
            if p.is_file() and p.suffix != '.tmp':
                z.write(p, arcname=str(p.relative_to(OUT.parent)))
    print('Result backup:', zip_path, '| size MB:',
          round(zip_path.stat().st_size / 1024**2, 1), flush=True)


if __name__ == '__main__':
    main()
