"""2026task2: audited B data + EfficientNet-B2, classifier dropout 0.5.

Kaggle: paste this entire file into ONE notebook cell, then Run.
GPU + Internet (for ImageNet weights on first run) are required.
No packages are installed; no Kaggle submission is sent automatically.

CLI examples:
  python fer_final.py
  python fer_final.py --dropout 0.3                 # matched B control
  python fer_final.py --resume /path/to/last.pt     # trusted own checkpoint
  python fer_final.py --tta                       # validate flip TTA first

Changes from historical D: stronger audit, diagnostics, separate loader RNGs,
epoch-seeded sampling/augmentation, nonpersistent workers for epoch resume.
These change random trajectories: compare 0.3/0.5 with THIS same script for
a strict matched control. Historical B=0.6715 is a reference, not a paired run.
Resume is at completed epoch boundaries, not in the middle of an epoch.
"""
import argparse
import hashlib
import inspect
import json
import math
import random
import sys
import time
import uuid
from datetime import datetime, timezone
from contextlib import nullcontext
from pathlib import Path, PurePosixPath

import numpy as np
import pandas as pd


LABEL_MAP = {'Angry': 0, 'Fear': 1, 'Happy': 2, 'Sad': 3, 'Surprise': 4, 'Neutral': 5}
CLASS_NAMES = list(LABEL_MAP)
EXPECTED_VAL_SIGNATURE = '5888dfdfe020603c697e634748d6979cab8ccb96888833cbbe9d35e2db32df64'


def require(condition, message):
    if not condition:
        raise ValueError(message)


def manifest_signature(frame):
    rows = sorted((str(r.sample_id), int(r.label)) for r in frame.itertuples())
    return hashlib.sha256(json.dumps(rows, ensure_ascii=False, separators=(',', ':')).encode()).hexdigest()


def file_sha256(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def build_manifests(folds, groups, expected_val_signature=None):
    require({'sample_id', 'class_name', 'label', 'group_id', 'fold'} <= set(folds), 'Missing fold columns')
    require({'sample_id', 'group_id', 'sha256'} <= set(groups), 'Missing group columns')
    require(folds.sample_id.is_unique and groups.sample_id.is_unique, 'Duplicate sample IDs')
    require(set(folds.sample_id) == set(groups.sample_id), 'Fold/group membership mismatch')
    require(folds.class_name.map(LABEL_MAP).eq(folds.label).all(), 'Label map mismatch')
    require(folds.fold.isin(range(5)).all(), 'Invalid fold number')
    require(folds.group_id.notna().all() and groups.sha256.notna().all(), 'Missing grouping metadata')
    for row in folds.itertuples():
        p = PurePosixPath(str(row.sample_id))
        require(not p.is_absolute() and len(p.parts) == 3 and p.parts[0] == 'train'
                and p.parts[1] == row.class_name and '..' not in p.parts
                and '\\' not in str(p), 'Unsafe/mismatched training path: ' + str(p))
    require(folds.groupby('group_id').fold.nunique().max() == 1, 'Known group crosses folds')
    merged = folds.merge(groups[['sample_id', 'group_id', 'sha256']], on='sample_id',
                         validate='one_to_one', suffixes=('', '_audit'))
    require(merged.group_id.eq(merged.group_id_audit).all(), 'Group IDs differ between files')
    require(merged.groupby('sha256').fold.nunique().max() == 1, 'Identical image crosses folds')
    original = merged.loc[merged.fold != 0].copy()
    val = merged.loc[merged.fold == 0].sort_values('sample_id').reset_index(drop=True)
    require(len(original) > 0 and len(val) > 0, 'Empty training or validation split')
    if expected_val_signature is not None:
        require(manifest_signature(val) == expected_val_signature, 'Wrong Fold 0 membership/labels; stop instead of repartitioning')
    counts = original.groupby('sha256').label.nunique()
    conflict = set(counts[counts > 1].index)
    nonconflict = original.loc[~original.sha256.isin(conflict)].sort_values('sample_id')
    train = pd.concat([nonconflict.drop_duplicates('sha256'),
                       original.loc[original.sha256.isin(conflict)]], ignore_index=True)
    train = train.sort_values('sample_id').reset_index(drop=True)
    removed = original.loc[~original.sample_id.isin(train.sample_id)].copy()
    require(not removed.sha256.isin(conflict).any(), 'Conflict records removed')
    require(set(train.sample_id).isdisjoint(val.sample_id), 'Train/validation overlap')
    return train, val, removed


def prediction_frame(frame, probs):
    probs = np.asarray(probs)
    require(probs.shape == (len(frame), 6), 'Prediction shape mismatch')
    require(np.isfinite(probs).all() and (probs >= 0).all()
            and np.allclose(probs.sum(axis=1), 1, atol=1e-5), 'Invalid probabilities')
    columns = [c for c in ['sample_id', 'group_id', 'label', 'ID'] if c in frame]
    result = frame[columns].reset_index(drop=True).copy()
    result['prediction'] = probs.argmax(axis=1)
    for index, name in enumerate(CLASS_NAMES):
        result['p_' + name] = probs[:, index]
    return result


def new_run_dir(root, name):
    path = Path(root) / (name + '_' + datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ') + '_' + uuid.uuid4().hex[:8])
    path.mkdir(parents=True, exist_ok=False)
    return path


def write_json(path, obj):
    Path(path).write_text(json.dumps(obj, indent=2, ensure_ascii=False, allow_nan=False), encoding='utf-8')


def atomic_save(torch, payload, path):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + '.tmp')
    torch.save(payload, temporary)
    temporary.replace(path)


def trusted_load(torch, path):
    # Full resume states contain NumPy/Python objects. Only load your own file.
    kwargs = {'map_location': 'cpu'}
    if 'weights_only' in inspect.signature(torch.load).parameters:
        kwargs['weights_only'] = False
    return torch.load(path, **kwargs)


def seed_all(torch, seed):
    random.seed(seed)
    np.random.seed(seed % (2**32))
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def seed_worker(worker_id):
    import torch
    seed = torch.initial_seed() % (2**32)
    random.seed(seed)
    np.random.seed(seed)


class Images:
    """Map-style dataset; training IDs already include train/class/filename."""
    def __init__(self, root, frame, transform, test=False):
        self.root = Path(root)
        self.transform = transform
        self.test = test
        self.ids = frame['ID' if test else 'sample_id'].astype(str).tolist()
        self.labels = None if test else frame.label.astype(int).tolist()

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, index):
        from PIL import Image
        path = self.root / 'test' / self.ids[index] if self.test else self.root / self.ids[index]
        with Image.open(path) as img:
            image = self.transform(img.convert('RGB'))
        return image if self.test else (image, self.labels[index])


def evaluate(model, loader, device, smoothing, tta=False):
    import torch
    import torch.nn.functional as F
    model.eval()
    smooth_sum, nll_sum, total = 0., 0., 0
    predictions = []
    with torch.inference_mode():
        for images, labels in loader:
            images, labels = images.to(device, non_blocking=True), labels.to(device, non_blocking=True)
            amp = torch.autocast(device_type='cuda', dtype=torch.float16) if device.type == 'cuda' else nullcontext()
            with amp:
                logits = model(images)
                other = model(images.flip(-1)) if tta else None
            log_probs = F.log_softmax(logits.float(), dim=1)
            if tta:
                log_probs = torch.logaddexp(log_probs, F.log_softmax(other.float(), dim=1)) - math.log(2)
            nll = -log_probs.gather(1, labels[:, None]).squeeze(1)
            smooth = (1 - smoothing) * nll - smoothing * log_probs.mean(dim=1)
            smooth_sum += smooth.sum().item()
            nll_sum += nll.sum().item()
            total += len(labels)
            predictions.append(log_probs.exp().cpu().numpy())
    return smooth_sum / total, nll_sum / total, np.concatenate(predictions)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-root', default='/kaggle/input/competitions/2026task2/fer_data/fer_data')
    parser.add_argument('--folds', default='/kaggle/input/datasets/zijin233/2026task2-folds-v3/2026task2_folds.csv')
    parser.add_argument('--groups', default='/kaggle/input/datasets/zijin233/2026task2-folds-v3/2026task2_groups.csv')
    parser.add_argument('--template', default='/kaggle/input/competitions/2026task2/submission.csv')
    parser.add_argument('--output-root', default='/kaggle/working/fer_final_runs')
    parser.add_argument('--dropout', type=float, choices=[0.3, 0.5], default=0.5)
    parser.add_argument('--seed', type=int, default=2026)
    parser.add_argument('--workers', type=int, default=2)
    parser.add_argument('--resume', default=None, help='Trusted last.pt generated by this script')
    parser.add_argument('--tta', action='store_true', help='Compare original/flip averaging on validation; use only if better')
    parser.add_argument('--audit-only', action='store_true', help='Read-only CPU data checks; no output files or model imports')
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    require(args.workers >= 0 and 0 <= args.seed < 2**32 - 1000, 'Invalid worker count or seed')
    folds, groups = pd.read_csv(args.folds), pd.read_csv(args.groups)
    require(len(folds) == len(groups) == 28275, 'Expected 28275 original training records')
    train, val, removed = build_manifests(folds, groups, EXPECTED_VAL_SIGNATURE)
    require((len(train), len(val), len(removed)) == (21712, 5656, 907), 'Wrong final_v3/B sample counts')
    require(set(train.label) == set(val.label) == set(range(6)), 'Missing class')
    print('Checking original TRAINING image hashes against audit (no test images used)...', flush=True)
    hashes = groups.set_index('sample_id').sha256
    for sid in folds.sample_id:
        path = Path(args.data_root) / sid
        require(path.is_file() and file_sha256(path) == hashes.loc[sid], 'Missing/changed training image: ' + sid)
    audit = {'original': len(folds) - len(val), 'train': len(train), 'val': len(val),
             'removed': len(removed), 'train_signature': manifest_signature(train),
             'val_signature': manifest_signature(val),
             'folds_sha256': file_sha256(args.folds), 'groups_sha256': file_sha256(args.groups),
             'unconfirmed_near_duplicates': 'Risk remains; no identity-disjoint claim'}
    print(json.dumps(audit, indent=2), flush=True)
    if args.audit_only:
        return

    # Lazy imports let audit-only and data-contract tests run without torchvision.
    import torch
    import torchvision
    from torch import nn
    from torch.utils.data import DataLoader
    from torchvision import models, transforms
    from torchvision.transforms import InterpolationMode
    from sklearn.metrics import classification_report, confusion_matrix
    require(torch.cuda.is_available(), 'Enable Kaggle GPU; CPU training is disabled')
    device = torch.device('cuda')
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    seed_all(torch, args.seed)
    config = {'recipe': 'D_final_v1', 'model': 'efficientnet_b2', 'weights': 'IMAGENET1K_V1',
              'dropout': args.dropout, 'seed': args.seed, 'image_size': 192, 'batch_size': 64,
              'epochs': 25, 'patience': 7, 'lr': 2e-4, 'min_lr': 1e-6, 'weight_decay': 1e-4,
              'label_smoothing': .05, 'clip_norm': 1., 'workers': args.workers,
              'persistent_workers': False, 'affine_interpolation': 'NEAREST',
              'label_map': LABEL_MAP, 'train_eval_size': 1024}
    environment = {'torch': str(torch.__version__), 'torchvision': torchvision.__version__,
                   'numpy': np.__version__, 'pandas': pd.__version__,
                   'python': sys.version, 'cuda': torch.version.cuda,
                   'cudnn': torch.backends.cudnn.version(), 'gpu': torch.cuda.get_device_name(0)}
    resume = trusted_load(torch, args.resume) if args.resume else None
    if resume:
        require(resume['config'] == config, 'Resume config differs; use original dropout/seed/workers')
        require(resume['audit'] == audit, 'Resume data differs')
        require(resume['environment'] == environment, 'Resume environment differs; exact continuation unavailable')
    run = new_run_dir(args.output_root, 'B_control' if args.dropout == .3 else 'D_final')
    print('Output:', run, '\nEnvironment:', environment, flush=True)
    write_json(run / 'config.json', {'training': config, 'environment': environment, 'arguments': vars(args), 'audit': audit})
    for name, frame in [('train_manifest', train), ('val_manifest', val), ('removed_exact_duplicates', removed)]:
        frame.to_csv(run / (name + '.csv'), index=False)
    try:
        source = Path(__file__)
        if source.is_file():
            (run / 'source.py').write_bytes(source.read_bytes())
    except NameError:
        pass  # In a pasted notebook cell the notebook is the source artifact.

    normalize = transforms.Normalize((.485, .456, .406), (.229, .224, .225))
    # Deliberately preserve D's augmentation. Test bilinear affine separately.
    train_transform = transforms.Compose([
        transforms.Resize((192, 192), interpolation=InterpolationMode.BILINEAR, antialias=True),
        transforms.RandomHorizontalFlip(.5),
        transforms.RandomAffine(10, translate=(.05, .05), scale=(.95, 1.05),
                                interpolation=InterpolationMode.NEAREST, fill=0),
        transforms.ColorJitter(brightness=.12, contrast=.12), transforms.ToTensor(), normalize])
    val_transform = transforms.Compose([
        transforms.Resize((192, 192), interpolation=InterpolationMode.BILINEAR, antialias=True),
        transforms.ToTensor(), normalize])
    train_generator = torch.Generator()
    def loader(frame, transform, shuffle=False, test=False, generator=None):
        if generator is None:
            generator = torch.Generator().manual_seed(args.seed + 700)
        return DataLoader(Images(args.data_root, frame, transform, test), batch_size=64,
                          shuffle=shuffle, num_workers=args.workers, pin_memory=True,
                          drop_last=False, persistent_workers=False,
                          worker_init_fn=seed_worker, generator=generator)
    train_loader = loader(train, train_transform, True, generator=train_generator)
    val_loader = loader(val, val_transform)
    diagnostic = train.sample(n=min(config['train_eval_size'], len(train)), random_state=args.seed).sort_values('sample_id')
    diagnostic.to_csv(run / 'train_diagnostic_manifest.csv', index=False)
    diagnostic_loader = loader(diagnostic, val_transform)

    model = models.efficientnet_b2(weights=None if resume else models.EfficientNet_B2_Weights.IMAGENET1K_V1)
    require(isinstance(model.classifier[0], nn.Dropout), 'Unexpected B2 classifier')
    model.classifier[0].p = args.dropout
    model.classifier[1] = nn.Linear(model.classifier[1].in_features, 6)
    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config['lr'], weight_decay=config['weight_decay'])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=25, eta_min=config['min_lr'])
    scaler = torch.amp.GradScaler('cuda') if hasattr(torch, 'amp') and hasattr(torch.amp, 'GradScaler') else torch.cuda.amp.GradScaler()
    criterion = nn.CrossEntropyLoss(label_smoothing=.05)
    history, best_state, best_epoch, best_correct, bad_epochs, start_epoch = [], None, 0, -1, 0, 1
    if resume:
        model.load_state_dict(resume['model'])
        optimizer.load_state_dict(resume['optimizer'])
        scheduler.load_state_dict(resume['scheduler'])
        scaler.load_state_dict(resume['scaler'])
        history, best_state = resume['history'], resume['best_model']
        best_epoch, best_correct, bad_epochs = resume['best_epoch'], resume['best_correct'], resume['bad_epochs']
        start_epoch = resume['epoch'] + 1
        atomic_save(torch, {'model': best_state, 'epoch': best_epoch, 'correct': best_correct,
                            'config': config, 'audit': audit}, run / 'best.pt')
        del resume
    for epoch in range(start_epoch, 26):
        if bad_epochs >= config['patience']:
            break
        # Epoch boundary reseeding + independent worker generator makes resume
        # independent of lost persistent-worker RNG states.
        seed_all(torch, args.seed + epoch)
        train_generator.manual_seed(args.seed + 100 * epoch)
        model.train()
        start = time.time()
        lr = optimizer.param_groups[0]['lr']
        total_loss, correct, count, skipped_steps = 0., 0, 0, 0
        for images, labels in train_loader:
            images, labels = images.to(device, non_blocking=True), labels.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type='cuda', dtype=torch.float16):
                logits = model(images)
                loss = criterion(logits, labels)
            require(bool(torch.isfinite(loss).item()), 'Nonfinite loss; stop and inspect before submitting')
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
            previous_scale = scaler.get_scale()
            scaler.step(optimizer)
            scaler.update()
            skipped_steps += int(scaler.get_scale() < previous_scale)
            total_loss += loss.item() * len(labels)
            correct += (logits.argmax(1) == labels).sum().item()
            count += len(labels)
        require(skipped_steps < len(train_loader), 'Every optimizer update was skipped')
        val_loss, val_nll, probs = evaluate(model, val_loader, device, .05)
        val_correct = int((probs.argmax(1) == val.label.to_numpy()).sum())
        _, diagnostic_nll, diagnostic_probs = evaluate(model, diagnostic_loader, device, .05)
        diagnostic_acc = float((diagnostic_probs.argmax(1) == diagnostic.label.to_numpy()).mean())
        # Strict accuracy improvement; keep earlier epoch on exact ties, as in D.
        if val_correct > best_correct:
            best_correct, best_epoch, bad_epochs = val_correct, epoch, 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            atomic_save(torch, {'model': best_state, 'epoch': epoch, 'correct': best_correct,
                                'config': config, 'audit': audit}, run / 'best.pt')
            prediction_frame(val, probs).to_csv(run / 'best_val_probabilities.csv', index=False)
        else:
            bad_epochs += 1
        scheduler.step()
        row = {'epoch': epoch, 'train_loss_smoothed': total_loss / count, 'train_acc_online': correct / count,
               'train_subset_acc_eval': diagnostic_acc, 'train_subset_nll': diagnostic_nll,
               'val_loss_smoothed': val_loss, 'val_nll': val_nll, 'val_acc': val_correct / len(val),
               'val_correct': val_correct, 'lr': lr, 'amp_skipped_steps': skipped_steps,
               'seconds': time.time() - start}
        history.append(row)
        pd.DataFrame(history).to_csv(run / 'history.csv', index=False)
        atomic_save(torch, {'model': model.state_dict(), 'optimizer': optimizer.state_dict(),
                            'scheduler': scheduler.state_dict(), 'scaler': scaler.state_dict(),
                            'epoch': epoch, 'history': history, 'best_model': best_state,
                            'best_epoch': best_epoch, 'best_correct': best_correct, 'bad_epochs': bad_epochs,
                            'config': config, 'audit': audit, 'environment': environment}, run / 'last.pt')
        print(f"Epoch {epoch:02d} | train={row['train_acc_online']:.4f} | train_eval={diagnostic_acc:.4f} "
              f"| val={row['val_acc']:.4f} | NLL={val_nll:.4f} | best={best_correct/len(val):.4f} "
              f"| LR={lr:.7f} | {row['seconds']:.0f}s | AMP skipped={skipped_steps}", flush=True)

    require(best_state is not None, 'No best checkpoint available')
    # Verify the file actually reloads, not just the in-memory snapshot.
    best = trusted_load(torch, run / 'best.pt')
    model.load_state_dict(best['model'])
    val_loss, val_nll, probs = evaluate(model, val_loader, device, .05)
    require(int((probs.argmax(1) == val.label.to_numpy()).sum()) == best_correct,
            'Reloaded best checkpoint gives different validation predictions/accuracy')
    prediction_frame(val, probs).to_csv(run / 'best_val_probabilities.csv', index=False)
    chosen_probs, use_tta, tta_accuracy = probs, False, None
    if args.tta:
        _, _, tta_probs = evaluate(model, val_loader, device, .05, tta=True)
        tta_correct = int((tta_probs.argmax(1) == val.label.to_numpy()).sum())
        tta_accuracy = tta_correct / len(val)
        prediction_frame(val, tta_probs).to_csv(run / 'val_flip_tta_probabilities.csv', index=False)
        if tta_correct > best_correct:
            chosen_probs, use_tta = tta_probs, True
    for name, probabilities in [('single', probs), ('selected', chosen_probs)]:
        pred = probabilities.argmax(1)
        report = classification_report(val.label, pred, labels=range(6), target_names=CLASS_NAMES,
                                       output_dict=True, zero_division=0)
        pd.DataFrame(report).T.to_csv(run / ('classification_report_' + name + '.csv'))
        pd.DataFrame(confusion_matrix(val.label, pred, labels=range(6)), index=CLASS_NAMES,
                     columns=CLASS_NAMES).to_csv(run / ('confusion_matrix_' + name + '.csv'))
    print(classification_report(val.label, chosen_probs.argmax(1), labels=range(6),
                                target_names=CLASS_NAMES, digits=4, zero_division=0), flush=True)

    submission = pd.read_csv(args.template, dtype={'ID': str})
    require(list(submission.columns) == ['ID', 'Emotion'] and len(submission) == 7065
            and submission.ID.notna().all() and submission.ID.is_unique, 'Invalid submission template')
    for filename in submission.ID:
        require(PurePosixPath(filename).name == filename and '\\' not in filename
                and filename not in {'', '.', '..'} and (Path(args.data_root) / 'test' / filename).is_file(),
                'Invalid/missing test filename: ' + filename)
    test_loader = loader(submission, val_transform, test=True)
    test_probs = []
    model.eval()
    with torch.inference_mode():
        for images in test_loader:
            images = images.to(device, non_blocking=True)
            with torch.autocast(device_type='cuda', dtype=torch.float16):
                logits = model(images)
                other = model(images.flip(-1)) if use_tta else None
            probabilities = logits.float().softmax(1)
            if use_tta:
                probabilities = (probabilities + other.float().softmax(1)) / 2
            test_probs.append(probabilities.cpu().numpy())
    test_probs = np.concatenate(test_probs)
    prediction_frame(submission, test_probs).to_csv(run / 'test_probabilities.csv', index=False)
    submission['Emotion'] = test_probs.argmax(1)
    submission.to_csv(run / 'submission.csv', index=False)
    pd.DataFrame(history).to_csv(run / 'history.csv', index=False)
    summary = {'best_epoch': best_epoch, 'single_val_accuracy': best_correct / len(val),
               'single_val_nll': val_nll, 'single_val_smoothed_loss': val_loss,
               'flip_tta_val_accuracy': tta_accuracy, 'submission_uses_flip_tta': use_tta,
               'historical_B_reference': .6715, 'historical_B_is_not_a_matched_control': True,
               'submission': str(run / 'submission.csv'), 'run_directory': str(run)}
    write_json(run / 'summary.json', summary)
    print(json.dumps(summary, indent=2), '\nNo automatic Kaggle submission performed.', flush=True)


if __name__ == '__main__':
    # Notebook kernels supply their own -f argument; use defaults when pasted.
    main([] if 'ipykernel' in sys.modules else None)
