"""Model-agnostic training loop shared by all four models.

- AdamW with linear warm-up then cosine decay (stepped per batch).
- Optional discriminative learning rates: parameters under `model.backbone` get
  `lr * backbone_lr_mult`, everything else (heads, physics decoder, ...) gets `lr`.
- Optional backbone freezing for the first `freeze_epochs` (head-only warm-up), the
  standard two-stage recipe for fine-tuning pretrained networks.
- Best epoch chosen on validation macro AUC; the test set is only touched once, at the end.
"""

import copy
import math
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from .metrics import classification_metrics, logits_of, predict
from .utils import count_params, save_json


@dataclass
class TrainConfig:
    epochs: int = 25
    lr: float = 1e-3
    backbone_lr_mult: float = 1.0
    weight_decay: float = 1e-4
    warmup_epochs: float = 1.0
    freeze_epochs: int = 0
    label_smoothing: float = 0.0
    grad_clip: float = 1.0
    patience: int | None = None          # early stopping on val macro AUC
    log_every: int = 0                   # batches; 0 = only per epoch
    extra: dict = field(default_factory=dict)


class CELoss(nn.Module):
    def __init__(self, label_smoothing=0.0):
        super().__init__()
        self.ce = nn.CrossEntropyLoss(label_smoothing=label_smoothing)

    def forward(self, out, y):
        loss = self.ce(logits_of(out), y)
        return loss, {}


def _backbone_params(model):
    """Parameters of `model.backbone`, excluding `model.head` even if it lives inside it
    (timm models keep their classifier within the backbone)."""
    backbone = getattr(model, "backbone", None)
    if backbone is None:
        return []
    head = getattr(model, "head", None)
    head_ids = {id(p) for p in head.parameters()} if head is not None else set()
    return [p for p in backbone.parameters() if id(p) not in head_ids]


def _param_groups(model, cfg: TrainConfig):
    bb_ids = {id(p) for p in _backbone_params(model)}
    groups = [
        {"params": [p for p in model.parameters() if id(p) in bb_ids], "lr": cfg.lr * cfg.backbone_lr_mult},
        {"params": [p for p in model.parameters() if id(p) not in bb_ids], "lr": cfg.lr},
    ]
    return [g for g in groups if g["params"]]


def _set_backbone_trainable(model, flag: bool):
    for p in _backbone_params(model):
        p.requires_grad_(flag)


def _run_epoch(model, loader, criterion, device, optimizer=None, scheduler=None, cfg=None):
    train = optimizer is not None
    model.train(train)
    totals, correct, seen = {}, 0, 0
    t0 = time.time()
    with torch.set_grad_enabled(train):
        for step, (x, y) in enumerate(loader, 1):
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            out = model(x)
            loss, terms = criterion(out, y)
            if train:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                if cfg.grad_clip:
                    nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
                optimizer.step()
                scheduler.step()
            bs = y.size(0)
            for k, v in {"loss": loss.item(), **terms}.items():
                totals[k] = totals.get(k, 0.0) + v * bs
            correct += (logits_of(out).argmax(1) == y).sum().item()
            seen += bs
            if train and cfg.log_every and step % cfg.log_every == 0:
                print(f"    step {step}/{len(loader)}  loss {totals['loss'] / seen:.4f}  "
                      f"{(time.time() - t0) / step:.2f}s/it", flush=True)
    stats = {k: v / seen for k, v in totals.items()}
    stats["acc"] = correct / seen
    return stats


def fit(model, loaders, cfg: TrainConfig, device, criterion=None, out_dir=None, run_name="run"):
    """Train, keep the best-val-AUC weights, evaluate on test. Returns (model, results)."""
    model.to(device)
    criterion = criterion or CELoss(cfg.label_smoothing)
    optimizer = torch.optim.AdamW(_param_groups(model, cfg), weight_decay=cfg.weight_decay)

    steps_per_epoch = len(loaders["train"])
    total = cfg.epochs * steps_per_epoch
    warm = max(1, int(cfg.warmup_epochs * steps_per_epoch))
    lr_lambda = lambda s: (s + 1) / warm if s < warm else 0.5 * (1 + math.cos(math.pi * (s - warm) / max(1, total - warm)))
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    history, best_auc, best_state, best_epoch, stale = {}, -1.0, None, 0, 0
    print(f"[{run_name}] {count_params(model):,} trainable params | device={device} | "
          f"train={len(loaders['train'].dataset)} val={len(loaders['val'].dataset)} test={len(loaders['test'].dataset)}")

    for epoch in range(1, cfg.epochs + 1):
        _set_backbone_trainable(model, epoch > cfg.freeze_epochs)
        t0 = time.time()
        tr = _run_epoch(model, loaders["train"], criterion, device, optimizer, scheduler, cfg)
        va = _run_epoch(model, loaders["val"], criterion, device)
        va_auc = classification_metrics(*predict(model, loaders["val"], device))["macro_auc"]

        for k, v in tr.items():
            history.setdefault(f"train_{k}", []).append(v)
        for k, v in va.items():
            history.setdefault(f"val_{k}", []).append(v)
        history.setdefault("val_macro_auc", []).append(va_auc)

        improved = va_auc > best_auc
        if improved:
            best_auc, best_epoch, stale = va_auc, epoch, 0
            best_state = copy.deepcopy({k: v.detach().cpu() for k, v in model.state_dict().items()})
        else:
            stale += 1
        extra = "  ".join(f"{k} {v:.4g}" for k, v in tr.items() if k not in ("loss", "acc"))
        print(f"  ep {epoch:>3}/{cfg.epochs}  train loss {tr['loss']:.4f} acc {tr['acc']:.4f}  "
              f"val loss {va['loss']:.4f} acc {va['acc']:.4f} AUC {va_auc:.4f}  {extra}  "
              f"[{time.time() - t0:.0f}s]{' *' if improved else ''}", flush=True)
        if cfg.patience and stale >= cfg.patience:
            print(f"  early stop: no val AUC improvement for {cfg.patience} epochs")
            break

    model.load_state_dict(best_state)
    probs, labels = predict(model, loaders["test"], device)
    results = {
        "run": run_name,
        "best_epoch": best_epoch,
        "best_val_macro_auc": best_auc,
        "test": classification_metrics(probs, labels),
        "params": count_params(model),
        "config": asdict(cfg),
    }
    print(f"[{run_name}] best epoch {best_epoch}  test acc {results['test']['accuracy']:.4f}  "
          f"test macro AUC {results['test']['macro_auc']:.4f}")

    if out_dir is not None:
        out = Path(out_dir) / run_name
        out.mkdir(parents=True, exist_ok=True)
        torch.save(best_state, out / "best.pt")
        save_json(results, out / "metrics.json")
        save_json(history, out / "history.json")
        np.savez_compressed(out / "test_predictions.npz", probs=probs, labels=labels)
    return model, {**results, "history": history, "probs": probs, "labels": labels}
