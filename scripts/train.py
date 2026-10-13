"""Train one model from the command line.

Examples
--------
Quick smoke test on a laptop (a few hundred images, 2 epochs):
    python scripts/train.py --model cnn --train-per-class 200 --test-per-class 100 --epochs 2

Full run (GPU recommended):
    python scripts/train.py --model efficientnet
    python scripts/train.py --model pinn --lambda-poisson 0.1
    python scripts/train.py --model deit
    python scripts/train.py --model pinn2 --amp --resume      # CUDA (e.g. Colab T4)
"""

import argparse
import sys
from dataclasses import replace
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from deeplense.data import DataConfig, build_loaders  # noqa: E402
from deeplense.models import PINNLoss, PINNv2Loss, build_model  # noqa: E402
from deeplense.train import TrainConfig, fit  # noqa: E402
from deeplense.utils import get_device, seed_everything  # noqa: E402

# Per-model defaults. Pretrained backbones get a head-only warm-up epoch and a lower
# backbone learning rate so ImageNet features aren't destroyed early on.
RECIPES = {
    "cnn":          TrainConfig(epochs=30, lr=1e-3, weight_decay=5e-2, warmup_epochs=1),
    "efficientnet": TrainConfig(epochs=25, lr=1e-3, backbone_lr_mult=0.3, weight_decay=1e-2,
                                warmup_epochs=1, freeze_epochs=1),
    "pinn":         TrainConfig(epochs=25, lr=1e-3, backbone_lr_mult=0.3, weight_decay=1e-2,
                                warmup_epochs=1, freeze_epochs=1),
    # ViTs collapse to a constant prediction if fine-tuned too hard: backbone LR 2e-4 did
    # exactly that (val acc stuck at 1/3, loss = ln 3). DeiT's recipe scales as
    # 5e-4 * batch / 512, i.e. ~3e-5 at batch 32, so: head 3e-4, backbone 3e-5.
    "deit":         TrainConfig(epochs=30, lr=3e-4, backbone_lr_mult=0.1, weight_decay=5e-2,
                                warmup_epochs=3, freeze_epochs=1),
    "pinn2":        TrainConfig(epochs=25, lr=1e-3, backbone_lr_mult=0.3, weight_decay=1e-2,
                                warmup_epochs=1, freeze_epochs=1),
}


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", required=True, choices=list(RECIPES))
    p.add_argument("--data-root", default="data/lensing")
    p.add_argument("--out-dir", default="results")
    p.add_argument("--run-name", default=None)
    p.add_argument("--train-per-class", type=int, default=None, help="subsample train/ (smoke tests)")
    p.add_argument("--test-per-class", type=int, default=None, help="subsample val/ used as test")
    p.add_argument("--batch-size", type=int, default=32, help="32 fits an 8 GB Mac; use 64+ on a GPU")
    p.add_argument("--num-workers", type=int, default=2)
    p.add_argument("--epochs", type=int)
    p.add_argument("--lr", type=float)
    p.add_argument("--patience", type=int)
    p.add_argument("--label-smoothing", type=float)
    p.add_argument("--lambda-poisson", type=float, default=0.1, help="PINN Poisson-loss weight")
    p.add_argument("--lambda-rec", type=float, default=1.0, help="PINN v2 image-reconstruction weight")
    p.add_argument("--lambda-sparse", type=float, default=0.01, help="PINN v2 kappa_sub sparsity weight")
    p.add_argument("--amp", action="store_true", help="mixed precision (CUDA only)")
    p.add_argument("--resume", action="store_true", help="continue from <out-dir>/<run>/last.pt")
    p.add_argument("--no-pretrained", action="store_true")
    p.add_argument("--device", default=None)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--log-every", type=int, default=0)
    a = p.parse_args(argv)

    seed_everything(a.seed)
    device = get_device(a.device)

    cfg = RECIPES[a.model]
    overrides = {k: v for k, v in dict(epochs=a.epochs, lr=a.lr, patience=a.patience,
                                       label_smoothing=a.label_smoothing, log_every=a.log_every,
                                       amp=a.amp or None).items()
                 if v is not None}
    cfg = replace(cfg, **overrides)

    loaders = build_loaders(DataConfig(root=a.data_root, train_per_class=a.train_per_class,
                                       test_per_class=a.test_per_class, batch_size=a.batch_size,
                                       num_workers=a.num_workers, seed=a.seed))

    kwargs = {} if a.model == "cnn" else {"pretrained": not a.no_pretrained}
    model = build_model(a.model, **kwargs)

    criterion = None
    run_name = a.run_name or a.model
    if a.model == "pinn":
        criterion = PINNLoss(a.lambda_poisson, cfg.label_smoothing)
        cfg = replace(cfg, extra={"lambda_poisson": a.lambda_poisson})
        run_name = a.run_name or f"pinn_lambda{a.lambda_poisson:g}"
    elif a.model == "pinn2":
        criterion = PINNv2Loss(a.lambda_rec, a.lambda_poisson, a.lambda_sparse, cfg.label_smoothing)
        cfg = replace(cfg, extra={"lambda_rec": a.lambda_rec, "lambda_poisson": a.lambda_poisson,
                                  "lambda_sparse": a.lambda_sparse})
        run_name = a.run_name or f"pinn2_lambda{a.lambda_poisson:g}"

    fit(model, loaders, cfg, device, criterion=criterion, out_dir=a.out_dir, run_name=run_name, resume=a.resume)


if __name__ == "__main__":
    main()
