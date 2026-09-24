import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import confusion_matrix, roc_auc_score, roc_curve

from . import CLASS_NAMES, NUM_CLASSES


def logits_of(out):
    return out["logits"] if isinstance(out, dict) else out


@torch.no_grad()
def predict(model, loader, device):
    """Return (softmax probs (N, C), labels (N,)) over a loader."""
    model.eval()
    probs, labels = [], []
    for x, y in loader:
        probs.append(torch.softmax(logits_of(model(x.to(device))), dim=1).float().cpu())
        labels.append(y)
    return torch.cat(probs).numpy(), torch.cat(labels).numpy()


def classification_metrics(probs: np.ndarray, labels: np.ndarray) -> dict:
    """Accuracy, one-vs-rest ROC AUC per class and macro AUC, confusion matrix."""
    onehot = np.eye(NUM_CLASSES)[labels]
    per_class = [float(roc_auc_score(onehot[:, c], probs[:, c])) for c in range(NUM_CLASSES)]
    return {
        "accuracy": float((probs.argmax(1) == labels).mean()),
        "auc_per_class": dict(zip(CLASS_NAMES, per_class)),
        "macro_auc": float(np.mean(per_class)),
        "confusion_matrix": confusion_matrix(labels, probs.argmax(1), labels=range(NUM_CLASSES)).tolist(),
        "n": int(len(labels)),
    }


def plot_roc(probs, labels, title="", ax=None):
    ax = ax or plt.subplots(figsize=(6, 5))[1]
    onehot = np.eye(NUM_CLASSES)[labels]
    aucs = []
    for c, name in enumerate(CLASS_NAMES):
        fpr, tpr, _ = roc_curve(onehot[:, c], probs[:, c])
        aucs.append(roc_auc_score(onehot[:, c], probs[:, c]))
        ax.plot(fpr, tpr, lw=2, label=f"{name} (AUC {aucs[-1]:.4f})")
    ax.plot([0, 1], [0, 1], "k--", lw=1)
    ax.set(xlabel="False positive rate", ylabel="True positive rate", xlim=(0, 1), ylim=(0, 1.01),
           title=f"{title}  macro AUC {np.mean(aucs):.4f}".strip())
    ax.legend(loc="lower right", fontsize=9)
    ax.grid(alpha=0.3)
    return ax


def plot_confusion(cm, title="", ax=None):
    cm = np.asarray(cm)
    ax = ax or plt.subplots(figsize=(5, 4.5))[1]
    norm = cm / cm.sum(1, keepdims=True).clip(min=1)
    ax.imshow(norm, cmap="Blues", vmin=0, vmax=1)
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, f"{cm[i, j]}\n{norm[i, j]:.1%}", ha="center", va="center",
                    color="white" if norm[i, j] > 0.5 else "black", fontsize=9)
    short = [n.split(" ")[0] for n in CLASS_NAMES]
    ax.set(xticks=range(len(short)), yticks=range(len(short)), xticklabels=short, yticklabels=short,
           xlabel="Predicted", ylabel="True", title=title)
    return ax


def plot_history(history: dict, title=""):
    """Loss / accuracy / val-AUC curves, plus any extra loss terms (e.g. Poisson)."""
    extra = [k for k in history if k.startswith("train_") and k not in ("train_loss", "train_acc")]
    n = 3 + bool(extra)
    fig, axes = plt.subplots(1, n, figsize=(5 * n, 4))
    ep = np.arange(1, len(history["train_loss"]) + 1)
    axes[0].plot(ep, history["train_loss"], label="train"); axes[0].plot(ep, history["val_loss"], label="val")
    axes[0].set_title("Loss")
    axes[1].plot(ep, history["train_acc"], label="train"); axes[1].plot(ep, history["val_acc"], label="val")
    axes[1].set_title("Accuracy")
    axes[2].plot(ep, history["val_macro_auc"], color="C2"); axes[2].set_title("Val macro AUC")
    if extra:
        for k in extra:
            axes[3].plot(ep, history[k], label=k.removeprefix("train_"))
        axes[3].set_yscale("log"); axes[3].set_title("Loss terms (train)")
    for ax in axes:
        ax.set_xlabel("epoch"); ax.grid(alpha=0.3)
        if ax.get_legend_handles_labels()[0]:
            ax.legend()
    fig.suptitle(title)
    fig.tight_layout()
    return fig
