"""
Teaching note
-------------
For a model this small (785 parameters, 397x30 inputs) the GPU is NOT the
reason to use PyTorch -- kernel-launch and host/device transfer overhead
dominate, and a fold runs in well under a second on either device. The big
win over the from-scratch scalar engine comes from *vectorization*, not the
GPU. This file exists to show how little it takes to opt into CUDA.

Caveat: CPU and GPU results are not bit-identical even with the same seed.

Task
----
Predict whether a tumor is malignant from 30 numeric features.

    label 1 = malignant  (positive class, 212 samples)
    label 0 = benign     (negative class, 357 samples)

Evaluation uses StratifiedKFold; features are standardized inside each fold
using only that fold's training statistics.
"""

import random

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

# globals
FOLDS = 5
EPOCHS = 60
HIDDEN = [16, 16, 1]
LR = 0.05
BATCH_SIZE = 32
ALPHA = 1e-4  # for L2 regularization
SEED = 1337
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"  # DEVICE


# --------------------------------------------------------------------------
# Model
# --------------------------------------------------------------------------
class MLP(nn.Module):
    """Linear + ReLU stacks, with a linear output layer (a logit).

    This mirrors `scalar_autograd.nn.MLP(X.shape[1], HIDDEN)`.
    """

    def __init__(self, nin, nouts):
        super().__init__()
        sizes = [nin] + nouts
        layers = []
        for i in range(len(nouts)):
            layers.append(nn.Linear(sizes[i], sizes[i + 1]))
            if i != len(nouts) - 1:
                layers.append(nn.ReLU())
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


# --------------------------------------------------------------------------
# Loss & evaluation helpers
# --------------------------------------------------------------------------
def bce_with_logits_loss(model, X, y):
    """Mean binary cross-entropy (with logits).

    PyTorch computes exactly the same numerically stable form as the scratch
    version: `softplus(z) - y*z`. No explicit sigmoid or log is needed.
    """
    logits = model(X).squeeze(-1)
    return F.binary_cross_entropy_with_logits(logits, y)


def l2_regularization(model, alpha):
    """L2 penalty over all parameters (equivalent to weight decay on the weights).

    Note: unlike the scratch version, each parameter is a full tensor, so we
    reduce each `p*p` to a scalar with `.sum()` before adding.
    """
    return alpha * sum((p * p).sum() for p in model.parameters())


@torch.no_grad()
def predict_proba(model, X):
    """Return P(malignant) for each row of `X`."""
    if not torch.is_tensor(X):
        X = torch.as_tensor(X, dtype=torch.float32, device=DEVICE)  # DEVICE
    return torch.sigmoid(model(X).squeeze(-1))


def accuracy(y_true, y_pred):
    return float((y_true == y_pred).float().mean())


# --------------------------------------------------------------------------
# Training
# --------------------------------------------------------------------------
def train(model, X_tr, y_tr, X_va, y_va):
    """Mini-batch SGD with a linearly decaying learning rate.

    Returns per-epoch train and validation BCE losses.
    """
    X_tr = torch.as_tensor(X_tr, dtype=torch.float32, device=DEVICE)  # DEVICE
    y_tr = torch.as_tensor(y_tr, dtype=torch.float32, device=DEVICE)  # DEVICE
    X_va = torch.as_tensor(X_va, dtype=torch.float32, device=DEVICE)  # DEVICE
    y_va = torch.as_tensor(y_va, dtype=torch.float32, device=DEVICE)  # DEVICE

    optimizer = torch.optim.SGD(model.parameters(), lr=LR)
    rng = np.random.default_rng(SEED)
    n = len(X_tr)
    history = {"train_loss": [], "val_loss": []}

    for epoch in range(EPOCHS):
        # learning rate decays linearly to 10% of its initial value
        lr = LR * (1.0 - 0.9 * epoch / max(1, EPOCHS - 1))
        for group in optimizer.param_groups:
            group["lr"] = lr

        perm = rng.permutation(n)
        epoch_loss = 0.0

        for s in range(0, n, BATCH_SIZE):
            idx = perm[s:s + BATCH_SIZE]
            data_loss = bce_with_logits_loss(model, X_tr[idx], y_tr[idx])
            total_loss = data_loss + l2_regularization(model, ALPHA)

            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()  # p.data -= lr * p.grad

            epoch_loss += data_loss.item() * len(idx)

        history["train_loss"].append(epoch_loss / n)
        with torch.no_grad():
            history["val_loss"].append(
                F.binary_cross_entropy_with_logits(
                    model(X_va).squeeze(-1), y_va).item())

    return history


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------
def plot_curves(histories):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    train_curves = np.array([h["train_loss"] for h in histories])
    val_curves = np.array([h["val_loss"] for h in histories])
    epochs = np.arange(1, train_curves.shape[1] + 1)

    fig, ax = plt.subplots(figsize=(7, 4))
    for curves, color, label in ((train_curves, "tab:red", "train loss"),
                                 (val_curves, "tab:blue", "val loss")):
        mean, std = curves.mean(axis=0), curves.std(axis=0)
        ax.plot(epochs, mean, color=color, label=label)
        ax.fill_between(epochs, mean - std, mean + std, color=color, alpha=0.15)
    ax.grid(True, linestyle='--', alpha=0.6)
    ax.set_xlabel("epoch")
    ax.set_ylabel("BCE loss")
    ax.legend(loc="upper right")

    fig.tight_layout()
    fig.savefig("cancer_training_torch.png", dpi=150)
    print("saved training curve to cancer_training_torch.png")


def main():
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)

    # DEVICE
    print(f"device: {DEVICE}"
          + (f" ({torch.cuda.get_device_name(0)})" if DEVICE == "cuda" else ""))

    # load the data
    data = load_breast_cancer()
    X = data.data.astype(np.float64)
    # sklearn encodes 0 = malignant, 1 = benign; swap that
    y = (data.target == 0).astype(np.float64)
    print(f"dataset: {X.shape[0]} samples, {X.shape[1]} features")
    print(f"classes: {int(np.sum(y == 1))} malignant, {int(np.sum(y == 0))} benign")

    splitter = StratifiedKFold(n_splits=FOLDS, shuffle=True,
                               random_state=SEED)
    histories = []

    for fold, (tr, va) in enumerate(splitter.split(X, y), start=1):
        scaler = StandardScaler()
        X_tr = scaler.fit_transform(X[tr])
        X_va = scaler.transform(X[va])

        model = MLP(X.shape[1], HIDDEN).to(DEVICE)  # DEVICE
        history = train(model, X_tr, y[tr], X_va, y[va])
        histories.append(history)

        # calculate the accuracy of the validation set
        with torch.no_grad():
            y_va = torch.as_tensor(y[va], dtype=torch.float32, device=DEVICE)  # DEVICE
            probs = predict_proba(model, X_va)
            preds = (probs > 0.5).float()
            accuracy_score = accuracy(y_va, preds)
        print(f"fold {fold}/{FOLDS}: "
              f"train loss {history['train_loss'][-1]:.4f}, "
              f"val loss {history['val_loss'][-1]:.4f}, "
              f"acc {accuracy_score * 100:5.2f}%")

    plot_curves(histories)


if __name__ == "__main__":
    main()
