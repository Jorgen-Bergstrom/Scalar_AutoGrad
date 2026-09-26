"""Binary classification of the Breast Cancer Wisconsin dataset
using the scalar autograd engine in `scalar_autograd/engine.py` and the MLP
in `scalar_autograd/nn.py`.

Task
----
Predict whether a tumor is malignant from 30 numeric features computed from a
digitized image of a fine needle aspirate (FNA) of a breast mass.

    label 1 = malignant  (positive class, 212 samples)
    label 0 = benign     (negative class, 357 samples)

Model & loss
------------
A logistic-regression / MLP classifier trained with **binary cross-entropy**.
The network outputs a single logit `z`; the probability is `sigmoid(z)`.
The loss is the numerically stable "BCE with logits":

    L = softplus(z) - y * z,   where softplus(z) = log(1 + e^z)

which is algebraically identical to the usual `-(y log p + (1-y) log(1-p))`
with `p = sigmoid(z)`, but avoids `log(0)` / overflow when the logit saturates.

Evaluation
----------
Instead of a single train/val/test split, the whole dataset is evaluated with
**StratifiedKFold** cross-validation: every sample is used for validation
exactly once, and the reported metrics are the mean +/- std over the folds.
Features are standardized inside each fold using only that fold's training
statistics, so no information leaks from the validation samples.
"""

import random
import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from scalar_autograd import Value
from scalar_autograd.nn import MLP

# globals
FOLDS = 5
EPOCHS = 60
HIDDEN = [16, 16, 1]
LR = 0.05
BATCH_SIZE = 32
ALPHA = 1e-4  # for L2 regularization
SEED = 1337


# --------------------------------------------------------------------------
# Loss & evaluation helpers
# --------------------------------------------------------------------------
def bce_with_logits_loss(model, X, y):
    """Mean binary cross-entropy (with logits), as a single autograd `Value`.
    `X` is an (n, d) float array of features and `y` an (n,) array of 0/1.
    We build the softplus branch explicitly so the argument of `log` is
    always >= 1 and the result never becomes +/-inf.
    """
    losses = []
    for xi, yi in zip(X, y):
        z = model([Value(v) for v in xi])

        if z.data > 0:
            # log(1 + e^z) = z + log(1 + e^-z)  (exponent stays <= 0)
            softplus = z + (1.0 + (-z).exp()).log()
        else:
            # log(1 + e^z)  (exponent stays <= 0)
            softplus = (1.0 + z.exp()).log()

        # softplus(z) - y*z  ==  BCE(sigmoid(z), y)
        losses.append(softplus - yi * z)

    return sum(losses) * (1.0 / len(losses))


def l2_regularization(model, alpha):
    """L2 penalty over all parameters, as a single `Value`."""
    return alpha * sum((p * p for p in model.parameters()))


def bce_from_probs(y, probs):
    """Mean BCE from predicted probabilities (no autograd graph).
    Probabilities are clipped so `log(0)` cannot occur.
    """
    y = np.asarray(y, dtype=float)
    p = np.clip(np.asarray(probs, dtype=float), 1e-12, 1.0 - 1e-12)
    return float(-np.mean(y * np.log(p) + (1.0 - y) * np.log(1.0 - p)))


def predict_proba(model, X):
    """Return P(malignant) for each row of `X`."""
    probs = []
    for xi in X:
        z = model([Value(v) for v in xi])
        probs.append(z.sigmoid().data)
    return np.asarray(probs)


def accuracy(y_true, y_pred):
    return float(np.mean(np.asarray(y_true) == np.asarray(y_pred)))


# --------------------------------------------------------------------------
# Training
# --------------------------------------------------------------------------
def train(model, X_tr, y_tr, X_va, y_va):
    """Mini-batch SGD with a linearly decaying learning rate.
    Returns per-epoch train and validation BCE losses.
    """
    rng = np.random.default_rng(SEED)
    n = len(X_tr)
    history = {"train_loss": [], "val_loss": []}

    for epoch in range(EPOCHS):
        # learning rate decays linearly to 10% of its initial value
        lr = LR * (1.0 - 0.9 * epoch / max(1, EPOCHS - 1))
        perm = rng.permutation(n)
        epoch_loss = 0.0

        for s in range(0, n, BATCH_SIZE):
            idx = perm[s:s + BATCH_SIZE]
            data_loss = bce_with_logits_loss(model, X_tr[idx], y_tr[idx])
            total_loss = data_loss + l2_regularization(model, ALPHA)

            model.zero_grad()
            total_loss.backward()
            for p in model.parameters():
                p.data -= lr * p.grad

            epoch_loss += data_loss.data * len(idx)

        history["train_loss"].append(epoch_loss / n)
        history["val_loss"].append(
            bce_from_probs(y_va, predict_proba(model, X_va)))

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
    fig.savefig("cancer_training.png", dpi=150)
    print("saved training curve to cancer_training.png")


def main():
    random.seed(SEED)
    np.random.seed(SEED)

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

        model = MLP(X.shape[1], HIDDEN)
        history = train(model, X_tr, y[tr], X_va, y[va])
        histories.append(history)

        # calculate the accuracy of the validation set
        probs = predict_proba(model, X_va)
        preds = (probs > 0.5).astype(float)
        accuracy_score = accuracy(y[va], preds)
        print(f"fold {fold}/{FOLDS}: "
                f"train loss {history['train_loss'][-1]:.4f}, "
                f"val loss {history['val_loss'][-1]:.4f}, "
                f"acc {accuracy_score * 100:5.2f}%")

    plot_curves(histories)


if __name__ == "__main__":
    main()
