"""Train and evaluate the Ridge baseline and C1 Stats MLP on P1's case splits."""
import argparse
import json
import os
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import RidgeCV
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "P1"))
from amr.data import norm_params

TARGET_NAMES = ["solid_area", "tip_radius_mean", "interface_length"]


def encode_inputs(X):
    """Normalize [delta, epsilonM, D, t] to [-1, 1]^4 using P1's PARAM_BOX and t in [0, 300]."""
    p_norm = norm_params(X[:, :3]).astype(np.float32)
    t_norm = (2.0 * (X[:, 3:4] / 300.0) - 1.0).astype(np.float32)
    return np.hstack([p_norm, t_norm])


class StatsMLP(nn.Module):
    def __init__(self, in_dim=4, hidden_dim=128, depth=3, out_dim=3):
        super().__init__()
        layers = [nn.Linear(in_dim, hidden_dim), nn.SiLU()]
        for _ in range(depth - 1):
            layers += [nn.Linear(hidden_dim, hidden_dim), nn.SiLU()]
        layers.append(nn.Linear(hidden_dim, out_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


def evaluate_predictions(y_true, y_pred, frames):
    """Compute RMSE, overall MAPE (t >= 10), and t=300 relative error for each statistic."""
    rmse = np.sqrt(np.mean((y_pred - y_true) ** 2, axis=0))
    mask_t10 = frames >= 10
    mape = np.mean(
        np.abs(y_pred[mask_t10] - y_true[mask_t10]) / np.maximum(np.abs(y_true[mask_t10]), 1e-6),
        axis=0,
    ) * 100.0
    mask_t300 = frames == 300
    err_t300 = np.mean(
        np.abs(y_pred[mask_t300] - y_true[mask_t300]) / np.maximum(np.abs(y_true[mask_t300]), 1e-6),
        axis=0,
    ) * 100.0

    metrics = {}
    for i, name in enumerate(TARGET_NAMES):
        metrics[name] = {
            "rmse": float(rmse[i]),
            "mape_t10_300_percent": float(mape[i]),
            "mape_t300_percent": float(err_t300[i]),
        }
    return metrics


def train_and_eval(data_path, out_dir, results_dir, epochs=250, batch_size=256, lr=1e-3, seed=539):
    torch.manual_seed(seed)
    np.random.seed(seed)

    data = np.load(data_path)
    X_raw, Y, splits, frames = data["X"], data["Y"], data["splits"], data["frames"]
    X = encode_inputs(X_raw)

    idx_train = splits == "train"
    idx_val = splits == "val"
    idx_test = splits == "test"
    idx_outer = splits == "outer"

    # Standardize targets using training split statistics only
    y_mean = Y[idx_train].mean(axis=0)
    y_std = Y[idx_train].std(axis=0) + 1e-8
    Y_norm = (Y - y_mean) / y_std

    # ---------------------------------------------------------------- 1. Ridge Baseline
    ridge = make_pipeline(
        PolynomialFeatures(degree=3, include_bias=False),
        StandardScaler(),
        RidgeCV(alphas=np.logspace(-4, 3, 25)),
    )
    ridge.fit(X[idx_train], Y_norm[idx_train])

    ridge_metrics = {}
    for split_name, mask in [("val", idx_val), ("test", idx_test), ("outer", idx_outer)]:
        pred = ridge.predict(X[mask]) * y_std + y_mean
        ridge_metrics[split_name] = evaluate_predictions(Y[mask], pred, frames[mask])

    # ---------------------------------------------------------------- 2. C1 Stats MLP
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = StatsMLP(in_dim=4, hidden_dim=128, depth=3, out_dim=3).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    loss_fn = nn.MSELoss()

    train_loader = DataLoader(
        TensorDataset(torch.from_numpy(X[idx_train]), torch.from_numpy(Y_norm[idx_train])),
        batch_size=batch_size,
        shuffle=True,
    )
    X_val_t = torch.from_numpy(X[idx_val]).to(device)
    Y_val_norm_t = torch.from_numpy(Y_norm[idx_val]).to(device)

    best_val_loss = float("inf")
    best_state = None
    history = {"train_loss": [], "val_loss": []}

    for epoch in range(1, epochs + 1):
        model.train()
        batch_losses = []
        for bx, by in train_loader:
            bx, by = bx.to(device), by.to(device)
            optimizer.zero_grad()
            loss = loss_fn(model(bx), by)
            loss.backward()
            optimizer.step()
            batch_losses.append(loss.item())
        scheduler.step()

        model.eval()
        with torch.no_grad():
            val_loss = float(loss_fn(model(X_val_t), Y_val_norm_t).item())

        history["train_loss"].append(float(np.mean(batch_losses)))
        history["val_loss"].append(val_loss)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {k: v.cpu() for k, v in model.state_dict().items()}

        if epoch % 50 == 0 or epoch == 1:
            print(f"Epoch {epoch:03d}/{epochs} | Train MSE: {history['train_loss'][-1]:.5f} | Val MSE: {val_loss:.5f}")

    model.load_state_dict(best_state)
    model.eval()

    mlp_metrics = {}
    val_preds_mlp = None
    for split_name, mask in [("val", idx_val), ("test", idx_test), ("outer", idx_outer)]:
        with torch.no_grad():
            pred_norm = model(torch.from_numpy(X[mask]).to(device)).cpu().numpy()
        pred = pred_norm * y_std + y_mean
        if split_name == "val":
            val_preds_mlp = pred
        mlp_metrics[split_name] = evaluate_predictions(Y[mask], pred, frames[mask])

    os.makedirs(out_dir, exist_ok=True)
    ckpt_path = os.path.join(out_dir, "c1_mlp.pt")
    torch.save(
        {
            "model_state": best_state,
            "y_mean": y_mean,
            "y_std": y_std,
            "target_names": TARGET_NAMES,
        },
        ckpt_path,
    )

    os.makedirs(results_dir, exist_ok=True)
    summary = {
        "best_val_norm_mse": best_val_loss,
        "ridge_baseline": ridge_metrics,
        "c1_mlp": mlp_metrics,
    }
    with open(os.path.join(results_dir, "c1_metrics.json"), "w") as fh:
        json.dump(summary, fh, indent=2)

    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    for i, name in enumerate(TARGET_NAMES):
        axes[i].scatter(Y[idx_val, i], val_preds_mlp[:, i], s=8, alpha=0.5)
        lo, hi = Y[idx_val, i].min(), Y[idx_val, i].max()
        axes[i].plot([lo, hi], [lo, hi], "k--", lw=1)
        axes[i].set_title(f"{name} (Val MAPE: {mlp_metrics['val'][name]['mape_t10_300_percent']:.2f}%)")
        axes[i].set_xlabel("True")
        axes[i].set_ylabel("C1 MLP Predicted")
    fig.tight_layout()
    fig.savefig(os.path.join(results_dir, "c1_val_parity.png"), dpi=140)
    print(f"Saved metrics to {results_dir}/c1_metrics.json")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", default=str(REPO_ROOT / "P3" / "outputs" / "stats_dataset.npz"))
    parser.add_argument("--out_dir", default=str(REPO_ROOT / "P3" / "outputs"))
    parser.add_argument("--results_dir", default=str(REPO_ROOT / "P3" / "results"))
    parser.add_argument("--epochs", type=int, default=250)
    args = parser.parse_args()
    train_and_eval(args.data_path, args.out_dir, args.results_dir, epochs=args.epochs)
