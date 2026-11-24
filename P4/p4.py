"""P4: CNN baseline (SI-ConvNeXt from LLNL NPS) trained on 256x256 grids of the AMR data.

Run from anywhere:
    python P4/p4.py --epochs 10

Data: split comes from P1/data/cases.csv; fields come from P1's cache (make_cache.py),
rasterized once to P4/grid_cache. See P4/README.md.
"""
import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parent
sys.path.insert(0, str(HERE))
from p4_data import FrameDataset, load_cases_table

NPS_URL = "https://github.com/llnl/NPS.git"


# Model
def build_model():
    """SI-ConvNeXt (isotropic ConvNeXt from NPS). Input: [phi, u, delta, epsilon_M, D]."""
    clone = HERE / "NPS"
    if not clone.exists():
        subprocess.run(["git", "clone", NPS_URL, str(clone)], check=True)
    # The repo's package is NPS/NPS/model, so the clone folder itself must be on sys.path
    # for `NPS.model` to resolve to the inner package.
    if (clone / "NPS" / "model").is_dir():
        root = clone
    # layout where the clone is the package
    elif (clone / "model").is_dir():
        root = HERE
    else:
        raise RuntimeError(f"Could not find the NPS 'model' package under {clone}")
    sys.path.insert(0, str(root))
    from NPS.model.convnext import ConvNeXtIsotropic   # noqa: E402
    return ConvNeXtIsotropic(in_chans=5, num_classes=2, depth=4, dim=64, DIM=2)


class ParamNorm:
    """Scale the three parameter channels (delta, epsilon_M, D) to [-1, 1] using the
    min/max of the TRAINING cases. Field channels (phi, u) are left unchanged."""

    def __init__(self, lo, hi, device):
        self.lo = torch.tensor(lo, dtype=torch.float32, device=device).view(1, 3, 1, 1)
        self.hi = torch.tensor(hi, dtype=torch.float32, device=device).view(1, 3, 1, 1)

    def __call__(self, x):
        x = x.clone()
        x[:, 2:5] = 2.0 * (x[:, 2:5] - self.lo) / (self.hi - self.lo) - 1.0
        return x

# Training / validation
def train_epoch(model, loader, optimizer, criterion, device, norm, noise_level):
    model.train()
    total = 0.0
    for x, y in loader:
        x = norm(x.to(device))
        y = y.to(device)
        # noise on phi and u only
        if noise_level > 0:
            x[:, :2] += torch.randn_like(x[:, :2]) * noise_level
        optimizer.zero_grad()
        loss = criterion(model(x), y)
        loss.backward()
        optimizer.step()
        total += loss.item() * x.size(0)
    return total / len(loader.dataset)


@torch.no_grad()
def validate(model, loader, criterion, device, norm):
    model.eval()
    total = 0.0
    for x, y in loader:
        x = norm(x.to(device))
        y = y.to(device)
        total += criterion(model(x), y).item() * x.size(0)
    return total / len(loader.dataset)


def case_subset_loader(dataset, n_cases, batch_size, shuffle, num_workers=0):
    """Loader over ALL frame pairs of n_cases randomly chosen cases (fresh draw per call)."""
    n_total = len(dataset.case_ids)
    chosen = np.random.choice(n_total, size=min(n_cases, n_total), replace=False)
    return DataLoader(Subset(dataset, dataset.pair_indices(chosen)), batch_size=batch_size,
                      shuffle=shuffle, num_workers=num_workers)


def train_model(model, train_ds, val_ds, optimizer, criterion, device, norm, out_dir, ckpt_extra,
                epochs, batch_size, noise_level, max_train_cases, max_val_cases, num_workers):
    out_dir.mkdir(parents=True, exist_ok=True)
    full_train = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    full_val = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    best = float("inf")
    history = {"train_loss": [], "val_loss": []}
    print(f"Training on {device}: {len(train_ds.case_ids)} train cases / {len(val_ds.case_ids)} val cases "
          f"({train_ds.n_pairs} frame pairs per case)")
    start_time = time.time()
    for epoch in range(1, epochs + 1):
        epoch_time = time.time()
        tl = (case_subset_loader(train_ds, max_train_cases, batch_size, True, num_workers)
              if max_train_cases else full_train)
        vl = (case_subset_loader(val_ds, max_val_cases, batch_size, False, num_workers)
              if max_val_cases else full_val)
        train_loss = train_epoch(model, tl, optimizer, criterion, device, norm, noise_level)
        val_loss = validate(model, vl, criterion, device, norm)
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        print(f"Epoch {epoch:02d}/{epochs:02d} | train {train_loss:.6f} | val {val_loss:.6f} "
              f"| {time.time() - epoch_time:.1f}s")
        if val_loss < best:
            best = val_loss
            torch.save({"model_state": model.state_dict(), **ckpt_extra, "val_loss": best},
                       out_dir / "best_convnext_isotropic.pt")
            print(f"  --> saved new best (val {best:.6f})")
    print(f"Done in {(time.time() - start_time) / 60:.1f} min | best val {best:.6f}")

    (out_dir / "history.json").write_text(json.dumps(history, indent=2))
    plt.figure(figsize=(10, 5))
    plt.plot(range(1, epochs + 1), history["train_loss"], marker="o", label="Train")
    plt.plot(range(1, epochs + 1), history["val_loss"], marker="s", label="Val")
    plt.xlabel("Epoch")
    plt.ylabel("MSE")
    plt.grid(True)
    plt.legend()
    plt.title("ConvNeXtIsotropic training and validation loss")
    plt.savefig(out_dir / "training_plot.png", dpi=300, bbox_inches="tight")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default=str(REPO_ROOT / "P1" / "data" / "cases.csv"))
    ap.add_argument("--cache-dir", default=str(REPO_ROOT / "cache"), help="P1 make_cache.py output")
    ap.add_argument("--grid-dir", default=str(HERE / "grid_cache"))
    ap.add_argument("--out-dir", default=str(HERE / "outputs"))
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--noise", type=float, default=0.01)
    ap.add_argument("--n-frames", type=int, default=301,
                    help="frames 0..n-1 per case (301 = every frame; fewer for quick tests)")
    ap.add_argument("--max-train-cases", type=int, default=None,
                    help="each epoch, train on ALL frame pairs of this many randomly chosen "
                         "train cases (fresh draw every epoch); default: all train cases")
    ap.add_argument("--max-val-cases", type=int, default=None,
                    help="same for validation; default: all val cases")
    ap.add_argument("--num-workers", type=int, default=0, help="DataLoader worker processes")
    ap.add_argument("--prepare-only", action="store_true",
                    help="rasterize the grid files for ALL splits (train/val/test/outer), then exit")
    ap.add_argument("--seed", type=int, default=539)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    frames = range(args.n_frames)
    if args.prepare_only:
        for split in ("train", "val", "test", "outer"):
            FrameDataset(args.cache_dir, args.grid_dir, args.csv, split, frames)
        print("Grid files ready in", args.grid_dir)
        return
    train_ds = FrameDataset(args.cache_dir, args.grid_dir, args.csv, "train", frames)
    val_ds = FrameDataset(args.cache_dir, args.grid_dir, args.csv, "val", frames)

    df = load_cases_table(args.csv)
    tr = df[df["set"] == "train"][["delta", "epsilonM", "D"]]
    lo = tr.min().tolist()
    hi = tr.max().tolist()
    norm = ParamNorm(lo, hi, device)

    model = build_model().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    train_model(model, train_ds, val_ds, optimizer, torch.nn.MSELoss(), device, norm,
                Path(args.out_dir), {"param_lo": lo, "param_hi": hi, "n_frames": args.n_frames},
                args.epochs, args.batch_size, args.noise,
                args.max_train_cases, args.max_val_cases, args.num_workers)


if __name__ == "__main__":
    main()
