"""P4 data pipeline: cases.csv splits + P1's epoch cache -> 256x256 grids -> torch Dataset.

P1's make_cache.py writes   <cache_dir>/<campaign>/<case>/epoch_XXX.npz   (adaptive mesh
per epoch: pos, phi, u, params, ...; frames 10e..10e+10 live on mesh e).
This module rasterizes the needed frames onto a uniform grid (once per case, saved to
<grid_dir>/<case>_g256.npz) and serves consecutive-frame pairs for the CNN.
"""
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

SPLITS = ("train", "val", "test", "outer")   # labels in the cases.csv "set" column
GRID = 256
DOMAIN = 500.0


# ----------------------------------------------------------------------------
# Splits
# ----------------------------------------------------------------------------
def load_cases_table(csv_path):
    """Read cases.csv (columns: case, campaign, delta, epsilonM, D, set) and validate it."""
    df = pd.read_csv(csv_path)
    df["set"] = df["set"].str.strip().str.lower()
    unknown = set(df["set"]) - set(SPLITS)
    if unknown:
        raise ValueError(f"Unexpected labels in 'set' column: {unknown}")
    if not df["case"].is_unique:
        raise ValueError("Duplicate case IDs in cases.csv")
    return df


def split_rows(df, split):
    return list(df.loc[df["set"] == split].itertuples())


# ----------------------------------------------------------------------------
# P1 epoch cache -> uniform grid
# ----------------------------------------------------------------------------
def find_case_dir(cache_dir, campaign, case):
    d = Path(cache_dir) / campaign / case
    if not (d / "epoch_000.npz").exists():
        raise FileNotFoundError(f"{d}/epoch_000.npz not found - run P1's make_cache.py "
                                f"(or point cache_dir at the shared cache)")
    return d


def rasterize_case(case_dir, frames, grid=GRID):
    """Interpolate the requested global frame numbers onto a grid x grid lattice.

    Frame f is read from the mesh of epoch e = min(f // 10, last), index f - 10e, so every
    frame is interpolated from its own mesh (frame 300 comes from the last epoch's end
    frame). Returns (len(frames), 2, grid, grid) float32 = [phi, u], and params (3,).
    """
    from scipy.interpolate import LinearNDInterpolator

    case_dir = Path(case_dir)
    n_epochs = len(list(case_dir.glob("epoch_*.npz")))
    with np.load(case_dir / "epoch_000.npz") as z:
        fpe = z["phi"].shape[0] - 1                    # frames per epoch (10)
        params = z["params"].astype(np.float32)        # [delta, epsilonM, D]
    delta = float(params[0])

    by_epoch = {}
    for out_i, f in enumerate(frames):
        e = min(f // fpe, n_epochs - 1)
        by_epoch.setdefault(e, []).append((out_i, f - e * fpe))

    gx, gy = np.mgrid[0:DOMAIN:complex(0, grid), 0:DOMAIN:complex(0, grid)]
    out = np.empty((len(frames), 2, grid, grid), dtype=np.float32)
    for e, items in by_epoch.items():
        with np.load(case_dir / f"epoch_{e:03d}.npz") as z:
            pos, phi, u = z["pos"].astype(np.float64), z["phi"], z["u"]
        ks = [k for _, k in items]
        vals = np.concatenate([phi[ks].T, u[ks].T], axis=1)       # (N, 2K)
        g = LinearNDInterpolator(pos, vals)(gx, gy)                # one triangulation per epoch
        K = len(ks)
        for j, (out_i, _) in enumerate(items):
            out[out_i, 0] = np.nan_to_num(g[..., j], nan=-1.0)          # liquid background
            out[out_i, 1] = np.nan_to_num(g[..., K + j], nan=-delta)    # undercooled background
    return out, params


def load_grids(cache_dir, grid_dir, campaign, case, frames):
    """Gridded fields for one case; rasterized on first use, then read from grid_dir."""
    frames = np.asarray(list(frames))
    path = Path(grid_dir) / f"{case}_g{GRID}.npz"
    if path.exists():
        with np.load(path) as z:
            if np.array_equal(z["frames"], frames):
                return z["fields"], z["params"]
    fields, params = rasterize_case(find_case_dir(cache_dir, campaign, case), frames)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, fields=fields, params=params, frames=frames)
    return fields, params


# ----------------------------------------------------------------------------
# Dataset
# ----------------------------------------------------------------------------
class FrameDataset(Dataset):
    """Yields (x, y) for consecutive frame pairs of every case in one split.

    x : (5, H, W) = [phi_t, u_t, delta, epsilon_M, D]     y : (2, H, W) = [phi_{t+1}, u_{t+1}]
    `frames` must be consecutive snapshot numbers (default 0..20 as in the old notebook).
    """

    def __init__(self, cache_dir, grid_dir, csv_path, split, frames=range(21)):
        if split not in SPLITS:
            raise ValueError(f"split must be one of {SPLITS}")
        frames = list(frames)
        if any(b - a != 1 for a, b in zip(frames, frames[1:])):
            raise ValueError("frames must be consecutive")
        df = load_cases_table(csv_path)
        rows = split_rows(df, split)
        self.case_ids = [r.case for r in rows]
        self.n_pairs = len(frames) - 1
        fields, params = [], []
        for r in rows:
            f, p = load_grids(cache_dir, grid_dir, r.campaign, r.case, frames)
            fields.append(f)
            params.append(p)
        self.fields = torch.from_numpy(np.stack(fields))     # (C, T, 2, H, W)
        self.params = torch.from_numpy(np.stack(params))     # (C, 3)

    def __len__(self):
        return len(self.case_ids) * self.n_pairs

    def __getitem__(self, i):
        c, t = divmod(i, self.n_pairs)
        cur, nxt = self.fields[c, t], self.fields[c, t + 1]
        H, W = cur.shape[-2:]
        p = self.params[c].view(3, 1, 1).expand(3, H, W)
        return torch.cat([cur, p], dim=0), nxt
