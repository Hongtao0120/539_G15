"""P4 data pipeline: cases.csv splits + P1's epoch cache -> 256x256 grids -> torch Dataset.

P1's make_cache.py writes   <cache_dir>/<campaign>/<case>/epoch_XXX.npz   (adaptive mesh
per epoch: pos, phi, u, params, ...; frames 10e..10e+10 live on mesh e).

Each case is rasterized ONCE onto a uniform grid for all requested frames and saved as
<grid_dir>/<case>_g256_f<first>-<last>.npy  (shape (T, 2, 256, 256), float32; ~158 MB per
case for 301 frames). The dataset reads frame pairs from these files with memory-mapping,
so only the pairs actually used are read from disk - the whole split is never held in RAM.
Delete the grid files to force a rebuild.
"""
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from scipy.interpolate import LinearNDInterpolator

SPLITS = ("train", "val", "test", "outer") # labels in the cases.csv "set" column
GRID = 256
DOMAIN = 500.0


# Splits
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


# P1 epoch cache -> uniform grid
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
    frame). Returns (len(frames), 2, grid, grid) float32 = [phi, u].
    """
    case_dir = Path(case_dir)
    n_epochs = len(list(case_dir.glob("epoch_*.npz")))
    with np.load(case_dir / "epoch_000.npz") as z:
        fpe = z["phi"].shape[0] - 1
        delta = float(z["params"][0])

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
        vals = np.concatenate([phi[ks].T, u[ks].T], axis=1)
        g = LinearNDInterpolator(pos, vals)(gx, gy)
        K = len(ks)
        for j, (out_i, _) in enumerate(items):
            # liquid background
            out[out_i, 0] = np.nan_to_num(g[..., j], nan=-1.0)
            # undercooled background
            out[out_i, 1] = np.nan_to_num(g[..., K + j], nan=-delta)
    return out


def grid_path(grid_dir, case, frames):
    return Path(grid_dir) / f"{case}_g{GRID}_f{frames[0]}-{frames[-1]}.npy"


def ensure_grid(cache_dir, grid_dir, campaign, case, frames):
    """Path of the gridded fields for one case; rasterized on first use."""
    path = grid_path(grid_dir, case, frames)
    if not path.exists():
        fields = rasterize_case(find_case_dir(cache_dir, campaign, case), frames)
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_name(path.name + ".tmp")
        # write then rename: no half-written files
        with open(tmp, "wb") as fh:
            np.save(fh, fields)
        tmp.replace(path)
        print(f"[grid] {case} -> {path.name}", flush=True)
    return path


# Dataset
class FrameDataset(Dataset):
    """Every consecutive frame pair of every case in one split (case-major order).

    x : (5, H, W) = [phi_t, u_t, delta, epsilon_M, D]     y : (2, H, W) = [phi_{t+1}, u_{t+1}]
    With frames=range(301) each case contributes 300 pairs. `frames` must be consecutive.
    Parameters come from cases.csv. Pairs are read lazily from the grid files.
    """

    def __init__(self, cache_dir, grid_dir, csv_path, split, frames=range(301)):
        if split not in SPLITS:
            raise ValueError(f"split must be one of {SPLITS}")
        frames = list(frames)
        if len(frames) < 2 or any(b - a != 1 for a, b in zip(frames, frames[1:])):
            raise ValueError("frames must be at least 2 consecutive snapshot numbers")
        rows = split_rows(load_cases_table(csv_path), split)
        self.case_ids = [r.case for r in rows]
        # pairs per case
        self.n_pairs = len(frames) - 1
        self.paths = [ensure_grid(cache_dir, grid_dir, r.campaign, r.case, frames) for r in rows]
        self.params = np.array([[r.delta, r.epsilonM, r.D] for r in rows], dtype=np.float32)
        self._mm = {}

    def __len__(self):
        return len(self.case_ids) * self.n_pairs

    def pair_indices(self, case_idx):
        """Dataset indices of all pairs belonging to the given case numbers (0..n_cases-1)."""
        return np.concatenate([np.arange(c * self.n_pairs, (c + 1) * self.n_pairs)
                               for c in case_idx])

    def _pair(self, i):
        c, t = divmod(i, self.n_pairs)
        if c not in self._mm:
            self._mm[c] = np.load(self.paths[c], mmap_mode="r")
        cur, nxt = np.array(self._mm[c][t:t + 2]) # copy just these two frames
        H, W = cur.shape[-2:]
        p = np.broadcast_to(self.params[c].reshape(3, 1, 1), (3, H, W))
        return np.concatenate([cur, p], axis=0), nxt

    def __getitem__(self, i):
        x, y = self._pair(i)
        return torch.from_numpy(x), torch.from_numpy(np.ascontiguousarray(y))
