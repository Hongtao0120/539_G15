"""Extract summary statistics (solid_area, tip_radius_mean, interface_length) across all cached cases."""
import argparse
import os
import sys
from pathlib import Path

import numpy as np

# Allow importing shared utilities from P1/
REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "P1"))

from amr.data import FRAMES_PER_EPOCH, N_EPOCHS, load_epoch, load_splits
from amr.pipeline import AREA, field_stats


def compute_interface_length(pos, cells, phi):
    """Contour length of phi = 0 via marching squares over the 4 corners of each Q1/Q3 cell.
    In deal.II FE_Q(3), local indices 0, 1, 3, 2 are the four cell corners in CCW order."""
    corners = cells[:, [0, 1, 3, 2]]
    c_pos = pos[corners]   # [n_cells, 4, 2]
    c_phi = phi[corners]   # [n_cells, 4]

    has_pos = np.any(c_phi >= 0.0, axis=1)
    has_neg = np.any(c_phi < 0.0, axis=1)
    active = np.where(has_pos & has_neg)[0]

    total_length = 0.0
    for cid in active:
        pts = c_pos[cid]
        vals = c_phi[cid]
        crossings = []
        for e in range(4):
            v0, v1 = vals[e], vals[(e + 1) % 4]
            if (v0 >= 0.0) != (v1 >= 0.0):
                t = v0 / (v0 - v1 + 1e-15)
                crossings.append(pts[e] + t * (pts[(e + 1) % 4] - pts[e]))
        if len(crossings) == 2:
            total_length += float(np.linalg.norm(crossings[0] - crossings[1]))
        elif len(crossings) == 4:
            total_length += float(
                np.linalg.norm(crossings[0] - crossings[1])
                + np.linalg.norm(crossings[2] - crossings[3])
            )
    return total_length


def extract_all_stats(cache_dir, split_path, out_path, stride=1):
    splits = load_splits(split_path)
    all_cases = [
        (split_name, case)
        for split_name in ("train", "val", "test", "outer")
        for case in splits[split_name]
    ]

    X_rows, Y_rows, case_rows, split_rows, frame_rows = [], [], [], [], []

    for idx, (split_name, case) in enumerate(all_cases, 1):
        for e in range(N_EPOCHS):
            d = load_epoch(cache_dir, case, e)
            delta, eps_m, diff_d = map(float, d["params"])
            # Frames 0..9 in epochs 0..28, plus frame 10 (t=300) in epoch 29 -> 301 unique frames
            k_end = FRAMES_PER_EPOCH + 1 if e == N_EPOCHS - 1 else FRAMES_PER_EPOCH
            for k in range(0, k_end, stride):
                frame = FRAMES_PER_EPOCH * e + k
                phi = d["phi"][k].astype(np.float64)

                fstats = field_stats(d, phi)
                solid_area = fstats["solid_fraction"] * AREA
                tip_pos = fstats["tip_radius_mean"]
                iface_len = compute_interface_length(
                    d["pos"].astype(np.float64), d["cells"].astype(np.int64), phi
                )

                X_rows.append([delta, eps_m, diff_d, float(frame)])
                Y_rows.append([solid_area, tip_pos, iface_len])
                case_rows.append(case)
                split_rows.append(split_name)
                frame_rows.append(frame)

        if idx % 20 == 0 or idx == len(all_cases):
            print(f"Processed {idx}/{len(all_cases)} cases ({len(X_rows)} samples)")

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    np.savez_compressed(
        out_path,
        X=np.asarray(X_rows, dtype=np.float32),
        Y=np.asarray(Y_rows, dtype=np.float32),
        cases=np.asarray(case_rows),
        splits=np.asarray(split_rows),
        frames=np.asarray(frame_rows, dtype=np.int32),
    )
    print(f"Saved dataset to {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache_dir", default="cache", help="Path to unpacked P1 cache directory")
    parser.add_argument("--split_path", default=str(REPO_ROOT / "P1" / "splits.json"))
    parser.add_argument("--out_path", default=str(REPO_ROOT / "P3" / "outputs" / "stats_dataset.npz"))
    parser.add_argument("--stride", type=int, default=1, help="Frame stride (1 = all 301 frames)")
    args = parser.parse_args()
    extract_all_stats(args.cache_dir, args.split_path, args.out_path, args.stride)
