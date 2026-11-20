"""Convert the raw amr_cloud.h5 files into the .npz cache read by load_epoch.

One file per (case, epoch): cache/<campaign>/<case>/epoch_XXX.npz holds the mesh and
graph of epoch e and the frames 10e .. 10e+10 on that mesh. Frame 10e+10 is stored by
the solver on the next mesh, so it is evaluated exactly (FE_Q(3)) at the nodes of mesh e.

    python make_cache.py --raw-dir <folder> [<folder> ...] --cache-dir ../../cache [--jobs 8]

Each raw folder and all its subfolders are searched for <case>/amr_cloud.h5 (for example
<raw-dir>/lhs100_q1/cases/lhs_000/amr_cloud.h5) or for <case>.h5. The output goes to
the campaign folder of each case, whatever folder the .h5 file came from.
"""
import argparse
import csv
import glob
import os
import sys
from multiprocessing import Pool

import h5py
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))   # P1 folder
from data.amr_data import FRAMES_PER_EPOCH, N_EPOCHS, Q3Field, build_graph, campaign_of   # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))


def find_h5(raw_dirs, case):
    hits = []
    for raw_dir in raw_dirs:
        hits += (glob.glob(os.path.join(raw_dir, "**", case, "amr_cloud.h5"), recursive=True)
                 + glob.glob(os.path.join(raw_dir, "**", f"{case}.h5"), recursive=True))
    return hits[0] if hits else None


def epoch_arrays(f, e):
    g = f[f"epoch_{e:03d}"]
    return (g["coordinates"][:], g["cell_dof_indices"][:].astype(np.int64),
            g["frames"][:], g["phi"][:], g["u"][:], g["weights"][:])


def convert_case(job):
    h5_path, case, cache_dir = job
    out_dir = os.path.join(cache_dir, campaign_of(case), case)
    if os.path.exists(os.path.join(out_dir, "done")):
        return case
    os.makedirs(out_dir, exist_ok=True)
    with h5py.File(h5_path, "r") as f:
        usp = f["unit_support_points"][:]
        a = f["parameters"].attrs
        params = np.array([a["delta"], a["epsilonM"], a["D"]])
        nxt = epoch_arrays(f, 0)
        for e in range(N_EPOCHS):
            xy, cells, frames, phi, u, w = nxt
            nxt = epoch_arrays(f, e + 1)
            assert list(frames) == list(range(FRAMES_PER_EPOCH * e, FRAMES_PER_EPOCH * (e + 1)))
            assert nxt[2][0] == FRAMES_PER_EPOCH * (e + 1)
            M = Q3Field(nxt[0], nxt[1], usp).interpolation_matrix(xy, len(nxt[0]))
            phi_end, u_end = (M @ nxt[3][0])[None], (M @ nxt[4][0])[None]
            edge_index, node_type, h = build_graph(xy, cells, usp)
            np.savez(os.path.join(out_dir, f"epoch_{e:03d}.npz"),
                     pos=xy.astype(np.float32), cells=cells.astype(np.int32),
                     edge_index=edge_index.astype(np.int32), node_type=node_type.astype(np.int8),
                     h=h.astype(np.float32), weights=w,
                     phi=np.concatenate([phi, phi_end]).astype(np.float32),
                     u=np.concatenate([u, u_end]).astype(np.float32),
                     params=params, first_frame=FRAMES_PER_EPOCH * e)
    open(os.path.join(out_dir, "done"), "w").close()
    return case


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw-dir", nargs="+", required=True,
                    help="one or more folders with the amr_cloud.h5 files (searched with all subfolders)")
    ap.add_argument("--cache-dir", default=os.path.join(HERE, "..", "..", "cache"))
    ap.add_argument("--cases", nargs="*", default=None, help="default: every case in cases.csv")
    ap.add_argument("--jobs", type=int, default=4)
    args = ap.parse_args()
    if args.cases is None:
        with open(os.path.join(HERE, "cases.csv")) as fh:
            args.cases = [r["case"] for r in csv.DictReader(fh)]
    jobs, missing = [], []
    for case in args.cases:
        path = find_h5(args.raw_dir, case)
        (jobs.append((path, case, args.cache_dir)) if path else missing.append(case))
    if missing:
        print(f"no .h5 file found for {len(missing)} cases: {' '.join(missing)}", flush=True)
    with Pool(args.jobs) as pool:
        for i, case in enumerate(pool.imap_unordered(convert_case, jobs), 1):
            print(f"[{i}/{len(jobs)}] {case}", flush=True)


if __name__ == "__main__":
    main()
