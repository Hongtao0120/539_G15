import argparse
import json
import math
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))   # P1 folder
from data.amr_data import FRAMES_PER_EPOCH, N_EPOCHS, epoch_of_frame, load_epoch, load_splits   # noqa: E402
from model.networks import GraphAutoencoder, frame_sample, to_device                              # noqa: E402
from model.pipeline import frame_errors                                                           # noqa: E402

N_FRAMES = FRAMES_PER_EPOCH * N_EPOCHS + 1
UT_SCALE = 0.41                       # spread of ut over the training data
QUICK_VAL_FRAMES = (50, 150, 300)
EVAL_FRAMES = range(10, N_FRAMES, 10)


class RandomFrames(torch.utils.data.IterableDataset):
    """Endless random (case, frame) samples from the training cases."""

    def __init__(self, cache_dir, cases, seed, enc_grid):
        self.cache_dir, self.cases, self.seed, self.enc_grid = cache_dir, cases, seed, enc_grid

    def __iter__(self):
        info = torch.utils.data.get_worker_info()
        rng = np.random.default_rng([self.seed, info.id if info else 0])
        while True:
            case = self.cases[rng.integers(len(self.cases))]
            yield frame_sample(self.cache_dir, case, int(rng.integers(0, N_FRAMES)), enc_grid=self.enc_grid)


def loss_fn(phi, ut, s):
    w = s["weights"]
    band = (s["phi"].abs() < 0.95).float()
    err_phi = (phi - s["phi"]) ** 2
    return ((w * err_phi).mean() + 2.0 * (band * err_phi).sum() / band.sum().clamp_min(1.0)
            + (w * (ut - s["ut"]) ** 2).mean() / UT_SCALE ** 2)


@torch.no_grad()
def quick_validation(model, cache_dir, cases, device):
    """Worst and median phase RMSE (%) over the validation cases at a few frames."""
    model.eval()
    worst = []
    for case in cases:
        per_frame = []
        for frame in QUICK_VAL_FRAMES:
            s = to_device(frame_sample(cache_dir, case, frame, enc_grid=model.enc_grid), device)
            phi, _ = model(s)
            w = s["weights"]
            per_frame.append(100 * float(torch.sqrt((w * (phi - s["phi"]) ** 2).sum() / w.sum())))
        worst.append(max(per_frame))
    model.train()
    return float(max(worst)), float(np.median(worst))


@torch.no_grad()
def full_evaluation(model, cache_dir, cases, device):
    """Scores of encode + decode on every 10th frame of the validation cases, on the solver meshes."""
    model.eval()
    rows = []
    for case in cases:
        cached = None
        for frame in EVAL_FRAMES:
            e = epoch_of_frame(frame)
            if cached is None or cached[0] != e:
                cached = (e, load_epoch(cache_dir, case, e))
            d = cached[1]
            s = to_device(frame_sample(cache_dir, case, frame, d, model.enc_grid), device)
            phi, ut = model(s)
            k = frame - FRAMES_PER_EPOCH * e
            delta = float(d["params"][0])
            rows.append({"case": case, "frame": frame,
                         **frame_errors(d["weights"].astype(np.float64), phi.double().cpu().numpy(),
                                        ut.double().cpu().numpy(), d["phi"][k].astype(np.float64),
                                        (d["u"][k].astype(np.float64) + delta) / delta)})
    return rows


def summarize(rows):
    """Median and worst over cases of each case's worst frame."""
    out = {}
    for key in ("phi_rmse_percent", "u_rmse_over_delta_percent", "solid_area_percent"):
        worst = {}
        for r in rows:
            worst[r["case"]] = max(worst.get(r["case"], 0.0), abs(r[key]))
        out[key] = {"median": float(np.median(list(worst.values()))), "worst": float(max(worst.values())),
                    "worst_case": max(worst, key=worst.get)}
    out["solid_iou_min"] = float(min(r["solid_iou"] for r in rows))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache-dir", default="../cache")
    ap.add_argument("--split", default=os.path.join(os.path.dirname(__file__), "..", "data", "splits.json"))
    ap.add_argument("--out", default="weights_new")
    ap.add_argument("--latent-size", type=int, default=64)
    ap.add_argument("--latent-channels", type=int, default=4)
    ap.add_argument("--enc-grid", type=int, default=64)
    ap.add_argument("--dec-grid", type=int, default=128)
    ap.add_argument("--steps", type=int, default=150000)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--eval-every", type=int, default=5000)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--resume", action="store_true", help="continue from <out>/last.pt")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    torch.manual_seed(args.seed)
    device = torch.device("cuda")
    split = load_splits(args.split)

    config = {"latent_size": args.latent_size, "latent_channels": args.latent_channels,
              "enc_grid": args.enc_grid, "dec_grid": args.dec_grid}
    model = GraphAutoencoder(**config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.0)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.steps, eta_min=args.lr * 0.01)
    report = {"args": vars(args), "train_cases": len(split["train"]), "history": []}
    best, start_step = math.inf, 0
    last_path = os.path.join(args.out, "last.pt")
    if args.resume and os.path.exists(last_path):
        last = torch.load(last_path, map_location=device, weights_only=True)
        model.load_state_dict(last["model"])
        optimizer.load_state_dict(last["optimizer"])
        scheduler.load_state_dict(last["scheduler"])
        best, start_step = last["best_val"], last["step"]
        report["history"] = last["history"]
        print(f"resumed at step {start_step}", flush=True)
    with open(os.path.join(args.out, "config.json"), "w") as fh:
        json.dump({"autoencoder": config}, fh, indent=1)

    loader = torch.utils.data.DataLoader(RandomFrames(args.cache_dir, split["train"], args.seed + start_step,
                                                      args.enc_grid),
                                         batch_size=None, num_workers=args.workers, prefetch_factor=4,
                                         persistent_workers=True)
    start, running = time.time(), []
    model.train()
    for step, s in enumerate(loader, start_step + 1):
        if step > args.steps:
            break
        s = to_device(s, device)
        phi, ut = model(s)
        loss = loss_fn(phi, ut, s)
        if not torch.isfinite(loss):
            raise RuntimeError(f"non-finite loss at step {step}")
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()
        running.append(loss.item())
        if step % args.eval_every == 0 or step == args.steps:
            worst, median = quick_validation(model, args.cache_dir, split["val"], device)
            report["history"].append({"step": step, "train_loss": float(np.mean(running)),
                                      "val_phase_worst": worst, "val_phase_median": median,
                                      "seconds": time.time() - start,
                                      "peak_memory_GB": torch.cuda.max_memory_allocated() / 2 ** 30})
            running = []
            print(json.dumps(report["history"][-1]), flush=True)
            if worst < best:
                best = worst
                torch.save(model.state_dict(), os.path.join(args.out, "best.pt"))
            torch.save({"step": step, "model": model.state_dict(), "optimizer": optimizer.state_dict(),
                        "scheduler": scheduler.state_dict(), "best_val": best, "history": report["history"]},
                       last_path)

    model.load_state_dict(torch.load(os.path.join(args.out, "best.pt"), map_location=device, weights_only=True))
    torch.save(model.state_dict(), os.path.join(args.out, "autoencoder.pt"))
    report["best_quick_val_phase_worst"] = best
    report["validation"] = summarize(full_evaluation(model, args.cache_dir, split["val"], device))
    with open(os.path.join(args.out, "report.json"), "w") as fh:
        json.dump(report, fh, indent=1)
    print(json.dumps(report["validation"], indent=1), flush=True)


if __name__ == "__main__":
    main()
