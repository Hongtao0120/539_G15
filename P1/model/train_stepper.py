import argparse
import json
import math
import os
import sys
import time

import numpy as np
import torch
import torch.utils.checkpoint

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))   # P1 folder
from data.amr_data import epoch_of_frame, load_epoch, load_splits, norm_params    # noqa: E402
from model.networks import GraphAutoencoder, LatentStepper, frame_sample, latent_rollout, to_device   # noqa: E402

N_FRAMES = 301


@torch.no_grad()
def encode_cases(model, cache_dir, cases, device):
    """Latents [cases, 301, C, L, L] and scaled parameters [cases, 3]."""
    latents, params = [], []
    for case in cases:
        zs, cached = [], None
        for frame in range(N_FRAMES):
            e = epoch_of_frame(frame)
            if cached is None or cached[0] != e:
                cached = (e, load_epoch(cache_dir, case, e))
            s = to_device(frame_sample(cache_dir, case, frame, cached[1], model.enc_grid), device)
            zs.append(model.encode(s)[0].cpu())
        latents.append(torch.stack(zs))
        params.append(norm_params(cached[1]["params"]).astype(np.float32))
        print(f"encoded {case}", flush=True)
    return torch.stack(latents), torch.from_numpy(np.stack(params))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache-dir", default="../cache")
    ap.add_argument("--split", default=os.path.join(os.path.dirname(__file__), "..", "data", "splits.json"))
    ap.add_argument("--model-dir", default="weights_new")
    ap.add_argument("--steps", type=int, default=100000)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--max-unroll", type=int, default=32)
    ap.add_argument("--noise", type=float, default=0.01)
    ap.add_argument("--hidden", type=int, default=128)
    ap.add_argument("--blocks", type=int, default=8)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--eval-every", type=int, default=5000)
    ap.add_argument("--grad-checkpoint", action=argparse.BooleanOptionalAction, default=True,
                    help="recompute each step's activations in the backward pass to save GPU memory")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--resume", action="store_true", help="continue from <model-dir>/stepper_last.pt")
    args = ap.parse_args()
    torch.manual_seed(args.seed)
    device = torch.device("cuda")
    split = load_splits(args.split)
    out = args.model_dir

    with open(os.path.join(out, "config.json")) as fh:
        config = json.load(fh)
    ae = GraphAutoencoder(**config["autoencoder"]).to(device).eval()
    ae.load_state_dict(torch.load(os.path.join(out, "autoencoder.pt"), map_location=device, weights_only=True))
    for p in ae.parameters():
        p.requires_grad_(False)

    latent_path = os.path.join(out, "latents.pt")
    if os.path.exists(latent_path):
        data = torch.load(latent_path, weights_only=True)
    else:
        Ztr, Ptr = encode_cases(ae, args.cache_dir, split["train"], device)
        Zva, Pva = encode_cases(ae, args.cache_dir, split["val"], device)
        data = {"Ztr": Ztr, "Ptr": Ptr, "Zva": Zva, "Pva": Pva}
        torch.save(data, latent_path)
    mean = data["Ztr"].mean(dim=(0, 1, 3, 4), keepdim=True)[0]                     # [1, C, 1, 1]
    std = data["Ztr"].std(dim=(0, 1, 3, 4), keepdim=True)[0].clamp_min(1e-6)
    torch.save({"mean": mean, "std": std}, os.path.join(out, "latent_normalization.pt"))
    Z = ((data["Ztr"] - mean) / std).to(device)                                     # [cases, 301, C, L, L]
    Zva = ((data["Zva"] - mean) / std).to(device)
    P, Pva = data["Ptr"].to(device), data["Pva"].to(device)

    stepper_args = {"channels": config["autoencoder"].get("latent_channels", 4), "hidden": args.hidden,
                    "blocks": args.blocks}
    stepper = LatentStepper(**stepper_args).to(device)
    opt = torch.optim.AdamW(stepper.parameters(), lr=args.lr, weight_decay=0.0)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, args.steps, eta_min=args.lr * 0.01)
    history, best_val, start_step = [], math.inf, 0
    last_path = os.path.join(out, "stepper_last.pt")
    if args.resume and os.path.exists(last_path):
        last = torch.load(last_path, map_location=device, weights_only=True)
        stepper.load_state_dict(last["stepper"])
        opt.load_state_dict(last["optimizer"])
        sched.load_state_dict(last["scheduler"])
        history, best_val, start_step = last["history"], last["best_val"], last["step"]
        print(f"resumed at step {start_step}", flush=True)

    def validate():
        errs = []
        for i in range(len(Zva)):
            zs = latent_rollout(stepper, Zva[i, :1], Pva[i:i + 1])
            errs.append(float((zs - Zva[i]).square().mean().sqrt()))
        stepper.train()
        return float(np.max(errs))

    rng = np.random.default_rng([args.seed, start_step])
    start = time.time()
    stepper.train()
    for step in range(start_step + 1, args.steps + 1):
        unroll = int(min(args.max_unroll, 1 + (args.max_unroll - 1) * step / (0.5 * args.steps)))
        ci = torch.from_numpy(rng.integers(len(Z), size=args.batch)).to(device)
        t0 = torch.from_numpy(rng.integers(N_FRAMES - unroll, size=args.batch)).to(device)
        z = Z[ci, t0]
        z = z + args.noise * torch.randn_like(z)
        p = P[ci]
        loss = 0.0
        for k in range(1, unroll + 1):
            if args.grad_checkpoint:
                z = torch.utils.checkpoint.checkpoint(stepper, z, p, use_reentrant=False)
            else:
                z = stepper(z, p)
            loss = loss + (z - Z[ci, t0 + k]).square().mean()
        loss = loss / unroll
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(stepper.parameters(), 1.0)
        opt.step()
        sched.step()
        if step % 1000 == 0:
            row = {"step": step, "loss": loss.item(), "unroll": unroll, "seconds": time.time() - start,
                   "peak_memory_GB": torch.cuda.max_memory_allocated() / 2 ** 30}
            if step % args.eval_every == 0 or step == args.steps:
                row["val_latent_rollout_rmse"] = validate()
                if row["val_latent_rollout_rmse"] < best_val:
                    best_val = row["val_latent_rollout_rmse"]
                    torch.save(stepper.state_dict(), os.path.join(out, "stepper_best.pt"))
                history.append(row)
                torch.save({"step": step, "stepper": stepper.state_dict(), "optimizer": opt.state_dict(),
                            "scheduler": sched.state_dict(), "best_val": best_val, "history": history},
                           last_path)
            else:
                history.append(row)
            print(json.dumps(row), flush=True)

    best_path = os.path.join(out, "stepper_best.pt")
    state = torch.load(best_path, map_location=device, weights_only=True) if os.path.exists(best_path) \
        else stepper.state_dict()
    torch.save(state, os.path.join(out, "stepper.pt"))
    config["stepper"] = stepper_args
    with open(os.path.join(out, "config.json"), "w") as fh:
        json.dump(config, fh, indent=1)
    with open(os.path.join(out, "stepper_report.json"), "w") as fh:
        json.dump({"args": vars(args), "train_cases": len(split["train"]), "best_val_latent_rollout_rmse": best_val,
                   "history": history}, fh, indent=1)
    print(f"best validation latent rollout RMSE {best_val:.4f}", flush=True)


if __name__ == "__main__":
    main()
