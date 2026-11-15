"""Final pipeline (model <-> remeshing loop): from (delta, epsilonM, D) and the initial seed to t = 300.

1. The model encodes the initial frame (the seed in the undercooled melt) into the
   latent grid and advances it frame by frame with the latent stepper.
2. At the end of every 10-frame epoch, the model decodes the latent on the current
   mesh; remeshing builds the next mesh from that phase field and the loop continues
   on the new mesh. Only these 30 decodes are needed, so the timing is the
   real online cost.
3. Every decoded frame is scored against the solver: solid IoU, phase and
   temperature errors, solid-area error and arm-tip position error; summary
   statistics (solid fraction, mean tip radius) are kept for the trust check.
"""
import math
import time

import numpy as np
import torch

from amr.model import GraphAutoencoder, LatentStepper, frame_sample, latent_rollout, mesh_sample, to_device
from amr.remesh import MESH_KEYS, next_mesh, transfer_matrix
from amr.data import FRAMES_PER_EPOCH, N_EPOCHS, interface_points, load_epoch, norm_params, state_frames

CENTRE = 250.0
THETA0 = 0.125                                        # crystal orientation (rad)
ARM_ANGLES = THETA0 + np.arange(4) * math.pi / 2
AREA = 500.0 * 500.0


def frame_errors(w, phi_p, ut_p, phi_t, ut_t):
    """Phase RMSE (%), u RMSE / delta (%), solid IoU and relative solid-area error (%).
    ut is the lifted temperature (u + delta) / delta, so the ut RMSE equals u RMSE / delta."""
    area = w.sum()
    p, t = phi_p > 0, phi_t > 0
    union = w[p | t].sum()
    solid_t = (w * (1 + phi_t) / 2).sum()
    return {"phi_rmse_percent": 100 * float(np.sqrt((w * (phi_p - phi_t) ** 2).sum() / area)),
            "u_rmse_over_delta_percent": 100 * float(np.sqrt((w * (ut_p - ut_t) ** 2).sum() / area)),
            "solid_iou": float(w[p & t].sum() / union) if union > 0 else 1.0,
            "solid_area_percent": 100 * float(((w * (1 + phi_p) / 2).sum() - solid_t) / solid_t)}


def arm_tips(pos, edge_index, phi):
    """Tip of each of the 4 arms: the phi = 0 point farthest along the arm direction
    within +-45 degrees of it. Returns [4, 2]; NaN rows for arms without interface points."""
    points = interface_points(np.asarray(pos, dtype=np.float64), edge_index, phi)
    tips = np.full((4, 2), np.nan)
    if len(points) == 0:
        return tips
    v = points - CENTRE
    angles = np.arctan2(v[:, 1], v[:, 0])
    for k, angle in enumerate(ARM_ANGLES):
        sector = np.abs(np.angle(np.exp(1j * (angles - angle)))) <= math.pi / 4
        if sector.any():
            direction = np.array([math.cos(angle), math.sin(angle)])
            tips[k] = points[sector][np.argmax(v[sector] @ direction)]
    return tips


def field_stats(mesh, phi):
    """Summary statistics of one frame for the trust check: solid fraction and mean tip radius."""
    w = mesh["weights"].astype(np.float64)
    tips = arm_tips(mesh["pos"], mesh["edge_index"], phi)
    radii = np.hypot(tips[:, 0] - CENTRE, tips[:, 1] - CENTRE)
    return {"solid_fraction": float((w * (1 + phi) / 2).sum() / AREA),
            "tip_radius_mean": float(np.nanmean(radii)) if np.isfinite(radii).any() else float("nan")}


def load_model(model_dir, device, latent_size=32, latent_channels=4):
    """Trained model: autoencoder, stepper and latent normalization from model_dir."""
    ae = GraphAutoencoder(latent_size, latent_channels).to(device).eval()
    ae.load_state_dict(torch.load(f"{model_dir}/autoencoder.pt", map_location=device, weights_only=True))
    stepper = LatentStepper(latent_channels).to(device).eval()
    stepper.load_state_dict(torch.load(f"{model_dir}/stepper.pt", map_location=device, weights_only=True))
    norm = torch.load(f"{model_dir}/latent_normalization.pt", map_location=device, weights_only=True)
    return ae, stepper, norm["mean"].to(device), norm["std"].to(device)


@torch.no_grad()
def run_case(ae, stepper, mean, std, cache_dir, case, device):
    """Autonomous rollout on the meshes built by remeshing; one row per decoded frame (10, 20, ..., 300) and timings."""
    timing = {"latent": 0.0, "decode": 0.0, "remesh": 0.0}
    t0 = time.perf_counter()
    d0 = load_epoch(cache_dir, case, 0)
    p = torch.from_numpy(norm_params(d0["params"]).astype(np.float32))[None].to(device)
    z0 = (ae.encode(to_device(frame_sample(cache_dir, case, 0, d0), device)) - mean) / std
    zs = latent_rollout(stepper, z0, p) * std + mean
    if device.type == "cuda":
        torch.cuda.synchronize()
    timing["latent"] = time.perf_counter() - t0
    rows, mesh = [], {k: d0[k] for k in MESH_KEYS}
    for e in range(N_EPOCHS):
        d = load_epoch(cache_dir, case, e)
        frame = FRAMES_PER_EPOCH * (e + 1)
        t1 = time.perf_counter()
        phi, ut = ae.decode(zs[frame][None], to_device(mesh_sample(mesh), device))
        phi, ut = phi.double().cpu().numpy(), ut.double().cpu().numpy()
        timing["decode"] += time.perf_counter() - t1
        # The solver's frame 10e + 10 (stored as index 10 of epoch e), moved onto the pipeline mesh.
        truth = transfer_matrix(d, mesh["pos"]) @ state_frames(d)[FRAMES_PER_EPOCH].astype(np.float64)
        tips_p = arm_tips(mesh["pos"], mesh["edge_index"], phi)
        tips_t = arm_tips(mesh["pos"], mesh["edge_index"], truth[:, 0])
        rows.append({"case": case, "frame": frame, "nodes": len(phi),
                     **frame_errors(mesh["weights"], phi, ut, truth[:, 0], truth[:, 1]),
                     "tip_position_error_max": float(np.nanmax(np.linalg.norm(tips_p - tips_t, axis=1))),
                     "stats_pred": field_stats(mesh, phi), "stats_true": field_stats(mesh, truth[:, 0])})
        if e < N_EPOCHS - 1:
            t2 = time.perf_counter()
            mesh = next_mesh(mesh, phi)
            timing["remesh"] += time.perf_counter() - t2
    timing["total"] = time.perf_counter() - t0
    return rows, timing
