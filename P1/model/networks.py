import math

import numpy as np
import torch
import torch.nn.functional as F
from scipy.spatial import cKDTree
from torch import nn

from data.amr_data import DOMAIN, FINEST, FRAMES_PER_EPOCH, epoch_of_frame, load_epoch

ENC_GRID = 64
K_NEAREST = 8
N_FRAMES = 301


# ---------------------------------------------------------------- graph inputs

def mesh_sample(mesh, phi=None, ut=None, enc_grid=ENC_GRID):
    """Graph tensors of a mesh (dict with pos, edge_index, node_type, h, weights), optionally with fields."""
    pos = mesh["pos"].astype(np.float64)
    ei = mesh["edge_index"].astype(np.int64)
    rel = pos[ei[0]] - pos[ei[1]]
    centres = (np.arange(enc_grid) + 0.5) * DOMAIN / enc_grid
    gx, gy = np.meshgrid(centres, centres)
    dist, idx = cKDTree(pos).query(np.c_[gx.ravel(), gy.ravel()], k=K_NEAREST)
    knn_w = 1.0 / (dist + 1e-6)
    knn_w /= knn_w.sum(1, keepdims=True)
    cell = np.minimum((pos / (DOMAIN / enc_grid)).astype(np.int64), enc_grid - 1)
    w = mesh["weights"].astype(np.float64)
    s = {"pos": torch.from_numpy((pos / DOMAIN).astype(np.float32)),
         "static": torch.from_numpy(np.concatenate([np.eye(3, dtype=np.float32)[mesh["node_type"].astype(np.int64)],
                                                    (np.log2(mesh["h"] / FINEST) / 6.0)[:, None].astype(np.float32)], 1)),
         "edge": torch.from_numpy((np.concatenate([rel, np.linalg.norm(rel, axis=1, keepdims=True)], 1) / 10.0).astype(np.float32)),
         "edge_index": torch.from_numpy(ei),
         "weights": torch.from_numpy((w / w.mean()).astype(np.float32)),
         "boundary": torch.from_numpy(mesh["node_type"] == 1),
         "knn_idx": torch.from_numpy(idx.astype(np.int64)),
         "knn_w": torch.from_numpy(knn_w.astype(np.float32)),
         "grid_cell": torch.from_numpy(cell[:, 1] * enc_grid + cell[:, 0])}
    if phi is not None:
        s["phi"] = torch.from_numpy(np.asarray(phi, dtype=np.float32))
        s["ut"] = torch.from_numpy(np.asarray(ut, dtype=np.float32))
    return s


def frame_sample(cache_dir, case, frame, d=None, enc_grid=ENC_GRID):
    """One stored frame on its own solver mesh."""
    e = epoch_of_frame(frame)
    d = d if d is not None else load_epoch(cache_dir, case, e)
    k = frame - FRAMES_PER_EPOCH * e
    delta = float(d["params"][0])
    return mesh_sample(d, d["phi"][k], (d["u"][k] + delta) / delta, enc_grid)


def to_device(s, device):
    return {k: v.to(device) for k, v in s.items()}


# ---------------------------------------------------------------- graph autoencoder

def mlp(n_in, hidden, n_out, norm=True):
    layers = [nn.Linear(n_in, hidden), nn.SiLU(), nn.Linear(hidden, hidden), nn.SiLU(), nn.Linear(hidden, n_out)]
    if norm:
        layers.append(nn.LayerNorm(n_out))
    return nn.Sequential(*layers)


class MPBlock(nn.Module):
    def __init__(self, hidden):
        super().__init__()
        self.edge = mlp(3 * hidden, hidden, hidden)
        self.node = mlp(2 * hidden, hidden, hidden)

    def forward(self, h, e, sender, receiver):
        e = e + self.edge(torch.cat([e, h[sender], h[receiver]], dim=1))
        h = h + self.node(torch.cat([h, torch.zeros_like(h).index_add_(0, receiver, e)], dim=1))
        return h, e


class GraphAutoencoder(nn.Module):
    def __init__(self, latent_size=64, latent_channels=4, hidden=128, enc_layers=4, dec_layers=6,
                 dec_grid=128, n_freq=8, enc_grid=ENC_GRID):
        super().__init__()
        self.enc_grid = enc_grid
        self.register_buffer("freqs", 2.0 ** torch.arange(n_freq) * math.pi)
        self.enc_node, self.enc_edge = mlp(6 + 4 * n_freq, hidden, hidden), mlp(3, hidden, hidden)
        self.enc_blocks = nn.ModuleList(MPBlock(hidden) for _ in range(enc_layers))
        convs = []
        for _ in range(int(round(math.log2(enc_grid // latent_size)))):
            convs += [nn.Conv2d(hidden, hidden, 3, stride=2, padding=1), nn.SiLU()]
        self.enc_cnn = nn.Sequential(*convs, nn.Conv2d(hidden, latent_channels, 3, padding=1))
        ups = [nn.Conv2d(latent_channels, hidden, 3, padding=1), nn.SiLU()]
        for _ in range(int(round(math.log2(dec_grid // latent_size)))):
            ups += [nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
                    nn.Conv2d(hidden, hidden, 3, padding=1), nn.SiLU()]
        self.dec_cnn = nn.Sequential(*ups)
        self.dec_node = mlp(hidden + 4 + 4 * n_freq, hidden, hidden)
        self.dec_edge = mlp(3, hidden, hidden)
        self.dec_blocks = nn.ModuleList(MPBlock(hidden) for _ in range(dec_layers))
        self.head = mlp(hidden, hidden, 2, norm=False)

    def fourier(self, p):
        ang = p[:, :, None] * self.freqs
        return torch.cat([torch.sin(ang), torch.cos(ang)], -1).reshape(len(p), -1)

    def encode(self, s):
        x = torch.cat([s["phi"][:, None], s["ut"][:, None], s["static"], self.fourier(s["pos"])], 1)
        h, e = self.enc_node(x), self.enc_edge(s["edge"])
        for block in self.enc_blocks:
            h, e = block(h, e, s["edge_index"][0], s["edge_index"][1])
        n_cells = self.enc_grid * self.enc_grid
        total = torch.zeros(n_cells, h.shape[1], device=h.device, dtype=h.dtype).index_add_(0, s["grid_cell"], h)
        count = torch.bincount(s["grid_cell"], minlength=n_cells).to(h.dtype)[:, None]
        nearest = (h[s["knn_idx"]] * s["knn_w"][..., None]).sum(1)
        g = torch.where(count > 0, total / count.clamp_min(1.0), nearest)
        return self.enc_cnn(g.T.reshape(1, -1, self.enc_grid, self.enc_grid))

    def decode(self, z, s):
        grid = self.dec_cnn(z)
        p = s["pos"]
        feat = F.grid_sample(grid, (p * 2 - 1)[None, :, None, :], mode="bilinear", align_corners=False)[0, :, :, 0].T
        h = self.dec_node(torch.cat([feat, s["static"], self.fourier(p)], 1))
        e = self.dec_edge(s["edge"])
        for block in self.dec_blocks:
            h, e = block(h, e, s["edge_index"][0], s["edge_index"][1])
        out = self.head(h)
        return torch.tanh(out[:, 0]), out[:, 1].masked_fill(s["boundary"], 0.0)

    def forward(self, s):
        return self.decode(self.encode(s), s)


class ResBlock(nn.Module):
    def __init__(self, ch, dilation):
        super().__init__()
        self.net = nn.Sequential(nn.GroupNorm(8, ch), nn.SiLU(), nn.Conv2d(ch, ch, 3, padding=dilation, dilation=dilation),
                                 nn.GroupNorm(8, ch), nn.SiLU(), nn.Conv2d(ch, ch, 3, padding=1))

    def forward(self, x):
        return x + self.net(x)


class LatentStepper(nn.Module):
    def __init__(self, channels, hidden=128, blocks=8):
        super().__init__()
        self.inp = nn.Conv2d(channels + 3, hidden, 3, padding=1)
        self.blocks = nn.Sequential(*[ResBlock(hidden, d) for d in ([1, 2, 4, 8] * math.ceil(blocks / 4))[:blocks]])
        self.out = nn.Conv2d(hidden, channels, 3, padding=1)
        nn.init.zeros_(self.out.weight)
        nn.init.zeros_(self.out.bias)

    def forward(self, z, p):
        maps = p[:, :, None, None].expand(-1, -1, z.shape[2], z.shape[3])
        return z + self.out(self.blocks(self.inp(torch.cat([z, maps], 1))))


@torch.no_grad()
def latent_rollout(stepper, z0, p, steps=N_FRAMES - 1):
    stepper.eval()
    zs = [z0]
    for _ in range(steps):
        zs.append(stepper(zs[-1], p))
    return torch.cat(zs)
