"""Data access: the AMR cache, case splits, and the FE_Q(3) mesh and graph.

Cache layout: <cache_dir>/<campaign>/<case>/epoch_XXX.npz, epochs 0..29.
Epoch e holds the mesh of frames 10e .. 10e+9 and frame 10e+10 evaluated
exactly on that mesh, so phi and u have shape [11, nodes].

Mesh: a quadtree of square cells (3 x 3 base cells on a 500 x 500 domain,
levels 0..6), each carrying a FE_Q(3) element on Gauss-Lobatto points, so a
nodal value is the field value at that node (16 nodes per cell).

State convention: phi in [-1, 1] and the lifted temperature
ut = (u + delta) / delta, which is exactly 0 on the Dirichlet boundary.
"""
import json
import os

import numpy as np
import scipy.sparse as sp

DOMAIN = 500.0
FINEST = DOMAIN / 3 / 2**6          # finest cell width, 2.604
N_FINEST = 3 * 2**6                 # finest cells per side, 192
N_EPOCHS = 30
FRAMES_PER_EPOCH = 10
PARAM_BOX = np.array([[0.60, 0.90], [0.0375, 0.0625], [0.5, 1.5]])   # delta, epsilonM, D (training box)
NODE_INTERIOR, NODE_BOUNDARY, NODE_HANGING = 0, 1, 2

GL_1D = np.array([0.0, (1 - 1 / np.sqrt(5)) / 2, (1 + 1 / np.sqrt(5)) / 2, 1.0])
# Positions of the 16 local nodes in the unit cell, in the solver's (deal.II FE_Q(3)) order:
# 4 vertices, then the inner nodes of the edges x = 0, x = 1, y = 0, y = 1, then the 4 interior nodes.
_LOCAL_IJ = [(0, 0), (3, 0), (0, 3), (3, 3), (0, 1), (0, 2), (3, 1), (3, 2),
             (1, 0), (2, 0), (1, 3), (2, 3), (1, 1), (2, 1), (1, 2), (2, 2)]
UNIT_SUPPORT = GL_1D[np.array(_LOCAL_IJ)]                              # [16, 2]


# ---------------------------------------------------------------- cases and cache

def campaign_of(case):
    if case.startswith("outer"):
        return "lhs20_outer_q1"
    if case.startswith("hd_"):
        return "lhs20_highdelta_q1"      # extra high-undercooling training cases
    return "lhs100_q1"


def load_splits(path="splits.json"):
    """Case lists: 100 train (80 main + 20 high-delta), 10 val, 10 test, 20 outer."""
    with open(path) as fh:
        split = json.load(fh)
    sizes = {k: len(split[k]) for k in ("train", "val", "test", "outer")}
    assert sizes == {"train": 100, "val": 10, "test": 10, "outer": 20}, sizes
    return split


def load_epoch(cache_dir, case, e):
    path = os.path.join(cache_dir, campaign_of(case), case, f"epoch_{e:03d}.npz")
    return dict(np.load(path))


def epoch_of_frame(frame):
    """Epoch whose own mesh carries this frame (frame 300 lives on epoch 29 as index 10)."""
    return min(frame // FRAMES_PER_EPOCH, N_EPOCHS - 1)


def state_frames(d):
    """[11, n, 2] array of (phi, ut) for the 11 frames stored in one epoch."""
    delta = d["params"][0]
    return np.stack([d["phi"], (d["u"] + delta) / delta], -1).astype(np.float32)


def norm_params(params):
    """(delta, epsilonM, D) scaled to [-1, 1] over the training box."""
    lo, hi = PARAM_BOX[:, 0], PARAM_BOX[:, 1]
    return 2 * (np.asarray(params) - lo) / (hi - lo) - 1


def interface_points(pos, edge_index, phi):
    """phi = 0 crossings along graph edges, linear in phi along each edge: [m, 2]."""
    a, b = edge_index
    cross = (phi[a] > 0) & (phi[b] <= 0)
    a, b = a[cross], b[cross]
    t = (phi[a] / (phi[a] - phi[b]))[:, None]
    return pos[a] + t * (pos[b] - pos[a])


# ---------------------------------------------------------------- mesh and graph

def local_lattice_index(unit_support_points=UNIT_SUPPORT):
    """(i, j) position of each local node in the 4 x 4 Gauss-Lobatto lattice."""
    ij = np.abs(unit_support_points[:, :, None] - GL_1D[None, None, :]).argmin(-1)
    assert np.allclose(GL_1D[ij], unit_support_points)
    return ij                                            # [16, 2]


def _lattice_pairs(ij):
    """Local node pairs of the 4 x 4 lattice: axis neighbours and both diagonals (42 pairs)."""
    pos = {tuple(p): k for k, p in enumerate(ij)}
    pairs = []
    for (i, j), k in pos.items():
        for di, dj in ((1, 0), (0, 1), (1, 1)):
            if (i + di, j + dj) in pos:
                pairs.append((k, pos[(i + di, j + dj)]))
        if (i + 1, j - 1) in pos:
            pairs.append((k, pos[(i + 1, j - 1)]))
    return np.array(pairs)


def cell_geometry(coordinates, cells):
    """Lower-left corner and width of each cell (local nodes 0 and 3 are opposite corners)."""
    x0 = coordinates[cells[:, 0]]
    x1 = coordinates[cells[:, 3]]
    width = x1[:, 0] - x0[:, 0]
    assert np.allclose(width, x1[:, 1] - x0[:, 1], atol=1e-3), "cells must be squares"
    return x0, width


def hanging_nodes(coordinates, cells, ij):
    """Nodes lying strictly inside an edge of a cell they do not belong to, and the 4 nodes of
    that edge. Includes a coarse cell's own edge nodes that lie inside the shorter edge of a fine
    neighbour; the graph uses all of them as cross-level links."""
    key = np.round(coordinates / FINEST * 1e6).astype(np.int64)
    by_line = [{}, {}]   # by_line[axis][constant coordinate key] -> (sorted varying keys, node ids)
    for axis in (0, 1):
        const, vary = key[:, 1 - axis], key[:, axis]
        order = np.lexsort((vary, const))
        c_sorted = const[order]
        starts = np.r_[0, np.flatnonzero(np.diff(c_sorted)) + 1, len(order)]
        for a, b in zip(starts[:-1], starts[1:]):
            ids = order[a:b]
            by_line[axis][c_sorted[a]] = (vary[ids], ids)

    hang, masters = [], []
    for axis, fixed in ((0, 0), (0, 3), (1, 0), (1, 3)):          # the four edges of a cell
        sel = ij[:, 1 - axis] == fixed
        loc = np.flatnonzero(sel)[np.argsort(ij[sel, axis])]
        edge_nodes = cells[:, loc]                                  # [n_cells, 4]
        lo = key[edge_nodes[:, 0], axis]
        hi = key[edge_nodes[:, 3], axis]
        cst = key[edge_nodes[:, 0], 1 - axis]
        for c in range(len(cells)):
            line = by_line[axis].get(cst[c])
            if line is None:
                continue
            vary, ids = line
            a = np.searchsorted(vary, lo[c], side="right")
            b = np.searchsorted(vary, hi[c], side="left")
            for n in ids[a:b]:
                if n not in edge_nodes[c]:
                    hang.append(n)
                    masters.append(edge_nodes[c])
    if not hang:
        return np.zeros(0, np.int64), np.zeros((0, 4), np.int64)
    return np.array(hang), np.array(masters)


def build_graph(coordinates, cells, unit_support_points=UNIT_SUPPORT):
    """Graph of one mesh: edges (both directions, no duplicates), node types and node cell size.

    Edges inside a cell link lattice neighbours along x, y and both diagonals; each hanging
    node is also linked to the 4 nodes of the edge it lies on.
    """
    ij = local_lattice_index(unit_support_points)
    e = cells[:, _lattice_pairs(ij)].reshape(-1, 2)
    hang, masters = hanging_nodes(coordinates, cells, ij)
    if len(hang):
        e = np.concatenate([e, np.stack([np.repeat(hang, 4), masters.ravel()], 1)])
    e = np.concatenate([e, e[:, ::-1]])
    e = np.unique(e, axis=0)
    e = e[e[:, 0] != e[:, 1]]

    n = len(coordinates)
    node_type = np.full(n, NODE_INTERIOR, np.int64)
    on_bnd = np.any((coordinates <= 1e-9) | (coordinates >= DOMAIN - 1e-9), axis=1)
    node_type[np.unique(hang)] = NODE_HANGING
    node_type[on_bnd] = NODE_BOUNDARY
    _, width = cell_geometry(coordinates, cells)
    h = np.full(n, np.inf)
    np.minimum.at(h, cells.ravel(), np.repeat(width, cells.shape[1]))
    return e.T.copy(), node_type, h                    # edge_index [2, n_edges]


class Q3Field:
    """Exact evaluation of the degree-3 field of one mesh at arbitrary points."""

    def __init__(self, coordinates, cells, unit_support_points=UNIT_SUPPORT):
        self.cells = cells
        self.ij = local_lattice_index(unit_support_points)
        self.x0, self.width = cell_geometry(coordinates, cells)
        lut = np.full((N_FINEST, N_FINEST), -1, np.int64)    # every cell covers whole finest bins
        b0 = np.rint(self.x0 / FINEST).astype(np.int64)
        nb = np.rint(self.width / FINEST).astype(np.int64)
        for c in range(len(cells)):
            lut[b0[c, 1]:b0[c, 1] + nb[c], b0[c, 0]:b0[c, 0] + nb[c]] = c
        assert (lut >= 0).all(), "cells do not tile the domain"
        self.lut = lut

    def locate(self, points):
        b = np.clip(np.floor(points / FINEST).astype(np.int64), 0, N_FINEST - 1)
        cell = self.lut[b[:, 1], b[:, 0]]
        xi = (points - self.x0[cell]) / self.width[cell][:, None]
        return cell, np.clip(xi, 0.0, 1.0)

    @staticmethod
    def lagrange(t):
        """1D Lagrange basis on GL_1D at points t, shape [len(t), 4]."""
        L = np.ones((len(t), 4))
        for a in range(4):
            for b in range(4):
                if a != b:
                    L[:, a] *= (t - GL_1D[b]) / (GL_1D[a] - GL_1D[b])
        return L

    def interpolation_matrix(self, points, n_nodes):
        """Sparse M [n_points, n_nodes] with field(points) = M @ nodal values."""
        cell, xi = self.locate(points)
        Lx, Ly = self.lagrange(xi[:, 0]), self.lagrange(xi[:, 1])
        vals = Lx[:, self.ij[:, 0]] * Ly[:, self.ij[:, 1]]
        rows = np.repeat(np.arange(len(points)), 16)
        return sp.csr_matrix((vals.ravel(), (rows, self.cells[cell].ravel())),
                             shape=(len(points), n_nodes))
