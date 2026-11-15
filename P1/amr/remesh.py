"""Remeshing: build the next adaptive mesh from a phase field and move fields between meshes.

A mesh is described by its quadtree leaves (x, y, s) in units of the finest
cell (FINEST = 500 / 192): lower-left bin and size s in {1, 2, ..., 64}.

Remesh rule (chosen in remesh.ipynb): mark every cell that lies within DISTANCE of
the phi = 0 interface; marked cells refine one level, groups of four unmarked
siblings coarsen one level; then 2:1 balance across faces and corners. Only
the interface location matters, so small errors of a predicted field in the
bulk do not change the mesh. The solver's own rule (`marking`: any nodal phi
in (-0.9999, 0.9999)) is kept for comparison.

Field transfer is exact FE_Q(3) evaluation of the old field at the new nodes:
exact on refined or unchanged cells, interpolation on coarsened cells (as in
the solver).
"""
from collections import Counter

import numpy as np
from scipy.spatial import cKDTree

from amr.data import (FINEST, N_FINEST, NODE_BOUNDARY, UNIT_SUPPORT, Q3Field, build_graph,
                        cell_geometry, interface_points, local_lattice_index)

DISTANCE = 8.0                                # refine within this distance of phi = 0
BALANCE_CORNERS = True
GL_W = np.array([1, 5, 5, 1]) / 12.0          # Gauss-Lobatto weights on [0, 1]
MAX_SIZE = 64
MESH_KEYS = ("pos", "cells", "edge_index", "node_type", "h", "weights", "params")


# ---------------------------------------------------------------- remesh

def leaves(d):
    """Quadtree leaves of a cached or generated mesh."""
    corners, widths = cell_geometry(d["pos"].astype(np.float64), d["cells"].astype(np.int64))
    bins = np.rint(corners / FINEST).astype(np.int64)
    sizes = np.rint(widths / FINEST).astype(np.int64)
    result = [(int(x), int(y), int(s)) for (x, y), s in zip(bins, sizes)]
    assert len(result) == len(set(result)), "duplicate cells"
    return result


def fill_luts(cells):
    """Owner index and cell size for every finest-level bin, [192, 192] each (row = y)."""
    owners = np.full((N_FINEST, N_FINEST), -1, dtype=np.int32)
    sizes = np.zeros_like(owners)
    for index, (x, y, s) in enumerate(cells):
        owners[y:y + s, x:x + s] = index
        sizes[y:y + s, x:x + s] = s
    assert (owners >= 0).all(), "cells do not tile the domain"
    return owners, sizes


def marking(cells_nodes, phi, threshold=0.9999):
    """Solver rule: flag a cell if any of its nodal phi is in (-threshold, threshold)."""
    local = phi[cells_nodes]
    return np.any((local > -threshold) & (local < threshold), axis=1)


def distance_marking(cells, points, distance=DISTANCE):
    """Flag a leaf if some part of it lies within `distance` of an interface point.
    Uses the centres of the finest bins (conservative by half a bin diagonal)."""
    flags = np.zeros(len(cells), dtype=bool)
    if len(points) == 0:
        return flags
    owners, _ = fill_luts(cells)
    centres = (np.arange(N_FINEST) + 0.5) * FINEST
    X, Y = np.meshgrid(centres, centres)                        # row = y, column = x
    reach = distance + FINEST / np.sqrt(2)
    dist, _ = cKDTree(points).query(np.c_[X.ravel(), Y.ravel()], distance_upper_bound=reach)
    flags[np.unique(owners.ravel()[dist <= reach])] = True
    return flags


def balance(cells, corners=BALANCE_CORNERS):
    """Split cells until neighbours differ by at most one level (2:1 rule)."""
    cells = set(cells)
    directions = [(1, 0), (0, 1)]
    if corners:
        directions += [(1, 1), (1, -1)]
    for iteration in range(8):
        ordered = sorted(cells)
        owners, sizes = fill_luts(ordered)
        split_ids = set()
        for dx, dy in directions:
            ya = slice(max(0, -dy), min(N_FINEST, N_FINEST - dy))
            xa = slice(max(0, -dx), min(N_FINEST, N_FINEST - dx))
            yb = slice(max(0, dy), min(N_FINEST, N_FINEST + dy))
            xb = slice(max(0, dx), min(N_FINEST, N_FINEST + dx))
            left, right = sizes[ya, xa], sizes[yb, xb]
            split_ids.update(np.unique(owners[ya, xa][left > 2 * right]).tolist())
            split_ids.update(np.unique(owners[yb, xb][right > 2 * left]).tolist())
        if not split_ids:
            return cells, iteration
        for index in split_ids:
            x, y, s = ordered[index]
            cells.remove((x, y, s))
            h = s // 2
            cells.update((x + ox, y + oy, h) for ox in (0, h) for oy in (0, h))
    raise RuntimeError("2:1 balancing did not converge")


def remesh(original, flags, corners=BALANCE_CORNERS):
    """Refine flagged leaves and coarsen complete groups of unflagged siblings, one level each."""
    assert len(original) == len(flags)
    current = set(original)
    refined = {cell for cell, flag in zip(original, flags) if flag and cell[2] > 1}
    coarsen = {cell for cell, flag in zip(original, flags) if not flag and cell[2] < MAX_SIZE}
    parents = set()
    for x, y, s in coarsen:
        px, py = x // (2 * s) * (2 * s), y // (2 * s) * (2 * s)
        siblings = {(px + ox, py + oy, s) for ox in (0, s) for oy in (0, s)}
        if siblings <= coarsen:
            parents.add((px, py, 2 * s))
    for px, py, s in parents:
        h = s // 2
        current.difference_update((px + ox, py + oy, h) for ox in (0, h) for oy in (0, h))
        current.add((px, py, s))
    for x, y, s in refined:
        current.remove((x, y, s))
        h = s // 2
        current.update((x + ox, y + oy, h) for ox in (0, h) for oy in (0, h))
    result, iterations = balance(current, corners)
    return result, {"marked_refine": len(refined), "coarsen_groups": len(parents),
                    "balance_iterations": iterations}


def build_mesh(cells, params, support=UNIT_SUPPORT):
    """FE_Q(3) nodes, cells, graph and lumped integration weights from quadtree leaves."""
    ordered = sorted(cells)
    bins = np.array(ordered, dtype=np.float64)
    local = (bins[:, None, :2] + bins[:, None, 2:3] * support[None]) * FINEST
    keys = np.rint(local.reshape(-1, 2) * 1e7).astype(np.int64)
    unique_keys, inverse = np.unique(keys, axis=0, return_inverse=True)
    pos = (unique_keys.astype(np.float64) / 1e7).astype(np.float32)
    cell_nodes = inverse.reshape(-1, 16).astype(np.int32)
    edges, node_type, h = build_graph(pos.astype(np.float64), cell_nodes, support)
    lattice = local_lattice_index(support)
    local_weight = GL_W[lattice[:, 0]] * GL_W[lattice[:, 1]]
    weights = np.zeros(len(pos), dtype=np.float64)
    np.add.at(weights, cell_nodes.ravel(),
              ((bins[:, 2:3] * FINEST) ** 2 * local_weight[None, :]).ravel())
    return {"pos": pos, "cells": cell_nodes, "edge_index": edges.astype(np.int32),
            "node_type": node_type.astype(np.int8), "h": h.astype(np.float32),
            "weights": weights, "params": np.asarray(params)}


def next_mesh(mesh, phi, distance=DISTANCE):
    """Remeshing step: the next mesh from the phase field phi on the current mesh."""
    own = leaves(mesh)
    points = interface_points(mesh["pos"].astype(np.float64), mesh["edge_index"], phi)
    cells, _ = remesh(own, distance_marking(own, points, distance))
    return build_mesh(cells, mesh["params"])


# ---------------------------------------------------------------- field transfer

def transfer_matrix(d_from, to_pos, support=UNIT_SUPPORT):
    """Sparse [n_to, n_from]: values on the new nodes = M @ values on the old nodes."""
    q = Q3Field(d_from["pos"].astype(np.float64), d_from["cells"].astype(np.int64), support)
    return q.interpolation_matrix(np.asarray(to_pos, dtype=np.float64), len(d_from["pos"]))


def transfer_state(d_from, state, d_to, delta=None):
    """Move a nodal state [n_from, c] onto mesh d_to; if delta is given, column 1 is u
    and is reset to the Dirichlet value -delta on the new boundary nodes."""
    out = transfer_matrix(d_from, d_to["pos"]) @ np.asarray(state, dtype=np.float64)
    if delta is not None:
        out[d_to["node_type"] == NODE_BOUNDARY, 1] = -delta
    return out


def enthalpy(w, phi, u):
    """H = integral of (u - phi / 2), conserved by the solver."""
    return float((w * (u - phi / 2)).sum())


# ---------------------------------------------------------------- mesh comparison

def compare(predicted, target):
    """Agreement of two leaf sets: exact cells and the fraction of area with the same cell size."""
    pred, truth = set(predicted), set(target)
    _, pred_sizes = fill_luts(sorted(pred))
    _, true_sizes = fill_luts(sorted(truth))
    return {
        "predicted_cells": len(pred), "true_cells": len(truth),
        "cell_recall": len(pred & truth) / len(truth),
        "area_same_level": float(np.mean(pred_sizes == true_sizes)),
        "area_coarser": float(np.mean(pred_sizes > true_sizes)),
        "area_finer": float(np.mean(pred_sizes < true_sizes)),
        "missing_by_size": dict(sorted(Counter(s for _, _, s in truth - pred).items())),
    }


def finest_coverage(cells, points):
    """Fraction of points (for example true phi = 0 points) that lie in finest cells."""
    if len(points) == 0:
        return 1.0
    _, sizes = fill_luts(sorted(cells))
    b = np.clip(np.floor(points / FINEST).astype(np.int64), 0, N_FINEST - 1)
    return float(np.mean(sizes[b[:, 1], b[:, 0]] == 1))
