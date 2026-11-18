import numpy as np
from scipy.spatial import cKDTree

from data.amr_data import (FINEST, N_FINEST, NODE_BOUNDARY, UNIT_SUPPORT, Q3Field, build_graph,
                        cell_geometry, interface_points, local_lattice_index)

DISTANCE = 8.0                                # refine within this distance of phi = 0
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


def distance_marking(cells, points):
    """Flag a leaf if some part of it lies within DISTANCE of an interface point.
    Uses the centres of the finest bins (conservative by half a bin diagonal)."""
    flags = np.zeros(len(cells), dtype=bool)
    if len(points) == 0:
        return flags
    owners, _ = fill_luts(cells)
    centres = (np.arange(N_FINEST) + 0.5) * FINEST
    X, Y = np.meshgrid(centres, centres)                        # row = y, column = x
    reach = DISTANCE + FINEST / np.sqrt(2)
    dist, _ = cKDTree(points).query(np.c_[X.ravel(), Y.ravel()], distance_upper_bound=reach)
    flags[np.unique(owners.ravel()[dist <= reach])] = True
    return flags


def balance(cells):
    """Split cells until neighbours (across faces and corners) differ by at most one level (2:1 rule)."""
    cells = set(cells)
    directions = [(1, 0), (0, 1), (1, 1), (1, -1)]
    for _ in range(8):
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
            return cells
        for index in split_ids:
            x, y, s = ordered[index]
            cells.remove((x, y, s))
            h = s // 2
            cells.update((x + ox, y + oy, h) for ox in (0, h) for oy in (0, h))
    raise RuntimeError("2:1 balancing did not converge")


def remesh(original, flags):
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
    return balance(current)


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


def next_mesh(mesh, phi):
    """Remeshing step: the next mesh from the phase field phi on the current mesh."""
    own = leaves(mesh)
    points = interface_points(mesh["pos"].astype(np.float64), mesh["edge_index"], phi)
    cells = remesh(own, distance_marking(own, points))
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
