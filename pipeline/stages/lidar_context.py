"""Build a coarse LiDAR reference layer outside the reconstructed area.

MODEL=<preview.bin> LAZ=<raw tile.laz>,... OUT=<lidar-coarse.bin> python lidar_context.py

Use raw public-domain USGS 3DEP tiles, not tiles classified with noncommercial
model weights. This reference layer is separate from reconstruction outputs.
Use 4 m cells near captures and 8 m cells farther away; omit covered cells.
"""
import json
import os
from pathlib import Path

import numpy as np
import laspy
ORIGIN = np.array([377249.9995, 4700249.9995, 150.0])
POINT = np.dtype([('xyz', '<f4', 3), ('rgba', 'u1', 4)])
# (cell size, how far past the walks this ring reaches, how far a cell must rise before it also gets a top point)
RINGS = [(4.0, 250.0, 2.5), (8.0, float('inf'), 4.0)]
CLEAR = float(os.environ.get('CLEAR', 3.0))
GROUND = [2, 9, 17]

model = np.fromfile(Path(os.environ['MODEL']), dtype=POINT)
lo, hi = model['xyz'].min(0), model['xyz'].max(0)
covered = set(map(tuple, np.floor(model['xyz'][:, [0, 2]] / RINGS[0][0]).astype(np.int64)))

def ring_of(x, z):
    """0 near the walks, 1 beyond: distance to the model's footprint in plan."""
    d = np.hypot(np.maximum(np.maximum(lo[0] - x, x - hi[0]), 0), np.maximum(np.maximum(lo[2] - z, z - hi[2]), 0))
    return (d > RINGS[0][1]).astype(np.int8)

acc = {}          # ring -> dict(keys, ground, top, count) accumulated across tiles
for path in os.environ['LAZ'].split(','):
    las = laspy.read(path.strip())
    cls = np.asarray(las.classification)
    P = np.column_stack([np.asarray(las.x) - ORIGIN[0], np.asarray(las.z) - ORIGIN[2], ORIGIN[1] - np.asarray(las.y)])
    r = ring_of(P[:, 0], P[:, 2])
    for ri, (cell, _, _) in enumerate(RINGS):
        m = r == ri
        if not m.any(): continue
        Q, cq = P[m], cls[m]
        c = np.floor(Q[:, [0, 2]] / cell).astype(np.int64)
        keys, inv = np.unique(c, axis=0, return_inverse=True)
        g = np.full(len(keys), np.inf); np.minimum.at(g, inv, np.where(np.isin(cq, GROUND), Q[:, 1], np.inf))
        t = np.full(len(keys), -np.inf); np.maximum.at(t, inv, Q[:, 1])
        n = np.bincount(inv, minlength=len(keys))
        a = acc.setdefault(ri, dict(keys=[], ground=[], top=[], count=[]))
        a['keys'].append(keys); a['ground'].append(g); a['top'].append(t); a['count'].append(n)
    print(json.dumps({'tile': Path(path).name, 'points': int(len(P))}), flush=True)

out, report = [], {}
for ri, (cell, _, rise) in enumerate(RINGS):
    if ri not in acc: continue
    a = acc[ri]
    keys = np.concatenate(a['keys']); ground = np.concatenate(a['ground']); top = np.concatenate(a['top']); count = np.concatenate(a['count'])
    uk, inv = np.unique(keys, axis=0, return_inverse=True)          # merge the seams where tiles meet
    G = np.full(len(uk), np.inf); np.minimum.at(G, inv, ground)
    T = np.full(len(uk), -np.inf); np.maximum.at(T, inv, top)
    N = np.zeros(len(uk), np.int64); np.add.at(N, inv, count)
    G = np.where(np.isfinite(G), G, T)
    reach = int(np.ceil(CLEAR / cell))
    scale = int(round(cell / RINGS[0][0]))
    near = np.array([any((a0 * scale + dx, b0 * scale + dz) in covered
                         for dx in range(-reach * scale, (reach + 1) * scale) for dz in range(-reach * scale, (reach + 1) * scale))
                     for a0, b0 in uk]) if ri == 0 else np.array(
                    [any((a0 * scale + dx, b0 * scale + dz) in covered for dx in range(0, scale) for dz in range(0, scale)) for a0, b0 in uk])
    keep = (N >= 2) & ~near
    centres = (uk + .5) * cell
    g = np.column_stack([centres[keep, 0], G[keep], centres[keep, 1]])
    shade = np.clip((g[:, 1] - np.percentile(g[:, 1], 2)) / max(1e-6, np.percentile(g[:, 1], 98) - np.percentile(g[:, 1], 2)), 0, 1)
    out.append((g, np.column_stack([72 + 26 * shade, 80 + 26 * shade, 92 + 26 * shade]).astype(np.uint8)))
    tall = keep & (T - G > rise)
    if tall.any():
        t = np.column_stack([centres[tall, 0], T[tall], centres[tall, 1]])
        out.append((t, np.tile(np.array([104, 112, 124], np.uint8), (len(t), 1))))
    report[f'ring_{ri}_{cell:g}m'] = dict(cells=int(len(uk)), dropped_walk_covered=int(near.sum()),
                                          ground=int(keep.sum()), elevated=int(tall.sum()))

xyz = np.concatenate([a for a, _ in out]); rgb = np.concatenate([b for _, b in out])
rec = np.zeros(len(xyz), POINT); rec['xyz'] = xyz; rec['rgba'][:, :3] = rgb; rec['rgba'][:, 3] = 200
rec.tofile(os.environ['OUT'])
print(json.dumps({'rings': report, 'total': int(len(rec)), 'megabytes': round(len(rec) * 16 / 1e6, 2),
                  'bounds': [xyz.min(0).round(0).tolist(), xyz.max(0).round(0).tolist()],
                  'source': 'USGS 3DEP NY_FingerLakes_1_2020 (public domain); raw tiles, not the 3dsem classification'}, indent=1))
