"""Register and merge reconstructed captures into one point cloud.

WALKS=<capture dir>,... OUT=<merged dir> python fuse.py

Align overlapping wall and ground surfaces with bounded point-to-plane fits.
Prefer nearby, sharper views, flatten shared seams, and deduplicate overlapping
cells. Writes preview.bin, walk-index.bin and fusion.json.
"""
import json
import os
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree
from sheets import align, collapse, normals, step

POINT = np.dtype([('xyz', '<f4', 3), ('rgba', 'u1', 4)])
WALLS, GROUND = [2, 3], [0, 1, 9]
MAX_SHIFT_M, MAX_YAW_DEG, MIN_PAIRS, ROUNDS = 6.0, 5.0, 1500, 4
COLLAPSE_M, GROUND_COLLAPSE_M, BEST_R = 2.5, .6, 2.5   # BEST_R: two walks' copies of a wall drift up to ~2 m apart at 15-25 m range
rng = np.random.default_rng(0)


def moved(P, x, c):
    return P @ step(x, c)[:3, :3].T + step(x, c)[:3, 3]



walks = [Path(p) for p in os.environ['WALKS'].split(',') if (Path(p) / 'block-consistent.bin').exists()]
out = Path(os.environ['OUT']); out.mkdir(parents=True, exist_ok=True)
clouds = [np.fromfile(w / 'block-consistent.bin', dtype=POINT) for w in walks]
ranges = [np.fromfile(w / 'block-range.bin', dtype=np.float16).astype(np.float32) if (w / 'block-range.bin').exists() else np.full(len(c), 10, np.float32)
          for w, c in zip(walks, clouds)]
ranges = [r if len(r) == len(c) else np.full(len(c), 10, np.float32) for r, c in zip(ranges, clouds)]
P = [c['xyz'].astype(np.float64) for c in clouds]; centre = [p.mean(0) for p in P]
report = {w.name: dict(points=len(c), shift=[0, 0, 0], yaw_deg=0.0, matrix=np.eye(4).tolist()) for w, c in zip(walks, clouds)}   # matrix: for tsdf.py

# ---- 1. registration: each overlapping walk moves halfway onto the others, a few rounds ----------------------------
def sample(i, n=40000):
    s = np.flatnonzero(np.isin(clouds[i]['rgba'][:, 3], WALLS + GROUND)); return P[i][s[rng.permutation(len(s))[:n]]]
cellsets = [set(map(tuple, np.floor(p[:, [0, 2]] / 4).astype(np.int64))) for p in P]
for rnd in range(ROUNDS if len(walks) > 1 else 0):
    for i in range(len(walks)):
        others = [k for k in range(len(walks)) if k != i and len(cellsets[i] & cellsets[k]) >= 10]
        if not others: continue
        dst = np.concatenate([sample(k, 60000) for k in others]); dst = dst[np.unique(np.floor(dst / .15).astype(np.int64), axis=0, return_index=True)[1]]
        fit = align(sample(i), dst, normals(dst), centre[i], MIN_PAIRS)
        if fit is None: report[walks[i].name]['registration'] = 'too little overlap'; continue
        x, pairs = fit; x = x / 2                                                                # meet halfway
        total = np.array(report[walks[i].name]['shift']) + x[1:]; yaw = report[walks[i].name]['yaw_deg'] + np.degrees(x[0])
        if np.hypot(total[0], total[2]) > MAX_SHIFT_M or abs(total[1]) > 1.5 or abs(yaw) > MAX_YAW_DEG:
            report[walks[i].name]['registration'] = f'refused: would move {np.hypot(total[0], total[2]):.1f} m / {yaw:.1f} deg'; continue
        P[i] = moved(P[i], x, centre[i]); report[walks[i].name]['matrix'] = (step(x, centre[i]) @ np.array(report[walks[i].name]['matrix'])).tolist(); cellsets[i] = set(map(tuple, np.floor(P[i][:, [0, 2]] / 4).astype(np.int64)))
        report[walks[i].name].update(shift=[round(float(v), 3) for v in total], yaw_deg=round(float(yaw), 3), pairs=pairs, registration='aligned', overlaps=[walks[k].name for k in others])

# ---- 2. fusion ----------------------------------------------------------------------------------------------------
xyz = np.concatenate(P); rgba = np.concatenate([c['rgba'] for c in clouds]); lab = rgba[:, 3]; rg = np.concatenate(ranges)
wid = np.concatenate([np.full(len(c), k, np.uint16) for k, c in enumerate(clouds)])
if len(walks) > 1:
    # (a) two walks never agree to the centimetre (each has its own depth calibration, so their copies of a facade
    # drift 0.2-1 m apart with range), and averaging the copies blends two views' colours into a mottle. The walk that saw a wall patch from nearest (range
    # smoothed over 1 m of its own wall) keeps it, and other walks' copies within BEST_R go. A new walk that passes closer
    # replaces the old view; one that passes farther changes nothing, so more walks can only sharpen a street.
    w = np.flatnonzero(np.isin(lab, WALLS)); Pw = xyz[w]; ww = wid[w]; rs = np.empty(len(w)); trees = {}
    for k in range(len(walks)):
        ik = np.flatnonzero(ww == k)
        if not len(ik): continue
        trees[k] = (cKDTree(Pw[ik]), ik); d, j = trees[k][0].query(Pw[ik], k=16, distance_upper_bound=1.0)
        rs[ik] = np.nanmedian(np.where(np.isfinite(d), rg[w][ik][np.minimum(j, len(ik) - 1)], np.nan), axis=1)
    # A walk's own wall noise says how far to trust it: its median wall thickness scales its range, so of two walks that
    # saw a patch from the same distance the sharper one keeps it (a sloppy walk can add coverage, never blur a street).
    for k, (tk, ik) in trees.items():
        s_ = ik[np.random.default_rng(k).permutation(len(ik))[:3000]]; th = []
        for nb in tk.query_ball_point(Pw[s_], 1.0):
            if len(nb) >= 20: Q = Pw[ik[nb]] - Pw[ik[nb]].mean(0); th.append(np.sqrt(max(np.linalg.eigvalsh(Q.T @ Q / len(nb))[0], 0)))
        report[walks[k].name]['wall_sigma_m'] = round(float(np.median(th)), 3) if th else None
    sig = {k: report[walks[k].name]['wall_sigma_m'] or 1.0 for k in trees}; s0 = min(sig.values())
    for k, (tk, ik) in trees.items(): rs[ik] *= sig[k] / s0
    drop = np.zeros(len(w), bool)
    for k, (tk, ik) in trees.items():
        for q, (tq, iq) in trees.items():
            if q == k: continue
            d, j = tq.query(Pw[ik], distance_upper_bound=BEST_R); hit = np.isfinite(d)
            drop[ik[hit]] |= rs[iq[j[hit]]] < rs[ik[hit]]
    keep = np.ones(len(xyz), bool); keep[w[drop]] = False
    xyz, rgba, lab, rg, wid = xyz[keep], rgba[keep], lab[keep], rg[keep], wid[keep]
    # (b) then, where two walks still meet, what is left flattens onto one sheet (2.5 m for walls, 0.6 m for the
    # ground). After (a) most of a street is one walk's, so this only closes the seams where two walks' patches touch.
    for cls, window in ((WALLS, COLLAPSE_M), (GROUND, GROUND_COLLAPSE_M)):
        m = np.flatnonzero(np.isin(lab, cls))
        c4 = np.floor(xyz[m][:, [0, 2]] / 4).astype(np.int64); key = c4[:, 0] * 1000003 + c4[:, 1]
        u, inv = np.unique(key, return_inverse=True); nw = np.zeros(len(u), int)
        for k in range(len(walks)): nw += np.isin(u, np.unique(key[wid[m] == k]))
        shared = m[nw[inv] > 1]
        if len(shared) > 1000: xyz[shared] = collapse(xyz[shared], window)
# One point per 0.1 m cell per class, but only BETWEEN walks: each walk was already thinned on its own grid, and
# refine.py slid it by a fraction of a cell, so re-gridding a walk against itself would merge its own neighbours.
ci = np.floor(xyz / .10).astype(np.int64); ck = (ci[:, 0] * 73856093) ^ (ci[:, 1] * 19349663) ^ (ci[:, 2] * 83492791) ^ (lab.astype(np.int64) << 40)
pair = np.unique(np.column_stack([ck, wid.astype(np.int64)]), axis=0); uk, nwk = np.unique(pair[:, 0], return_counts=True)
shared = np.isin(ck, uk[nwk > 1])
o = np.lexsort((rg, ck)); ks = ck[o]                                     # a shared cell keeps its nearest view; other cells keep all
first = np.ones(len(o), bool); first[1:] = ~(shared[o][1:] & (ks[1:] == ks[:-1]))
sel = np.sort(o[first])
city = np.zeros(len(sel), POINT); city['xyz'] = xyz[sel]; city['rgba'] = rgba[sel]; wid = wid[sel]
city.tofile(out / 'preview.bin'); wid.tofile(out / 'walk-index.bin')
for k, w in enumerate(walks): report[w.name]['kept'] = int((wid == k).sum())
(out / 'fusion.json').write_text(json.dumps(dict(walks=report, points_in=int(sum(len(c) for c in clouds)), points_out=len(city)), indent=1))
print(json.dumps(dict(points_in=int(sum(len(c) for c in clouds)), points_out=len(city), walks={k: {x: v[x] for x in ('shift', 'yaw_deg', 'kept') if x in v} | {'reg': v.get('registration')} for k, v in report.items()})))
