"""Fuse calibrated depth maps into a shared truncated signed-distance volume.

RUN=<capture dir with frames-dump/ and global-frame.json> python tsdf.py
RUNS=<capture dir>,... OUT=<dir> [REGISTER=<fusion.json>] python tsdf.py

Register overlapping chunks, accumulate depth and free-space votes, then rebuild
the volume from rays consistent with its strongest surfaces. Extract coloured
zero crossings, prefer nearby views, and apply LiDAR footprint and height checks.
Writes block-tsdf.bin with semantic classes in the alpha byte. Running without
RUN or RUNS performs the synthetic fusion check.
"""
import json
import os
import time
from pathlib import Path

import numpy as np

POINT = np.dtype([('xyz', '<f4', 3), ('rgba', 'u1', 4)])
VOXEL, TRUNC, TOL, NEAR, SKY_M, STRIDE = float(os.environ.get('VOXEL', .1)), .4, .5, .3, 40.0, 2
F_STRONG, F_MIN, MAX_VOXELS, BATCH = 3, 2, 4e8, 4096
BUCKET = {0: 0, 1: 0, 9: 0, 2: 1, 3: 1, 8: 2}               # ground, wall, vegetation
OUT_CLASS = np.array([0, 2, 8], np.uint8)
MIN_SCORE, SKY = .60, 10
INSIDE_M = 2.5                                                # refine.py's rule: no wall this deep inside a LiDAR footprint
GRAZE_DEG = 6.0                                               # the rule filter.py applies to its points
CEIL_M, CEIL_SUP = 3.0, 3                                     # nothing stands CEIL_M over every LiDAR return around it, unless more than CEIL_SUP frames saw it
OPEN_M, OPEN_H, OPEN_VIEWS = 4.0, 2.5, 9                      # a wall OPEN_M out from every building and OPEN_H up needs OPEN_VIEWS frames; low walls and fences stay
TILE_M, MARGIN_M, REACH_M, WORKERS = 80.0, 6.0, 60.0, int(os.environ.get('WORKERS', 4))   # city tiles; frames within REACH_M vote
REG_ROUNDS, REG_MAX_M, REG_MAX_DEG, REG_STRIDE = 3, 2.0, 3.0, 8           # sections meet halfway; a bigger total move is a wrong match


def add(arr, idx, w=None):
    u, inv = np.unique(idx, return_inverse=True); arr[u] += np.bincount(inv.ravel(), weights=w, minlength=len(u)).astype(arr.dtype)


def rays(fr):
    """(origin, unit-depth directions in the city frame, depth, bucket, rgb) for the surface pixels; directions of sky pixels."""
    D, K, T = fr['depth'], fr['K'], fr['T']; H, W = D.shape
    v, u = np.mgrid[0:H:STRIDE, 0:W:STRIDE]; v, u = v.ravel(), u.ravel()
    d_cam = np.column_stack([(u - K[0, 2]) / K[0, 0], (v - K[1, 2]) / K[1, 1], np.ones(len(u))])   # a point at depth z is d_cam * z
    dirs = d_cam @ T[:3, :3].T; C = T[:3, 3]
    lab = fr['labels'][v, u]; sc = fr['score'][v, u].astype(np.float32); z = D[v, u]
    surf = fr['valid'][v, u] & (z > NEAR) & np.isin(lab, list(BUCKET)) & (sc >= MIN_SCORE)
    surf &= ~(np.isin(lab, [0, 1, 9]) & (np.abs(dirs[:, 1]) / np.linalg.norm(dirs, axis=1) < np.sin(np.radians(GRAZE_DEG))))
    sky = (lab == SKY) & (sc >= MIN_SCORE)
    b = np.array([BUCKET.get(int(x), 0) for x in range(256)], np.int8)[lab]
    return C, dirs[surf], z[surf], b[surf], fr['colors'][v, u][surf], dirs[sky]


def fuse(frames, lo, hi, voxel=VOXEL, log=print):
    """frames: dicts with depth, valid, colors, labels, score, K, T (camera -> city 4x4). Returns POINT array."""
    shape = np.ceil((hi - lo) / voxel).astype(int) + 1; n = int(np.prod(shape)); stride = np.array([shape[1] * shape[2], shape[2], 1])
    assert n <= MAX_VOXELS, f'volume of {n:.3g} cells; raise VOXEL'
    def flat(P):
        ijk = np.floor((P - lo) / voxel).astype(np.int64); ok = np.all((ijk >= 0) & (ijk < shape), axis=-1)
        return (ijk[..., 0] * shape[1] + ijk[..., 1]) * shape[2] + ijk[..., 2], ok
    band = np.arange(-TRUNC, TRUNC + 1e-6, voxel); sdf = -band / TRUNC; core = np.abs(band) <= voxel   # core: cells at the surface itself

    def vote(V, C, dirs, z, sky):
        """One frame's rays into V: W (weight), WD (weighted distance), NF (frames with a band here)."""
        seen = []
        for s in range(0, len(z), BATCH):
            d, zz = dirs[s:s + BATCH], z[s:s + BATCH]
            idx, ok = flat(C + d[:, None] * (zz[:, None] + band)[..., None])
            add(V['W'], idx[ok]); add(V['WD'], idx[ok], np.broadcast_to(sdf, idx.shape)[ok]); seen.append(idx[ok])
            steps = np.arange(NEAR, zz.max() - TRUNC, voxel)
            if len(steps):
                idx, ok = flat(C + d[:, None] * steps[None, :, None]); ok &= steps[None] < (zz - TRUNC)[:, None]
                add(V['W'], idx[ok]); add(V['WD'], idx[ok])
        for s in range(0, len(sky), BATCH):
            idx, ok = flat(C + sky[s:s + BATCH, None] * np.arange(NEAR, SKY_M, voxel)[None, :, None]); add(V['W'], idx[ok]); add(V['WD'], idx[ok])
        if seen: V['NF'][np.unique(np.concatenate(seen))] += 1

    def volume():
        return dict(W=np.zeros(n, np.float32), WD=np.zeros(n, np.float32), NF=np.zeros(n, np.uint16))

    # ---- pass 1: every ray votes ------------------------------------------------------------------------------------
    t0 = time.time(); R = [rays(fr) for fr in frames]; V = volume()
    for C, dirs, z, b, rgb, sky in R: vote(V, C, dirs, z, sky)
    D1 = np.divide(V['WD'], V['W'], out=np.zeros(n, np.float32), where=V['W'] > 0); strong = (V['NF'] >= F_STRONG) & (D1 <= 0)
    log(f'pass 1: {int((V["NF"] > 0).sum()):,} surface cells, {int(strong.sum()):,} strong ({time.time() - t0:.0f} s)'); del V, D1

    def first_hit(C, dirs, stop):
        """Depth of the first strong cell along each ray before its stop depth; inf where there is none."""
        out = np.full(len(dirs), np.inf)
        for s in range(0, len(dirs), BATCH):
            d, st = dirs[s:s + BATCH], stop[s:s + BATCH]; march = np.arange(NEAR, st.max(), voxel)
            if not len(march): continue
            idx, ok = flat(C + d[:, None] * march[None, :, None])
            hit = ok & strong[np.where(ok, idx, 0)] & (march[None] < st[:, None]); k = np.argmax(hit, 1); has = hit[np.arange(len(k)), k]
            out[s:s + BATCH][has] = march[k[has]]
        return out

    # ---- pass 2: keep the rays that agree with the consensus, and rebuild ----------------------------------------------
    t0 = time.time(); V = volume(); kept = total = sky_kept = sky_total = 0; G = []
    for C, dirs, z, b, rgb, sky in R:
        first = first_hit(C, dirs, z + TOL + 1); good = ~np.isfinite(first) | (np.abs(z - first) <= TOL)
        sgood = ~np.isfinite(first_hit(C, sky, np.full(len(sky), SKY_M)))
        kept += int(good.sum()); total += len(z); sky_kept += int(sgood.sum()); sky_total += len(sky)
        vote(V, C, dirs[good], z[good], sky[sgood]); G.append(good)
    del strong
    log(f'pass 2: kept {kept:,} of {total:,} rays ({100 * kept / max(total, 1):.1f}%), {sky_kept:,} of {sky_total:,} sky rays ({time.time() - t0:.0f} s)')

    # ---- surface: each kept ray's crossing near its own depth ----------------------------------------------------------
    W, NF = V['W'], V['NF']; D = np.divide(V['WD'], W, out=np.zeros(n, np.float32), where=W > 0); del V
    corners = np.array([[i, j, k] for i in (0, 1) for j in (0, 1) for k in (0, 1)])
    def trilinear(P):
        g = (P - lo) / voxel - .5; i0 = np.floor(g).astype(np.int64); f = g - i0; acc = np.zeros(len(P)); ws = np.zeros(len(P))
        for c in corners:
            q = i0 + c; ok = np.all((q >= 0) & (q < shape), axis=1); ci = np.where(ok, (q[:, 0] * shape[1] + q[:, 1]) * shape[2] + q[:, 2], 0)
            w = np.prod(np.where(c, f, 1 - f), axis=1) * (ok & (W[ci] > 0)); acc += w * D[ci]; ws += w
        return acc / np.maximum(ws, 1e-9)
    window = np.arange(-1., TOL + 1e-6, voxel)                                     # relative to each ray's own depth
    pts, cols, cls, wts, grp, sup = [], [], [], [], [], []
    for fr, (C, dirs, z, b, rgb, _), good in zip(frames, R, G):
        dirs, z, b, rgb = dirs[good], z[good], b[good], rgb[good]
        for s in range(0, len(z), BATCH):
            d, zz = dirs[s:s + BATCH], z[s:s + BATCH]; t = np.maximum(zz[:, None] + window, NEAR)
            idx, ok = flat(C + d[:, None] * t[..., None]); idx = np.where(ok, idx, 0)
            neg = ok & (W[idx] > 0) & (D[idx] <= 0); cross = neg[:, 1:] & (NF[idx[:, 1:]] >= F_MIN) & ~neg[:, :-1]
            k = np.argmax(cross, 1); r = np.flatnonzero(cross[np.arange(len(k)), k]); k = k[r]
            ta, tb = t[r, k], t[r, k + 1]; Da, Db = trilinear(C + d[r] * ta[:, None]), trilinear(C + d[r] * tb[:, None])
            th = ta + (tb - ta) * np.where((Da > 0) & (Db <= 0), Da / np.where(Da > Db, Da - Db, 1), 1.)
            pts.append(C + d[r] * th[:, None]); cols.append(rgb[s:s + BATCH][r]); cls.append(b[s:s + BATCH][r]); wts.append(1 / np.maximum(th, 1)); grp.append(np.full(len(r), fr.get('group', 0)))
            sup.append(NF[idx[r, k + 1]])                                             # frames that put their band where this crossing is
    P, rgb, b, w, grp, sup = np.concatenate(pts), np.concatenate(cols).astype(np.float64), np.concatenate(cls), np.concatenate(wts), np.concatenate(grp), np.concatenate(sup)
    key, _ = flat(P); u, inv = np.unique(key, return_inverse=True); cnt = np.bincount(inv); ws = np.bincount(inv, weights=w)
    out = np.zeros(len(u), POINT)
    out['xyz'] = np.column_stack([np.bincount(inv, weights=P[:, k]) / cnt for k in range(3)]).astype(np.float32)
    out['rgba'][:, :3] = np.clip(np.column_stack([np.bincount(inv, weights=w * rgb[:, k]) / ws for k in range(3)]), 0, 255).astype(np.uint8)
    out['rgba'][:, 3] = OUT_CLASS[np.argmax(np.column_stack([np.bincount(inv, weights=(b == k).astype(float), minlength=len(u)) for k in range(3)]), 1)]
    log(f'surface: {len(out):,} points from {len(P):,} ray crossings')
    o = np.lexsort((1 / w, inv)); near = o[np.r_[0, np.flatnonzero(np.diff(inv[o])) + 1]]              # each cell's nearest crossing
    support = np.zeros(len(u), np.int64); np.maximum.at(support, inv, sup.astype(np.int64))
    return out, grp[near], (1 / w[near]).astype(np.float32), support


def nearest_view_wins(P, grp, rng, R=1.5, across=.5, same=.2):
    """Where two sections (a walk's ~30 m chunks, of one walk or of several) put a wall at two depths, the copy seen from
    nearer stays: a point goes when another section has wall within R of it along its normal, and within `across` of
    that line, that it saw from nearer (depths smoothed over each section's 16 nearest own points). Copies closer than
    `same` are one surface already and both stay. P: wall points; returns the mask of points to drop."""
    from scipy.spatial import cKDTree
    drop = np.zeros(len(P), bool); sec = {}
    for g in np.unique(grp):
        i = np.flatnonzero(grp == g)
        if len(i) < 16: continue
        t = cKDTree(P[i]); _, j = t.query(P[i], k=16); Q = P[i][j] - P[i][j].mean(1, keepdims=True)
        sec[g] = (t, i, np.linalg.eigh(np.einsum('nki,nkj->nij', Q, Q))[1][:, :, 0], np.median(rng[i][j], axis=1), P[i].min(0) - R, P[i].max(0) + R)
    for g, (_, ig, n, rg, lo, hi) in sec.items():
        for q, (tq, iq, _, rq, lo2, hi2) in sec.items():
            if q == g or np.any(hi < lo2) or np.any(hi2 < lo): continue
            d, j = tq.query(P[ig], k=8, distance_upper_bound=R); ok = np.isfinite(d); j = np.where(ok, j, 0)
            v = P[iq][j] - P[ig][:, None]; along = np.einsum('nkj,nj->nk', v, n); off = np.linalg.norm(v - along[..., None] * n[:, None], axis=2)
            drop[ig] |= (ok & (np.abs(along) >= same) & (off < across) & (rq[j] < rg[:, None])).any(1)
    return drop


def lidar_rules(xyz, cls, support, ground, ceiling, to_edge):
    """(sail, street) masks of points the airborne LiDAR rules out: more than CEIL_M over every return around them, or a
    wall more than OPEN_M out from every building outline and OPEN_H over the ground; either stays when enough frames saw
    it (a new building, a real overhang). Per point: LiDAR ground height, highest return nearby, metres to the nearest
    outline; nan where unknown, and unknown never removes anything."""
    with np.errstate(invalid='ignore'):
        sail = (xyz[:, 1] > ceiling + CEIL_M) & (support <= CEIL_SUP)
        street = (cls == 2) & (to_edge > OPEN_M) & (xyz[:, 1] - ground > OPEN_H) & (support < OPEN_VIEWS)
    return sail, street


if __name__ == '__main__' and 'RUN' not in os.environ and 'RUNS' not in os.environ:
    # Six cameras 1.5 m above a ground plane face a wall at 10 m (2% depth noise). Two bad frames draw the wall 1.5 m too
    # deep; two draw a slab at 5 m in front of part of it. The fusion must return the wall at 10 and the ground only, and
    # keep the ground, which every camera sees at a grazing angle (cells 0.5 m deep: the rows of all six cameras land on the
    # same depths, about 0.2-0.4 m apart there).
    H, W = 240, 320; K = np.array([[300., 0, W / 2], [0, 300., H / 2], [0, 0, 1]]); rng = np.random.default_rng(0)
    v, u = np.mgrid[0:H, 0:W]; ground = np.where(v > H / 2, 1.5 * 300 / np.maximum(v - H / 2, 1e-3), np.inf)
    def frame(x, D):
        T = np.eye(4); T[0, 3] = x; D = D * (1 + .02 * rng.standard_normal(D.shape))
        return dict(depth=D.astype(np.float32), valid=np.ones((H, W), bool), colors=np.full((H, W, 3), 200, np.uint8),
                    labels=np.full((H, W), 2, np.uint8), score=np.ones((H, W), np.float16), K=K, T=T)
    scene = np.minimum(10., ground)
    frames = [frame(x, scene) for x in (-1.5, -1, -.5, 0, .5, 1)]
    frames += [frame(x, np.minimum(11.5, ground)) for x in (.25, -.75)]                                # the bad frames
    frames += [frame(x, np.where((u < W // 2) & (v < 160), 5., scene)) for x in (-.25, .75)]           # the slab
    pts = fuse(frames, np.array([-12., -12., 0.]), np.array([12., 12., 14.]), .1, log=lambda m: None)[0]
    x, y, z = pts['xyz'].T.astype(float); wall, grd = np.abs(z - 10) < .4, np.abs(y - 1.5) < .15
    mid = (np.abs(x) < 5) & (y > -1.8)                                    # where 3+ good views see, and see through the slab
    bad, slab = mid & (z > 11.1), mid & (z < 5.6) & (y < 1.2)
    print(f'surface points {len(pts)}: off the wall and ground {(~wall & ~grd & mid).mean():.2%}, bad sheet {bad.mean():.2%}, slab {slab.mean():.2%}')
    assert (wall | grd)[mid].mean() > .99, 'the fused surface must be the wall at 10 m and the ground'
    assert bad.mean() < .005 and slab.mean() < .005, 'the deep frames and the slab must be voted out'
    assert (wall & (y < 1)).sum() > 1000, 'the wall must survive'
    g = grd & (np.abs(x) < 1) & (z > 5) & (z < 9); cov = len(np.unique(np.floor(np.column_stack([x[g], z[g]]) / [.1, .5]), axis=0)) / 160
    assert cov > .9, f'the ground seen at grazing angles must survive ({cov:.0%} of its cells)'
    # sail: seen by 2 / by 5 / no LiDAR; walls: out in the street high up / low / next to a building / unknown outline
    xyz = np.array([[0, 20, 0], [0, 20, 0], [0, 20, 0], [0, 4, 0], [0, 1, 0], [0, 4, 0], [0, 4, 0]], float)
    sail, street = lidar_rules(xyz, np.array([2, 2, 2, 2, 2, 2, 2]), np.array([2, 5, 1, 3, 3, 3, 3]), np.zeros(7),
                               np.array([10, 10, np.nan, 30, 30, 30, 30]), np.array([0, 0, 0, 6, 6, 2, np.nan]))
    assert sail.tolist() == [True, False, False, False, False, False, False] and street.tolist() == [False, False, False, True, False, False, False], (sail, street)
    print(f'ground cells covered {cov:.0%}, wall depth median {np.median(z[wall & (y < 1)]):.3f}\ntsdf ok')
else:
    def walk_frames(R, M=np.eye(4), walk=0):
        """(frame file, camera -> city 4x4) of every frame a walk kept: place.py's pose, then refine.py's rigid corrections for
        its tracked part and its chunk. A chunk refine.py dropped as poorly shot is left out."""
        g = json.loads((R / 'global-frame.json').read_text()).get('refinement', {}); I = np.eye(4)
        Mc = {int(k): np.array(v['matrix']) for k, v in g.get('components', {}).items() if 'matrix' in v}
        Mk = {int(k): np.array(v['matrix']) for k, v in g.get('chunks', {}).items() if 'matrix' in v}
        dropped = {c['chunk'] for c in g.get('chunks_dropped', [])}; out = []
        for p in sorted((R / 'frames-dump').glob('*.npz')):
            f = np.load(p); ch, cp = int(f['chunk']), int(f['comp'])
            if ch not in dropped: out.append((str(p), M @ Mk.get(ch, I) @ Mc.get(cp, I) @ f['G'] @ f['c2w'], walk * 1000 + ch))
        return out

    def section_cloud(path, T):
        """One frame's walls and ground within 30 m, every REG_STRIDE pixels, in the city frame."""
        f = np.load(path); D, K = f['depth'], f['K']; v, u = np.mgrid[0:D.shape[0]:REG_STRIDE, 0:D.shape[1]:REG_STRIDE]
        z, lab = D[v, u], f['labels'][v, u]
        ok = f['valid'][v, u] & (z > NEAR) & (z < 30) & np.isin(lab, [0, 1, 9, 2, 3]) & (f['score'][v, u].astype(np.float32) >= MIN_SCORE)
        return np.column_stack([(u[ok] - K[0, 2]) / K[0, 0] * z[ok], (v[ok] - K[1, 2]) / K[1, 1] * z[ok], z[ok]]) @ T[:3, :3].T + T[:3, 3]

    def register_sections(frames):
        """Every section (a walk's ~30 m chunk) moves halfway onto the sections that overlap it, of any walk, for
        REG_ROUNDS rounds, so two sections that saw one wall meet on it instead of stacking two copies 0.5-1.5 m apart
        (each was placed on its own against the LiDAR). A total move beyond REG_MAX_M / REG_MAX_DEG, or 1 m up or down,
        is a wrong match and refused. Returns {section: 4x4}."""
        from sheets import align, normals, step
        S = {}
        for p, T, g in frames: S.setdefault(g, []).append(section_cloud(p, T))
        S = {g: (lambda Q: Q[np.unique(np.floor(Q / .15).astype(np.int64), axis=0, return_index=True)[1]])(np.concatenate(v)) for g, v in S.items()}
        cells = lambda Q: set(map(tuple, np.floor(Q[:, [0, 2]] / 4).astype(np.int64)))
        M = {g: np.eye(4) for g in S}; P = dict(S); cs = {g: cells(Q) for g, Q in P.items()}; rng = np.random.default_rng(0)
        for _ in range(REG_ROUNDS):
            for g in P:
                others = [k for k in P if k != g and len(cs[g] & cs[k]) >= 10]
                if not others or len(P[g]) < 2000: continue
                dst = np.concatenate([P[k] for k in others]); c = P[g].mean(0)
                fit = align(P[g][rng.permutation(len(P[g]))[:20000]], dst, normals(dst), c)
                if fit is None: continue
                Mn = step(fit[0] / 2, c) @ M[g]; c0 = S[g].mean(0); d = Mn[:3, :3] @ c0 + Mn[:3, 3] - c0
                if abs(np.degrees(np.arctan2(Mn[0, 2], Mn[0, 0]))) > REG_MAX_DEG or np.hypot(d[0], d[2]) > REG_MAX_M or abs(d[1]) > 1.0: continue
                M[g] = Mn; P[g] = S[g] @ Mn[:3, :3].T + Mn[:3, 3]; cs[g] = cells(P[g])
        moves = [np.hypot(*(M[g][:3, :3] @ S[g].mean(0) + M[g][:3, 3] - S[g].mean(0))[[0, 2]]) for g in S if not np.allclose(M[g], np.eye(4))]
        print(f'registration: {len(moves)} of {len(S)} sections moved onto their neighbours' + (f', median {np.median(moves):.2f} m, largest {max(moves):.2f} m' if moves else ''), flush=True)
        return M

    def tile(job):
        """Fuse one tile from every frame that can see into it; keep the points in its core."""
        (lo, hi, c0, c1), frames = job
        voxel = VOXEL
        while np.prod(np.ceil((hi - lo) / voxel) + 1) > MAX_VOXELS: voxel *= 1.1
        out, grp, rng, support = fuse([dict(np.load(p), T=T, group=g) for p, T, g in frames], lo, hi, voxel, log=lambda m: None)
        xz = out['xyz'][:, [0, 2]]; k = np.all((xz >= c0) & (xz < c1), axis=1); return out[k], grp[k], rng[k], voxel, support[k]

if __name__ == '__main__' and ('RUN' in os.environ or 'RUNS' in os.environ):
    from concurrent.futures import ProcessPoolExecutor
    from scipy import ndimage
    runs = [Path(p) for p in os.environ.get('RUNS', os.environ.get('RUN', '')).split(',') if p]; OUT = Path(os.environ.get('OUT', runs[0]))
    reg = json.loads(Path(os.environ['REGISTER']).read_text()).get('walks', {}) if os.environ.get('REGISTER') else {}
    frames = [f for i, R in enumerate(runs) for f in walk_frames(R, np.array(reg.get(R.name, {}).get('matrix', np.eye(4))), i)]
    M = register_sections(frames); frames = [(p, M[g] @ T, g) for p, T, g in frames]; cams = np.array([T[:3, 3] for _, T, _ in frames])
    refs = [np.fromfile(R / 'block-consistent.bin', dtype=POINT)['xyz'] for R in runs]; ref = np.concatenate(refs)
    lo = np.min([np.percentile(r, .5, 0) for r in refs], 0); hi = np.max([np.percentile(r, 99.5, 0) for r in refs], 0)
    jobs = []
    for x0 in np.arange(lo[0], hi[0], TILE_M):
        for z0 in np.arange(lo[2], hi[2], TILE_M):
            c0 = np.array([x0, z0]); c1 = c0 + TILE_M
            m = np.all((ref[:, [0, 2]] >= c0 - MARGIN_M) & (ref[:, [0, 2]] < c1 + MARGIN_M), axis=1)
            if m.sum() < 1000: continue                                                          # nobody walked here
            y0, y1 = np.percentile(ref[m, 1], [.5, 99.5])
            near = np.flatnonzero(np.all(np.abs(cams[:, [0, 2]] - np.clip(cams[:, [0, 2]], c0, c1)) <= REACH_M, axis=1))
            jobs.append(((np.array([x0 - MARGIN_M, y0 - 3, z0 - MARGIN_M]), np.array([x0 + TILE_M + MARGIN_M, y1 + 4, z0 + TILE_M + MARGIN_M]), c0, c1),
                         [frames[j] for j in near]))
    print(f'{len(runs)} walks, {len(frames)} frames, {len(jobs)} tiles of {TILE_M:.0f} m (frames per tile {min(len(j[1]) for j in jobs)}-{max(len(j[1]) for j in jobs)})', flush=True)
    with ProcessPoolExecutor(WORKERS) as ex: res = list(ex.map(tile, jobs))
    out, grp, rng, support = (np.concatenate([r[k] for r in res]) for k in (0, 1, 2, 4))
    wall = np.flatnonzero(out['rgba'][:, 3] == 2); gone = wall[nearest_view_wins(out['xyz'][wall].astype(float), grp[wall], rng[wall])]
    out = np.delete(out, gone); support = np.delete(support, gone); print(f'dropped {len(gone):,} wall points another section saw nearer, at another depth', flush=True)
    # refine.py's LiDAR rule, from each walk's cleaned footprints: a street camera cannot see a wall deep behind a facade
    deep = np.zeros(len(out), bool); to_edge = np.full(len(out), np.nan)
    for R in runs:
        if not (R / 'footprints.npz').exists(): continue
        fp = np.load(R / 'footprints.npz'); grid = float(fp['grid']); depth_in = ndimage.distance_transform_edt(fp['occ']) * grid
        edge = fp['occ'] & ~ndimage.binary_erosion(fp['occ'], iterations=int(round(1 / grid))); d_edge = ndimage.distance_transform_edt(~edge) * grid
        ij = np.floor((out['xyz'][:, [0, 2]] - fp['origin']) / grid).astype(int); ok = np.flatnonzero(np.all((ij >= 0) & (ij < depth_in.shape), axis=1))
        deep[ok[depth_in[ij[ok, 0], ij[ok, 1]] > INSIDE_M]] = True
        to_edge[ok] = np.fmin(to_edge[ok], d_edge[ij[ok, 0], ij[ok, 1]])
    drop = deep & (out['rgba'][:, 3] == 2)
    print(f'dropped {int(drop.sum()):,} wall points more than {INSIDE_M} m inside a LiDAR footprint')
    if os.environ.get('LAZ'):
        from scipy.spatial import cKDTree
        from laz import read_city
        P = out['xyz'].astype(np.float64); lo2, hi2 = P.min(0) - 30, P.max(0) + 30
        L, lc = read_city(os.environ['LAZ']); keep = (L[:, 0] > lo2[0]) & (L[:, 0] < hi2[0]) & (L[:, 2] > lo2[2]) & (L[:, 2] < hi2[2]); L, lc = L[keep], lc[keep]
        G = L[np.isin(lc, [2, 17])]; dg, j = cKDTree(G[:, [0, 2]]).query(P[:, [0, 2]], k=8); ground = np.where(dg[:, 0] < 10, np.median(G[j, 1], axis=1), np.nan)
        cell = np.floor((L[:, [0, 2]] - lo2[[0, 2]]) / 2.0).astype(int); top = np.full(tuple(cell.max(0) + 1), -np.inf)
        np.maximum.at(top, (cell[:, 0], cell[:, 1]), L[:, 1]); top = ndimage.maximum_filter(top, 3, mode='constant', cval=-np.inf)   # the highest return within ~3 m
        pc = np.floor((P[:, [0, 2]] - lo2[[0, 2]]) / 2.0).astype(int); inside = np.all((pc >= 0) & (pc < top.shape), axis=1); ceiling = np.full(len(P), np.nan)
        ceiling[inside] = top[pc[inside, 0], pc[inside, 1]]; ceiling[~np.isfinite(ceiling)] = np.nan                                  # no LiDAR there: no verdict
        sail, street = lidar_rules(P, out['rgba'][:, 3], support, ground, ceiling, to_edge)
        drop |= sail | street; print(f'dropped {int((sail & ~deep).sum()):,} points over every LiDAR return around them and {int((street & ~sail).sum()):,} high walls out in the open street')
    out = out[~drop]
    OUT.mkdir(parents=True, exist_ok=True); out.tofile(OUT / 'block-tsdf.bin')
    print(json.dumps({'points': len(out), 'voxel_m': round(max(r[3] for r in res), 3), 'classes': {int(k): int(c) for k, c in zip(*np.unique(out['rgba'][:, 3], return_counts=True))}}))
