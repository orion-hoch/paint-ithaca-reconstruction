"""Refine component and chunk placement against classified LiDAR building outlines.

RUN=<capture dir with block-consistent.bin and manifest.json>
LIDAR_DIR=<dir of classified_*.laz>

Search bounded yaw and translation corrections using wall distance and orientation
relative to roof outlines. Penalize ground inside buildings, remove unsupported
walls, and flatten retained facades. Updates the point cloud and its sidecars,
saves footprints.npz, and records corrections in global-frame.json.
"""
import json
import os
from pathlib import Path

import numpy as np
import laspy
from scipy import ndimage
from scipy.spatial import cKDTree
from sheets import collapse
POINT = np.dtype([('xyz', '<f4', 3), ('rgba', 'u1', 4)])
ORIGIN = np.array([377249.9995, 4700249.9995, 150.0])
MIN_EDGES, MIN_WALL, MIN_SIGMA, MAX_YAW_DEG, GAIN, INSIDE_M, GRID, PENALTY, MIN_FOOTPRINT_M2 = 200, 2000, 3.0, 4.0, 1.2, 2.5, .25, 2.0, 40
WIDE_YAW_DEG = 25.0    # searches reach this far: a short track's GPS heading is loose, and ARKit's heading can jump inside a walk
CHUNK_PRIOR = .03      # evidence a chunk's move must buy per half squared (size / expected error): a chunk can find an outline anywhere
CHUNK_YAW_SIGMA, CHUNK_SHIFT_SIGMA = 8.0, 3.0   # a chunk against its tracked part: ARKit keeps them together, bar the odd jump
OPEN_M, OPEN_SUPPORT = 3.0, 6          # a wall standing in open street this far from any building needs this many frames behind it
CHUNK_REACH_M, MIN_CHUNK_WALL, MAX_CHUNK_LOSS = 3.0, 1500, .6   # then each ~30 m chunk: a turn and shift of its own
NEAR_M, SKEW_DEG = 8.0, 12.0
SNAP_M, SNAP_BLOCK_M, PARALLEL_DEG = 3.0, 2.0, 25.0   # a facade within SNAP_M of an outline becomes a flat vertical plane, per SNAP_BLOCK_M of outline           # near buildings a wall runs along their grid (parallel or square); one skewed more is a rotated copy
R = Path(os.environ['RUN']); LIDAR = Path(os.environ['LIDAR_DIR'])

pts = np.fromfile(R / 'block-consistent.bin', dtype=POINT); xyz = pts['xyz'].astype(np.float64); lab = pts['rgba'][:, 3]
side = {name: (np.fromfile(R / name, dtype=dt) if (R / name).exists() else None) for name, dt in (('block-range.bin', np.float16), ('block-comp.bin', np.uint8), ('block-support.bin', np.uint8), ('block-chunk.bin', np.uint16))}
side = {k: v for k, v in side.items() if v is not None and len(v) == len(pts)}
comp = side.get('block-comp.bin', np.zeros(len(pts), np.uint8)).astype(int)
manifest = json.loads((R / 'manifest.json').read_text())
acc = [f['location']['horizontal_accuracy_m'] for f in manifest.get('frames', []) if f.get('location')]
sigma = max(MIN_SIGMA, float(np.median(acc))) if acc else 8.0; reach = 2 * sigma
lo, hi = xyz.min(0) - reach - 10, xyz.max(0) + reach + 10
roofs, ground = [], []
for path in sorted(LIDAR.glob('classified_*.laz')):
    with laspy.open(path) as h:                                  # skip tiles that cannot overlap the walk
        if h.header.maxs[0] - ORIGIN[0] < lo[0] or h.header.mins[0] - ORIGIN[0] > hi[0] or ORIGIN[1] - h.header.mins[1] < lo[2] or ORIGIN[1] - h.header.maxs[1] > hi[2]: continue
    las = laspy.read(path); cls = np.asarray(las.classification)
    L = np.column_stack([np.asarray(las.x) - ORIGIN[0], np.asarray(las.z) - ORIGIN[2], ORIGIN[1] - np.asarray(las.y)])
    m = (L[:, 0] > lo[0]) & (L[:, 0] < hi[0]) & (L[:, 2] > lo[2]) & (L[:, 2] < hi[2])
    roofs.append(L[m & (cls == 6)][:, [0, 2]]); ground.append(L[m & np.isin(cls, [2, 9, 17])])
roofs = np.concatenate(roofs) if roofs else np.zeros((0, 2)); ground = np.concatenate(ground) if ground else np.zeros((0, 3))
record = dict(gps_sigma_m=round(sigma, 2), roof_returns=int(len(roofs)), components={}, applied=False)
gf = R / 'global-frame.json'; path_m = {int(k): v.get('path_m', 0.0) for k, v in (json.loads(gf.read_text()).get('components', {}) if gf.exists() else {}).items()}

# roof occupancy on a GRID raster (1 m roof cells, as the LiDAR spacing allows), its outline, and distance maps
o = lo[[0, 2]]; shape = tuple(np.ceil((hi[[0, 2]] - o) / GRID).astype(int) + 1)
occ = np.zeros(shape, bool)
if len(roofs):
    c1 = np.unique(np.floor(roofs).astype(np.int64), axis=0)          # 1 m cells with a roof return
    for dx in np.arange(0, 1, GRID):
        for dz in np.arange(0, 1, GRID):
            ij = np.floor((c1 + [dx, dz] - o) / GRID).astype(int); ok = (ij >= 0).all(1) & (ij < shape).all(1); occ[ij[ok, 0], ij[ok, 1]] = True
# Buildings are big solid blocks. The LiDAR's building class also has speckle (tree crowns, awnings) that would put
# fake outlines in the street, so keep only solid footprints: open away anything thinner than ~2 m, fill small holes,
# drop blobs under MIN_FOOTPRINT_M2.
occ = ndimage.binary_closing(ndimage.binary_opening(occ, iterations=int(round(1 / GRID))), iterations=int(round(1 / GRID)))
lbl, n = ndimage.label(occ)
if n:
    area = ndimage.sum(occ, lbl, index=np.arange(1, n + 1)) * GRID * GRID; occ = np.isin(lbl, 1 + np.flatnonzero(area >= MIN_FOOTPRINT_M2))
record['footprints'] = int((ndimage.label(occ)[1]))
np.savez_compressed(R / 'footprints.npz', occ=occ, origin=o, grid=GRID)   # the cleaned footprints, for tsdf.py
edge = occ & ~ndimage.binary_erosion(occ, iterations=int(round(1 / GRID)))
d_edge = ndimage.distance_transform_edt(~edge) * GRID if edge.any() else None      # metres to the nearest outline
depth_in = ndimage.distance_transform_edt(occ) * GRID                              # metres inside a footprint
def at(grid, P2):
    ij = np.clip(np.floor((P2 - o) / GRID).astype(int), 0, np.array(shape) - 1); return grid[ij[:, 0], ij[:, 1]]
def trespass(P2): return float(np.mean(at(depth_in, P2) > .5)) if len(P2) else 0.0   # walked ground inside a building: impossible
def tensor_angle(img, sigma_px, fold):
    gx = ndimage.sobel(img, 0); gz = ndimage.sobel(img, 1); th = np.arctan2(gz, gx); mag = np.hypot(gx, gz)
    C = ndimage.gaussian_filter(mag * np.cos(fold * th), sigma_px); S = ndimage.gaussian_filter(mag * np.sin(fold * th), sigma_px)
    M = ndimage.gaussian_filter(mag, sigma_px); return np.arctan2(S, C) / fold, np.hypot(C, S) / np.maximum(M, 1e-9)
# Score proximity and parallel orientation equally so walls cannot improve their fit by crossing roof outlines.
edge_n = tensor_angle(ndimage.gaussian_filter(occ.astype(float), 2), 4 / GRID, 2)[0]            # outline normal, mod 180
def wall_normals(P2):
    """Each wall point's own line normal (radians, mod 180), from a density raster of these walls."""
    ij = np.clip(np.floor((P2 - o) / GRID).astype(int), 0, np.array(shape) - 1); dens = np.zeros(shape); np.add.at(dens, (ij[:, 0], ij[:, 1]), 1.0)
    return tensor_angle(ndimage.gaussian_filter(dens, 1), 1 / GRID, 2)[0][ij[:, 0], ij[:, 1]]
def evidence(P2, th): return float(np.mean(np.exp(-.5 * at(d_edge, P2) ** 2) * (.5 + .5 * np.cos(th - at(edge_n, P2)) ** 2)))
def score(Wp, th, Gp): return evidence(Wp, th) - PENALTY * trespass(Gp)
def rigid(deg, c, dx, dz, dy=0.0):
    """The 4x4 (city frame) of yaw(., deg, c) + [dx, dz] in plan and dy in height."""
    t = np.radians(deg); ca, sa = np.cos(t), np.sin(t); M = np.eye(4)
    M[0, 0], M[0, 2], M[2, 0], M[2, 2] = ca, -sa, sa, ca
    M[0, 3] = c[0] - ca * c[0] + sa * c[1] + dx; M[2, 3] = c[1] - sa * c[0] - ca * c[1] + dz; M[1, 3] = dy; return M
def yaw(P2, deg, c):
    t = np.radians(deg); Rm = np.array([[np.cos(t), -np.sin(t)], [np.sin(t), np.cos(t)]]); return (P2 - c) @ Rm.T + c
def search(S, G2, c, span, reach, sy, ss, prior, th):
    """Best move of walls S and ground G2 (yaw about c, then shift): LiDAR evidence minus PRIOR times the move's squared
    size against its expected error (sy deg, ss m), so a move beyond what GPS and ARKit allow needs strong support.
    Coarse 1 deg / 1 m over +-span, +-reach, then 0.25 deg / 0.25 m. Returns (value, deg, dx, dz)."""
    def val(deg, dx, dz, Y, YG): return score(Y + [dx, dz], th + np.radians(deg), YG + [dx, dz]) - prior * ((dx * dx + dz * dz) / (2 * ss * ss) + deg * deg / (2 * sy * sy))
    best = (score(S, th, G2), 0.0, 0.0, 0.0)
    for deg in np.arange(-round(span), round(span) + .01, 1.0):
        Y, YG = yaw(S, deg, c), yaw(G2, deg, c)
        for dx in np.arange(-reach, reach + .01, 1.0):
            for dz in np.arange(-reach, reach + .01, 1.0):
                if dx * dx + dz * dz <= reach * reach: best = max(best, (val(deg, dx, dz, Y, YG), deg, dx, dz))
    _, d0, x0, z0 = best
    for deg in np.arange(d0 - .75, d0 + .76, .25):
        Y, YG = yaw(S, deg, c), yaw(G2, deg, c)
        for dx in np.arange(x0 - 1, x0 + 1.01, .25):
            for dz in np.arange(z0 - 1, z0 + 1.01, .25): best = max(best, (val(deg, dx, dz, Y, YG), deg, dx, dz))
    return best

# ---- 1. placement, per tracked part -------------------------------------------------------------------------------
wall = np.isin(lab, [2, 3])
if d_edge is not None and int(edge.sum()) * GRID >= MIN_EDGES:
    for k in np.unique(comp):
        mk = comp == k; W2 = xyz[mk & wall][:, [0, 2]]
        if len(W2) < MIN_WALL: record['components'][str(k)] = dict(wall_points=int(len(W2)), applied=False, why='too little wall'); continue
        S = W2[np.random.default_rng(0).permutation(len(W2))[:12000]]; c = S.mean(0)
        G2 = xyz[mk & np.isin(lab, [0, 1, 9])][:, [0, 2]]; G2 = G2[np.random.default_rng(1).permutation(len(G2))[:8000]]
        th = wall_normals(S); stay = score(S, th, G2)
        # place.py took this part's heading from its GPS track: good to ~atan(1.4 sigma / path), a few degrees on a long
        # walk but 10 deg or worse on a 25 m start where the phone mostly stood still
        sy = max(float(np.degrees(np.arctan2(1.4 * sigma, max(path_m.get(int(k), 0.0), 1.0)))), 2.0)
        span = float(np.clip(2.5 * sy, MAX_YAW_DEG, WIDE_YAW_DEG)); best = search(S, G2, c, span, reach, sy, sigma, 0.0, th)
        e, deg, dx, dz = best; moved_w = evidence(yaw(S, deg, c) + [dx, dz], th + np.radians(deg)); stay_w = evidence(S, th)
        ok = bool(deg or dx or dz) and moved_w >= GAIN * stay_w and e >= stay + .05
        rec = dict(wall_points=int(len(W2)), wall_on_outline_stay=round(stay_w, 3), wall_on_outline_moved=round(moved_w, 3),
                   ground_inside_stay=round(trespass(G2), 3), ground_inside_moved=round(trespass(yaw(G2, deg, c) + [dx, dz]), 3),
                   yaw_deg=round(float(deg), 2), shift_m=[round(float(dx), 2), round(float(dz), 2)], yaw_span_deg=round(span, 1), applied=bool(ok))
        if ok:
            P2 = yaw(xyz[mk][:, [0, 2]], deg, c) + [dx, dz]; xyz[np.flatnonzero(mk)[:, None], [0, 2]] = P2
            if len(ground) > 100:                                  # the ground under the new spot may sit at a different height
                gt = cKDTree(ground[:, [0, 2]]); g = mk & np.isin(lab, [0, 1, 9]); sel = np.flatnonzero(g)[:: max(1, int(g.sum()) // 20000)]
                if len(sel):
                    _, j = gt.query(xyz[sel][:, [0, 2]], k=8); dy = float(np.clip(np.median(np.median(ground[j, 1], axis=1) - xyz[sel, 1]), -1.5, 1.5))
                    xyz[mk, 1] += dy; rec['height_resettle_m'] = round(dy, 3)
            record['applied'] = True
            rec['matrix'] = rigid(deg, c, dx, dz, rec.get('height_resettle_m', 0.0)).tolist()
        record['components'][str(k)] = rec
else: record['why'] = 'too few roof outlines here'

# ---- 1b. per chunk: undo the rotation a group of frames got wrong (it drew the same facade a few degrees off, or ARKit's
# heading jumped by 20 deg or more mid-walk and the stabiliser kept the frames in one part) ------------------------------
if d_edge is not None and 'block-chunk.bin' in side and record.get('applied') is not None:
    chunk = side['block-chunk.bin'].astype(int); record['chunks'] = {}
    for k in np.unique(chunk):
        mk = chunk == k; W2 = xyz[mk & wall][:, [0, 2]]
        if len(W2) < MIN_CHUNK_WALL: continue
        S = W2[np.random.default_rng(0).permutation(len(W2))[:8000]]; c = S.mean(0)
        G2 = xyz[mk & np.isin(lab, [0, 1, 9])][:, [0, 2]]; G2 = G2[np.random.default_rng(1).permutation(len(G2))[:5000]]
        th = wall_normals(S); stay = score(S, th, G2); best = search(S, G2, c, WIDE_YAW_DEG, max(CHUNK_REACH_M, reach), CHUNK_YAW_SIGMA, CHUNK_SHIFT_SIGMA, CHUNK_PRIOR, th)
        e, deg, dx, dz = best; moved_w = evidence(yaw(S, deg, c) + [dx, dz], th + np.radians(deg)); stay_w = evidence(S, th)
        ok = bool(deg or dx or dz) and moved_w >= 1.1 * stay_w and e >= stay + .03
        record['chunks'][str(k)] = dict(wall_points=int(len(W2)), on_outline=[round(stay_w, 3), round(moved_w, 3)], yaw_deg=round(float(deg), 2), shift_m=[round(float(dx), 2), round(float(dz), 2)], applied=bool(ok))
        if ok: xyz[np.flatnonzero(mk)[:, None], [0, 2]] = yaw(xyz[mk][:, [0, 2]], deg, c) + [dx, dz]; record['chunks'][str(k)]['matrix'] = rigid(deg, c, dx, dz).tolist()

# ---- 2. impossible walls: deep inside a building footprint --------------------------------------------------------
keep = np.ones(len(pts), bool)
if len(roofs):
    inside = wall & (at(depth_in, xyz[:, [0, 2]]) > INSIDE_M); keep &= ~inside
    record['inside_walls_dropped'] = int(inside.sum()); record['wall_points'] = int(wall.sum())
if d_edge is not None and 'block-support.bin' in side:
    # a slab standing in open street, far from every building outline, with few frames behind it: a depth guess that
    # went wrong together (fences and real free-standing walls are seen from many frames and stay)
    sup = side['block-support.bin']; open_ = wall & keep & (at(depth_in, xyz[:, [0, 2]]) == 0) & (at(d_edge, xyz[:, [0, 2]]) > OPEN_M)
    slab = open_ & (sup < OPEN_SUPPORT); keep &= ~slab
    record.update(open_walls=int(open_.sum()), open_slabs_dropped=int(slab.sum()))
if len(roofs) and wall.any():
    # rotated copies: buildings here follow a local grid (their footprint edges run at some angle and square to it).
    # A wall that is clearly a straight line yet sits more than SKEW_DEG off that grid is the same facade seen through
    # frames whose rotation came out wrong. Both directions come from structure tensors: the footprint's edge angles
    # averaged modulo 90 deg over ~6 m (the local grid), and each wall's own line direction over ~1 m.
    grid_ang, grid_str = tensor_angle(ndimage.gaussian_filter(occ.astype(float), 2), 6 / GRID, 4)       # building grid, mod 90
    wi = np.flatnonzero(wall & keep); ijw = np.clip(np.floor((xyz[wi][:, [0, 2]] - o) / GRID).astype(int), 0, np.array(shape) - 1)
    dens = np.zeros(shape); np.add.at(dens, (ijw[:, 0], ijw[:, 1]), 1.0)
    wall_ang, wall_str = tensor_angle(ndimage.gaussian_filter(dens, 1), 1 / GRID, 2)                   # wall line normal, mod 180
    a = wall_ang[ijw[:, 0], ijw[:, 1]] - grid_ang[ijw[:, 0], ijw[:, 1]]
    dev = np.degrees(np.abs((a + np.pi / 4) % (np.pi / 2) - np.pi / 4))                                 # off the grid, 0..45 deg
    skew = (dev > SKEW_DEG) & (wall_str[ijw[:, 0], ijw[:, 1]] > .6) & (grid_str[ijw[:, 0], ijw[:, 1]] > .5) & (at(d_edge, xyz[wi][:, [0, 2]]) < NEAR_M)
    keep[wi[skew]] = False; record.update(rotated_copies_dropped=int(skew.sum()))
if 'block-chunk.bin' in side:
    # a chunk that lost most of its wall to the checks above was poorly shot (its depth or its pose is wrong), and its
    # ground and trees came from the same frames: drop the whole chunk. Not every frame has to be used.
    chunk = side['block-chunk.bin'].astype(int); record['chunks_dropped'] = []
    for k in np.unique(chunk):
        mk = chunk == k; n0 = int((mk & wall).sum())
        if n0 >= MIN_CHUNK_WALL and (mk & wall & keep).sum() < (1 - MAX_CHUNK_LOSS) * n0:
            keep &= ~mk; record['chunks_dropped'].append(dict(chunk=int(k), wall_points=n0, wall_kept=int((mk & wall & keep).sum())))
if edge.any() and (wall & keep).any():
    # ---- 3. one wall per facade. The cleaned outline splits into straight segments (edge pixels with the same normal,
    # 15 deg bins, connected); each segment gets one straight line. Its wall points (running along it, within SNAP_M)
    # vote along the line's normal, and the densest layer wins: the walk saw the facade there most often. Every layer
    # of that facade moves onto it, so a facade is one flat vertical plane instead of a stack.
    dist, (ei, ej) = ndimage.distance_transform_edt(~edge, return_indices=True)
    n_ang, _ = tensor_angle(ndimage.gaussian_filter(occ.astype(float), 2), 4 / GRID, 2)            # outline normal, mod 180, smooth
    bins = np.round(np.degrees(n_ang) % 180 / 15).astype(int) % 12; seg = np.zeros(shape, int); nseg = 0
    for b_ in range(12):
        lbl, n_ = ndimage.label(edge & (bins == b_), structure=np.ones((3, 3))); seg[lbl > 0] = lbl[lbl > 0] + nseg; nseg += n_
    wi = np.flatnonzero(wall & keep); ijw = np.clip(np.floor((xyz[wi][:, [0, 2]] - o) / GRID).astype(int), 0, np.array(shape) - 1)
    near = dist[ijw[:, 0], ijw[:, 1]] * GRID < SNAP_M; wi, ijw = wi[near], ijw[near]
    dens = np.zeros(shape); np.add.at(dens, (ijw[:, 0], ijw[:, 1]), 1.0)
    w_ang, _ = tensor_angle(ndimage.gaussian_filter(dens, 1), 1 / GRID, 2)                          # the wall's own normal
    sid = seg[ei[ijw[:, 0], ijw[:, 1]], ej[ijw[:, 0], ijw[:, 1]]]; moved_ids = []
    for k in np.unique(sid):
        if k == 0: continue
        ep = np.argwhere(seg == k) * GRID + o + GRID / 2
        if len(ep) < int(2 / GRID): continue                                                         # under 2 m of outline: not a facade
        cq = ep.mean(0); _, vec = np.linalg.eigh(np.cov((ep - cq).T)); nrm = vec[:, 0]                # the segment's straight line and normal
        m = sid == k; P2 = xyz[wi[m]][:, [0, 2]]
        wn = np.column_stack([np.cos(w_ang[ijw[m, 0], ijw[m, 1]]), np.sin(w_ang[ijw[m, 0], ijw[m, 1]])])
        par = np.abs(wn @ nrm) > np.cos(np.radians(PARALLEL_DEG))                                    # runs along the facade
        if par.sum() < 50: continue
        off = (P2[par] - cq) @ nrm
        h, edges_ = np.histogram(off, bins=np.arange(off.min() - .2, off.max() + .3, .1)); h = ndimage.gaussian_filter1d(h.astype(float), 2)
        mode = edges_[np.argmax(h)] + .05                                                            # the densest layer
        idx = wi[m][par]; xyz[idx[:, None], [0, 2]] += ((mode - off)[:, None] * nrm); moved_ids.append(idx)
    wi = np.concatenate(moved_ids) if moved_ids else np.zeros(0, int)
    record.update(facade_segments=int(len(moved_ids)), facade_points_flattened=int(len(wi)))
    flat = np.zeros(len(pts), bool); flat[wi] = True
else: flat = np.zeros(len(pts), bool)
# walls no outline speaks for (no LiDAR building next to them) still become one sheet: their layers collapse onto the
# median sheet within 2.5 m. Colours from the layers blend a little; one wall is worth more than a clean texture.
rest = np.flatnonzero(wall & keep & ~flat)
if len(rest) > 1000: xyz[rest] = collapse(xyz[rest], 2.5); record['other_walls_collapsed'] = int(len(rest))
if d_edge is not None:
    w2 = xyz[wall & keep][:, [0, 2]]; record['walls_within_1m_of_outline'] = round(float(np.mean(at(d_edge, w2) < 1)), 3) if len(w2) else None
pts['xyz'] = xyz.astype(np.float32); pts = pts[keep]; pts.tofile(R / 'block-consistent.bin')
for name, v in side.items(): v[keep].tofile(R / name)
gf = R / 'global-frame.json'; g = json.loads(gf.read_text()) if gf.exists() else {}
g['refinement'] = record; gf.write_text(json.dumps(g, indent=1))
print('refinement', json.dumps(record))
