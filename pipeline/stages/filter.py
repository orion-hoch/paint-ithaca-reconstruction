"""Filter and fuse depth estimates using multi-view consistency and LiDAR support.

Merge duplicate frame depths, calibrate scale against ARKit feature points, and
retain semantic classes supported by other views or nearby LiDAR ground. Select
the nearest wall surfaces and remove unsupported sheets above the LiDAR ceiling.
Writes block-consistent.bin, point sidecars and consistency-stats.json; optionally
retains frame evidence for tsdf.py when DUMP_FRAMES=1.
"""
import json, os
from collections import defaultdict
from pathlib import Path
import numpy as np
from sheets import collapse, farther, nearest_sheet

R = Path(os.environ['RUN'])
REL_TOL, MIN_AGREE, MIN_SCORE = float(os.environ.get('REL_TOL', .05)), int(os.environ.get('MIN_AGREE', 1)), .60
SRC = 'mapanything-posed'; LAZ = os.environ.get('LAZ')
FUSE = os.environ.get('FUSE') == '1'
REFINE = int(os.environ.get('REFINE', 0)); REFINE_TOL = .25            # ray-consensus passes; loose gate for candidates
BASELINE_M = float(os.environ.get('BASELINE_M', 0))                    # a corroborating view must stand this far away (sky lands on the same pixels otherwise)
GROUND_BYPASS_M = 15.0                                                  # ponytail: fixed range for the LiDAR-ground pass; tie it to the depth calibration if streets still fan out
DEPTH_CALIB = os.environ.get('DEPTH_CALIB'); calib = None
GRAZE_DEG = float(os.environ.get('GRAZE_DEG', 0))                     # ground seen flatter than this is dropped: its range error smears along the ray (fans)
CELL = {0: .10, 1: .10, 9: .10, 2: .10, 3: .10, 8: .30}
POINT = np.dtype([('xyz', '<f4', 3), ('rgba', 'u1', 4)])
NAMES = {0: 'road', 1: 'sidewalk', 2: 'building', 3: 'wall', 8: 'vegetation', 9: 'terrain'}
ORIGIN = np.array([377249.9995, 4700249.9995, 150.0])
def tf(P, M): return P @ M[:3, :3].T + M[:3, 3]

PLACEMENT = f'placement-{SRC}.json'
def _ok(p):
    if not (p / PLACEMENT).exists(): return False
    g = json.loads((p / PLACEMENT).read_text())['icp'].get('median_after_m'); return g is not None and g <= 1.5
chunks = sorted(p for p in (R / 'chunks').glob('c*') if _ok(p))
print('held (ground > 1.5 m):', [p.name for p in sorted((R / 'chunks').glob('c*')) if (p / PLACEMENT).exists() and not _ok(p)])
if not chunks: raise SystemExit('no chunk settled onto the LiDAR ground; nothing to keep')
G_of = {c.name: np.array(json.loads((c / PLACEMENT).read_text())['matrix_model_to_city']) for c in chunks}
comp_of = {c.name: json.loads((c / 'index.json').read_text())['component'] for c in chunks}   # one ARKit frame per component
chunk_ix_all = {c.name: k for k, c in enumerate(chunks)}

# ---- load frames -------------------------------------------------------------------------------------
F = {}; by_file = defaultdict(list); scale0 = {}
for c in chunks:
    idx = json.loads((c / 'index.json').read_text()); byname = {f['file']: f for f in idx['frames']}
    res = json.loads((c / SRC / 'result.json').read_text()); pl = json.loads((c / PLACEMENT).read_text())
    scale0[c.name] = pl['scale']['feature_points'] or pl['scale']['used']
    for i, cam in enumerate(res['cameras']):
        f = np.load(c / SRC / f'frame_{i:04d}.npz')
        F[(c.name, i)] = dict(D=f['depth'].squeeze().astype(np.float64), valid=f['valid'].squeeze(), conf=f['confidence'].squeeze().astype(np.float32),
                              labels=f['labels'].squeeze(), score=f['label_score'].squeeze(), colors=f['colors'], K=np.array(cam['intrinsics'], float),
                              c2w=np.array(cam.get('c2w_used') or byname[cam['image']]['c2w_cv'], float), file=cam['image'])
        by_file[cam['image']].append((c.name, i))

# ---- one depth map per frame ----------------------------------------------------------------------------------------
# Chunks overlap by half, so every frame was inferred twice, and two inferences of one image disagree by 5-10% in depth:
# kept side by side that is a second facade 0.5-1 m behind the first, which the agreement test cannot tell from a real
# wall because each version is backed by its own chunk. Merge the versions per pixel and keep the frame in ONE chunk
# (the one it sits most centrally in); the chunk links below still let its points be checked against both chunks.
link = defaultdict(set)                                                  # chunks that shared a frame stay each other's witnesses
for file, occ in by_file.items():
    for a in occ:
        for b in occ: link[a[0]].add(b[0])
n_in = defaultdict(int)
for (c, i) in F: n_in[c] += 1
versions = sum(len(o) for o in by_file.values())
for file, occ in by_file.items():
    if len(occ) < 2: continue
    Ds = np.stack([F[k]['D'] for k in occ]); valids = np.stack([F[k]['valid'] for k in occ])
    merged = np.nanmedian(np.where(valids, Ds, np.nan), axis=0); vm = valids.sum(0) * 2 > len(occ)   # majority valid
    merged[~vm] = 0
    keep = max(occ, key=lambda k: min(k[1], n_in[k[0]] - 1 - k[1]))
    F[keep]['D'] = merged; F[keep]['valid'] = vm
    for k in occ:
        if k != keep: del F[k]
by_file = {file: [k for k in occ if k in F] for file, occ in by_file.items()}
print(f'frames: {len(F)} kept from {versions} chunk versions')

# ---- depth-response calibration from ARKit metric feature points (VIO): log(true) = a + b*log(pred) ----------------
# Fit one robust log-log depth correction across the capture; calibrated chunk scale anchors become 1.
if DEPTH_CALIB:
    man = json.loads((R / 'capture/manifest.json').read_text()); byidx = {fr['video_frame_index']: fr for fr in man['frames']}
    ROTC = np.array([[0., 1., 0.], [-1., 0., 0.], [0., 0., 1.]]); xs, ys = [], []
    for c in chunks:
        idx = json.loads((c / 'index.json').read_text()); byname = {fr['file']: fr for fr in idx['frames']}; res = json.loads((c / SRC / 'result.json').read_text())
        for i, cam in enumerate(res['cameras']):
            if (c.name, i) not in F: continue
            f = F[(c.name, i)]; raw = byidx[byname[cam['image']]['video_frame_index']]; fp = raw.get('feature_points') or {}; pt = np.array(fp.get('points') or fp.get('xyz') or [], float)
            if pt.ndim != 2 or len(pt) < 5: continue
            Traw = np.array(raw['transform'], float); Pf = (pt - Traw[:3, 3]) @ Traw[:3, :3] * np.array([1, -1, -1])
            if cam.get('upright'): Pf = Pf @ ROTC
            dep = Pf[:, 2]; ok = dep > .5; K = f['K']; H, W = f['D'].shape
            pu = np.round(K[0, 0] * Pf[ok, 0] / dep[ok] + K[0, 2]).astype(int); pv = np.round(K[1, 1] * Pf[ok, 1] / dep[ok] + K[1, 2]).astype(int); dep = dep[ok]
            inb = (pu >= 0) & (pu < W) & (pv >= 0) & (pv < H); pu, pv, dep = pu[inb], pv[inb], dep[inb]
            md = f['D'][pv, pu]; good = f['valid'][pv, pu] & (md > .1); xs += list(np.log(md[good])); ys += list(np.log(dep[good]))
    xs, ys = np.array(xs), np.array(ys); keep = np.ones(len(xs), bool); mode = DEPTH_CALIB; a_, b_, dmax = 0.0, 1.0, None
    # Keep per-chunk scales with too few features; use scale-only calibration with too few bins or an implausible slope.
    if len(xs) < 30: mode = 'none'
    else:
        for _ in range(3):                                                   # LS line with 2 rounds of outlier trimming (|res| > 2.5 MAD)
            b_, a_ = np.polyfit(xs[keep], ys[keep], 1); res_ = ys - (a_ + b_ * xs); mad = np.median(np.abs(res_[keep] - np.median(res_[keep]))) + 1e-6; keep = np.abs(res_) < 2.5 * 1.4826 * mad
        if mode == 'binned':                                                 # equal weight per range bin (of predicted depth), so the far bins count
            be = np.log([1, 1.5, 2, 3, 5, 8, 12, 20]); bx, by, top = [], [], None
            for lo, hi in zip(be[:-1], be[1:]):
                m = keep & (xs >= lo) & (xs < hi)
                if m.sum() >= 15: bx.append(np.median(xs[m])); by.append(np.median(ys[m])); top = hi
            if len(bx) >= 3:
                b_, a_ = np.polyfit(bx, by, 1); dmax = float(np.exp(top)); print('  bin medians (pred m -> true m):', [(round(float(np.exp(x)), 1), round(float(np.exp(y)), 1)) for x, y in zip(bx, by)])
            else: mode = 'scale-only'
        if mode == 'binned' and not .8 <= b_ <= 1.35: mode = 'scale-only'
        if mode == 'scale-only': b_ = 1.0; a_ = float(np.median(ys[keep] - xs[keep]))
        res_ = ys - (a_ + b_ * xs)
    if mode != 'none':
        # Past the last calibrated range the power law is an extrapolation; CALIB_FAR=hold keeps the ratio it reached there.
        hold = os.environ.get('CALIB_FAR', 'power') == 'hold' and dmax is not None
        print(f'depth calibration ({mode}) from {int(keep.sum())}/{len(xs)} ARKit features: true = {np.exp(a_):.3f} * pred^{b_:.3f}' + (f', ratio held past {dmax:.0f} m' if hold else '') + f'  (residual IQR {np.percentile(res_[keep], 25):+.3f}..{np.percentile(res_[keep], 75):+.3f})')
        for k, f in F.items():
            D = f['D']; ok = D > .1
            D[ok] = np.exp(a_) * (np.minimum(D[ok], dmax) ** b_ * np.maximum(D[ok] / dmax, 1) if hold else D[ok] ** b_)
        scale0 = {n: 1.0 for n in scale0}
    else: print(f'depth calibration skipped: {len(xs)} ARKit features; chunk scales from place.py')
    calib = dict(mode=mode, a=round(float(a_), 4), b=round(float(b_), 4), features=int(len(xs)), calibrated_to_m=dmax)

# ---- A. scale chaining -------------------------------------------------------------------------------
names = [c.name for c in chunks]; pairs = {}
for file, occ in by_file.items():
    for (ca, ia) in occ:
        for (cb, ib) in occ:
            if ca >= cb: continue
            A, B = F[(ca, ia)], F[(cb, ib)]; m = A['valid'] & B['valid'] & (A['D'] > .1) & (B['D'] > .1)
            if m.sum() > 500: pairs.setdefault((ca, cb), []).append(float(np.median(np.log(A['D'][m]) - np.log(B['D'][m]))))
rows, rhs, w = [], [], []; ix = {n: k for k, n in enumerate(names)}
for (a, b), v in pairs.items():
    r = np.zeros(len(names)); r[ix[b]] = 1; r[ix[a]] = -1; rows.append(r); rhs.append(float(np.median(v))); w.append(10.0)
for n in names:
    r = np.zeros(len(names)); r[ix[n]] = 1; rows.append(r); rhs.append(float(np.log(scale0[n]))); w.append(1.0)
Aw = np.array(rows) * np.array(w)[:, None]; bw = np.array(rhs) * np.array(w)
x = np.linalg.lstsq(Aw, bw, rcond=None)[0]; scale = {n: float(np.exp(x[ix[n]])) for n in names}
print('scale chaining (feature estimate -> chained):', {n: (round(scale0[n], 3), round(scale[n], 3)) for n in names})

# ---- B. back-project + reprojection consistency (+ optional fusion) ---------------------------------------
pts, pix2idx = {}, {}
for key, f in F.items():
    D, K = f['D'], f['K']; H, W = D.shape; v, u = np.mgrid[0:H, 0:W]; s = scale[key[0]]
    X = (u - K[0, 2]) / K[0, 0] * D * s; Y = (v - K[1, 2]) / K[1, 1] * D * s; Z = D * s
    P = np.stack([X, Y, Z], -1)[f['valid']]; pts[key] = tf(P, f['c2w'])
    m = np.full((H, W), -1, np.int64); m[f['valid']] = np.arange(int(f['valid'].sum())); pix2idx[key] = m
    f['w2c'] = np.linalg.inv(f['c2w']); f['Ds'] = D * s

# ---- ray-consensus refinement: each frame's depth -> median of what the other frames imply along its rays ----------
def refine_pass():
    new_pts = {}
    for key, P in pts.items():
        f = F[key]; w2c_i = f['w2c']; z_i = tf(P, w2c_i)[:, 2]; cands = [z_i.copy()]
        for other, g in F.items():
            if other == key or other[0] not in link[key[0]]: continue
            Pc = tf(P, g['w2c']); z = Pc[:, 2]; ok = z > .1; K = g['K']; H, W = g['D'].shape
            u = np.round(K[0, 0] * Pc[:, 0] / np.where(ok, z, 1) + K[0, 2]).astype(int); v = np.round(K[1, 1] * Pc[:, 1] / np.where(ok, z, 1) + K[1, 2]).astype(int)
            inb = ok & (u >= 0) & (u < W) & (v >= 0) & (v < H)
            zc = np.full(len(P), np.nan)
            if inb.any():
                j = pix2idx[other][v[inb], u[inb]]; good = j >= 0
                Pj = pts[other][j[good]]                                   # frame j's own surface point at that pixel
                zj_i = tf(Pj, w2c_i)[:, 2]                                 # its depth along (near) ray i
                sel = np.flatnonzero(inb)[good]; rel = np.abs(zj_i - z_i[sel]) / np.maximum(z_i[sel], 1e-6)
                keep = rel < REFINE_TOL; zc[sel[keep]] = zj_i[keep]
            cands.append(zc)
        Z = np.vstack(cands); zmed = np.nanmedian(Z, axis=0)                # own estimate + every consistent-ish neighbour
        ratio = zmed / np.maximum(z_i, 1e-6); new_pts[key] = f['c2w'][:3, 3] + (P - f['c2w'][:3, 3]) * ratio[:, None]   # slide along the ray
    return new_pts
for it in range(REFINE):
    pts = refine_pass()
    for key, f in F.items():                                                # refresh depth maps used as reprojection targets
        z = tf(pts[key], f['w2c'])[:, 2]; Ds = np.zeros_like(f['Ds']); Ds[f['valid']] = z; f['Ds'] = Ds
    print(f'refine pass {it + 1} done')

if os.environ.get('DUMP_FRAMES') == '1':                                      # the per-frame evidence, for volumetric fusion (tsdf.py)
    dump = R / 'frames-dump'; dump.mkdir(exist_ok=True)
    for (cn, i), f in F.items():
        np.savez_compressed(dump / f'{cn}_{i:04d}.npz', depth=f['Ds'].astype(np.float32), valid=f['valid'], colors=f['colors'], labels=f['labels'], score=f['score'],
                 K=f['K'], c2w=f['c2w'], G=G_of[cn], chunk=chunk_ix_all[cn], comp=comp_of[cn])
    print(f'dumped {len(F)} frames for fusion')
agree, fused = {}, {}
for key, P in pts.items():
    n = np.zeros(len(P), np.int16); acc = P.copy() if FUSE else None
    for other, g in F.items():
        if other == key or other[0] not in link[key[0]]: continue
        if BASELINE_M and np.linalg.norm(g['c2w'][:3, 3] - F[key]['c2w'][:3, 3]) < BASELINE_M: continue
        Pc = tf(P, g['w2c']); z = Pc[:, 2]; ok = z > .1
        K = g['K']; H, W = g['D'].shape
        u = np.round(K[0, 0] * Pc[:, 0] / np.where(ok, z, 1) + K[0, 2]).astype(int); v = np.round(K[1, 1] * Pc[:, 1] / np.where(ok, z, 1) + K[1, 2]).astype(int)
        inb = ok & (u >= 0) & (u < W) & (v >= 0) & (v < H)
        d = np.full(len(P), np.nan); d[inb] = g['Ds'][v[inb], u[inb]]; vv = np.zeros(len(P), bool); vv[inb] = g['valid'][v[inb], u[inb]]
        a = vv & (np.abs(d - z) / np.maximum(z, 1e-6) < REL_TOL); n += a.astype(np.int16)
        if FUSE and a.any(): acc[a] += pts[other][pix2idx[other][v[a], u[a]]]
    agree[key] = n
    if FUSE: fused[key] = acc / (1 + n)[:, None]

# ---- ground reference (optional) --------------------------------------------------------------------------
gh = None
if LAZ:
    from scipy.spatial import cKDTree
    from laz import read_city
    LA, cls = read_city(LAZ); gr = LA[np.isin(cls, [2, 17])]
    gtree = cKDTree(gr[:, [0, 2]])
    def gh(xz): _, i = gtree.query(xz, k=8); return np.median(gr[i, 1], axis=1)

# ---- C. gather, class gate, cells with max-confidence winner ---------------------------------------------
xyz, rgb, lab, conf, ag, rng_, cls_stats = [], [], [], [], [], [], defaultdict(lambda: [0, 0]); xa, comp_, chunk_ = [], [], []; chunk_ix = {c.name: k for k, c in enumerate(chunks)}
for key, P in pts.items():
    f = F[key]; m = f['valid']; Pw = fused[key] if FUSE else P
    l = f['labels'][m]; q = f['score'][m]; c = f['conf'][m]; col = f['colors'][m]
    keep = np.isin(l, list(CELL)) & (q >= MIN_SCORE)
    for cl in np.unique(l[keep]):
        sel = keep & (l == cl); cls_stats[int(cl)][0] += int(sel.sum()); cls_stats[int(cl)][1] += int((sel & (agree[key] >= MIN_AGREE)).sum())
    ok = agree[key] >= MIN_AGREE
    if gh is not None:
        # Ground may pass by lying on the LiDAR ground, but only near the camera: a road sheet predicted at the wrong
        # range still lies on the ground plane, and those wrong-range sheets are the fans that radiate down the street.
        near = tf(Pw, f['w2c'])[:, 2] <= GROUND_BYPASS_M
        Pc = tf(Pw, G_of[key[0]]); ground_like = np.isin(l, [0, 1, 9]); sel = np.flatnonzero(ground_like & keep & near)
        if len(sel): ok[sel] |= np.abs(gh(Pc[sel][:, [0, 2]]) - Pc[sel, 1]) <= .35
    keep &= ok
    if GRAZE_DEG:
        ray = Pw - f['c2w'][:3, 3]; sin_el = np.abs(ray[:, 1]) / np.maximum(np.linalg.norm(ray, axis=1), 1e-6)   # ARKit world is gravity-aligned, ground normal = +y
        keep &= ~(np.isin(l, [0, 1, 9]) & (sin_el < np.sin(np.radians(GRAZE_DEG))))
    xa.append(Pw[keep]); comp_.append(np.full(int(keep.sum()), comp_of[key[0]])); chunk_.append(np.full(int(keep.sum()), chunk_ix[key[0]]))
    xyz.append(tf(Pw[keep], G_of[key[0]])); rgb.append(col[keep]); lab.append(l[keep]); conf.append(c[keep]); ag.append(agree[key][keep]); rng_.append(np.linalg.norm(Pw[keep] - f['c2w'][:3, 3], axis=1))
xa = np.concatenate(xa); comp_ = np.concatenate(comp_); chunk_ = np.concatenate(chunk_); xyz = np.concatenate(xyz); rgb = np.concatenate(rgb); lab = np.concatenate(lab); conf = np.concatenate(conf); ag = np.concatenate(ag); rng_ = np.concatenate(rng_)
if not len(xyz): raise SystemExit('no point survived the agreement test')
cell = np.array([CELL[int(l)] for l in lab]); ci = np.floor(xyz / cell[:, None]).astype(np.int64)
key = (ci[:, 0] * 73856093) ^ (ci[:, 1] * 19349663) ^ (ci[:, 2] * 83492791) ^ (lab.astype(np.int64) << 40)
o = np.lexsort((-conf, key)); ks = key[o]; first = np.concatenate([[True], ks[1:] != ks[:-1]]); win = o[first]
out = np.zeros(len(win), POINT); out['xyz'] = xyz[win]; out['rgba'][:, :3] = rgb[win]; out['rgba'][:, 3] = lab[win]; out_rng = rng_[win]; out_comp = comp_[win]; out_chunk = chunk_[win]
# ---- visibility vote: a wall is real when the walk keeps seeing it there -----------------------------------------
# The agreement test above asks one other frame of a neighbouring chunk, so a slab that a few frames got wrong
# together passes. Here EVERY frame of the walk that can see a wall point votes: its own depth at that pixel agrees
# (support), lies clearly beyond the point (it sees through it: nothing is there), or lies in front (occluded: no
# vote). A wall point stays when at least VOTE_MIN frames see it there and more frames see it than see through it.
VOTE_MIN, VOTE_TOL, SEE_THROUGH = int(os.environ.get('VOTE_MIN', 0)), .10, .20
out_sup = np.full(len(out), 255, np.uint8)       # frames that saw each wall point there (255 = not a wall)
if True:
    w = np.flatnonzero(np.isin(out['rgba'][:, 3], [2, 3])); Xa = xa[win][w]; Cw = comp_[win][w]
    sup = np.zeros(len(w), np.int32); thr = np.zeros(len(w), np.int32)
    for key, f in F.items():
        i = np.flatnonzero(Cw == comp_of[key[0]])
        if not len(i): continue
        Pc = tf(Xa[i], f['w2c']); z = Pc[:, 2]; K = f['K']; H, W = f['Ds'].shape; ok = z > .3
        u = np.round(K[0, 0] * Pc[:, 0] / np.where(ok, z, 1) + K[0, 2]).astype(int); v = np.round(K[1, 1] * Pc[:, 1] / np.where(ok, z, 1) + K[1, 2]).astype(int)
        ok &= (u >= 0) & (u < W) & (v >= 0) & (v < H); i, z, u, v = i[ok], z[ok], u[ok], v[ok]
        d = f['Ds'][v, u]; seen = f['valid'][v, u] & (d > .1)
        sup[i[seen & (np.abs(d - z) < VOTE_TOL * z)]] += 1; thr[i[seen & (d > z * (1 + SEE_THROUGH))]] += 1
    out_sup[w] = np.minimum(sup, 254)
    phantom = ((sup < VOTE_MIN) | (thr >= sup)) if VOTE_MIN else np.zeros(len(w), bool)
    keep_v = np.ones(len(out), bool); keep_v[w[phantom]] = False; out = out[keep_v]; out_rng = out_rng[keep_v]; out_comp = out_comp[keep_v]; out_sup = out_sup[keep_v]; out_chunk = out_chunk[keep_v]
    print(f'visibility vote: dropped {int(phantom.sum()):,} of {len(w):,} wall points (support median {int(np.median(sup))}, see-through median {int(np.median(thr))})')
# ---- one sheet per wall ---------------------------------------------------------------------------------------
# Prefer the nearer view of duplicate facades to preserve depth and colour detail, then flatten the remaining noise.
COLLAPSE_M, COLLAPSE_TUBE_M = float(os.environ.get('COLLAPSE_M', 1.0)), .6
NEARER_M, NEARER_RATIO = float(os.environ.get('NEARER_M', 2.5)), .6          # drop a wall point when a view 40% nearer saw the same surface within NEARER_M
if np.isin(out['rgba'][:, 3], [2, 3]).sum() > 1000:
    w = np.flatnonzero(np.isin(out['rgba'][:, 3], [2, 3])); ks = nearest_sheet(out['xyz'][w].astype(np.float64), out_rng[w].astype(np.float64))
    keep2 = np.ones(len(out), bool); keep2[w[~ks]] = False; out = out[keep2]; out_rng = out_rng[keep2]; out_comp = out_comp[keep2]; out_sup = out_sup[keep2]; out_chunk = out_chunk[keep2]
    print(f'sheet select: dropped {int((~ks).sum()):,} of {len(w):,} wall points on a farther sheet')
    walls = np.isin(out['rgba'][:, 3], [2, 3])
    out['xyz'][walls] = collapse(out['xyz'][walls].astype(np.float64), COLLAPSE_M, COLLAPSE_TUBE_M).astype(np.float32)
    ci2 = np.floor(out['xyz'] / .10).astype(np.int64); k2 = (ci2[:, 0] * 73856093) ^ (ci2[:, 1] * 19349663) ^ (ci2[:, 2] * 83492791) ^ (out['rgba'][:, 3].astype(np.int64) << 40)
    o2 = np.lexsort((out_rng, k2)); ks2 = k2[o2]; first2 = o2[np.concatenate([[True], ks2[1:] != ks2[:-1]])]   # cells merge again; the nearest view keeps each
    before = len(out); sel2 = np.sort(first2); out = out[sel2]; out_rng = out_rng[sel2]; out_comp = out_comp[sel2]; out_sup = out_sup[sel2]; out_chunk = out_chunk[sel2]
    print(f'sheet flatten: {before:,} -> {len(out):,} cells')
    if NEARER_M:                                                                                 # a far view of a wall yields to a much nearer one
        w = np.flatnonzero(np.isin(out['rgba'][:, 3], [2, 3])); nearer = farther(out['xyz'][w].astype(np.float64), out_rng[w], NEARER_M, NEARER_RATIO)
        keep2 = np.ones(len(out), bool); keep2[w[nearer]] = False; out = out[keep2]; out_rng = out_rng[keep2]; out_comp = out_comp[keep2]; out_sup = out_sup[keep2]; out_chunk = out_chunk[keep2]
        print(f'nearer-view gate: dropped {int(nearer.sum()):,} wall points seen only from far away')

# ---- sail cut: drop thin unsupported sheets that float above what the LiDAR can see ---------------------------
# Misclassified sky can survive multi-view agreement. Require thin geometry and missing support as well as height
# above the LiDAR ceiling, since a height-only cut would remove real buildings with sparse LiDAR coverage.
sail_stats = None
if LAZ:
    from scipy import sparse
    # LA holds every class: the ceiling is whatever the scan saw
    lo_, hi_ = out['xyz'].min(0) - 30, out['xyz'].max(0) + 30
    LA = LA[(LA[:, 0] > lo_[0]) & (LA[:, 0] < hi_[0]) & (LA[:, 2] > lo_[2]) & (LA[:, 2] < hi_[2])]
    sail_stats = dict(lidar_points=int(len(LA)), above_ceiling=0, clusters=0, cut=0)
    if len(LA) > 1000:
        CEIL_CELL, ABOVE_M, THIN_M, SUPPORT = 2.0, 3.0, .60, .50
        lc = np.floor(LA[:, [0, 2]] / CEIL_CELL).astype(np.int64)
        lk, li = np.unique(lc, axis=0, return_inverse=True)
        topz = np.full(len(lk), -1e9); np.maximum.at(topz, li, LA[:, 1])
        at = {tuple(k): i for i, k in enumerate(lk)}
        pcell = np.floor(out['xyz'][:, [0, 2]] / CEIL_CELL).astype(np.int64)
        ceil = np.full(len(out), np.nan)
        for dx in (-1, 0, 1):                                    # 3x3 dilation absorbs the placement slack
            for dz in (-1, 0, 1):
                j = np.array([at.get((a + dx, b + dz), -1) for a, b in pcell]); ok = j >= 0
                ceil[ok] = np.fmax(ceil[ok], topz[j[ok]])
        high = np.flatnonzero(np.isfinite(ceil) & (out['xyz'][:, 1] - ceil > ABOVE_M))
        sail_stats['above_ceiling'] = int(len(high))
        if len(high) > 50:
            grounded = set(map(tuple, np.floor(out['xyz'][np.isfinite(ceil) & (out['xyz'][:, 1] - ceil < 1.0)][:, [0, 2]]).astype(np.int64)))
            Qh = out['xyz'][high].astype(np.float64); kt = cKDTree(Qh)
            adj = kt.sparse_distance_matrix(kt, .35, output_type='coo_matrix')
            ncl, comp = sparse.csgraph.connected_components(sparse.csr_matrix((np.ones(adj.nnz), (adj.row, adj.col)), shape=(len(Qh),) * 2), directed=False)
            sail_stats['clusters'] = int(ncl); drop = np.zeros(len(out), bool)
            for k in range(ncl):
                mk = comp == k
                if mk.sum() < 50: continue
                C = Qh[mk] - Qh[mk].mean(0); thin = float(np.sqrt(max(np.linalg.eigvalsh(C.T @ C / mk.sum())[0], 0)))
                cols = np.unique(np.floor(Qh[mk][:, [0, 2]]).astype(np.int64), axis=0)
                if thin < THIN_M and np.mean([tuple(a) in grounded for a in cols]) < SUPPORT: drop[high[mk]] = True
            sail_stats['cut'] = int(drop.sum()); out = out[~drop]; out_rng = out_rng[~drop]; out_comp = out_comp[~drop]; out_sup = out_sup[~drop]; out_chunk = out_chunk[~drop]
    print('sail cut:', json.dumps(sail_stats))

out.tofile(R / 'block-consistent.bin'); out_rng.astype(np.float16).tofile(R / 'block-range.bin'); out_comp.astype(np.uint8).tofile(R / 'block-comp.bin'); out_sup.tofile(R / 'block-support.bin'); out_chunk.astype(np.uint16).tofile(R / 'block-chunk.bin')   # which ARKit component (tracked part) each point came from, for refine.py   # per-point viewing range: lets the city fusion prefer nearer views across walks
stats = dict(depth_calibration=calib, graze_deg=GRAZE_DEG, rel_tol=REL_TOL, min_agree=MIN_AGREE, baseline_m=BASELINE_M, fuse=FUSE, refine=REFINE, refine_tol=REFINE_TOL, depth_calib=DEPTH_CALIB,
             chained_scale=scale, feature_scale=scale0,
             kept_fraction_by_class={NAMES[k]: (round(v[1] / max(1, v[0]), 3), v[0]) for k, v in sorted(cls_stats.items())},
             consistent_points=int(len(xyz)), cells=int(len(win)), cells_by_class={NAMES[int(k)]: int((lab[win] == k).sum()) for k in np.unique(lab[win])},
             sail_cut=sail_stats, agree_hist={int(k): int(v) for k, v in zip(*np.unique(ag[win], return_counts=True))})
(R / 'consistency-stats.json').write_text(json.dumps(stats, indent=1)); print(json.dumps(stats, indent=1))
