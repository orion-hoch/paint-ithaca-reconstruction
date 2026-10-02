"""Place each continuous ARKit component in the shared geographic frame.

Fit yaw, east, south and height against GPS and LiDAR ground. Back-project
MapAnything depth through each ARKit pose; estimate chunk scale from feature
points or camera height, then settle component heights onto LiDAR ground.
Writes global-frame.json and per-chunk placement-mapanything-posed.json.
"""
import json, os
from pathlib import Path
import numpy as np
from scipy.spatial import cKDTree
from laz import read_city

R = Path(os.environ['RUN'])
LAZ = os.environ['LAZ']                       # one tile, or the tiles the walk touches joined with commas
ORIGIN = np.array([377249.9995, 4700249.9995, 150.0])
CAMERA_HEIGHT_M, MIN_SCORE = 1.65, .60
GROUND_LIKE = [0, 1, 9]
SRC = 'mapanything-posed'

def city_from_EN(E, N, Z): return np.column_stack([E - ORIGIN[0], Z - ORIGIN[2], ORIGIN[1] - N])
def tf(P, M): return P @ M[:3, :3].T + M[:3, 3]

L, cls = read_city(LAZ); ground = L[np.isin(cls, [2, 17])]
gtree = cKDTree(ground[:, [0, 2]])
def ground_h(xz): _, i = gtree.query(xz, k=8); return np.median(ground[i, 1], axis=1)

man = json.loads((R / 'capture/manifest.json').read_text()); byidx = {f['video_frame_index']: f for f in man['frames']}
chunks = sorted(p for p in (R / 'chunks').glob('c*') if (p / SRC / 'cloud.npz').exists())

# ---- 4-DoF per rigid component: corrected ARKit (x east, y up, z south) -> city ---------------------------------
# A rigid component is one ARKit frame, so it gets its own translation + height. Yaw needs a long enough GPS track
# (>= YAW_MIN_PATH_M) to be determinable. Shorter components are held out when the walk has a long one; when it has
# none they inherit the capture-wide yaw (ARKit heading alignment is shared across the session).
YAW_MIN_PATH_M = 15.0
def fit2d(a2, tgt, take):
    ms, md = a2[take].mean(0), tgt[take].mean(0); U, S_, Vt = np.linalg.svd((tgt[take] - md).T @ (a2[take] - ms))
    d = np.sign(np.linalg.det(U @ Vt)); R2 = U @ np.diag([1, d]) @ Vt; return R2, md - R2 @ ms
def trimmed_yaw(a2, tgt):
    held = np.arange(len(a2)) % 5 == 2; train = ~held; keep = train.copy()
    for _ in range(5):
        R2, t2 = fit2d(a2, tgt, keep); err = np.linalg.norm(a2 @ R2.T + t2 - tgt, axis=1)
        keep = train & (err <= max(3.0, 1.5 * np.quantile(err[train], .8)))
    R2, t2 = fit2d(a2, tgt, keep); err = np.linalg.norm(a2 @ R2.T + t2 - tgt, axis=1)
    return R2, t2, float(np.sqrt(np.mean(err[train] ** 2))), (float(np.sqrt(np.mean(err[held] ** 2))) if held.any() else None)
comp_of = {}; cams_by = {}; gps_by = {}
for chunk in chunks:
    idx = json.loads((chunk / 'index.json').read_text()); comp_of[chunk.name] = idx['component']
    for f in idx['frames']:
        if f['gps']['E'] is None: continue
        cams_by.setdefault(idx['component'], []).append(np.array(f['c2w_cv'])[:3, 3]); gps_by.setdefault(idx['component'], []).append([f['gps']['E'], f['gps']['N']])
if not cams_by: print('no chunk has GPS-tagged frames: the walk cannot be placed on the map'); raise SystemExit(3)
allc = np.array(sum(cams_by.values(), [])); allg = np.array(sum(gps_by.values(), []), float)
R2c, _, tr_c, ho_c = trimmed_yaw(allc[:, [0, 2]], city_from_EN(allg[:, 0], allg[:, 1], np.zeros(len(allg)))[:, [0, 2]])
yaw_capture = float(np.degrees(np.arctan2(R2c[1, 0], R2c[0, 0])))
frames_json = {'capture_yaw_deg': round(yaw_capture, 2), 'capture_train_rmse_m': round(tr_c, 2), 'capture_held_out_rmse_m': None if ho_c is None else round(ho_c, 2), 'components': {}}
G_of = {}
path_of = {c: float(np.linalg.norm(np.diff(np.array(cams_by[c])[:, [0, 2]], axis=0), axis=1).sum()) for c in cams_by}
placeable = {c for c in cams_by if path_of[c] >= YAW_MIN_PATH_M and len(cams_by[c]) >= 8}
for comp in sorted(cams_by):
    a3 = np.array(cams_by[comp]); a2 = a3[:, [0, 2]]; tgt = city_from_EN(np.array(gps_by[comp])[:, 0], np.array(gps_by[comp])[:, 1], np.zeros(len(a2)))[:, [0, 2]]
    path = path_of[comp]
    if placeable and comp not in placeable:
        # A fragment a few metres long has no heading of its own and its height would come from the ground under a
        # handful of GPS fixes; placed alone it lands rotated or floating. When the walk has a real component, leave it out.
        frames_json.setdefault('held_out', []).append(dict(component=comp, cameras=int(len(a2)), path_m=round(path, 1)))
        print(f'component {comp}: {len(a2)} cameras, path {path:.1f} m, held out (too short to place on its own)'); continue
    if comp in placeable:
        R2, t2, tr, ho = trimmed_yaw(a2, tgt); yaw_src = 'component GPS track'
    else:
        R2 = R2c; t2 = tgt.mean(0) - R2 @ a2.mean(0); err = np.linalg.norm(a2 @ R2.T + t2 - tgt, axis=1); tr, ho = float(np.sqrt(np.mean(err ** 2))), None; yaw_src = 'capture-wide yaw (short component)'
    G = np.eye(4); G[np.ix_([0, 2], [0, 2])] = R2; G[[0, 2], 3] = t2
    G[1, 3] = float(np.median(ground_h(a2 @ R2.T + t2) + CAMERA_HEIGHT_M - a3[:, 1]))
    G_of[comp] = G
    frames_json['components'][str(comp)] = dict(cameras=int(len(a2)), path_m=round(path, 1), yaw_deg=round(float(np.degrees(np.arctan2(R2[1, 0], R2[0, 0]))), 2), yaw_source=yaw_src,
                                                train_rmse_m=round(tr, 2), held_out_rmse_m=None if ho is None else round(ho, 2), height_offset_m=round(float(G[1, 3]), 2), matrix=G.tolist())
    print(f"component {comp}: {len(a2)} cameras, path {path:.1f} m, yaw {frames_json['components'][str(comp)]['yaw_deg']} ({yaw_src}), train {tr:.2f} m held-out {ho}, height {G[1,3]:.2f}")

# ---- per chunk: depth -> ARKit pose -> city, scale only, then height-only settle onto LiDAR ground ------------------
def place_chunks(G_of):
    """Scale every chunk and settle it onto the LiDAR ground; writes placement-<SRC>.json per chunk."""
    print(f"{'chunk':5s}     {'frames':>6s} {'s_feat':>7s} {'n_feat':>6s} {'s_camh':>7s} {'s used':>7s} {'points':>10s}")
    pending = []
    for chunk in chunks:
        if comp_of[chunk.name] not in G_of: print(f'{chunk.name}: component {comp_of[chunk.name]} held out'); continue
        idx = json.loads((chunk / 'index.json').read_text()); byname = {f['file']: f for f in idx['frames']}
        res = json.loads((chunk / SRC / 'result.json').read_text())
        per_frame = []; s_feat_samples = []; s_camh_samples = []
        for i, cam in enumerate(res['cameras']):
            fr = byname[cam['image']]; f = np.load(chunk / SRC / f'frame_{i:04d}.npz')
            D = f['depth'].squeeze().astype(np.float64); valid = f['valid'].squeeze() & np.isfinite(D) & (D > .05)
            H, W = D.shape; K = np.array(cam['intrinsics'], float)
            v, u = np.mgrid[0:H, 0:W]
            X = (u - K[0, 2]) / K[0, 0] * D; Y = (v - K[1, 2]) / K[1, 1] * D          # OpenCV camera frame, unit scale
            Pc = np.stack([X, Y, D], -1)[valid]
            c2w = np.array(cam.get('c2w_used') or fr['c2w_cv'], float)                # pose the model was given (upright-rotated if UPRIGHT)
            lab = f['labels'].squeeze()[valid]; sc = f['label_score'].squeeze()[valid]
            # camera-height cue: predicted ground below this camera, in model units
            g = np.isin(lab, GROUND_LIKE) & (sc >= MIN_SCORE)
            if g.sum() > 200:
                Pw1 = tf(Pc[g], c2w); h_model = c2w[1, 3] - np.median(Pw1[:, 1])
                if h_model > .2: s_camh_samples.append(CAMERA_HEIGHT_M / h_model)
            # ARKit metric feature points seen by this frame (raw frame + raw pose share coordinates)
            raw = byidx[fr['video_frame_index']]; fp = raw.get('feature_points') or {}
            pts = np.array(fp.get('points') or fp.get('xyz') or [], float)
            if pts.ndim == 2 and len(pts) >= 5:
                # features live in the RAW ArKit world; the pose we use may be corrected, so go raw-world -> raw camera, then
                # (if rotated) into the rotated camera axes, and project with the model grid's own intrinsics
                Traw = np.array(raw['transform'], float); Pf = (pts - Traw[:3, 3]) @ Traw[:3, :3]; Pf = Pf * np.array([1, -1, -1])   # ARKit cam -> OpenCV cam
                if cam.get('upright'): Pf = Pf @ np.array([[0., 1., 0.], [-1., 0., 0.], [0., 0., 1.]])                                  # old cam -> rotated cam
                dep = Pf[:, 2]
                uu = K[0, 0] * Pf[:, 0] / np.where(dep > 0, dep, 1) + K[0, 2]; vv = K[1, 1] * Pf[:, 1] / np.where(dep > 0, dep, 1) + K[1, 2]
                pu = np.round(uu).astype(int); pv = np.round(vv).astype(int)
                ok = (dep > .5) & (pu >= 0) & (pu < W) & (pv >= 0) & (pv < H)
                if ok.any():
                    md = D[pv[ok], pu[ok]]; vv2 = f['valid'].squeeze()[pv[ok], pu[ok]] & (md > .1)
                    s_feat_samples += list(dep[ok][vv2] / md[vv2])
            per_frame.append((Pc, c2w, lab, sc))
        s_feat = float(np.median(s_feat_samples)) if len(s_feat_samples) >= 30 else float('nan')
        s_camh = float(np.median(s_camh_samples)) if s_camh_samples else float('nan')
        s = s_feat if np.isfinite(s_feat) and .3 < s_feat < 3 else s_camh
        if not np.isfinite(s): s = 1.0
        xyz, lab, sc = [], [], []
        for Pc, c2w, l, q in per_frame:
            Pw = tf(Pc * s, c2w)                                                        # scale depth, keep ARKit pose
            xyz.append(tf(Pw, G_of[comp_of[chunk.name]])); lab.append(l); sc.append(q)
        xyz = np.concatenate(xyz); lab = np.concatenate(lab); sc = np.concatenate(sc)
        # height settle: how far the chunk's ground-labelled points sit from the LiDAR ground. The correction itself is
        # ONE number per component (median over its chunks, below): the ARKit frame is level and continuous, so a
        # per-chunk shift would only step the road and the walls at every chunk seam.
        g = np.isin(lab, GROUND_LIKE) & (sc >= MIN_SCORE); gh_sel = y_sel = None
        if g.sum() > 200:
            gsel = np.flatnonzero(g); gsel = gsel[:: max(1, len(gsel) // 20000)]
            gh_sel = ground_h(xyz[gsel][:, [0, 2]]); y_sel = xyz[gsel, 1]
        points = int((np.linalg.norm(xyz - xyz.mean(0), axis=1) < 250).sum())
        pending.append(dict(chunk=chunk, comp=comp_of[chunk.name], gh=gh_sel, y=y_sel, rep=dict(chunk=chunk.name, version=3, frames=len(per_frame), points=points,
                   shared_frame='ARKit session frame; one 4-DoF to city per component (global-frame.json)',
                   scale=dict(feature_points=None if not np.isfinite(s_feat) else round(s_feat, 4), feature_samples=len(s_feat_samples),
                              camera_height=None if not np.isfinite(s_camh) else round(s_camh, 4), used=round(float(s), 4)),
                   arkit_to_city=dict(component=comp_of[chunk.name], **{k: v for k, v in frames_json['components'][str(comp_of[chunk.name])].items() if k != 'matrix'}))))
        print(f"{chunk.name:5s} {len(per_frame):6d} {s_feat:7.3f} {len(s_feat_samples):6d} {s_camh:7.3f} {s:7.3f} {points:10,d}")
    # one height correction per component, then each chunk's residual against it decides whether the chunk is held
    own = {p['chunk'].name: float(np.median(p['gh'] - p['y'])) for p in pending if p['gh'] is not None}
    for comp in {p['comp'] for p in pending}:
        vals = [own[p['chunk'].name] for p in pending if p['comp'] == comp and p['chunk'].name in own]
        dy = float(np.clip(np.median(vals), -1.0, 1.0)) if vals else 0.0
        G = G_of[comp].copy(); G[1, 3] += dy
        for p in pending:
            if p['comp'] != comp: continue
            after = float(np.median(np.abs(p['gh'] - (p['y'] + dy)))) if p['gh'] is not None else float('nan')
            rep = dict(p['rep'], height_settle_m=round(dy, 3), height_settle_own_m=round(own.get(p['chunk'].name, float('nan')), 3),
                       icp=dict(applied=True, method='height-only settle onto LiDAR class-2 ground, one shift per component', median_after_m=round(after, 3) if np.isfinite(after) else None),
                       matrix_model_to_city=G.tolist())
            (p['chunk'] / f'placement-{SRC}.json').write_text(json.dumps(rep, indent=1))
            print(f"{p['chunk'].name}: component {comp} shift {dy:+.2f} m (own {own.get(p['chunk'].name, float('nan')):+.2f}), ground after {after:.3f} m")


frames_json['matrix'] = G_of[max(G_of, key=lambda c: len(cams_by[c]))].tolist()   # the largest PLACED component
(R / 'global-frame.json').write_text(json.dumps(frames_json, indent=1))
place_chunks(G_of)
