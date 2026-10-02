"""Split a capture into overlapping 30 m chunks of usable keyframes.

RUN=<capture dir>, FRAMES=<extracted jpg dir, named by video_frame_index>,
GAUGE=<candidate-poses.npz from stabilize.py>, MAX_FRAMES=<frames per window>.
"""
import json, os, shutil
from pathlib import Path
import numpy as np
from pyproj import Transformer
R = Path(os.environ['RUN']); FR = Path(os.environ['FRAMES']); CAP = R / 'capture'
# Reject long exposures and high ISO/exposure combinations that compromise depth estimation.
MAX_EXPOSURE_S = 1 / 60
MAX_DARK_INDEX = 2.0         # ISO x exposure seconds
MIN_USABLE_FRACTION = 0.6

def light_ok(f):
    e = f.get('exif_numeric') or {}
    t, iso = e.get('ExposureTime'), e.get('ISOSpeedRatings')
    if t is None or iso is None: return True                      # no EXIF: do not invent a reason to reject
    return t <= MAX_EXPOSURE_S and t * iso <= MAX_DARK_INDEX

WINDOW_M, STEP_M, MIN_FRAMES, MAX_FRAMES = 30.0, 15.0, 6, int(os.environ.get('MAX_FRAMES', 40))
TO_UTM = Transformer.from_crs('EPSG:4326', 'EPSG:6347', always_xy=True)
man = json.loads((CAP / 'manifest.json').read_text()); t0 = man['session_started_timestamp']
# Load the usable ARKit poses and continuous components selected by stabilize.py.
g = np.load(os.environ['GAUGE']); gauge_pose = {}; gauge_comp = {}
for k, i in enumerate(g['feature_indices']): gauge_pose[int(i)] = g['feature_poses_corrected'][k]; gauge_comp[int(i)] = int(g['feature_components'][k])
recs = []; dark = 0
for f in man['frames']:
    src = FR / f"{f['video_frame_index']:06d}.jpg"
    if not src.exists() or f.get('tracking_state') != 'normal' or f['video_frame_index'] not in gauge_pose: continue
    if not light_ok(f): dark += 1; continue
    T = np.array(gauge_pose[f['video_frame_index']], float); cv = T.copy(); cv[:3, 1] *= -1; cv[:3, 2] *= -1
    loc = f.get('location') or {}; E, N = TO_UTM.transform(loc['longitude'], loc['latitude']) if loc.get('latitude') is not None else (None, None)
    recs.append(dict(file=src.name, video_frame_index=f['video_frame_index'], t=f['timestamp'] - t0, K=f['intrinsics'],
                     calibration_dims=f['calibration_image_dimensions'], c2w_arkit_corrected=T.tolist(), c2w_cv=cv.tolist(), c2w_arkit_raw=f['transform'],
                     gps=dict(lat=loc.get('latitude'), lon=loc.get('longitude'), E=E, N=N, horizontal_accuracy_m=loc.get('horizontal_accuracy_m'), altitude_m=loc.get('altitude_m')),
                     heading=f.get('heading') or {}, exposure=f.get('exposure'), feature_point_count=f['feature_point_count'], component=gauge_comp[f['video_frame_index']]))
recs.sort(key=lambda r: r['t'])
chunks = []
for c in sorted({r['component'] for r in recs}):
    rows = [r for r in recs if r['component'] == c]
    if len(rows) < MIN_FRAMES: continue
    P = np.array([np.array(r['c2w_cv'])[:3, 3] for r in rows]); s = np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(P, axis=0), axis=1))])
    # Subsample ONCE per component (uniform stride so <= MAX_FRAMES land in any 30 m window), THEN window: overlapping
    # windows then contain identical frames, which is what lets scale chaining and cross-chunk consistency work.
    per_window = max((np.searchsorted(s, s0 + WINDOW_M, 'right') - i0) for i0, s0 in enumerate(s))
    stride = max(1, int(np.ceil(per_window / MAX_FRAMES))); keep_idx = set(range(0, len(rows), stride)) | {len(rows) - 1}
    start = 0.0
    while True:
        sel = [i for i in range(len(rows)) if i in keep_idx and start <= s[i] <= start + WINDOW_M]
        if len(sel) >= MIN_FRAMES:
            chunks.append(dict(component=c, path_start_m=round(start, 1), path_m=round(float(s[sel[-1]] - s[sel[0]]), 1), frames=[rows[i] for i in sel]))
        if start + WINDOW_M >= s[-1]: break
        start += STEP_M
out = R / 'chunks'; out.mkdir(exist_ok=True); summary = []
for n, ch in enumerate(chunks):
    cid = f"c{n:02d}"; d = out / cid; (d / 'images').mkdir(parents=True, exist_ok=True)
    for r in ch['frames']:
        dst = d / 'images' / r['file']
        if not dst.exists(): shutil.copyfile(FR / r['file'], dst)
    gps = np.array([[r['gps']['E'], r['gps']['N']] for r in ch['frames'] if r['gps']['E'] is not None])
    (d / 'index.json').write_text(json.dumps(dict(chunk=cid, capture=man['id'], component=ch['component'], path_start_m=ch['path_start_m'], path_m=ch['path_m'],
        frames=ch['frames'], gps_centroid_EN=gps.mean(0).tolist() if len(gps) else None, pose_source='ARKit poses with motion filtering and component splitting'), indent=1))
    summary.append(dict(chunk=cid, component=ch['component'], frames=len(ch['frames']), path_m=ch['path_m'], t=[round(ch['frames'][0]['t'], 1), round(ch['frames'][-1]['t'], 1)]))
(R / 'batch-input.json').write_text(json.dumps(dict(capture=man['id'], created_at=man['created_at'], frames_total=sum(c['frames'] for c in summary), chunks=summary,
    rule=dict(window_m=WINDOW_M, step_m=STEP_M, min_frames=MIN_FRAMES, max_frames=MAX_FRAMES)), indent=1))
usable = len(recs) / max(1, len(recs) + dark)
if dark and usable < MIN_USABLE_FRACTION:                       # exit 3 marks an unusable capture
    print(f"{dark} of {len(recs) + dark} frames are too dark or too blurred to reconstruct ({usable:.0%} usable). "
          "Record this walk again in daylight. A low-light capture produces geometry that looks plausible and is wrong."); raise SystemExit(3)
print(f"keyframes {len(recs)} | components {len({r['component'] for r in recs})} | chunks {len(summary)}" + (f" | dark frames dropped {dark}" if dark else "") + f" | frames used {sum(c['frames'] for c in summary)}")
for c in summary: print(f"  {c['chunk']} comp {c['component']:2d} frames {c['frames']:2d} path {c['path_m']:5.1f} m t {c['t'][0]:5.1f}-{c['t'][1]:5.1f}s")
