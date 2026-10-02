"""Infer depth and semantic labels for each chunk using ARKit pose and intrinsics priors.

RUN=<capture dir>; writes chunks/cXX/mapanything-posed/ with frame arrays,
cloud.npz and result.json for place.py and filter.py.
"""
import json, os, sys, time
from pathlib import Path
import numpy as np, torch
from PIL import Image
from mapanything.models import MapAnything
from mapanything.utils.image import preprocess_inputs
import segment

R = Path(os.environ['RUN']); OUT = 'mapanything-posed'; UPRIGHT = os.environ.get('UPRIGHT') == '1'
# 90 deg clockwise image rotation: u' = H-1-v, v' = u. New camera axes in old camera coords: x' = -y, y' = +x, z' = z.
ROT_CAM = np.array([[0., 1., 0.], [-1., 0., 0.], [0., 0., 1.]])          # columns = new axes expressed in the old frame
def upright(img, K, c2w):
    H, W = img.shape[:2]
    img2 = np.ascontiguousarray(np.rot90(img, k=-1))
    K2 = np.array([[K[1, 1], 0., H - 1 - K[1, 2]], [0., K[0, 0], K[0, 2]], [0., 0., 1.]], np.float32)
    c2w2 = c2w.copy(); c2w2[:3, :3] = c2w[:3, :3] @ ROT_CAM
    return img2, K2, c2w2
DEVICE = os.environ.get('DEVICE', 'cuda')
AMP_DTYPE = os.environ.get('AMP_DTYPE', 'bf16')
model = MapAnything.from_pretrained('facebook/map-anything-apache', revision=segment.WEIGHTS_REVISION).to(DEVICE).eval()
segmenter = segment.load_segmenter(DEVICE)
if segmenter is None: sys.exit('the segmenter did not load (see above); without labels no chunk can settle on the ground')

for chunk in sorted(p for p in (R / 'chunks').glob('c*') if (p / 'images').is_dir()):
    out = chunk / OUT; out.mkdir()
    idx = json.loads((chunk / 'index.json').read_text()); frames = sorted(idx['frames'], key=lambda f: f['t'])
    views, t0 = [], time.monotonic()
    for f in frames:
        img = np.asarray(Image.open(chunk / 'images' / f['file']).convert('RGB'))
        K = np.array(f['K'], np.float32); W0, H0 = f['calibration_dims']; c2w = np.array(f['c2w_cv'], np.float32)
        assert img.shape[1] == W0 and img.shape[0] == H0, f"{f['file']}: image is {img.shape[1]}x{img.shape[0]}, the intrinsics were calibrated for {W0}x{H0}"
        if UPRIGHT: img, K, c2w = upright(img, K, c2w)
        f['c2w_used'] = c2w.tolist()
        views.append({'img': img, 'intrinsics': K, 'camera_poses': c2w, 'is_metric_scale': torch.tensor([True])})
    views = preprocess_inputs(views, resize_mode='fixed_mapping')
    with torch.inference_mode():
        preds = model.infer(views, memory_efficient_inference=True, use_amp=DEVICE == 'cuda', amp_dtype=AMP_DTYPE,
                            apply_mask=True, mask_edges=True, apply_confidence_mask=True, confidence_percentile=15)
    inf_s = time.monotonic() - t0
    arr = lambda v: v.detach().float().cpu().numpy()
    cams, P, C, L, S, FI = [], [], [], [], [], []
    for i, (f, pr) in enumerate(zip(frames, preds)):
        pts = arr(pr['pts3d'])[0]; conf = arr(pr['conf'])[0].squeeze(); mask = arr(pr['mask'])[0].squeeze().astype(bool)
        colors = arr(pr['img_no_norm'])[0]; colors = colors.transpose(1, 2, 0) if colors.shape[0] == 3 else colors
        colors = np.clip(colors * 255 if colors.max() <= 1.01 else colors, 0, 255).astype(np.uint8)
        valid = mask & np.isfinite(pts).all(-1) & np.isfinite(conf)
        labels, raw_labels, score = segment.segment_frame(colors, valid, segmenter)
        depth = arr(pr['depth_z'])[0].squeeze(); cam = arr(pr['camera_poses'])[0]; Kp = arr(pr['intrinsics'])[0]
        prior = np.array(f['c2w_used']); dpos = float(np.linalg.norm(cam[:3, 3] - prior[:3, 3]))
        dang = float(np.degrees(np.arccos(np.clip((np.trace(cam[:3, :3].T @ prior[:3, :3]) - 1) / 2, -1, 1))))
        np.savez(out / f'frame_{i:04d}.npz', points=pts, confidence=conf, valid=valid, depth=depth, colors=colors, labels=labels, raw_labels=raw_labels, label_score=score)
        cams.append({'image': f['file'], 'cam2world': cam.tolist(), 'intrinsics': Kp.tolist(), 'valid_points': int(valid.sum()), 'image_size': list(pts.shape[:2]),
                     'frame_file': f'frame_{i:04d}.npz', 'c2w_used': f['c2w_used'], 'upright': UPRIGHT, 'moved_from_prior_m': round(dpos, 3), 'rotated_from_prior_deg': round(dang, 2)})
        P.append(pts[valid]); C.append(colors[valid]); L.append(labels[valid]); S.append(score[valid]); FI.append(np.full(int(valid.sum()), i, np.uint32))
    P = np.concatenate(P); keep = np.linalg.norm(P, axis=1) < 250
    np.savez(out / 'cloud.npz', points=P[keep], colors=np.concatenate(C)[keep], labels=np.concatenate(L)[keep], label_score=np.concatenate(S)[keep], frame_index=np.concatenate(FI)[keep])
    moved = [c['moved_from_prior_m'] for c in cams]; rot = [c['rotated_from_prior_deg'] for c in cams]
    res = {'backend': 'facebook/map-anything-apache', 'priors': 'ARKit camera_poses (OpenCV c2w) + intrinsics, is_metric_scale=True', 'upright_input': UPRIGHT, 'cameras': cams,
           'point_count': int(keep.sum()), 'inference_s': inf_s, 'weights_revision': segment.WEIGHTS_REVISION,
           'camera_drift_from_prior': {'pos_median_m': float(np.median(moved)), 'pos_max_m': float(np.max(moved)), 'rot_median_deg': float(np.median(rot)), 'rot_max_deg': float(np.max(rot))},
           'segmentation': {'model': segment.SEGMENTER, 'revision': segment.SEGMENTER_REVISION, 'labels': segment.LABELS, 'unknown_label': segment.UNKNOWN_LABEL, 'confidence_threshold': segment.MIN_LABEL_CONFIDENCE}}
    (out / 'result.json').write_text(json.dumps(res, indent=1))
    print(json.dumps({'chunk': chunk.name, 'frames': len(cams), 'points': res['point_count'], 'inference_s': round(inf_s, 1), 'camera_drift_from_prior': res['camera_drift_from_prior']}), flush=True)
print('ALL CHUNKS DONE', flush=True)
