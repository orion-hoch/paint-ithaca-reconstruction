"""Filter implausible ARKit motion and split captures at pose jumps or recording gaps.

RUN=<capture dir>; reads capture/manifest.json and writes gauge/candidate-poses.npz
for stage.py. Running without RUN performs the synthetic motion check.

ARKit poses retain their recorded gravity and scale. Keyframes are rejected when
path speed exceeds GPS speed by GPS_SLACK_MPS, or WALK_MPS when GPS course is
unreliable. Remaining poses are grouped into continuous rigid components.
"""
import json
import os
from pathlib import Path

import numpy as np

JUMP_M, GAP_S = 2.0, 3.0            # ARKit moved more than 2 m between snapshots (2 m/s over longer gaps), or they are 3 s apart: a relocalisation, not a walk
WALK_MPS, GPS_SLACK_MPS = 2.5, 1.0


def gauge(manifest):
    """(kept keyframes, their rigid components, jumps, dropped keyframes)."""
    frames = [f for f in manifest['frames'] if f.get('feature_points', {}).get('points')]
    ps = manifest.get('pose_samples') or []                              # ARKit at 7.5-10 Hz, between the 2-3 Hz keyframes
    st = np.array([p['timestamp'] for p in ps]); sp = np.array([np.array(p['transform'])[:3, 3] for p in ps]).reshape(-1, 3)

    def too_fast(f):
        if len(st) < 2: return False
        a, b = max(f['timestamp'] - .5, st[0]), min(f['timestamp'] + .5, st[-1])      # path, not displacement: a snap out and back cancels
        q = np.r_[a, st[(st > a) & (st < b)], b]; speed = np.linalg.norm(np.diff(np.column_stack([np.interp(q, st, sp[:, k]) for k in (0, 2)]), axis=0), axis=1).sum() / max(b - a, .25)
        loc = f.get('location') or {}; course = loc.get('course_accuracy_deg'); gps = loc.get('speed_mps', -1)
        return speed > (gps + GPS_SLACK_MPS if course is not None and 0 <= course < 90 and gps >= 0 else WALK_MPS)
    def jumped(a, b):                                                    # between keyframes a and b ARKit relocalised, or the recording gapped
        step = float(np.linalg.norm(np.array(frames[b]['transform'])[:3, 3] - np.array(frames[a]['transform'])[:3, 3]))
        gap = float(frames[b]['video_timestamp_s'] - frames[a]['video_timestamp_s'])
        return gap > GAP_S or step > JUMP_M * max(1.0, gap)
    keep = [i for i, f in enumerate(frames) if not too_fast(f)]
    components, jumps = [0], 0
    for a, b in zip(keep, keep[1:]):
        # Split only if a recorded jump persists across the dropped frames; a jump that snaps back stays one component.
        cut = jumped(a, b) and any(jumped(i, i + 1) for i in range(a, b))
        components.append(components[-1] + int(cut)); jumps += int(cut)
    kept = set(keep)
    return [frames[i] for i in keep], components[:len(keep)], jumps, [f for i, f in enumerate(frames) if i not in kept]


if __name__ == '__main__' and 'RUN' not in os.environ:
    # A 30 s walk east at 1.3 m/s. At 10-11 s ARKit runs off south at 5 m/s and snaps back; at 14-15 s it snaps 0.8 m
    # sideways four times and keeps the 3.2 m; at 20 s it jumps 10 m and stays there; at 24-27 s GPS loses its course
    # and reports 0.1 m/s. Only the three episodes may go, and only the jump that stays may cut the walk (the 14 s snaps
    # never jumped 2 m at once, so the walk as recorded was not cut there either).
    def pose(x, z): T = np.eye(4); T[0, 3], T[2, 3] = x, z; return T.tolist()
    def at(t): return 1.3 * t, 5 * min(max(t - 10, 0), 1) * (t < 11.1) + .8 * min(max(np.floor((t - 14) / .25) + 1, 0), 4) + 10 * (t >= 20)
    ts = np.arange(0, 30, 1 / 7.5); kt = np.arange(.2, 30, 1 / 3)
    man = dict(pose_samples=[dict(timestamp=t, transform=pose(*at(t))) for t in ts],
               frames=[dict(timestamp=t, video_timestamp_s=t, video_frame_index=i, transform=pose(*at(t)), feature_points=dict(points=[[0, 0, 0]]),
                            location=dict(speed_mps=.1 if 24 <= t < 27 else 1.3, course_accuracy_deg=180 if 24 <= t < 27 else 30)) for i, t in enumerate(kt)])
    kept, comp, jumps, fast = gauge(man); gone = np.array([f['timestamp'] for f in fast])
    assert all(9.5 < t < 11.7 or 13.4 < t < 15.4 or 19.4 < t < 20.6 for t in gone), f'dropped outside the episodes: {gone}'
    assert all(any(abs(g - t) < 1e-9 for g in gone) for t in kt if 10.1 < t < 11.0 or 14.2 < t < 14.9 or 19.7 < t < 20.3), 'a keyframe inside an episode survived'
    assert jumps == 1 and len(set(comp)) == 2 and all(c == (f['timestamp'] > 20) for f, c in zip(kept, comp)), (jumps, comp)
    print(f'stabilize ok: dropped {len(fast)} keyframes at {gone.min():.1f}-{gone.max():.1f} s, {len(set(comp))} components')
if __name__ == '__main__' and 'RUN' in os.environ:
    RUN = Path(os.environ['RUN'])
    OUT = RUN / 'gauge'
    OUT.mkdir(parents=True, exist_ok=True)
    frames, components, jumps, fast = gauge(json.loads((RUN / 'capture/manifest.json').read_text()))
    indices = np.array([f['video_frame_index'] for f in frames]); poses = np.array([f['transform'] for f in frames], float)
    np.savez_compressed(OUT / 'candidate-poses.npz', feature_indices=indices, feature_poses_corrected=poses, feature_components=components)
    print(f'feature snapshots {len(frames) + len(fast)} | dropped {len(fast)} where ARKit outran the walk | rigid components {len(set(components))} | pose jumps {jumps}')
