"""Expose extracted video frames as <video_frame_index>.jpg symlinks (ffmpeg numbers them from 1)."""
import json, os, sys
man, raw, out = sys.argv[1:4]
os.makedirs(out, exist_ok=True)
n = 0
for f in json.load(open(man))['frames']:
    dst = os.path.join(out, f"{f['video_frame_index']:06d}.jpg")
    src = os.path.join(os.path.abspath(raw), f"{f['video_frame_index'] + 1:06d}.jpg")
    if not os.path.exists(dst) and os.path.exists(src):
        os.symlink(src, dst); n += 1
print('frames linked:', n, '| total in dir:', len(os.listdir(out)))
