"""Reconstruct ARKit video captures and fuse them into a colored 3D point cloud."""
import argparse
import fcntl
import json
import math
import os
import re
import shutil
import signal
import subprocess
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
POINT = np.dtype([('xyz', '<f4', 3), ('rgba', 'u1', 4)])
SETTINGS = dict(src='mapanything-posed', upright=1, depth_calib='binned', calib_far='hold',
                graze_deg=6, refine=2, fuse=1, rel_tol=0.08, min_agree=1, baseline_m=1.5,
                sail_cut=1, volume=1)
STAGE_TIMEOUT_S = 3 * 3600
CHILD = None


class InvalidCapture(Exception):
    """Capture metadata or image quality prevents reconstruction."""


def log(message):
    print(f'[{datetime.now(timezone.utc):%Y-%m-%d %H:%M:%S}] {message}', flush=True)


def on_term(signum, frame):
    if CHILD and CHILD.poll() is None:
        CHILD.terminate()
        try: CHILD.wait(30)
        except subprocess.TimeoutExpired: CHILD.kill()
    raise SystemExit(128 + signum)


def run(command, env=None):
    global CHILD
    CHILD = subprocess.Popen(command, env={**os.environ, **(env or {})},
                             stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    try:
        output, _ = CHILD.communicate(timeout=STAGE_TIMEOUT_S)
        code = CHILD.returncode
    except subprocess.TimeoutExpired:
        CHILD.kill(); CHILD.communicate()
        raise RuntimeError(f'{Path(command[1]).name} exceeded {STAGE_TIMEOUT_S // 3600} hours')
    finally:
        CHILD = None
    lines = output.strip().splitlines()
    if code == 3: raise InvalidCapture(lines[-1] if lines else 'capture rejected')
    if code: raise RuntimeError('\n'.join(lines[-8:]) or f'command exited with {code}')
    for line in lines[-3:]: log(f'  {line}')


def read_capture(path):
    manifest = json.loads((path / 'manifest.json').read_text())
    cid = manifest.get('id', '')
    if not isinstance(cid, str) or not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_-]{0,127}', cid):
        raise InvalidCapture('manifest id must contain only letters, numbers, hyphens or underscores')
    video = path / 'capture.mov'
    if not video.is_file(): video = path / 'video/capture.mov'
    if not video.is_file(): raise InvalidCapture('capture.mov is missing')
    fixes = [f['location'] for f in manifest.get('frames', [])
             if f.get('location') and f['location'].get('horizontal_accuracy_m', 99) <= 25]
    points = [(float(f['latitude']), float(f['longitude'])) for f in fixes]
    metres = sum(math.dist(a, b) for a, b in zip(points, points[1:])) * 111320
    seconds = float((manifest.get('video') or {}).get('duration_s') or 0)
    if seconds < 15 or metres < 20:
        raise InvalidCapture(f'capture needs at least 15 seconds and 20 metres of GPS coverage ({seconds:.0f} s, {metres:.0f} m)')
    return manifest, video


class Reconstruction:
    def __init__(self, args):
        self.args = args
        self.root = args.output
        self.walks, self.work = self.root / 'walks', self.root / 'work'
        self.merged, self.model = self.root / 'merged', self.root / 'model'
        self.work.mkdir(parents=True, exist_ok=True)
        self.walks.mkdir(exist_ok=True)
        import laspy
        self.tiles = []
        for path in sorted(args.lidar.glob('classified_*.laz')):
            with laspy.open(path) as reader:
                self.tiles.append((path, reader.header.mins[:2], reader.header.maxs[:2]))
        if not self.tiles:
            raise ValueError(f'no classified_*.laz tiles in {args.lidar}')

    def laz_for(self, manifest, margin=60):
        from pyproj import Transformer
        fixes = [f['location'] for f in manifest['frames'] if f.get('location')]
        east, north = Transformer.from_crs(4326, 6347, always_xy=True).transform(
            [f['longitude'] for f in fixes], [f['latitude'] for f in fixes])
        lo = np.array([min(east), min(north)]) - margin
        hi = np.array([max(east), max(north)]) + margin
        tiles = [str(path) for path, lower, upper in self.tiles if (lo <= upper).all() and (hi >= lower).all()]
        if not tiles: raise InvalidCapture('no classified LiDAR tile overlaps the capture')
        return ','.join(tiles)

    def process(self, source, manifest, video):
        cid = manifest['id']; out = self.walks / cid
        if (out / 'block-consistent.bin').exists() and not self.args.force:
            log(f'{cid}: using cached reconstruction'); return
        laz = self.laz_for(manifest)
        work = Path(tempfile.mkdtemp(prefix=f'{cid}-', dir=self.work))
        env = {'RUN': str(work)}
        stages = HERE / 'stages'
        log(f'{cid}: reconstructing')
        try:
            (work / 'capture').mkdir()
            shutil.copy2(source / 'manifest.json', work / 'capture/manifest.json')
            (work / 'raw').mkdir()
            # Preserve one image per encoded frame so camera poses keep their frame indices.
            run(['ffmpeg', '-nostdin', '-loglevel', 'error', '-y', '-noautorotate', '-i', str(video),
                 '-fps_mode', 'passthrough', '-q:v', '2', str(work / 'raw/%06d.jpg')])
            expected = manifest['video'].get('frame_count'); got = len(list((work / 'raw').glob('*.jpg')))
            if expected and got != expected:
                raise InvalidCapture(f'video has {got} frames, manifest declares {expected}')
            run([sys.executable, str(stages / 'link_frames.py'), str(work / 'capture/manifest.json'), str(work / 'raw'), str(work / 'frames')])
            run([sys.executable, str(stages / 'stabilize.py')], env)
            run([sys.executable, str(stages / 'stage.py')], {**env, 'FRAMES': str(work / 'frames'),
                 'GAUGE': str(work / 'gauge/candidate-poses.npz'), 'MAX_FRAMES': str(self.args.frames_per_chunk)})
            if not json.loads((work / 'batch-input.json').read_text())['chunks']:
                raise InvalidCapture('no chunk has enough usable frames')
            shutil.rmtree(work / 'raw'); shutil.rmtree(work / 'frames')
            run([sys.executable, str(stages / 'infer.py')], {**env, 'UPRIGHT': str(SETTINGS['upright']),
                 'DEVICE': self.args.device, 'AMP_DTYPE': self.args.amp_dtype, 'PYTHONUNBUFFERED': '1'})
            run([sys.executable, str(stages / 'place.py')], {**env, 'LAZ': laz})
            run([sys.executable, str(stages / 'filter.py')], {**env, 'LAZ': laz, 'DEPTH_CALIB': SETTINGS['depth_calib'],
                 'REFINE': str(SETTINGS['refine']), 'FUSE': str(SETTINGS['fuse']), 'REL_TOL': str(SETTINGS['rel_tol']),
                 'MIN_AGREE': str(SETTINGS['min_agree']), 'BASELINE_M': str(SETTINGS['baseline_m']),
                 'CALIB_FAR': SETTINGS['calib_far'], 'GRAZE_DEG': str(SETTINGS['graze_deg']), 'DUMP_FRAMES': str(SETTINGS['volume'])})
            shutil.copy2(work / 'capture/manifest.json', work / 'manifest.json')
            run([sys.executable, str(stages / 'refine.py')], {**env, 'LIDAR_DIR': str(self.args.lidar)})
            staged = Path(tempfile.mkdtemp(prefix=f'.{cid}-', dir=self.walks))
            for name in ('block-consistent.bin', 'block-range.bin', 'consistency-stats.json', 'global-frame.json',
                         'batch-input.json', 'manifest.json', 'footprints.npz'):
                if (work / name).exists(): shutil.copy2(work / name, staged / name)
            if (work / 'frames-dump').is_dir(): shutil.copytree(work / 'frames-dump', staged / 'frames-dump')
            (staged / 'settings.json').write_text(json.dumps(SETTINGS, indent=2) + '\n')
            previous = self.walks / f'.{cid}.previous'
            if out.exists():
                if previous.exists(): shutil.rmtree(previous)
                out.rename(previous)
            staged.rename(out)
        except BaseException:
            log(f'{cid}: intermediates retained at {work}')
            raise
        shutil.rmtree(work)
        log(f'{cid}: {(out / "block-consistent.bin").stat().st_size // 16:,} points kept')

    def merge(self):
        dirs = [d for d in sorted(self.walks.iterdir()) if not d.name.startswith('.') and (d / 'block-consistent.bin').exists()]
        if not dirs: raise RuntimeError('no reconstructed captures to merge')
        self.merged.mkdir(exist_ok=True)
        run([sys.executable, str(HERE / 'stages/fuse.py')], {'WALKS': ','.join(map(str, dirs)), 'OUT': str(self.merged)})
        merged = np.fromfile(self.merged / 'preview.bin', dtype=POINT)
        indices = np.fromfile(self.merged / 'walk-index.bin', dtype=np.uint16)
        walks = []
        for k, directory in enumerate(dirs):
            cloud = merged[indices == k]
            if not len(cloud): continue
            walks.append({'capture': directory.name, 'points': len(cloud),
                          'bounds_city': [cloud['xyz'].min(0).tolist(), cloud['xyz'].max(0).tolist()],
                          'classes': {str(int(c)): int(n) for c, n in zip(*np.unique(cloud['rgba'][:, 3], return_counts=True))}})
        volume = [k for k, d in enumerate(dirs) if (d / 'frames-dump').is_dir()]
        if volume:
            run([sys.executable, str(HERE / 'stages/tsdf.py')], {'RUNS': ','.join(str(dirs[k]) for k in volume),
                 'OUT': str(self.merged / 'volume'), 'REGISTER': str(self.merged / 'fusion.json'),
                 'WORKERS': str(self.args.volume_workers), 'LAZ': ','.join(str(t[0]) for t in self.tiles)})
            merged = np.concatenate([np.fromfile(self.merged / 'volume/block-tsdf.bin', dtype=POINT), merged[~np.isin(indices, volume)]])
            merged.tofile(self.merged / 'preview.bin')
        (self.merged / 'model.json').write_text(json.dumps({'name': '3D scene', 'point_count': len(merged), 'walks': walks, 'settings': SETTINGS}, indent=2) + '\n')
        reference = self.merged / 'lidar-coarse.bin'
        reference.unlink(missing_ok=True)
        raw_tiles = sorted(self.args.lidar.glob('u_*.laz'))
        if raw_tiles:
            run([sys.executable, str(HERE / 'stages/lidar_context.py')], {'MODEL': str(self.merged / 'preview.bin'),
                 'LAZ': ','.join(map(str, raw_tiles)), 'OUT': str(reference)})
        staged = Path(tempfile.mkdtemp(prefix='.model-', dir=self.root))
        run([sys.executable, str(HERE / 'export.py'), '--merged', str(self.merged), '--output', str(staged)])
        previous = self.root / 'model.previous'
        if self.model.exists():
            if previous.exists(): shutil.rmtree(previous)
            self.model.rename(previous)
        staged.rename(self.model)
        log(f'model: {len(merged):,} points from {len(dirs)} captures → {self.model / "preview.ply"}')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('captures', type=Path, help='capture folder, or directory containing capture folders')
    parser.add_argument('--lidar', type=Path, required=True, help='directory containing classified_*.laz tiles')
    parser.add_argument('--output', type=Path, default=Path('output'), help='output directory (default: output)')
    parser.add_argument('--only', help='process one capture by id or unique id prefix; merge all cached captures')
    parser.add_argument('--force', action='store_true', help='recompute selected captures even if cached')
    parser.add_argument('--frames-per-chunk', type=int, default=40)
    parser.add_argument('--volume-workers', type=int, default=4)
    parser.add_argument('--device', choices=('cuda', 'cpu'), default='cuda')
    parser.add_argument('--amp-dtype', choices=('bf16', 'fp16'), default='bf16')
    args = parser.parse_args()
    for key in ('captures', 'lidar', 'output'):
        path = getattr(args, key).expanduser().resolve()
        if ',' in str(path): parser.error('paths cannot contain commas')
        setattr(args, key, path)
    if not args.captures.is_dir(): parser.error('captures directory does not exist')
    if not args.lidar.is_dir(): parser.error('LiDAR directory does not exist')
    for source in (args.captures, args.lidar):
        if args.output == source or args.output in source.parents or source in args.output.parents:
            parser.error('output must be separate from capture and LiDAR directories')
    if args.frames_per_chunk < 6 or args.volume_workers < 1:
        parser.error('frames-per-chunk must be at least 6 and volume-workers at least 1')
    sources = [args.captures] if (args.captures / 'manifest.json').is_file() else sorted(p.parent for p in args.captures.glob('*/manifest.json'))
    if not sources: parser.error('no capture manifests found')
    captures = {}; failures = 0
    for source in sources:
        try:
            manifest, video = read_capture(source)
        except (InvalidCapture, ValueError, KeyError, TypeError, AttributeError) as error:
            log(f'{source.name}: skipped, {error}'); failures += 1; continue
        cid = manifest['id']
        if cid in captures: parser.error(f'duplicate capture id: {cid}')
        captures[cid] = (source, manifest, video)
    selected = [args.only] if args.only in captures else [cid for cid in captures if not args.only or cid.startswith(args.only)]
    if args.only and len(selected) != 1: parser.error('--only must match exactly one valid capture')
    if not selected: parser.error('no valid captures found')
    args.output.mkdir(parents=True, exist_ok=True)
    with (args.output / '.lock').open('w') as lock:
        try: fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError: parser.exit(1, 'another reconstruction is using this output directory\n')
        signal.signal(signal.SIGTERM, on_term)
        signal.signal(signal.SIGINT, on_term)
        try: reconstruction = Reconstruction(args)
        except ValueError as error: parser.error(str(error))
        for cid in selected:
            try: reconstruction.process(*captures[cid])
            except Exception as error:
                failures += 1; log(f'{cid}: failed, {error}')
        try: reconstruction.merge()
        except Exception as error:
            failures += 1; log(f'model export failed: {error}')
    if failures: parser.exit(1, f'{failures} capture or export failure(s)\n')


if __name__ == '__main__':
    main()
