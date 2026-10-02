"""Export a merged 3D scene as PLY, streaming tiles, and georeferenced metadata."""
import argparse
import hashlib
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

POINT = np.dtype([('xyz', '<f4', 3), ('rgba', 'u1', 4)])
ORIGIN = [377249.9995, 4700249.9995, 150.0]        # fixed Ithaca frame: east / up / south metres, EPSG:6347
CLASS_NAMES = {0: 'road', 1: 'sidewalk', 2: 'building', 3: 'wall', 8: 'vegetation', 9: 'terrain'}
TILE_M, MIN_TILE, COARSE_STRIDE = 96, 2000, 12


def sha256(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''): h.update(chunk)
    return h.hexdigest()


def geo(bounds):
    """lat/lon -> model x,z as an affine fit. The local UTM frame is affine to well under a metre at this size."""
    from pyproj import Transformer
    to_wgs84 = Transformer.from_crs(6347, 4326, always_xy=True)
    (x0, _, z0), (x1, _, z1) = bounds
    xs, zs = np.meshgrid(np.linspace(x0, x1, 5), np.linspace(z0, z1, 5)); xs, zs = xs.ravel(), zs.ravel()
    lon, lat = to_wgs84.transform(ORIGIN[0] + xs, ORIGIN[1] - zs)
    basis = np.c_[lat, lon, np.ones_like(lat)]
    ax = np.linalg.lstsq(basis, xs, rcond=None)[0]; az = np.linalg.lstsq(basis, zs, rcond=None)[0]
    residual = float(max(abs(basis @ ax - xs).max(), abs(basis @ az - zs).max()))
    assert residual < 1, f'affine geo fit residual {residual:.2f} m'
    clon, clat = to_wgs84.transform([ORIGIN[0] + x0, ORIGIN[0] + x1, ORIGIN[0] + x1, ORIGIN[0] + x0], [ORIGIN[1] - z1, ORIGIN[1] - z1, ORIGIN[1] - z0, ORIGIN[1] - z0])
    return {'affine': [*map(float, ax), *map(float, az)], 'residual_m': residual, 'corners_wgs84': [[float(a), float(b)] for a, b in zip(clat, clon)]}


def tiles(points, out):
    """A coarse sample plus a grid of TILE_M tiles, so the viewer streams nearest-first."""
    (out / 'tiles').mkdir(exist_ok=True)
    coarse = points[::COARSE_STRIDE]; coarse.tofile(out / 'tiles/coarse.bin')
    cells = np.floor(points['xyz'][:, [0, 2]] / TILE_M).astype(np.int64); entries, rest = [], []
    for cx, cz in np.unique(cells, axis=0):
        chunk = points[(cells[:, 0] == cx) & (cells[:, 1] == cz)]
        if len(chunk) < MIN_TILE: rest.append(chunk); continue
        name = f'tiles/{cx}_{cz}.bin'; chunk.tofile(out / name)
        entries.append({'file': name, 'points': len(chunk), 'center': [float(v) for v in chunk['xyz'].mean(0)]})
    if rest:
        chunk = np.concatenate(rest); chunk.tofile(out / 'tiles/rest.bin')
        entries.append({'file': 'tiles/rest.bin', 'points': len(chunk), 'center': [float(v) for v in chunk['xyz'].mean(0)]})
    return {'tile_m': TILE_M, 'coarse': {'file': 'tiles/coarse.bin', 'points': len(coarse)}, 'tiles': entries}


def build(merged, out):
    merged, out = Path(merged), Path(out)
    scene = json.loads((merged / 'model.json').read_text())
    cloud = np.fromfile(merged / 'preview.bin', dtype=POINT)
    assert len(cloud) == scene['point_count'], f"preview.bin has {len(cloud)} points, model.json says {scene['point_count']}"
    if out.exists(): shutil.rmtree(out)
    out.mkdir(parents=True)
    with (out / 'preview.ply').open('wb') as ply:
        ply.write(f'ply\nformat binary_little_endian 1.0\ncomment Video to 3D: {len(scene["walks"])} captures aligned to LiDAR\n'
                  f'element vertex {len(cloud)}\nproperty float x\nproperty float y\nproperty float z\n'
                  'property uchar red\nproperty uchar green\nproperty uchar blue\nend_header\n'.encode())
        ply.write(np.ascontiguousarray(cloud).view(np.uint8).reshape(-1, 16)[:, :15].tobytes())
    walks, focus = [], None
    for w in scene['walks']:
        lo, hi = (np.array(b, float) for b in w['bounds_city'])
        walks.append({'capture': w['capture'], 'points': w['points'], 'classes': {CLASS_NAMES.get(int(k), k): v for k, v in w['classes'].items()}})
        if focus is None or w['points'] > focus['points']:      # focus the viewer on the densest capture
            focus = {'points': w['points'], 'center': [float(v) for v in (lo + hi) / 2], 'radius': float(np.linalg.norm(hi - lo) / 2)}
    lo, hi = cloud['xyz'].min(0), cloud['xyz'].max(0); bounds = [[float(v) for v in lo], [float(v) for v in hi]]
    now = datetime.now(timezone.utc).isoformat(timespec='seconds')
    model = {'schema_version': 1, 'name': scene.get('name', '3D scene'), 'snapshot': now, 'updated_at': now,
             'point_count': len(cloud), 'preview_points': len(cloud), 'preview_download': 'preview.ply',
             'alignment': 'lidar_fitted', 'walks': walks, 'focus': focus, 'bounds': bounds,
             'classes': {CLASS_NAMES.get(int(k), int(k)): int(v) for k, v in zip(*np.unique(cloud['rgba'][:, 3], return_counts=True))},
             'download': {'label': 'Download 3D scene (PLY)', 'name': 'scene.ply', 'file': 'preview.ply',
                          'bytes': (out / 'preview.ply').stat().st_size, 'sha256': sha256(out / 'preview.ply')},
             'tiles': tiles(cloud, out), 'geo': geo(bounds)}
    ref = merged / 'lidar-coarse.bin'
    if ref.exists():
        shutil.copy2(ref, out / ref.name)
        model['reference'] = {'file': ref.name, 'points': ref.stat().st_size // 16, 'cell_m': 4.0, 'label': 'Unmapped ground (LiDAR)',
                              'note': 'Airborne LiDAR shown only where no walk covers the ground. Reference only: not part of point_count and not in the download.',
                              'source': 'USGS 3DEP NY_FingerLakes_1_2020 (public domain)'}
    (out / 'coordinates.json').write_text(json.dumps({'origin_projected': ORIGIN, 'horizontal_epsg': 6347, 'axes': ['east', 'up', 'south'], 'units': 'metres',
                                                     'alignment': 'surveyed', 'note': 'USGS 3DEP NY_FingerLakes_1_2020 LiDAR (May 2020), NAVD88 / GEOID12B. The origin is frozen.'}, indent=2) + '\n')
    (out / 'model.json').write_text(json.dumps(model, indent=2) + '\n')
    return model


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__); ap.add_argument('--merged', required=True); ap.add_argument('--output', required=True)
    a = ap.parse_args(); m = build(a.merged, a.output)
    print(json.dumps({'points': m['point_count'], 'tiles': len(m['tiles']['tiles']), 'download_mb': round(m['download']['bytes'] / 1e6, 1), 'walks': len(m['walks'])}))
