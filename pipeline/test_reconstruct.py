"""Run the cached-capture CLI through real fusion and export: python pipeline/test_reconstruct.py."""
import json
import subprocess
import sys
import tempfile
from pathlib import Path

import laspy
import numpy as np

from reconstruct import POINT


def check_cached_reconstruction():
    with tempfile.TemporaryDirectory(prefix='reconstruct-check-') as temporary:
        root = Path(temporary)
        captures, lidar, output = (root / name for name in ('captures', 'lidar', 'output'))
        lidar.mkdir()
        header = laspy.LasHeader(point_format=3, version='1.2')
        header.offsets = [377250, 4700250, 150]
        tile = laspy.LasData(header)
        tile.x, tile.y, tile.z = np.array([[377250, 377300], [4700250, 4700300], [150, 150]])
        tile.classification = [2, 2]
        tile.write(lidar / 'classified_test.laz')
        for cid in ('capture-1', 'capture-10'):
            source = captures / cid
            source.mkdir(parents=True)
            (source / 'manifest.json').write_text(json.dumps({
                'id': cid, 'created_at': '2026-01-01T00:00:00Z', 'video': {'duration_s': 20},
                'frames': [{'location': {'latitude': latitude, 'longitude': -76.5,
                                         'horizontal_accuracy_m': 1}} for latitude in (42.44, 42.441)],
            }))
            (source / 'capture.mov').write_bytes(b'cached capture: decoding must be skipped')
        source_bytes = {p: p.read_bytes() for directory in (captures, lidar)
                        for p in directory.rglob('*') if p.is_file()}
        cloud = np.zeros(4, POINT)
        cloud['xyz'] = [[1, 2, 3], [2, 2, 3], [1, 3, 4], [2, 3, 4]]
        cloud['rgba'] = [40, 80, 120, 2]
        cached = output / 'walks/capture-1'
        cached.mkdir(parents=True)
        cloud.tofile(cached / 'block-consistent.bin')
        command = [sys.executable, str(Path(__file__).with_name('reconstruct.py')),
                   str(captures), '--lidar', str(lidar), '--output', str(output), '--only', 'capture-1']

        def run(expected):
            result = subprocess.run(command, capture_output=True, text=True)
            assert result.returncode == 0, result.stdout + result.stderr
            model = json.loads((output / 'model/model.json').read_text())
            assert model['point_count'] == len(expected), model
            header, payload = (output / 'model/preview.ply').read_bytes().split(b'end_header\n', 1)
            assert f'element vertex {len(expected)}\n'.encode() in header
            assert payload == expected.view(np.uint8).reshape(-1, 16)[:, :15].tobytes()
            assert not any((output / 'work').iterdir()), 'a cached capture entered reconstruction'

        run(cloud)
        previous = {p.relative_to(output / 'model'): p.read_bytes()
                    for p in (output / 'model').rglob('*') if p.is_file()}
        # A second cached capture joins the next export even when --only selects an exact first ID.
        second = cloud.copy()
        second['xyz'] += [10, 0, 10]
        cached = output / 'walks/capture-10'
        cached.mkdir()
        second.tofile(cached / 'block-consistent.bin')
        run(np.concatenate([cloud, second]))
        assert previous == {p.relative_to(output / 'model.previous'): p.read_bytes()
                            for p in (output / 'model.previous').rglob('*') if p.is_file()}
        assert source_bytes == {p: p.read_bytes() for p in source_bytes}
        print('cached reconstruction ok: 4 → 8 exported points; source files and previous model preserved')


if __name__ == '__main__':
    check_cached_reconstruction()
