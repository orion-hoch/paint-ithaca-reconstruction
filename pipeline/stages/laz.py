"""Classified LiDAR in the city frame (x east, y up, z south), from one tile or several joined with commas."""
import numpy as np
import laspy

ORIGIN = np.array([377249.9995, 4700249.9995, 150.0])


def read_city(paths):
    """(xyz, classification) for every tile in `paths`, so a walk near a tile edge sees both sides."""
    xyz, cls = [], []
    for path in str(paths).split(','):
        las = laspy.read(path)
        xyz.append(np.column_stack([np.asarray(las.x) - ORIGIN[0], np.asarray(las.z) - ORIGIN[2], ORIGIN[1] - np.asarray(las.y)]))
        cls.append(np.asarray(las.classification))
    return np.concatenate(xyz), np.concatenate(cls)
