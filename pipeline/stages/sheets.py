"""One surface where several views (or several walks) saw one: shared by filter.py (inside a walk) and fuse.py (across walks)."""
import numpy as np
from scipy.spatial import cKDTree


def collapse(P, window, tube=.6, iterations=2):
    """Move every point onto the median sheet within `window` along the local normal (normals from a 0.3 m thinned copy).
    Copies of one facade 0.7-1.6 m apart become one sheet; relief deeper than about half the window is flattened."""
    for _ in range(iterations):
        vox = np.unique(np.floor(P / .3).astype(np.int64), axis=0, return_index=True)[1]; Q = P[vox]
        if len(Q) < 200: return P
        tq = cKDTree(Q); _, nn = tq.query(Q, k=min(24, len(Q)))
        R = Q[nn] - Q[:, None]; C = np.einsum('nki,nkj->nij', R, R) / nn.shape[1]
        n = np.linalg.eigh(C)[1][:, :, 0]                                                    # smallest-variance axis
        centre = np.zeros(len(Q))
        for i, nb in enumerate(tq.query_ball_point(Q, window)):
            d = (Q[nb] - Q[i]) @ n[i]; lat = np.linalg.norm((Q[nb] - Q[i]) - np.outer(d, n[i]), axis=1)
            d = d[lat < tube]
            if len(d) >= 4: centre[i] = np.median(d)
        _, j = tq.query(P); off = np.einsum('ij,ij->i', P - Q[j], n[j])                       # each point's own offset from its anchor's sheet
        P = P + n[j] * (centre[j] - off)[:, None]
    return P


def farther(P, rng, radius, ratio=.6):
    """True where a view `1/ratio` times nearer saw a surface within `radius`: the far point yields to it."""
    d, j = cKDTree(P).query(P, k=32, distance_upper_bound=radius); j = np.where(np.isfinite(d), j, 0)
    return (np.isfinite(d) & (rng[j] < ratio * rng[:, None])).any(1)


def cells(xyz, lab, size=.10, prefer=None):
    """Index of one point per `size` cell per class; the point with the smallest `prefer` wins (else the first)."""
    ci = np.floor(xyz / size).astype(np.int64); k = (ci[:, 0] * 73856093) ^ (ci[:, 1] * 19349663) ^ (ci[:, 2] * 83492791) ^ (lab.astype(np.int64) << 40)
    o = np.lexsort((prefer, k)) if prefer is not None else np.argsort(k, kind='stable'); ks = k[o]
    return np.sort(o[np.concatenate([[True], ks[1:] != ks[:-1]])])


def nearest_sheet(P, rng, reach=2.5, off=.3, probe=.25):
    """Where one wall was seen as two sheets, keep the sheet seen from nearer and drop the other. Averaging the sheets
    (collapse) blends colours from views that disagree sideways; picking one keeps a single view's surface, and the
    nearer view's depth is the more accurate one. Returns a keep mask. Ranges are smoothed over 0.2 m voxels, and the
    other sheet is looked for along the local normal, OFF the point's own sheet (a wall's own neighbours never count)."""
    key = np.floor(P / .2).astype(np.int64); uk, vox, inv = np.unique(key, axis=0, return_index=True, return_inverse=True); Q = P[vox]
    if len(Q) < 200: return np.ones(len(P), bool)
    tq = cKDTree(Q); _, nn = tq.query(Q, k=min(12, len(Q))); R = Q[nn] - Q[:, None]   # ~0.4 m patch: stays on its own sheet
    nQ = np.linalg.eigh(np.einsum('nki,nkj->nij', R, R))[1][:, :, 0]
    rq = np.zeros(len(Q)); cnt = np.zeros(len(Q)); np.add.at(rq, inv.ravel(), rng); np.add.at(cnt, inv.ravel(), 1); rq /= np.maximum(cnt, 1)
    drop = np.zeros(len(Q), bool)
    for t in np.arange(off + .1, reach + .01, .3):                    # probe both ways along the normal
        for sgn in (1, -1):
            d, j = tq.query(Q + sgn * t * nQ, distance_upper_bound=probe); hit = np.isfinite(d)
            drop[hit] |= rq[j[hit]] < rq[hit]
    return ~drop[inv.ravel()]


def normals(Q):
    _, nn = cKDTree(Q).query(Q, k=min(16, len(Q))); R = Q[nn] - Q[:, None]
    return np.linalg.eigh(np.einsum('nki,nkj->nij', R, R))[1][:, :, 0]


def step(x, c):
    """The 4x4 of one correction: yaw x[0] about c, then shift x[1:]."""
    th = x[0]; M = np.eye(4); M[:3, :3] = [[np.cos(th), 0, np.sin(th)], [0, 1, 0], [-np.sin(th), 0, np.cos(th)]]; M[:3, 3] = c - M[:3, :3] @ c + x[1:]; return M


def align(src, dst, dst_n, c, min_pairs=1500):
    """4-DoF (yaw about c, x, y, z) taking src onto dst by trimmed point-to-plane steps; None when the overlap is too small
    to trust. A step moves only along directions the geometry pins: in a straight street of parallel facades nothing
    fixes the slide along the street, and that component stays put instead of wandering off (degeneracy-aware steps)."""
    tree = cKDTree(dst); x = np.zeros(4)
    for dmax in (2.0, 1.2, .6):
        for _ in range(8):
            M = step(x, c); q = src @ M[:3, :3].T + M[:3, 3]; d, j = tree.query(q, distance_upper_bound=dmax); ok = np.isfinite(d)
            if ok.sum() < min_pairs: return None
            n = dst_n[j[ok]]; v = q[ok] - c; r = np.einsum('ij,ij->i', n, q[ok] - dst[j[ok]])
            keep = np.abs(r) < 3 * 1.4826 * np.median(np.abs(r)) + 1e-3
            L = np.sqrt(np.mean(v[keep][:, 0] ** 2 + v[keep][:, 2] ** 2)) + 1e-9               # yaw in metres of lever, like the shifts
            J = np.column_stack([(n[:, 0] * v[:, 2] - n[:, 2] * v[:, 0]) / L, n])[keep]      # d(n.q)/d(yaw) = n . (y x v)
            w, V = np.linalg.eigh(J.T @ J); pinned = w > 1e-2 * w.max()
            dx = V[:, pinned] @ ((V[:, pinned].T @ (J.T @ r[keep])) / w[pinned]); dx[0] /= L; x -= dx
    return x, int(keep.sum())


if __name__ == '__main__':
    # two sheets of one wall: the nearer (range 5) stays, the farther (range 9) goes, at any gap; a lone sheet and a
    # single wall seen at an angle (range changing along it) stay whole
    g = np.stack(np.meshgrid(np.arange(0, 6, .1), np.arange(0, 4, .1)), -1).reshape(-1, 2); n = len(g)
    near = np.column_stack([g[:, 0], g[:, 1], np.zeros(n)]); lone = near + [20, 0, 0]
    for gap in (.5, .8, 1.5, 2.2):
        k = nearest_sheet(np.concatenate([near, near + [0, 0, gap], lone]), np.concatenate([np.full(n, 5.), np.full(n, 9.), np.full(n, 9.)]))
        assert k[:n].mean() > .99 and k[n:2 * n].mean() < .05 and k[2 * n:].mean() > .99, gap
    assert nearest_sheet(np.column_stack([g[:, 0], g[:, 1], .3 * g[:, 0]]), 3 + g[:, 0]).mean() > .99
    print('sheets ok')

    # align: a straight street (two facades, ground) shifted 0.6 m across and turned 0.4 deg comes back, and does not slide
    # along the street, which nothing pins
    rng = np.random.default_rng(0); x = rng.uniform(0, 60, 20000); y = rng.uniform(0, 8, 20000)
    dst = np.vstack([np.column_stack([x, y, np.zeros(20000)]), np.column_stack([x, y, np.full(20000, 15.)]), np.column_stack([x, np.zeros(20000), rng.uniform(0, 15, 20000)])])
    dst += rng.normal(0, .03, dst.shape); c = dst.mean(0); M = step(np.array([np.radians(.4), 0, 0, .6]), c)
    fit, _ = align(dst[rng.permutation(len(dst))[:15000]] @ M[:3, :3].T + M[:3, 3], dst, normals(dst), c)
    assert abs(np.degrees(fit[0]) + .4) < .1 and abs(fit[3] + .6) < .1 and abs(fit[1]) < .3, fit
    print('align ok')
