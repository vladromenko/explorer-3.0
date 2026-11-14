"""Navigation on the shared lidar map (explorer3.fleet_map): path planning around furniture,
path following, a collision guard from the latest scan, and frontiers (where the map ends)
for autonomous exploration. Works on a 10 cm planning grid derived from the 5 cm map."""
import heapq
import math

import cv2
import numpy as np

from .fleet_map import RES, SIZE, wrap

STEP = 2                      # planning cell = 2 map cells = 10 cm
PRES = RES * STEP
RADIUS = {'rosmaster': .22, 'tank': .18}          # half-width + margin (a 0.9 m doorway must stay open)
BODY_FRONT = {'rosmaster': .24, 'tank': .17}
BODY_HALF_W = {'rosmaster': .18, 'tank': .14}


def _cell(xy):
    return (int(math.floor(xy[0] / PRES)) + SIZE // (2 * STEP), int(math.floor(xy[1] / PRES)) + SIZE // (2 * STEP))


def _xy(c):
    return ((c[0] - SIZE // (2 * STEP) + .5) * PRES, (c[1] - SIZE // (2 * STEP) + .5) * PRES)


def layers(grid, robot):
    """(blocked, free, unknown) boolean planning grids, indexed [y, x]."""
    n = SIZE // STEP
    occ = grid.occupied().reshape(n, STEP, n, STEP).any(axis=(1, 3))
    free = grid.known_free().reshape(n, STEP, n, STEP).any(axis=(1, 3)) & ~occ
    r = max(1, int(round(RADIUS[robot] / PRES)))
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1, 2 * r + 1))
    blocked = cv2.dilate(occ.astype(np.uint8), k).astype(bool)
    unknown = ~free & ~occ
    return blocked, free, unknown


def layers_occ(grid):
    n = SIZE // STEP
    return grid.occupied().reshape(n, STEP, n, STEP).any(axis=(1, 3)).astype(np.uint8)


def plan(grid, robot, start, goal, allow_unknown=True):
    """A* from start to goal (room metres). Returns a list of waypoints or None."""
    blocked, free, unknown = layers(grid, robot)
    n = blocked.shape[0]
    s, g = _cell(start), _cell(goal)
    if not (0 <= s[0] < n and 0 <= s[1] < n and 0 <= g[0] < n and 0 <= g[1] < n):
        return None
    blocked = blocked.copy()
    # The robot stands where it stands: clear its own footprint so it can leave a tight spot.
    r = int(math.ceil(RADIUS[robot] / PRES))
    blocked[max(0, s[1] - r):s[1] + r + 1, max(0, s[0] - r):s[0] + r + 1] = False
    if blocked[g[1], g[0]]:
        # Goal inside furniture inflation (an object next to a wall): nearest reachable cell.
        ys, xs = np.nonzero(~blocked & (free | (unknown if allow_unknown else False)))
        if len(xs) == 0:
            return None
        i = int(np.argmin((xs - g[0]) ** 2 + (ys - g[1]) ** 2))
        if math.hypot(xs[i] - g[0], ys[i] - g[1]) * PRES > .6:
            return None
        g = (int(xs[i]), int(ys[i]))
    cost = np.where(free, 1., 3. if allow_unknown else np.inf)
    # Soft clearance: hard inflation keeps doorways open, this keeps the path in the middle of
    # passages and away from furniture whenever there is room (up to +4 per cell at the edge).
    occ = layers_occ(grid)
    dist = cv2.distanceTransform((1 - occ).astype(np.uint8), cv2.DIST_L2, 3) * PRES
    cost = cost + 8. * np.clip((.55 - dist) / .3, 0., 1.) ** 2
    moves = [(1, 0, 1.), (-1, 0, 1.), (0, 1, 1.), (0, -1, 1.), (1, 1, 1.414), (1, -1, 1.414), (-1, 1, 1.414), (-1, -1, 1.414)]
    dist = {s: 0.}
    came = {}
    heap = [(0., s)]
    while heap:
        _, c = heapq.heappop(heap)
        if c == g:
            break
        d0 = dist[c]
        for dx, dy, m in moves:
            nx, ny = c[0] + dx, c[1] + dy
            if not (0 <= nx < n and 0 <= ny < n) or blocked[ny, nx]:
                continue
            step = m * cost[ny, nx]
            if not math.isfinite(step):
                continue
            nd = d0 + step
            if nd < dist.get((nx, ny), math.inf):
                dist[(nx, ny)] = nd
                came[(nx, ny)] = c
                heapq.heappush(heap, (nd + math.hypot(g[0] - nx, g[1] - ny), (nx, ny)))
    if g not in dist:
        return None
    cells = [g]
    while cells[-1] != s:
        cells.append(came[cells[-1]])
    cells.reverse()
    # Straight shortcuts only with one extra cell of clearance: otherwise they graze the hard
    # inflation edge and undo the soft clearance cost (A* cells themselves may stay tight, e.g. doorways).
    clear = cv2.dilate(blocked.astype(np.uint8), np.ones((3, 3), np.uint8)).astype(bool)
    return [_xy(c) for c in _shortcut(cells, clear)]


def _shortcut(cells, blocked):
    """Drop waypoints that can be skipped along a straight free line."""
    out = [cells[0]]
    i = 0
    while i < len(cells) - 1:
        j = len(cells) - 1
        while j > i + 1 and not _line_free(cells[i], cells[j], blocked):
            j -= 1
        out.append(cells[j])
        i = j
    return out


def _line_free(a, b, blocked):
    k = max(abs(b[0] - a[0]), abs(b[1] - a[1]))
    for t in np.linspace(0, 1, k + 1):
        x, y = int(round(a[0] + (b[0] - a[0]) * t)), int(round(a[1] + (b[1] - a[1]) * t))
        if blocked[y, x]:
            return False
    return True


def follow(pose, path, v_max=.15, w_max=.8, lookahead=.12):
    """Waypoint following: straight segments, turning in place at the corners, so the robot
    does not cut a corner into the furniture. Returns (v, w, remaining metres)."""
    x, y, t = pose
    # Drop waypoints already reached.
    while len(path) > 1 and math.hypot(path[0][0] - x, path[0][1] - y) < lookahead:
        path.pop(0)
    tx, ty = path[0]
    remaining = math.hypot(tx - x, ty - y) + sum(math.dist(path[i], path[i + 1]) for i in range(len(path) - 1))
    b = wrap(math.atan2(ty - y, tx - x) - t)
    w = max(-w_max, min(w_max, 1.6 * b))
    v = 0. if abs(b) > .6 else v_max * math.cos(b) ** 2 * min(1., remaining / .4 + .3)
    return v, w, remaining


def guard(points, robot, v, w, stop_m=.18):
    """Collision guard from the latest scan (robot base frame). Stops forward motion when
    something is in the corridor ahead and turns toward the freer side.
    Returns (v, w, blocked_reason or None)."""
    if points is None or v <= 0:
        return v, w, None
    front, half = BODY_FRONT[robot], BODY_HALF_W[robot] + .04
    reach = front + stop_m + v * .8
    ahead = points[(points[:, 0] > front - .02) & (points[:, 0] < reach) & (np.abs(points[:, 1]) < half)]
    if len(ahead) < 2:
        return v, w, None
    gap = float(ahead[:, 0].min() - front)
    left = points[(points[:, 0] > 0) & (points[:, 0] < reach + .3) & (points[:, 1] > 0) & (points[:, 1] < .6)]
    right = points[(points[:, 0] > 0) & (points[:, 0] < reach + .3) & (points[:, 1] < 0) & (points[:, 1] > -.6)]
    turn = .6 if len(left) <= len(right) else -.6
    return 0., (w if abs(w) > .2 else turn), f'препятствие впереди в {gap:.2f} м'


def frontiers(grid, robot, pose, min_size_m=.3, banned=(), visual=False):
    """Reachable places where known free space meets unknown space, best first.
    visual=True (object search): also floor the cameras have not looked at yet counts as unknown,
    so the search goes on until the cameras — not only the lidars — have covered the room.
    Returns [(x, y, size_m, path_m)]."""
    blocked, free, unknown = layers(grid, robot)
    free_all = free
    if visual:
        n = SIZE // STEP
        seen = grid.seen.reshape(n, STEP, n, STEP).any(axis=(1, 3))
        unseen = free & ~seen
        # unseen floor next to walls is fine (cameras look at the floor from a distance)
        unknown = unknown | (unseen & ~blocked)
        free = free & seen
    n = blocked.shape[0]
    un = unknown.astype(np.uint8)
    near_unknown = cv2.dilate(un, np.ones((3, 3), np.uint8)).astype(bool)
    edge = free & near_unknown & ~blocked
    count, labels, stats, cents = cv2.connectedComponentsWithStats(edge.astype(np.uint8), connectivity=8)
    if count <= 1:
        return []
    # Travel distance from the robot over known free, unblocked space (BFS).
    s = _cell(pose[:2])
    passable = (free_all | edge) & ~blocked
    r = int(math.ceil(RADIUS[robot] / PRES))
    passable[max(0, s[1] - r):s[1] + r + 1, max(0, s[0] - r):s[0] + r + 1] = True
    dist = np.full((n, n), np.inf)
    dist[s[1], s[0]] = 0
    queue = [s]
    for c in queue:
        d = dist[c[1], c[0]] + 1
        for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            nx, ny = c[0] + dx, c[1] + dy
            if 0 <= nx < n and 0 <= ny < n and passable[ny, nx] and dist[ny, nx] > d:
                dist[ny, nx] = d
                queue.append((nx, ny))
    # A long boundary is cut into ~1 m pieces so that two robots can take different parts of it.
    tile = max(1, int(round(1.0 / PRES)))
    pieces = []
    for i in range(1, count):
        if stats[i, cv2.CC_STAT_AREA] * PRES < min_size_m:
            continue
        ys, xs = np.nonzero(labels == i)
        keys = (ys // tile) * 100000 + xs // tile
        for k in np.unique(keys):
            sel = keys == k
            pieces.append((xs[sel], ys[sel]))
    out = []
    for xs, ys in pieces:
        size = len(xs) * PRES
        d = dist[ys, xs]
        if not np.isfinite(d).any():
            continue
        # The cell of the cluster closest to its centre among reachable ones.
        ok = np.isfinite(d)
        cx, cy = xs.mean(), ys.mean()
        j = int(np.argmin(np.where(ok, (xs - cx) ** 2 + (ys - cy) ** 2, np.inf)))
        x, y = _xy((int(xs[j]), int(ys[j])))
        if any(math.hypot(x - bx, y - by) < .4 for bx, by in banned):
            continue
        out.append((x, y, size, float(d[j]) * PRES))
    # Close and large first: a big opening nearby is worth more than a crack far away.
    return sorted(out, key=lambda f: f[3] - 1.5 * min(f[2], 2.))
