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
    Also the sides: a wall right beside the body (within 5 cm) forbids turning the body into it and
    slows the robot (front corners sweep sideways when it turns).
    Returns (v, w, blocked_reason or None)."""
    if points is None:
        return v, w, None
    side_note = None
    front_b, half_b = BODY_FRONT[robot], BODY_HALF_W[robot]
    beside = points[(np.abs(points[:, 0]) < front_b) & (np.abs(points[:, 1]) >= half_b - .02) & (np.abs(points[:, 1]) < half_b + .05)]
    if len(beside) >= 2:
        near_left = bool((beside[:, 1] > 0).sum() >= 2)
        near_right = bool((beside[:, 1] < 0).sum() >= 2)
        # turning left swings the front corner left and the rear corner right (and vice versa)
        if (w > 0 and near_left) or (w < 0 and near_right):
            w = 0.
            side_note = 'стена сбоку: не поворачиваю в неё'
        v = min(v, .05)
    if v <= 0:
        return v, w, side_note
    front, half = BODY_FRONT[robot], BODY_HALF_W[robot] + .04
    reach = front + stop_m + v * .8
    ahead = points[(points[:, 0] > front - .02) & (points[:, 0] < reach) & (np.abs(points[:, 1]) < half)]
    if len(ahead) < 2:
        return v, w, side_note
    gap = float(ahead[:, 0].min() - front)
    left = points[(points[:, 0] > 0) & (points[:, 0] < reach + .3) & (points[:, 1] > 0) & (points[:, 1] < .6)]
    right = points[(points[:, 0] > 0) & (points[:, 0] < reach + .3) & (points[:, 1] < 0) & (points[:, 1] > -.6)]
    turn = .6 if len(left) <= len(right) else -.6
    return 0., (w if abs(w) > .2 else turn), f'препятствие впереди в {gap:.2f} м'


def frontiers(grid, robot, pose, min_size_m=.3, banned=(), visual=False, look_m=.8):
    """Reachable places where known free space meets unknown space, best first.
    visual=True (object search): also floor the cameras have not looked at yet counts as unknown,
    so the search goes on until the cameras — not only the lidars — have covered the room.
    Returns [(x, y, size_m, path_m, look_x, look_y)]: the place to drive to and the middle of the
    boundary piece to turn the camera to (they differ when the piece is in a gap too narrow to enter)."""
    blocked, free, unknown = layers(grid, robot)
    free_all = free
    n = blocked.shape[0]
    un = unknown.astype(np.uint8)
    near_unknown = cv2.dilate(un, np.ones((3, 3), np.uint8)).astype(bool)
    # Boundary cells inside the inflation band count too: a pocket between furniture and a wall
    # that the robot cannot drive into can still be looked into by the lidar from its mouth.
    edge = free & near_unknown
    if visual:
        # Object search: every reachable piece of floor the cameras have not looked at is a goal
        # too (not only where the lidar map ends); driving there points the camera at it.
        seen = grid.seen.reshape(n, STEP, n, STEP).any(axis=(1, 3))
        edge = edge | (free & ~seen)               # narrow gaps too: looked into from the nearest reachable spot
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
    look = int(round(look_m / PRES))
    clear = cv2.distanceTransform((~blocked).astype(np.uint8), cv2.DIST_L2, 3)   # cells to the nearest blocked one
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
        if np.isfinite(d).any():
            # The cell of the cluster closest to its centre among reachable ones.
            ok = np.isfinite(d)
            cx, cy = xs.mean(), ys.mean()
            j = int(np.argmin(np.where(ok, (xs - cx) ** 2 + (ys - cy) ** 2, np.inf)))
            x, y = _xy((int(xs[j]), int(ys[j])))
            path_m = float(d[j]) * PRES
        else:
            # Not drivable into (inflation band): the nearest reachable cell within look_m of it.
            x0, x1, y0, y1 = max(0, xs.min() - look), min(n, xs.max() + look + 1), max(0, ys.min() - look), min(n, ys.max() + look + 1)
            mask = np.ones((y1 - y0, x1 - x0), np.uint8)
            mask[ys - y0, xs - x0] = 0
            near = cv2.distanceTransform(mask, cv2.DIST_L2, 3)
            sub = dist[y0:y1, x0:x1]
            cand = np.isfinite(sub) & (near <= look)
            roomy = cand & (clear[y0:y1, x0:x1] >= 2)        # not right at the edge of the inflation band
            if roomy.any():
                cand = roomy
            if not cand.any():
                continue
            score = np.where(cand, near * 3 + sub, np.inf)
            jy, jx = np.unravel_index(int(np.argmin(score)), score.shape)
            x, y = _xy((int(jx + x0), int(jy + y0)))
            path_m = float(sub[jy, jx]) * PRES
        if any(math.hypot(x - bx, y - by) < .4 for bx, by in banned):
            continue
        lx, ly = _xy((int(round(xs.mean())), int(round(ys.mean()))))
        out.append((x, y, size, path_m, lx, ly))
    # Close and large first: a big opening nearby is worth more than a crack far away.
    return sorted(out, key=lambda f: f[3] - 1.5 * min(f[2], 2.))


def choose(fronts, peers, spread_m=1.5):
    """Coordinated choice of a frontier (Burgard et al. 2005; Voronoi split as in MMPF/RACER):
    fronts — frontiers() output, best first; peers — [(peer_xy, peer_target_xy or None)].
    A frontier near a peer's target loses value (the peer's lidar will map it anyway), and one the
    peer is clearly closer to is left to the peer. Returns the chosen (x, y, size, path_m) or None."""
    if not fronts:
        return None
    best, score = None, math.inf
    for f in fronts:
        x, y, size, path = f[:4]
        s = path - 1.5 * min(size, 2.)
        for at, target in peers:
            if target is not None:
                d = math.hypot(x - target[0], y - target[1])
                if d < spread_m:
                    s += 3. * (1 - d / spread_m)
            if at is not None and math.hypot(x - at[0], y - at[1]) < .7 * path:
                s += 1.5
        if s < score:
            best, score = f, s
    return best
