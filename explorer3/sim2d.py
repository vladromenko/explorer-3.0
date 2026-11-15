"""2D rooms for the fleet benchmark and the swarm's «без роботов» mode: walls as segments, a lidar
by ray casting. Rooms: random size, furniture boxes, an internal wall with a 0.9 m doorway."""
import math

import numpy as np


def make_room(rng):
    w, h = rng.uniform(4.5, 7.5), rng.uniform(3.5, 6.)
    segs = [((0, 0), (w, 0)), ((w, 0), (w, h)), ((w, h), (0, h)), ((0, h), (0, 0))]
    # internal wall with a 0.9 m doorway (two rooms)
    x = rng.uniform(.4, .6) * w
    door = rng.uniform(.8, h - 1.7)
    segs += [((x, 0), (x, door)), ((x, door + .9), (x, h))]
    boxes = []
    for _ in range(rng.integers(3, 7)):
        bw, bh = rng.uniform(.3, 1.2), rng.uniform(.3, .9)
        for _try in range(20):
            bx, by = rng.uniform(.2, w - bw - .2), rng.uniform(.2, h - bh - .2)
            if abs(bx + bw / 2 - x) < bw / 2 + .5 or any(bx < c + d + .6 and c < bx + bw + .6 and by < e + f + .6 and e < by + bh + .6
                                                        for c, e, d, f in boxes):
                continue
            boxes.append((bx, by, bw, bh))
            segs += [((bx, by), (bx + bw, by)), ((bx + bw, by), (bx + bw, by + bh)),
                     ((bx + bw, by + bh), (bx, by + bh)), ((bx, by + bh), (bx, by))]
            break
    return {'w': w, 'h': h, 'segs': np.array(segs, float), 'wall_x': x, 'door': door, 'boxes': boxes}


def wall_distance(room, xy):
    a, b = room['segs'][:, 0], room['segs'][:, 1]
    ab = b - a
    t = np.clip(((xy - a) * ab).sum(1) / (ab * ab).sum(1), 0, 1)
    return float(np.linalg.norm(a + t[:, None] * ab - xy, axis=1).min())


def raycast(room, origin, angles):
    d = np.stack([np.cos(angles), np.sin(angles)], 1)[:, None, :]
    p, e = room['segs'][None, :, 0], room['segs'][None, :, 1] - room['segs'][None, :, 0]
    den = d[..., 0] * -e[..., 1] + d[..., 1] * e[..., 0]
    q = p - np.asarray(origin)
    with np.errstate(divide='ignore', invalid='ignore'):
        r = (q[..., 0] * -e[..., 1] + q[..., 1] * e[..., 0]) / den
        u = (d[..., 0] * q[..., 1] - d[..., 1] * q[..., 0]) / den
    return np.where((abs(den) > 1e-9) & (r > .05) & (u >= 0) & (u <= 1), r, np.inf).min(1)


def inside_furniture(room, p):
    return any(bx <= p[0] <= bx + bw and by <= p[1] <= by + bh for bx, by, bw, bh in room['boxes'])


def free_spot(room, rng, clearance=.45, avoid=()):
    for _ in range(500):
        p = np.array([rng.uniform(.4, room['w'] - .4), rng.uniform(.4, room['h'] - .4)])
        if (wall_distance(room, p) > clearance and not inside_furniture(room, p)
                and all(np.linalg.norm(p - a) > 1.2 for a in avoid)):
            return p
    raise RuntimeError('no free spot')


def lidar_packet(room, mounts, pose, rng, beams=450, noise=.01):
    """The robot's lidar streams at `pose` as a sensor-agent packet for FleetMap.update."""
    x, y, th = pose
    arrays, streams = {}, {}
    for name, (mx, my, myaw, direction) in mounts.items():
        o = (x + math.cos(th) * mx - math.sin(th) * my, y + math.sin(th) * mx + math.cos(th) * my)
        ang = th + myaw + direction * 2 * math.pi * np.arange(beams) / beams
        rr = raycast(room, o, ang)
        rr = np.where(rr < 12, rr + rng.normal(0, noise, beams), np.inf)
        arrays[name] = rr.astype(np.float32)
        streams[name] = {'angle_min': 0., 'angle_increment': 2 * math.pi / beams, 'range_min': .05}
    return {'arrays': arrays, 'meta': {'streams': streams}}


def visible(room, pose, xy, reach=2.5, fov_deg=35):
    """(distance, bearing) if a camera at pose sees the point xy in line of sight, else None."""
    x, y, th = pose
    d = np.asarray(xy, float) - np.array([x, y])
    dist = float(np.linalg.norm(d))
    a = math.atan2(d[1], d[0])
    bearing = math.atan2(math.sin(a - th), math.cos(a - th))
    if dist < reach and abs(bearing) < math.radians(fov_deg) and raycast(room, (x, y), np.array([a]))[0] > dist - .05:
        return dist, bearing
    return None
