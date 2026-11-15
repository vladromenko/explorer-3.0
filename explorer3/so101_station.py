"""The SO-101 table station as a KNOWLEDGE helper: it practises / measures an object and returns
robot-independent grasp knowledge (metric size, grasp height) for the recipient (ROSMASTER).

Real station: reuses the existing, physically used SO-101 pick-and-place project
(~/Projects/so101_autonomous_pick_place, service on :8010): its overview camera + SAM 3 fit the
object's 3D box (cubes: all eight vertices), and its collect run is a real grasp attempt.
This adapter only calls that service's HTTP API; it does not drive the arm itself.
Status: the adapter is code-checked; the end-to-end physical run is still to be accepted.
"""
import time

import httpx

from . import knowledge

URL = 'http://127.0.0.1:8010'


def _sizes(scan):
    """Metric sizes of detected objects in a scan result, robust to the service's field names."""
    out = []
    def walk(v):
        if isinstance(v, dict):
            for key in ('size_m', 'edge_m', 'cube_size_m', 'side_m'):
                if isinstance(v.get(key), (int, float)) and 0.005 < v[key] < .3:
                    out.append(float(v[key]))
            for x in v.values():
                walk(x)
        elif isinstance(v, list):
            for x in v:
                walk(x)
    walk(scan)
    return out


class SO101Station:
    def __init__(self, log=None):
        self.log = log or (lambda m: None)

    def available(self):
        try:
            return httpx.get(URL + '/api/status', timeout=3).status_code == 200
        except Exception:
            return False

    def practice(self, prompt, grasp=False):
        """Measure (and optionally grasp) `prompt` on the table. Returns a knowledge item or a reason."""
        if not self.available():
            return {'ok': False, 'reason': 'стенд SO-101 не запущен (пульт → «SO-101 · стенд захвата»)'}
        self.log(f'SO-101: измеряю «{prompt}» верхней камерой')
        try:
            scan = httpx.post(URL + '/api/scan', timeout=120).json()
        except Exception as exc:
            return {'ok': False, 'reason': f'стенд SO-101: {exc}'}
        sizes = _sizes(scan)
        if not sizes:
            return {'ok': False, 'reason': 'стенд SO-101 не нашёл и не измерил предмет'}
        size = sorted(sizes)[len(sizes) // 2]
        success = None
        if grasp:
            self.log('SO-101: пробный захват')
            r = httpx.post(URL + '/api/collect', params={'expected_cubes': 1}, timeout=30)
            started = r.status_code == 200
            end = time.monotonic() + 180
            while started and time.monotonic() < end:
                st = httpx.get(URL + '/api/status', timeout=5).json()
                if not st.get('busy', st.get('running', False)):
                    success = bool(st.get('last_success', st.get('success')))
                    break
                time.sleep(2)
        params = {'width_m': round(size, 3), 'height_m': round(size, 3), 'grasp_height_m': round(max(.012, size / 2), 3)}
        item = knowledge.add(prompt, 'so101_station', params, evidence={'scan_objects': len(sizes)}, success=success)
        return {'ok': True, 'knowledge': item}


class SimSO101Station:
    """Virtual station for the swarm simulation: measures the true object size with small noise."""
    def __init__(self, world, log=None):
        self.world, self.log = world, log or (lambda m: None)

    def available(self):
        return True

    def practice(self, prompt, grasp=True):
        size = self.world.sizes.get(prompt)
        if size is None:
            return {'ok': False, 'reason': f'«{prompt}» нет на столе SO-101'}
        self.log(f'SO-101: пробую взять «{prompt}» и измеряю')
        time.sleep(2.)
        import random
        s = size * (1 + random.uniform(-.05, .05))
        item = knowledge.add(prompt, 'so101_station', {'width_m': round(s, 3), 'height_m': round(s, 3),
                                                       'grasp_height_m': round(max(.012, s / 2), 3)},
                             evidence={'sim': True}, success=True, domain='sim')
        return {'ok': True, 'knowledge': item}
