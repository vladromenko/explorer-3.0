"""The SO-101 table station as a KNOWLEDGE helper: it measures an object on its table and returns
robot-independent grasp knowledge (metric width / length / height, grasp height) for the recipient.

Real station: the measuring mechanics are COPIED from the SO-101 pick-and-place project into
explorer3/so101_vendor (fixed overview camera, validated hand-eye calibration, eight-vertex box fit
to the SAM 3 silhouette). Explorer does not call or change that project or Wristlab. The station
does not move the arm: the knowledge is 'measured', not yet 'grasp-verified' (success=None); it is
confirmed or refuted by the recipient's own grasp (knowledge.record_use).
"""
import time

from . import knowledge


class SO101Station:
    def __init__(self, log=None, device='cuda'):
        self.log = log or (lambda m: None)
        self.device = device

    def available(self):
        from .so101_vendor import measure
        return measure.CALIBRATION.is_file() and __import__('os').path.exists(measure.CAMERA)

    def practice(self, prompt):
        """Measure `prompt` on the table. Returns {'ok': True, 'knowledge': item} or {'ok': False, 'reason'}."""
        from .so101_vendor import measure
        if not self.available():
            return {'ok': False, 'reason': 'верхняя камера SO-101 не подключена'}
        self.log(f'SO-101: измеряю «{prompt}» верхней камерой (SAM 3 + калибровка стола)')
        try:
            m = measure.measure(prompt, device=self.device)
        except Exception as exc:
            return {'ok': False, 'reason': f'стенд SO-101: {type(exc).__name__}: {exc}'}
        if not m['ok']:
            return m
        params = {'width_m': m['width_m'], 'length_m': m['length_m'], 'height_m': m['height_m'],
                  'grasp_height_m': round(max(.012, min(m['height_m'], .06) / 2), 3)}
        self.log(f'SO-101: «{prompt}» {m["width_m"] * 100:.1f}×{m["length_m"] * 100:.1f} см, высота {m["height_m"] * 100:.1f} см ({m["model"]})')
        item = knowledge.add(prompt, 'so101_station', params, success=None,
                             evidence={k: m[k] for k in ('model', 'silhouette_rmse_px', 'score', 'overlay', 'dir', 'height_sigma_m')})
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
        s = size * (1 + self.world.measure_rng.uniform(-.05, .05))
        item = knowledge.add(prompt, 'so101_station', {'width_m': round(s, 3), 'height_m': round(s, 3),
                                                       'grasp_height_m': round(max(.012, s / 2), 3)},
                             evidence={'sim': True}, success=True, domain='sim')
        return {'ok': True, 'knowledge': item}
