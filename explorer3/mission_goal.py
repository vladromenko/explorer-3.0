"""Explicit terminal conditions; no robot commands or alternate dispatcher."""
import math


def first_floor_goal(domain):
    return {'kind': 'floor_zone', 'domain': domain, 'label': 'Квадрат на свободном полу',
            'size_m': [.4, .4], 'center_xy_m': None, 'approach_pose': None,
            'place_pose': None, 'frame': 'fleet_map',
            'selection': 'owner delegated terminal-zone choice on 2026-11-16',
            'success': 'object released and independently observed inside zone'}


def preparation_reason(goal, domain):
    if not isinstance(goal, dict) or goal.get('kind') != 'floor_zone' or goal.get('domain') != domain or goal.get('frame') != 'fleet_map':
        return 'Конечная зона должна быть задана в текущей карте и режиме.'
    for key, length in (('center_xy_m', 2), ('approach_pose', 3), ('place_pose', 6), ('size_m', 2)):
        values = goal.get(key)
        if not isinstance(values, (list, tuple)) or len(values) != length or any(type(v) not in (int, float) or not math.isfinite(v) for v in values):
            return 'Нужно измерить место на полу, подъезд и позу выкладки.'
    if any(not .2 <= value <= .8 for value in goal['size_m']):
        return 'Сторона конечной зоны должна быть от 20 до 80 см.'
    distance = math.dist(goal['center_xy_m'], goal['approach_pose'][:2])
    if not .16 <= distance <= .35:
        return 'Точка подъезда должна находиться в пределах досягаемости руки от центра зоны.'
    if domain == 'real':
        return 'Для реальной доставки ещё нужны проверка позы выкладки и независимое наблюдение предмета на полу.'
    return None


def delivery_verified(goal, observation, *, domain, object_class, since):
    """Require a fresh released-object measurement, not the commanded arm pose."""
    if not isinstance(observation, dict):
        return False
    if observation.get('domain') != domain or observation.get('object_class') != object_class or observation.get('frame') != 'fleet_map':
        return False
    if observation.get('released') is not True or observation.get('inside_height_band') is not True:
        return False
    if type(observation.get('at')) not in (int, float) or not since <= observation['at']:
        return False
    if observation.get('source') not in ('sim2d_object_state', 'independent_visual_measurement'):
        return False
    if domain == 'real' and observation['source'] != 'independent_visual_measurement':
        return False
    xy, sigma = observation.get('xy_m'), observation.get('position_sigma_m')
    half = observation.get('object_half_extent_m')
    if type(half) not in (int, float) or not math.isfinite(half) or not 0 < half <= .1:
        return False
    if not isinstance(xy, (list, tuple)) or len(xy) != 2 or type(sigma) not in (int, float) or not math.isfinite(sigma) or sigma < 0:
        return False
    return all(type(x) in (int, float) and math.isfinite(x) and abs(x-c)+sigma+half <= side/2
               for x,c,side in zip(xy, goal['center_xy_m'], goal['size_m']))
