"""Physical placement constraints, with unknown measurements kept explicit."""
import json
import math
from explorer3.paths import ROOT


def placement():
    config=json.loads((ROOT/'configs/physical-layout.json').read_text())
    support=config['so101_support'];measured=support['measured_height_m']
    if support['kind']!='table':raise ValueError('SO-101 requires table support')
    if measured is not None and not support.get('measurement_source'):
        raise ValueError('A measured table height needs its measurement source')
    height=support['nominal_simulation_height_m'] if measured is None else measured
    if isinstance(height,bool) or not isinstance(height,(int,float)) or not math.isfinite(height) or not .2<=height<=1.5:
        raise ValueError('Invalid simulation support height')
    for robot in ('rosmaster','tank'):
        if config[robot+'_support']!={'kind':'floor','height_m':0.0}:raise ValueError('Mobile platforms must remain on the floor')
    return {**config,'effective_so101_height_m':height,
        'so101_height_status':'assumed_for_simulation' if measured is None else 'measured_height_only',
        'common_world_transform_verified':False}
