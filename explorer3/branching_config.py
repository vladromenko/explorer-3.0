"""Finite simulation comparison budgets, safe to import in the UI process."""
import math


def validate_config(config):
    for key,low,high in [('warmup_s',0,20),('delay_s',0,10),('correction_s',.05,10),('horizon_s',.1,30)]:
        value=config[key]
        if not isinstance(value,(int,float)) or not math.isfinite(value) or not low<=value<=high:
            raise ValueError('Invalid branch budget: '+key)
    if config['delay_s']+config['correction_s']>config['horizon_s']:
        raise ValueError('Horizon must include the same waiting and correction window')
    if type(config['repeats']) is not int or not 2<=config['repeats']<=3:
        raise ValueError('Use 2 or 3 matched repeats to verify restoration')

