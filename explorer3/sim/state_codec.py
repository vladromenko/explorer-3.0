"""Typed JSON snapshots: arrays and RNG tuples, never executable pickle objects."""
import base64
import hashlib
import json
from pathlib import Path
import numpy as np


def encode(value):
    if isinstance(value,np.ndarray):
        if value.dtype.hasobject:raise ValueError('Object arrays are not snapshots')
        return {'type':'array','dtype':value.dtype.str,'shape':list(value.shape),
                'data':base64.b64encode(value.tobytes()).decode()}
    if isinstance(value,np.generic):return encode(value.item())
    if type(value).__module__.startswith('torch'):
        return {'type':'torch','value':encode(value.detach().cpu().numpy())}
    if isinstance(value,dict):return {'type':'dict','value':[[encode(k),encode(v)] for k,v in value.items()]}
    if isinstance(value,(tuple,list)):return {'type':'tuple' if isinstance(value,tuple) else 'list','value':[encode(x) for x in value]}
    if value is None or isinstance(value,(str,bool,int,float)):return value
    raise TypeError('Unsupported snapshot state: '+str(type(value)))


def decode(value):
    if not isinstance(value,dict):return value
    kind=value['type']
    if kind=='array':
        dtype=np.dtype(value['dtype'])
        if dtype.hasobject:raise ValueError('Object arrays are not snapshots')
        return np.frombuffer(base64.b64decode(value['data'],validate=True),dtype=dtype).reshape(value['shape']).copy()
    if kind=='torch':
        import torch
        return torch.from_numpy(decode(value['value']))
    if kind=='dict':return {decode(k):decode(v) for k,v in value['value']}
    if kind in {'list','tuple'}:
        items=[decode(x) for x in value['value']]
        return tuple(items) if kind=='tuple' else items
    raise ValueError('Unknown snapshot type')


def dump(path,value):
    raw=json.dumps(encode(value),sort_keys=True,separators=(',',':'),allow_nan=False).encode()
    temporary=Path(str(path)+'.tmp');temporary.write_bytes(raw);temporary.replace(path)
    return hashlib.sha256(raw).hexdigest()


def load(path):return decode(json.loads(Path(path).read_text()))
