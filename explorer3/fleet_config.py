"""Single operator-owned hardware admission document; absence never grants access."""
from contextlib import contextmanager
import copy
import fcntl
import json
import os
from pathlib import Path
import tempfile
import time
from .paths import DATA

ROBOTS = ('so101', 'rosmaster', 'tank')
NAMES = {'so101':'SO-101','rosmaster':'ROSMASTER','tank':'Tank'}


def path(root=None):
    return Path(root if root is not None else DATA)/'configs/fleet.json'


def validate(value):
    if not isinstance(value,dict) or value.get('version')!=1:
        raise ValueError('Неизвестный формат допуска парка')
    robots=value.get('robots')
    if not isinstance(robots,dict) or set(robots)!=set(ROBOTS):
        raise ValueError('В допуске должны быть SO-101, ROSMASTER и Tank')
    for name in ROBOTS:
        if not isinstance(robots[name],dict) or type(robots[name].get('enabled')) is not bool:
            raise ValueError('Неверный переключатель '+NAMES[name])
    if robots['so101'].get('owner') not in ('explorer','legacy_8020'):
        raise ValueError('Не указан владелец SO-101')
    if robots['tank'].get('mode')!='floor':
        raise ValueError('Неизвестный режим Tank')
    if type(value.get('training_authorized')) is not bool:
        raise ValueError('Не указан допуск обучения')
    return value


def load(root=None):
    try:return validate(json.loads(path(root).read_text()))
    except (OSError,json.JSONDecodeError,ValueError) as exc:
        raise ValueError('Допуск парка не настроен: откройте «Готовность»') from exc


def robot_allowed(robot,root=None):
    if robot not in ROBOTS:return False,'Выберите робота'
    try:value=load(root)
    except ValueError as exc:return False,str(exc)
    entry=value['robots'][robot]
    if not entry['enabled']:return False,NAMES[robot]+' выключен в «Готовности»'
    if robot=='so101' and entry['owner']!='explorer':
        return False,'SO-101 у прежней системы: нажмите «Взять SO-101 в Explorer»'
    return True,'Разрешён запуск пользователем'


def require_robot(robot,root=None):
    allowed,reason=robot_allowed(robot,root)
    if not allowed:raise PermissionError(reason)


@contextmanager
def locked(root=None):
    folder=path(root).parent
    folder.mkdir(parents=True,exist_ok=True)
    with (folder/'fleet.lock').open('a') as handle:
        fcntl.flock(handle,fcntl.LOCK_EX)
        try:yield
        finally:fcntl.flock(handle,fcntl.LOCK_UN)


def atomic_write(destination,value):
    destination=Path(destination);destination.parent.mkdir(parents=True,exist_ok=True)
    fd,name=tempfile.mkstemp(prefix=destination.name+'.',suffix='.tmp',dir=destination.parent)
    try:
        with os.fdopen(fd,'w') as stream:
            json.dump(value,stream,ensure_ascii=False,indent=2)
            stream.write('\n');stream.flush();os.fsync(stream.fileno())
        os.replace(name,destination)
        directory=os.open(destination.parent,os.O_RDONLY)
        try:os.fsync(directory)
        finally:os.close(directory)
    finally:
        if os.path.exists(name):os.unlink(name)


def update_robot(robot,*,enabled=None,owner=None,root=None):
    if robot not in ROBOTS:raise ValueError('Неизвестный робот')
    with locked(root):
        value=copy.deepcopy(load(root))
        if enabled is not None:
            if type(enabled) is not bool:raise ValueError('Нужен переключатель включения')
            value['robots'][robot]['enabled']=enabled
        if owner is not None:
            if robot!='so101':raise ValueError('Передача владельца предусмотрена только для SO-101')
            value['robots'][robot]['owner']=owner
        value.update(updated_by='user',updated_at=time.time())
        atomic_write(path(root),validate(value))
        return value
