"""Bounded read-only recordings of the common ROS observation bus."""
import json
import shutil
import subprocess
import threading
import time
import uuid
from pathlib import Path
from .paths import DATA,ROOT

_lock=threading.Lock()
_active=None


def directory():
    path=DATA/'fleet-captures';path.mkdir(parents=True,exist_ok=True);return path


def entry(identifier):
    if len(identifier)!=32 or any(c not in '0123456789abcdef' for c in identifier):
        raise ValueError('Неверный номер записи')
    path=directory()/identifier
    if not path.is_dir():raise ValueError('Запись не найдена')
    return path


def records():
    result=[]
    for path in directory().glob('*/manifest.json'):
        try:
            value=json.loads(path.read_text())
            if value['phase']=='recording' and time.time()-value['created_at']>value.get('requested_seconds',60)+150:
                value={**value,'phase':'interrupted','error':'Процесс не подтвердил завершение записи'}
            result.append(value)
        except (OSError,ValueError):continue
    return sorted(result,key=lambda x:x['created_at'],reverse=True)[:50]


def command(identifier,seconds):
    return ['docker','run','--rm','--name','explorer3-capture-'+identifier,
        '--network','host','--user','1000:1000','--read-only','--cap-drop','ALL',
        '--security-opt','no-new-privileges','--tmpfs','/tmp:rw,nosuid,nodev,size=64m',
        '-e','FASTRTPS_DEFAULT_PROFILES_FILE=/workspace/agents/ros2_hub/fastdds.xml',
        '--mount',f'type=bind,src={ROOT},dst=/workspace,readonly',
        '--mount',f'type=bind,src={entry(identifier)},dst=/capture',
        'explorer3-ros2-hub:jazzy','python3','scripts/capture_ros2_observations.py',
        '--seconds',str(seconds)]


def start(seconds=30):
    global _active
    from .fleet_live import read_hub_status
    if type(seconds) is not int or seconds not in (15,30,60):raise ValueError('Выберите 15, 30 или 60 секунд')
    hub=read_hub_status()
    if not hub.get('running'):raise ValueError('Общая шина пока не отвечает')
    if not any(not r.get('error') and r.get('ages_s',{}).get('rgb',99)<2 for r in hub.get('robots',{}).values()):
        raise ValueError('Нет свежих данных роботов для записи')
    with _lock:
        if _active or any(r['phase']=='recording' for r in records()):raise ValueError('Общая запись уже идёт')
        if shutil.disk_usage(directory()).free<1024**3:raise ValueError('Для записи нужен 1 ГБ свободного места')
        identifier=uuid.uuid4().hex;path=directory()/identifier;path.mkdir()
        manifest={'id':identifier,'created_at':time.time(),'requested_seconds':seconds,'phase':'recording',
            'kind':'diagnostic_rosbag','training_eligible':False,'motion_commands_sent':0,
            'common_map_calibrated':False,'initial_hub_status':hub}
        (path/'manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2))
        _active=identifier
        threading.Thread(target=_run,args=(identifier,seconds),daemon=True).start()
    return {'id':identifier,'phase':'recording'}


def _run(identifier,seconds):
    global _active
    path=entry(identifier)
    try:
        with (path/'capture.log').open('w') as log:
            process=subprocess.run(command(identifier,seconds),stdout=log,stderr=subprocess.STDOUT,timeout=seconds+90)
        if process.returncode:raise RuntimeError('Не удалось сохранить общую запись; журнал сохранён')
        manifest=json.loads((path/'manifest.json').read_text())
        if manifest.get('phase')!='complete':raise RuntimeError('Запись завершилась без отчёта')
    except Exception as exc:
        # Stop only this bounded diagnostic reader, never the hub or robot services.
        try:subprocess.run(['docker','stop','--time','3','explorer3-capture-'+identifier],capture_output=True,timeout=8)
        except (OSError,subprocess.TimeoutExpired):pass
        manifest=json.loads((path/'manifest.json').read_text())
        if manifest.get('phase')=='complete' and manifest.get('readback',{}).get('passed'):
            manifest['cleanup_warning']='Файл сохранён и прочитан; завершение процесса записи потребовало дополнительной остановки'
        else:
            message='Превышено время подготовки или сохранения записи' if isinstance(exc,subprocess.TimeoutExpired) else str(exc)
            manifest.update(phase='failed',error=message)
        (path/'manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2))
    finally:
        with _lock:_active=None


def state():
    values=records()
    active=_active or next((r['id'] for r in values if r['phase']=='recording'),None)
    return {'active':active,'records':values}
