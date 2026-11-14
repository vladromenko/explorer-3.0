"""Reversible migration of old admission files, never robot I/O or service control."""
import hashlib
import fcntl
from contextlib import contextmanager
import json
from pathlib import Path
import shutil
import time
import uuid
from . import fleet_config

LEGACY = (
    'configs/workflow.json','configs/local-only.json','configs/so101-external-owner.json',
    'configs/so101-hardware-enabled.json','configs/rosmaster-hardware-enabled.json',
    'configs/tank-hardware-enabled.json','configs/sensor-observation-enabled.json',
    'stage-b-authorized.json','stage-c-authorized.json','stage-d-authorized.json',
    'stage-c-authorization-suspended-incident.json','validation/tank-motion-incident.json',
    'configs/bench-authorized.json',
)


def digest(path):return hashlib.sha256(path.read_bytes()).hexdigest()


@contextmanager
def idle(root):
    folder=Path(root)/'locks';folder.mkdir(parents=True,exist_ok=True)
    with (folder/'physical-activity.lock').open('a') as handle:
        try:fcntl.flock(handle,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError as exc:raise RuntimeError('Сначала завершите управление и запись') from exc
        try:yield
        finally:fcntl.flock(handle,fcntl.LOCK_UN)


def migrate(root):
    root=Path(root)
    with idle(root), fleet_config.locked(root):
        destination=fleet_config.path(root)
        journal=root/'configs/permission-migration.json'
        if journal.exists():
            record=json.loads(journal.read_text())
            if record.get('status')=='completed':
                fleet_config.load(root)
                return record
            raise ValueError('Миграция прервана: восстановите её архив перед повтором')
        if destination.exists():
            fleet_config.load(root)
            return {'status':'existing_fleet_preserved','fleet':str(destination)}
        stamp=time.strftime('%Y%m%dT%H%M%SZ',time.gmtime())+'-'+uuid.uuid4().hex[:8]
        archive=root/'archive'/('permissions-'+stamp)
        archive.mkdir(parents=True,exist_ok=False)
        entries=[]
        for relative in LEGACY:
            source=root/relative
            if source.exists():
                if not source.is_file() or source.is_symlink():raise ValueError('Недопустимый файл допуска: '+relative)
                backup=archive/relative;backup.parent.mkdir(parents=True,exist_ok=True)
                shutil.copy2(source,backup)
                entries.append({'path':relative,'sha256':digest(backup)})
        # Migration never assumes legacy service ownership has been relinquished.
        value={'version':1,'robots':{'so101':{'enabled':True,'owner':'legacy_8020'},
            'rosmaster':{'enabled':True},'tank':{'enabled':True,'mode':'floor'}},
            'training_authorized':False,'updated_by':'user','updated_at':time.time()}
        record={'status':'prepared','archive':str(archive),'files':entries,
            'reason':'RECORDING_READINESS_PROMPT_RU.md R1; user decision 2026-11-14',
            'performed_by':'explorer3 migration','created_at':time.time(),'fleet':value,
            'fleet_sha256':hashlib.sha256((json.dumps(value,ensure_ascii=False,indent=2)+'\n').encode()).hexdigest()}
        fleet_config.atomic_write(archive/'README.json',record)
        fleet_config.atomic_write(journal,record)
        fleet_config.atomic_write(destination,fleet_config.validate(value))
        record['fleet_sha256']=digest(destination)
        fleet_config.atomic_write(archive/'README.json',record)
        fleet_config.atomic_write(journal,record)
        for entry in entries:
            source=root/entry['path']
            if digest(source)!=entry['sha256']:raise ValueError('Допуск изменён во время миграции: '+entry['path'])
            source.unlink()
        record['status']='completed'
        fleet_config.atomic_write(archive/'README.json',record)
        fleet_config.atomic_write(journal,record)
        return record


def rollback(root,archive):
    root=Path(root).resolve();archive=Path(archive).resolve()
    if archive.parent!=root/'archive' or not archive.name.startswith('permissions-'):
        raise ValueError('Архив должен находиться в archive/permissions-*')
    with idle(root), fleet_config.locked(root):
        record=json.loads((archive/'README.json').read_text())
        destination=fleet_config.path(root)
        if record.get('status')=='rolled_back':return record
        if destination.exists() and digest(destination)!=record.get('fleet_sha256'):
            raise ValueError('fleet.json изменён после миграции; автоматический откат не затрёт изменения')
        for entry in record['files']:
            if entry['path'] not in LEGACY:raise ValueError('Неизвестный путь в архиве')
            backup=archive/entry['path'];target=root/entry['path']
            if digest(backup)!=entry['sha256']:raise ValueError('Архив повреждён: '+entry['path'])
            if target.exists() and digest(target)!=entry['sha256']:
                raise ValueError('Файл изменён после миграции: '+entry['path'])
        for entry in record['files']:
            target=root/entry['path'];target.parent.mkdir(parents=True,exist_ok=True)
            shutil.copy2(archive/entry['path'],target)
        if destination.exists():destination.unlink()
        journal=root/'configs/permission-migration.json'
        if journal.exists():journal.unlink()
        record['status']='rolled_back';record['rolled_back_at']=time.time()
        fleet_config.atomic_write(archive/'README.json',record)
        return record
