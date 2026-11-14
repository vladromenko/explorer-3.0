"""Disk reserves, SQLite backup API and incremental verified local copies. No deletion."""
from contextlib import closing
from datetime import datetime,timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import sqlite3
import threading
import time
from .paths import DATA
from .fleet_config import atomic_write

START_FREE=8*1024**3
FINALIZE_RESERVE=2*1024**3
_lock=threading.Lock()


def require_space(*,starting=True):
    free=shutil.disk_usage(DATA).free
    threshold=START_FREE if starting else FINALIZE_RESERVE
    if free<threshold:
        raise RuntimeError('Недостаточно места: для начала записи нужно 8 ГБ' if starting else
            'Остался резерв 2 ГБ: запись завершается, чтобы сохранить уже полученные данные')
    return free


def sha256(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(1024*1024),b''):h.update(block)
    return h.hexdigest()


def external_destination():
    cfg=DATA/'configs/backup.json'
    if cfg.exists():
        value=json.loads(cfg.read_text()).get('external_path')
        if value:
            path=Path(value).expanduser().resolve()
            if not path.is_dir():raise ValueError('Подключите внешний диск по указанному пути')
            if path==DATA or path.is_relative_to(DATA) or DATA.is_relative_to(path):
                raise ValueError('Для внешней копии нужен отдельный каталог, не хранилище Explorer')
            return path
    for base in (Path('/media')/os.environ.get('USER','jetsonnx'),Path('/run/media')/os.environ.get('USER','jetsonnx')):
        if base.is_dir():
            for mount in sorted(base.iterdir()):
                candidate=mount/'explorer3-backup'
                if os.path.ismount(mount) and candidate.is_dir():return candidate.resolve()
    return None


def independent_storage(path):
    return Path(path).stat().st_dev!=DATA.stat().st_dev


def configure_external(value):
    value=value.strip()
    if value:
        path=Path(value).expanduser().resolve()
        if not path.is_dir():raise ValueError('Сначала подключите диск и создайте на нём папку explorer3-backup')
        if path==DATA or path.is_relative_to(DATA) or DATA.is_relative_to(path):raise ValueError('Выберите отдельный каталог внешней копии')
        value=str(path)
    atomic_write(DATA/'configs/backup.json',{'external_path':value or None,'updated_by':'user','updated_at':time.time()})


def verified_copy(source,target,expected,root):
    if not target.resolve().is_relative_to(root.resolve()):raise ValueError('Путь копии выходит из выбранного каталога')
    target.parent.mkdir(parents=True,exist_ok=True)
    if target.exists():
        if sha256(target)==expected:return False
        raise ValueError('Ранее сохранённая внешняя копия изменилась: '+str(target))
    temporary=target.with_name(target.name+'.copying')
    shutil.copyfile(source,temporary)
    if sha256(temporary)!=expected:raise IOError('Хеш скопированного файла не совпал')
    with temporary.open('rb') as handle:os.fsync(handle.fileno())
    os.replace(temporary,target)
    return True


def backup_now(*,force=False):
    from .resources import ResourceLease
    with ResourceLease('physical-activity'),ResourceLease('compute'),_lock:
        today=datetime.now(timezone.utc).strftime('%Y-%m-%d')
        root=DATA/'backups'/today;root.mkdir(parents=True,exist_ok=True)
        existing=sorted(root.glob('*/manifest.json'))
        if existing and not force:
            report=json.loads(existing[-1].read_text())
            atomic_write(DATA/'backups/latest.json',report)
            return report
        stamp=time.strftime('%H%M%S',time.gmtime())+'-'+str(time.time_ns()%1_000_000_000)
        target=root/stamp;target.mkdir()
        source=DATA/'catalog.sqlite3'
        if not source.exists():raise ValueError('Каталог записей пока не создан')
        with closing(sqlite3.connect(source)) as db,closing(sqlite3.connect(target/'catalog.sqlite3')) as out:
            db.backup(out)
        with closing(sqlite3.connect(target/'catalog.sqlite3')) as db:
            db.row_factory=sqlite3.Row
            episodes=[dict(r) for r in db.execute('SELECT id,dataset FROM episodes')]
        files={};records={}
        for entry in episodes:
            folder=Path(entry['dataset']).resolve()
            if not folder.is_relative_to(DATA/'datasets'):raise ValueError('Запись находится вне хранилища Explorer')
            records[entry['id']]=str(folder.relative_to(DATA))
            for path in sorted(folder.rglob('*')):
                if path.is_file():
                    if not path.resolve().is_relative_to(folder):raise ValueError('Файл записи выходит из своего каталога')
                    files[str(path.relative_to(DATA))]={'sha256':sha256(path),'bytes':path.stat().st_size}
        manifest={'version':1,'created_at':time.time(),'catalog_sha256':sha256(target/'catalog.sqlite3'),
                  'snapshot':str(target),'episodes':records,'files':files,'external':None,
                  'warning':'Внешний диск не подключён; пока есть только локальная копия каталога'}
        # Durable local result exists even if the external drive fills/disconnects.
        atomic_write(target/'manifest.json',manifest)
        try:
            external=external_destination()
            if external:
                copied=0
                for relative,record in files.items():
                    copied+=verified_copy(DATA/relative,external/relative,record['sha256'],external)
                ext_snapshot=external/'catalogues'/today/stamp
                verified_copy(target/'catalog.sqlite3',ext_snapshot/'catalog.sqlite3',manifest['catalog_sha256'],external)
                manifest.update(external=str(external),copied_files=copied,
                    independent_storage=independent_storage(external),
                    warning=None if independent_storage(external) else 'Копия находится на том же диске; подключите независимый внешний накопитель')
                atomic_write(ext_snapshot/'manifest.json',manifest)
        except (OSError,ValueError) as exc:manifest['warning']='Внешняя копия не завершена: '+str(exc)
        atomic_write(target/'manifest.json',manifest)
        atomic_write(DATA/'backups/latest.json',manifest)
        return manifest


def restore_catalog(backup,destination):
    """Restore only into a NEW file. Live catalog is never overwritten by this helper."""
    backup=Path(backup);destination=Path(destination)
    if destination.exists():raise ValueError('Каталог назначения уже существует; перезапись запрещена')
    manifest=json.loads((backup/'manifest.json').read_text())
    if sha256(backup/'catalog.sqlite3')!=manifest['catalog_sha256']:raise ValueError('Повреждена резервная копия каталога')
    destination.parent.mkdir(parents=True,exist_ok=True)
    with closing(sqlite3.connect(backup/'catalog.sqlite3')) as source,closing(sqlite3.connect(destination)) as target:
        source.backup(target)
    with closing(sqlite3.connect(destination)) as restored:
        if restored.execute('PRAGMA integrity_check').fetchone()[0]!='ok':raise ValueError('Восстановленный SQLite не прошёл проверку')
    return manifest


def status():
    try:
        report=json.loads((DATA/'backups/latest.json').read_text())
        age=time.time()-report['created_at']
        return {'available':age<86400,'age_s':age,'warning':report.get('warning'),'external':report.get('external')}
    except (OSError,ValueError,KeyError):return {'available':False,'warning':'Резервная копия ещё не создана','external':None}
