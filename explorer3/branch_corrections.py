"""Review-bound admission of bounded sim expert segments; no automatic labels."""
import hashlib
import json
from pathlib import Path
from .paths import DATA


def source_hash(root):
    digest=hashlib.sha256()
    paths=[root/'manifest.json',root/'samples.jsonl',*sorted(root.glob('*.png'))]
    for path in paths:
        digest.update(path.name.encode());digest.update(path.read_bytes())
    return digest.hexdigest()


def candidate(report):
    if report.get('domain')!='sim' or not report.get('replay_verified'):
        raise ValueError('Нужен проверенный симуляционный отчёт')
    trials=[r for r in report['outcomes'] if r['branch']=='correction' and r['repeat']==0]
    if len(trials)!=1 or trials[0]['outcome']!='success':
        raise ValueError('Исправление не завершилось подтверждённым успехом')
    entries=report.get('correction_candidates',[])
    if len(entries)!=1 or entries[0]['frames']<1:raise ValueError('Нет отдельного сегмента исправления')
    root=Path(entries[0]['spool']).resolve()
    if not root.is_relative_to(DATA/'staging/sim'):raise ValueError('Недопустимый источник исправления')
    meta=json.loads((root/'manifest.json').read_text())
    if meta['domain']!='sim' or meta['command_source']!='scripted_expert' or meta['segment_role']!='bounded_correction':
        raise ValueError('Источник не является симуляционным экспертным исправлением')
    if meta['lineage']!=report['lineage'] or meta['session']!=report['lineage'] or meta['snapshot_sha256']!=report['snapshot_sha256']:
        raise ValueError('Происхождение исправления не совпадает с отчётом')
    return root,meta


def validate_review(root,review):
    if review.get('decision')!='accept_sim_scripted_correction' or review.get('source_sha256')!=source_hash(root):
        raise ValueError('Нет проверки исправления или источник изменился после проверки')
    meta=json.loads((root/'manifest.json').read_text())
    if review['lineage']!=meta['lineage'] or review['snapshot_sha256']!=meta['snapshot_sha256']:
        raise ValueError('Проверка относится к другой ситуации')
    if meta['domain']!='sim' or meta['outcome']!='success' or meta['command_source']!='scripted_expert':
        raise ValueError('Нельзя принять этот источник в симуляционные исправления')
