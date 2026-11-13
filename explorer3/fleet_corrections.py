"""Review gate for shared-world expert corrections; numerical fixtures excluded."""
import json
from pathlib import Path
from .paths import DATA
from .branch_corrections import source_hash
from .vision_assets import digest


def validate(root,review=None):
    root=Path(root).resolve()
    if not root.is_relative_to(DATA/'staging/sim'):raise ValueError('Invalid fleet spool location')
    meta=json.loads((root/'manifest.json').read_text())
    if meta.get('segment_role')!='fleet_correction' or meta.get('domain')!='sim' or meta.get('command_source')!='scripted_expert':
        raise ValueError('Это не исправление общей симуляции')
    if meta.get('selection_origin')!='frozen_checkpoint':raise ValueError('Программный пример не включается в обучающий набор')
    if meta.get('outcome')!='success':raise ValueError('Задача не завершилась подтверждённым успехом')
    if meta['lineage']!=meta['session'] or digest(root/'entry-snapshot.json')!=meta['entry_snapshot_sha256']:
        raise ValueError('Происхождение или снимок исправления изменились')
    if review is not None:
        if review.get('decision')!='accept_fleet_sim_correction' or review.get('source_sha256')!=source_hash(root) or review.get('lineage')!=meta['lineage']:
            raise ValueError('Нет действительной проверки исправления')
    return meta
