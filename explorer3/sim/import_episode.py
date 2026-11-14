"""Runs in the existing LeRobot environment; simulation domain is mandatory."""
import json
import sys
from pathlib import Path
import cv2
from explorer3.paths import DATA
from explorer3.recording import EpisodeWriter
from explorer3.validation import validate


def ingest(root,review=None):
    root=Path(root).resolve()
    if not root.is_relative_to(DATA/'staging/sim'):
        raise ValueError('Not an Explorer simulation spool')
    meta=json.loads((root/'manifest.json').read_text())
    if meta['domain']!='sim' or meta['frames']<1:
        raise ValueError('Empty or invalid simulation capture')
    if meta.get('segment_role')=='fleet_correction':
        from explorer3.fleet_corrections import validate as validate_fleet
        if review is None:raise ValueError('Fleet correction requires explicit review')
        validate_fleet(root,review);meta.update(training_eligible=True,review=review)
    if meta.get('segment_role')=='bounded_correction':
        from explorer3.branch_corrections import validate_review
        validate_review(root,review or {})
        meta.update(training_eligible=True,review=review)
    if meta.get('segment_role')=='noise_demonstration':
        from explorer3.noise_demonstrations import validate_review
        validate_review(root,review or {});meta.update(training_eligible=True,review=review)
    elif meta.get('training_label') is not None:raise ValueError('Unexpected expert target override')
    if (root/'imported.json').exists():
        return json.loads((root/'imported.json').read_text())
    writer=EpisodeWriter(meta['profile'],meta['session'],meta['scene'],
                         category='correction' if meta.get('segment_role') in {'bounded_correction','fleet_correction','noise_demonstration'} else 'demonstration',domain='sim',training_label=meta.get('training_label'))
    for line in (root/'samples.jsonl').read_text().splitlines():
        sample=json.loads(line)
        filename=(root/sample.pop('image_path')).resolve()
        if filename.parent!=root:
            raise ValueError('Invalid frame path')
        bgr=cv2.imread(str(filename))
        if bgr is None:
            raise ValueError('Missing simulator image')
        sample['images']={meta['camera']:{**sample.pop('camera'),'image':cv2.cvtColor(bgr,cv2.COLOR_BGR2RGB)}}
        writer.add(sample)
    ident=writer.finish(meta['outcome'],meta)
    result=validate(ident,domain='sim')
    (root/'imported.json').write_text(json.dumps(result,indent=2))
    return result


if __name__=='__main__':
    print(json.dumps(ingest(sys.argv[1])))
