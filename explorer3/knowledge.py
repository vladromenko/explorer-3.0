"""Knowledge gaps and who can fill them — the root of the research.

A robot about to act on an object first asks itself what it does not know:
  WHERE  — is there a fresh, confident position of the object on the shared map?
  HOW    — is there grasp knowledge for this object that fits ITS gripper (width, grasp height,
           approach), verified by a successful grasp, by itself or by another robot?
  SKILL  — does it have an admitted way to execute (learned policy that passed the gate, or the
           geometric grasp)?
Each gap has helpers that can produce that kind of knowledge (HELPERS): Tank and the other robot
look for objects (WHERE); the SO-101 table station practises grasping the object and measures it
(HOW); the person can provide anything. The AssistanceBroker picks among them; the help memory and
the outcome verifier say whether the help really helped the recipient.

Object grasp knowledge is stored per object name and is robot-independent (metric object size,
where to close the fingers relative to the object); each robot converts it to its own gripper
(rosmaster_arm.grip_command / grasp_target). Every entry keeps its source, evidence and outcomes.
"""
import json
import math
import time
import uuid

from .paths import DATA

STORE = DATA / 'world' / 'object_knowledge.json'      # swapped for a simulation copy by swarm_sim
WHERE_MAX_AGE_S = 900.          # an older sighting is a hint, not knowledge
WHERE_MAX_UNCERTAINTY_M = .35

# who can fill which gap (robot -> kinds of knowledge it can produce)
HELPERS = {'tank': ['where'], 'rosmaster': ['where'], 'so101': ['how'], 'human': ['where', 'how', 'skill']}


def _load():
    try:
        return json.loads(STORE.read_text())
    except (OSError, ValueError):
        return []


def _save(items):
    STORE.parent.mkdir(parents=True, exist_ok=True)
    tmp = STORE.with_suffix('.tmp')
    tmp.write_text(json.dumps(items[-2000:], ensure_ascii=False, indent=1))
    tmp.replace(STORE)


def add(prompt, source, params, evidence=None, success=None, domain='real', *, object_instance=None, applicability=None):
    """params: metric, robot-independent, e.g. {'width_m': .04, 'height_m': .04, 'grasp_height_m': .02}."""
    item = {'id': uuid.uuid4().hex[:12], 'at': time.time(), 'prompt': prompt, 'source': source, 'domain': domain,
            'params': params, 'evidence': evidence or {}, 'success': success, 'uses': [],
            'object_instance': object_instance, 'object_class': prompt, 'applicability': applicability or {},
            'verification_status': 'measured' if success is None else 'legacy_outcome_requires_evidence'}
    items = _load()
    items.append(item)
    _save(items)
    return item


def record_use(knowledge_id, robot, success, note=''):
    """Outcome of a robot using this knowledge (the recipient's result, measured by its own check)."""
    items = _load()
    for it in items:
        if it['id'] == knowledge_id:
            it['uses'].append({'at': time.time(), 'robot': robot, 'success': bool(success), 'note': note[:200]})
            _save(items)
            return it
    return None


def best(prompt, domain='real'):
    """Most useful grasp knowledge for the object: verified (success) first, newest, not refuted by uses."""
    cands = [it for it in _load() if it['prompt'] == prompt and it['domain'] == domain and it['success'] is not False]
    def score(it):
        uses = it['uses']
        good = sum(u['success'] for u in uses)
        bad = len(uses) - good
        return (bad <= good, bool(it['success']), good, it['at'])
    cands.sort(key=score, reverse=True)
    if cands and score(cands[0])[0]:
        return cands[0]
    return None


def assess(robot, prompt, domain='real', *, context=None, executor=None):
    """Explicit readiness facts; proposals from a language model cannot remove them.

    Legacy where/how keys remain for the K experiments. A missing motor skill is
    separate from missing object measurements. A Python function is not evidence.
    """
    from . import world_memory
    from .cognitive.registry import acquired_pick
    context = context or {}
    out = {'gaps': [], 'reasons': {}, 'readiness': {}, 'executor': executor}
    item = world_memory.target_for(None, prompt, WHERE_MAX_AGE_S)
    if item is not None and item.get('at', 0) < context.get('resume_after', 0):
        item = None
    uncertainty = item.get('uncertainty_m') if item else None
    unknown_location = item is None or type(uncertainty) not in (int, float) or not math.isfinite(uncertainty)
    poor_location = not unknown_location and uncertainty > WHERE_MAX_UNCERTAINTY_M
    if unknown_location or poor_location:
        out['gaps'].append('where')
        out['reasons']['where'] = 'не знает где: нет свежего положения с известной допустимой погрешностью'
    out['where'] = item
    out['readiness']['where_missing'] = unknown_location or poor_location
    measured = best(prompt, domain)
    if measured is not None and context.get('object_instance') and measured.get('object_instance') != context['object_instance']:
        measured = None
    out['how'] = measured
    properties = measured.get('params', {}) if measured else {}
    properties_missing = any(type(properties.get(key)) not in (int, float) or not math.isfinite(properties[key]) or properties[key] <= 0
                             for key in ('width_m', 'height_m'))
    if robot == 'rosmaster' and properties_missing:
        out['gaps'].append('how')
        out['reasons']['how'] = f'не хватает измерений подходящего экземпляра «{prompt}»'
    out['readiness']['object_properties_missing'] = properties_missing
    out['property_uncertainty_m'] = (measured or {}).get('evidence', {}).get('size_sigma_m')
    out['readiness']['object_identity_unknown'] = not context.get('object_instance')
    # Only a proved constraint can establish infeasibility. Unknown mechanics
    # produces uncertainty, not a fabricated impossibility or success.
    feasibility = context.get('feasibility') or {}
    infeasible = feasibility.get('reachable') is False and bool(feasibility.get('evidence_ref'))
    out['readiness']['physically_infeasible'] = infeasible
    if infeasible:
        out['reasons']['physically_infeasible'] = feasibility.get('reason', 'задача не допускается механикой или сценой')
    candidate = acquired_pick(robot, prompt, domain, context)
    out['registered_skill'] = candidate
    available = candidate.get('state') == 'available'
    uncertain = candidate.get('state') == 'uncertain'
    out['readiness'].update(skill_missing=not available and not uncertain, skill_uncertain=uncertain)
    out['baseline'] = {'name': 'geometric_baseline', 'selected': executor == 'geometric_baseline',
                       'learned_skill': False, 'physical_success_verified': False}
    if available:
        out['skill'] = candidate['skill_id']
    elif executor == 'geometric_baseline':
        out['skill'] = 'явно выбран геометрический baseline; приобретение навыка не подтверждено'
    else:
        out['skill'] = 'навык вне проверенных условий' if uncertain else 'нет проверенного захвата'
        if robot == 'rosmaster':
            out['gaps'].append('skill')
            out['reasons']['skill'] = out['skill']
    return out


def helpers_for(gap, available):
    """Robots (and the person) able to fill `gap`, among the available ones; the person last."""
    robots = [r for r in available if gap in HELPERS.get(r, [])]
    return robots + ['human']


def add_packet(packet):
    """Accept a traced artifact; its donor success never grants recipient admission."""
    import hashlib
    from pathlib import Path
    from . import catalog
    from .datasets import fingerprint
    from .skill_packets import admission
    from .cognitive.store import digest
    decision = admission(packet)
    if not decision['eligible']:
        raise ValueError('Incomplete donor packet: ' + ', '.join(decision['reasons']))
    rows = {e['id']: e for e in catalog.rows('episodes')}
    for source in packet['donor']['episodes']:
        entry = rows.get(source['id'])
        if entry is None or entry['profile'] not in ('so101_pick_place', 'so101_pick_place_sim', 'so101_handover_sim'):
            raise ValueError('Donor source is not a catalogued SO-101 episode')
        if entry['outcome'] != 'success' or fingerprint(entry['dataset']) != source['sha256']:
            raise ValueError('Donor episode is unsuccessful or has changed')
        provenance = json.loads((Path(entry['dataset']) / 'provenance.json').read_text())
        if provenance.get('domain', 'real') != packet['donor']['domain']:
            raise ValueError('Donor source domain mismatch')
    verification = packet['donor_verification']
    evidence = (DATA / verification['evidence_ref']).resolve()
    if not evidence.is_relative_to(DATA.resolve()) or not evidence.is_file():
        raise ValueError('Donor verification artifact missing')
    if hashlib.sha256(evidence.read_bytes()).hexdigest() != verification['evidence_sha256']:
        raise ValueError('Donor verification artifact changed')
    sha = digest(packet)
    path = DATA / 'world' / 'skill_packets' / (sha + '.json')
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and json.loads(path.read_text()) != packet:
        raise ValueError('Skill packet identity collision')
    if not path.exists():
        path.write_text(json.dumps(packet, ensure_ascii=False, indent=2, allow_nan=False))
    matches = [it for it in _load() if it.get('skill_packet_sha256') == sha]
    if matches:
        return matches[0]
    obj = packet['object']
    item = {'id': 'packet-' + sha[:20], 'at': time.time(), 'prompt': obj['class'],
            'source': 'so101_skill_packet', 'domain': packet['donor']['domain'],
            'object_instance': obj['instance_id'], 'object_class': obj['class'],
            'params': dict(zip(('width_m', 'length_m', 'height_m'), obj['size_m'])),
            'evidence': verification, 'success': True, 'uses': [],
            'skill_packet_sha256': sha, 'skill_packet_ref': str(path.relative_to(DATA)),
            'verification_status': 'donor_grasp_verified', 'recipient_verified': False,
            'recipient_policy_trained': False, 'applicability': packet.get('applicability', {})}
    items = _load()
    items.append(item)
    _save(items)
    return item


def load_packet(sha):
    from .cognitive.store import digest
    from .skill_packets import validate
    if not isinstance(sha, str) or len(sha) != 64 or any(c not in '0123456789abcdef' for c in sha):
        raise ValueError('Invalid packet identifier')
    packet = json.loads((DATA / 'world' / 'skill_packets' / (sha + '.json')).read_text())
    if digest(packet) != sha:
        raise ValueError('Skill packet changed')
    return validate(packet)
