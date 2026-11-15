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


def add(prompt, source, params, evidence=None, success=None, domain='real'):
    """params: metric, robot-independent, e.g. {'width_m': .04, 'height_m': .04, 'grasp_height_m': .02}."""
    item = {'id': uuid.uuid4().hex[:12], 'at': time.time(), 'prompt': prompt, 'source': source, 'domain': domain,
            'params': params, 'evidence': evidence or {}, 'success': success, 'uses': []}
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


def assess(robot, prompt, domain='real'):
    """What the robot does not know before acting on `prompt`. Returns
    {'gaps': [...], 'where': item|None, 'how': knowledge|None, 'skill': str, 'reasons': {...}}."""
    from . import world_memory
    out = {'gaps': [], 'reasons': {}}
    item = world_memory.target_for(None, prompt, WHERE_MAX_AGE_S)
    if item is None:
        out['gaps'].append('where')
        out['reasons']['where'] = f'нет свежего места «{prompt}» на общей карте (моложе {int(WHERE_MAX_AGE_S / 60)} мин)'
    elif (item.get('uncertainty_m') or 0) > WHERE_MAX_UNCERTAINTY_M:
        out['gaps'].append('where')
        out['reasons']['where'] = f'место «{prompt}» известно с погрешностью {item["uncertainty_m"]:.2f} м'
    out['where'] = item
    k = best(prompt, domain)
    out['how'] = k
    if robot == 'rosmaster' and k is None:
        out['gaps'].append('how')
        out['reasons']['how'] = f'нет проверенного знания, как брать «{prompt}» (размер, высота захвата)'
    if robot == 'rosmaster':
        from .rosmaster_arm import learned_pick_job
        job = learned_pick_job()
        out['skill'] = f'обученная модель {job[:8]}' if job else 'геометрический захват'
    else:
        out['skill'] = 'нет руки'
    return out


def helpers_for(gap, available):
    """Robots (and the person) able to fill `gap`, among the available ones; the person last."""
    robots = [r for r in available if gap in HELPERS.get(r, [])]
    return robots + ['human']
