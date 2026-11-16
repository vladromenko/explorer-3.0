"""Cards for existing components, with explicit sim and real admission."""
import hashlib
import json
import math
from explorer3.paths import ROOT
from explorer3.profiles import profile,PROFILES
from .schemas import SkillCard
from .store import digest


def file_hash(relative):return hashlib.sha256((ROOT/relative).read_bytes()).hexdigest()


def real_cards():
    """Register existing operator paths honestly; no fabricated learned skills."""
    result={}
    sources={'so101':'explorer3/session.py','tank':'explorer3/session.py','rosmaster':'explorer3/rosmaster_session.py'}
    for name,cfg in PROFILES.items():
        robot=cfg['robot'];source=sources[robot];ident=name+'.manual.v1'
        version=digest({'profile':cfg,'executor':file_hash(source),'trials':file_hash('explorer3/trials.py')})
        result[ident]=SkillCard(skill_id=ident,robot=robot,version=version,kind='manual',source_ref=source,
            profile=name,camera_sha256=digest({'camera':cfg['camera'],'profile':cfg}),
            state_schema={'channels':cfg.get('state_channels',cfg['channels']),'provenance':cfg.get('state_provenance','profile contract')},
            action_schema={'channels':cfg['channels'],'units':cfg['units']},
            goals=['fixed_box' if 'pick_place' in name else 'fixed_dock'],cameras=[cfg['camera']],
            verified_scope=['existing isolated operator controller; no trained autonomous skill implied'],
            preconditions=['physical_permission','exclusive_owner','fresh_measurements','incident_cleared'],
            postconditions=['operator_outcome_and_original_recording'],resources=[robot],timeout_s=60,
            cancel_method='existing controller STOP and browser lease',verifier='operator-labelled recorded episode',
            sim_admission='blocked',real_admission='blocked',
            limitation='Manual data collection through existing panel only; agent execution requires a separately pinned trial and current physical authorization')
    return result


def cards():
    result={}
    model_sources=['explorer3/sim/factory.py','explorer3/sim/environment.py','explorer3/sim/models.py',
        'explorer3/sim/mobile.py','explorer3/sim/rosmaster_arm.py','explorer3/sim/fleet.py',
        'explorer3/sim/tank_model.py','configs/tank-hardware-evidence.json',
        'explorer3/sim/room.py','explorer3/sim/vendor_props/manifest.json',
        'explorer3/sim/layout.py','configs/physical-layout.json']
    geometry={name:file_hash(name) for name in model_sources}
    entries=[('tank','dock_fixed','tank_dock_sim','explorer3/sim/mobile.py','DockExpert'),
             ('rosmaster','dock_fixed','rosmaster_dock_sim','explorer3/sim/mobile.py','DockExpert'),
             ('so101','pick_place','so101_pick_place_sim','explorer3/sim/expert.py','ScriptedSO101'),
             ('rosmaster','pick_place','rosmaster_pick_place_sim','explorer3/sim/expert.py','ScriptedRosmaster')]
    for robot,name,profile_id,source,symbol in entries:
        cfg=profile(profile_id);identifier=f'{robot}.{name}.scripted.v1';sha=file_hash(source)
        camera=digest({'profile':profile_id,'camera':cfg['camera'],'contract':cfg,'geometry':geometry})
        card=SkillCard(skill_id=identifier,robot=robot,version=digest({'code':sha,'profile':cfg,'geometry':geometry}),kind='scripted',
            source_ref=source+':'+symbol,profile=profile_id,policy_sha256=sha,processor_sha256=None,
            calibration_sha256=None,camera_sha256=camera,
            state_schema={'channels':cfg.get('state_channels',cfg['channels']),'units':cfg.get('state_units',cfg['units'])},
            action_schema={'channels':cfg['channels'],'units':cfg['units']},
            goals=['fixed_dock' if name=='dock_fixed' else 'fixed_box'],cameras=[cfg['camera']],
            verified_scope=['existing nominal MuJoCo task','privileged scripted teacher; not a learned policy'],
            preconditions=['fresh_evidence','recipient_stationary','fixed_scene'],
            postconditions=['stable_in_dock_zone' if name=='dock_fixed' else 'released_object_inside_box'],
            resources=[robot,'simulation_world'],timeout_s=40.,cancel_method='existing simulator hold_agent / cancellation',
            verifier='existing environment.info/complete_step independent of planner text',sim_admission='available',
            real_admission='blocked',limitation='AG0–AG2 Homebrain only; scripted teacher uses simulator state; physical authorization absent')
        result[identifier]=card
    source='explorer3/sim/cooperation.py';sha=file_hash(source)
    base=result['rosmaster.pick_place.scripted.v1']
    for name,description,timeout in [('scan_dock','bounded fixed camera scan',5.),('observe_dock','direct fixed camera observation',3.5)]:
        ident=f'rosmaster.{name}.geometric.v1'
        result[ident]=base.model_copy(update={'skill_id':ident,'kind':'geometric',
            'source_ref':source+':CooperationWorld.observe','version':digest({'code':sha,'parent':base.version,'name':name}),
            'policy_sha256':sha,'goals':['fixed_dock_marker'],'postconditions':['target_visible_in_robot_rgb'],
            'timeout_s':timeout,'verified_scope':[description,'fixed shared simulation scene only'],
            'limitation':'Fixed camera poses from existing ArmModel IK; no arbitrary target or real actuation'})
    base=result['tank.dock_fixed.scripted.v1'];ident='tank.dock_observed.scripted.v1'
    result[ident]=base.model_copy(update={'skill_id':ident,
        'version':digest({'parent':base.version,'admission_source':sha}),
        'preconditions':[*base.preconditions,'target_observed_by_helper']})
    from explorer3 import catalog
    with catalog.connect() as db:
        exists = db.execute("SELECT 1 FROM sqlite_master WHERE type='table' AND name='cognitive_skills'").fetchone()
        rows = db.execute('SELECT body FROM cognitive_skills').fetchall() if exists else []
    accepted = {record['skill_id']: SkillCard.model_validate(record['card'])
                for row in rows for record in [json.loads(row['body'])]}
    return {**result, **real_cards(), **accepted}


def get(identifier):
    if identifier.startswith('rosmaster.acquired_pick.'):
        from explorer3 import catalog
        with catalog.connect() as db:
            row = db.execute('SELECT body FROM cognitive_skills WHERE id=?', (identifier,)).fetchone()
        if row is None:
            raise ValueError('Unknown acquired skill')
        return SkillCard.model_validate(json.loads(row['body'])['card'])
    if identifier.startswith('trial.'):
        from explorer3 import trials
        plan=trials.load(identifier.removeprefix('trial.'))
        base=real_cards().get(plan['profile']+'.manual.v1')
        if base is None:raise ValueError('Unknown physical trial profile')
        return base.model_copy(update={'skill_id':identifier,'version':digest({'trial':plan,'executor_version':base.version}),
            'kind':'learned' if plan['kind']=='policy' else 'scripted',
            'policy_sha256':plan.get('checkpoint_bundle_sha256'),
            'source_ref':'explorer3/trials.py:'+('PolicyActions' if plan['kind']=='policy' else 'ReplayActions'),
            'real_admission':'available','preconditions':['current_trial_authorization','fresh_executor_observation','exclusive_owner'],
            'limitation':'Prepared trial only; runtime rechecks permission, incident, weights and measured initial pose. Replay is not a learned skill.'})
    result=cards()
    if identifier not in result:raise ValueError('Unknown skill: '+identifier)
    return result[identifier]


def validate_graph(graph,domain,*,shadow=False):
    if domain!='sim' and not shadow:raise PermissionError('Real graph needs the separately authorized physical executor')
    result=[]
    for node in graph.nodes:
        card=get(node.skill_id)
        if node.skill_version!=card.version:raise ValueError('Skill version changed')
        if node.robot!=card.robot:raise ValueError('Skill belongs to another robot')
        if node.goal_id not in card.goals:raise ValueError('Skill does not accept this goal')
        if node.deadline_s>card.timeout_s:raise ValueError('Skill deadline exceeds its contract')
        admission=card.sim_admission if domain=='sim' else card.real_admission
        if admission!='available':raise ValueError('Skill is '+admission)
        result.append(card)
    return result


def acquired_pick(robot, object_class, domain, context):
    """Search accepted cards in the existing registry; never pick newest weights.

    Registration is a separate verification step. No card is created by this
    lookup, by a model proposal, or just because a training job completed.
    """
    from explorer3 import catalog
    with catalog.connect() as db:
        exists = db.execute("SELECT 1 FROM sqlite_master WHERE type='table' AND name='cognitive_skills'").fetchone()
        records = [json.loads(r['body']) for r in db.execute('SELECT body FROM cognitive_skills ORDER BY id')] if exists else []
    relevant = [r for r in records if r.get('robot') == robot and r.get('object_class') == object_class and r.get('domain') == domain]
    accepted = []
    rejected = []
    for record in relevant:
        problems = []
        for key in ('camera_sha256', 'calibration_sha256', 'surface'):
            if context.get(key) is None or context[key] != record.get('scope', {}).get(key):
                problems.append(key + '_unknown_or_changed')
        size = context.get('size_m')
        bounds = record.get('scope', {}).get('size_bounds_m')
        if not isinstance(size, list) or len(size) != 3 or not isinstance(bounds, list) or len(bounds) != 3:
            problems.append('object_dimensions_unknown')
        elif any(type(v) not in (int, float) or not math.isfinite(v) or not lo <= v <= hi for v, (lo, hi) in zip(size, bounds)):
            problems.append('object_dimensions_outside_scope')
        if record.get('verification', {}).get('autonomous_without_donor') is not True:
            problems.append('independent_recipient_execution_not_verified')
        if record.get('verification', {}).get('status') != 'accepted':
            problems.append('recipient_skill_not_accepted')
        try:
            from explorer3.training_drafts import checkpoint_hash
            if checkpoint_hash(record['checkpoint']) != record.get('checkpoint_sha256'):
                problems.append('checkpoint_changed')
        except (OSError, ValueError, KeyError):
            problems.append('checkpoint_unavailable')
        if problems:
            rejected.append({'skill_id': record['skill_id'], 'reasons': problems})
        else:
            accepted.append(record)
    selected = context.get('skill_id')
    matches = [r for r in accepted if selected is None or r['skill_id'] == selected]
    if len(matches) == 1:
        return {'state': 'available', **matches[0], 'rejected': rejected}
    if matches or relevant:
        return {'state': 'uncertain', 'rejected': rejected, 'reason': 'select one accepted compatible skill' if matches else 'no compatible verified scope'}
    return {'state': 'missing', 'rejected': []}


def register_acquired_pick(job_id, packet_shas, scope, evidence_ids, *, domain):
    """Verifier-only registration in this registry, not an LLM tool or motor grant.

    Requires three independent measured executions with donor assistance disabled.
    The checkpoint, source packets, camera, calibration and accepted domain are pinned.
    """
    from explorer3 import catalog, knowledge
    from explorer3.training_drafts import checkpoint_hash
    from . import store
    if domain not in ('real', 'sim') or not packet_shas:
        raise ValueError('Recipient domain and donor lineage required')
    for sha in packet_shas:
        knowledge.load_packet(sha)
    jobs = [j for j in catalog.rows('jobs') if j['id'] == job_id and j['status'] == 'completed']
    if len(jobs) != 1 or not jobs[0]['checkpoint'] or jobs[0]['policy'] not in ('act', 'smolvla'):
        raise ValueError('Completed recipient policy training required')
    job = jobs[0]
    config = json.loads(job['config'])
    if set(config.get('donor_packet_shas', [])) != set(packet_shas):
        raise ValueError('Training lineage differs from the donor packets')
    if profile(job['profile'])['robot'] != 'rosmaster':
        raise ValueError('Recipient policy must belong to ROSMASTER')
    sha = checkpoint_hash(job['checkpoint'])
    if domain == 'real':
        from explorer3 import deploy_gate
        gate = deploy_gate.latest(job_id)
        if not gate or gate.get('verdict') != 'ready' or gate.get('checkpoint_bundle_sha256') != sha:
            raise ValueError('Current deploy gate required before physical skill registration')
    required = ('object_class', 'camera_sha256', 'calibration_sha256', 'surface', 'size_bounds_m')
    if any(not scope.get(k) for k in required):
        raise ValueError('Complete accepted scope required')
    bounds = scope['size_bounds_m']
    if len(bounds) != 3 or any(len(b) != 2 or not all(type(v) in (int, float) and math.isfinite(v) for v in b) or not 0 < b[0] <= b[1] for b in bounds):
        raise ValueError('Invalid object size bounds')
    items = store.evidence(evidence_ids, domain=domain, fresh=False)
    scenes = set()
    for item in items:
        p = item.payload
        if item.source != 'fixed_outcome_verifier' or p.get('checkpoint_sha256') != sha or p.get('scope_sha256') != digest(scope):
            raise ValueError('Independent execution evidence must match weights and scope')
        if p.get('donor_online') is not False or p.get('geometric_correction') is not False or p.get('recipient') != 'success':
            raise ValueError('Autonomous recipient success without donor or fallback required')
        for key, minimum in (('lift_m', .05), ('held_s', 1.)):
            value = p.get(key)
            if type(value) not in (int, float) or not math.isfinite(value) or value < minimum:
                raise ValueError('Missing measured lift/retention')
        if not p.get('scene_id'):
            raise ValueError('Independent scene identity required')
        scenes.add(p['scene_id'])
    if len(scenes) < 3:
        raise ValueError('At least three independent recipient scenes required')
    identifier = 'rosmaster.acquired_pick.' + digest({'job': job_id, 'weights': sha, 'scope': scope, 'domain': domain})[:20]
    cfg = profile(job['profile'])
    value = {'skill_id': identifier, 'robot': 'rosmaster', 'object_class': scope['object_class'],
             'domain': domain, 'job_id': job_id, 'checkpoint': job['checkpoint'], 'checkpoint_sha256': sha,
             'donor_packet_shas': sorted(packet_shas), 'scope': scope,
             'verification': {'status': 'accepted', 'autonomous_without_donor': True,
                              'evidence_ids': sorted(evidence_ids), 'independent_scenes': len(scenes)}}
    value['card'] = SkillCard(skill_id=identifier, robot='rosmaster', version=digest(value), kind='learned',
        source_ref='explorer3.policy_runtime:PolicyCore', profile=job['profile'], policy_sha256=sha,
        camera_sha256=scope['camera_sha256'], calibration_sha256=scope['calibration_sha256'],
        state_schema={'channels': cfg.get('state_channels', cfg['channels'])},
        action_schema={'channels': cfg['channels'], 'units': cfg['units']}, goals=['named_object'], cameras=[cfg['camera']],
        verified_scope=[json.dumps(scope, sort_keys=True)], preconditions=['current_motion_permission', 'fresh_scene', 'compatible_scope'],
        postconditions=['measured_lift_and_retention'], resources=['rosmaster'], timeout_s=60.,
        cancel_method='existing controller STOP', verifier='fixed_outcome_verifier',
        sim_admission='available' if domain == 'sim' else 'blocked', real_admission='available' if domain == 'real' else 'blocked',
        limitation='Accepted only in recorded scope; registration is not permission to move or a guarantee of future success').model_dump(mode='json')
    store.initialize()
    with catalog.connect() as db:
        old = db.execute('SELECT body FROM cognitive_skills WHERE id=?', (identifier,)).fetchone()
        if old and json.loads(old['body']) != value:
            raise ValueError('Accepted skill is immutable; make a new version')
        db.execute('INSERT OR IGNORE INTO cognitive_skills VALUES(?,?)', (identifier, store.encoded(value)))
    return value
