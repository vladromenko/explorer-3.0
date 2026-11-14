"""Cards for existing components, with explicit sim and real admission."""
import hashlib
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
    return {**result,**real_cards()}


def get(identifier):
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
