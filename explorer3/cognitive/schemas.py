"""Small data contracts. Observations and model text never grant permissions."""
from typing import Annotated,Literal
from pydantic import BaseModel,ConfigDict,Field,model_validator

Identifier=Annotated[str,Field(pattern=r'^[A-Za-z0-9_.:-]{1,128}$')]
Digest=Annotated[str,Field(pattern=r'^[0-9a-f]{64}$')]
Robot=Literal['so101','rosmaster','tank']
Domain=Literal['sim','real']
Role=Literal['MissionCoordinator','SceneObserver','AssistanceBroker','MemoryCurator','OutcomeVerifier','LearningAndExperimentCurator']
Outcome=Literal['success','failure','unknown','cancelled','hardware_error']


class Contract(BaseModel):
    model_config=ConfigDict(extra='forbid',allow_inf_nan=False,frozen=True)


class ObservationEvidence(Contract):
    id:Identifier
    domain:Domain
    robot:Robot
    source:str=Field(min_length=1,max_length=160)
    sample_id:Identifier
    frame_id:str=Field(min_length=1,max_length=160)
    camera_id:str|None=None
    source_timestamp_ns:int|None=Field(default=None,ge=0)
    received_timestamp_ns:int=Field(ge=0)
    clock_domain:str
    boot_id:str|None=None
    created_at:float
    expires_at:float
    units:dict[str,str]=Field(default_factory=dict)
    status:Literal['unknown','hypothesis','observed','estimated']
    quality:dict[str,str|bool|float|None]=Field(default_factory=dict)
    artifact_ref:str|None=None
    artifact_sha256:Digest|None=None
    content_sha256:Digest
    payload:dict

    @model_validator(mode='after')
    def valid_interval(self):
        if self.expires_at<=self.created_at:raise ValueError('Evidence must expire after creation')
        if self.artifact_ref and (self.artifact_ref.startswith('/') or '..' in self.artifact_ref.split('/') or '://' in self.artifact_ref):
            raise ValueError('Evidence references must stay inside the existing data directory')
        return self


class WorldFact(Contract):
    id:Identifier
    subject:Identifier
    relation:Identifier
    value:dict
    domain:Domain
    frame_id:str
    evidence_ids:list[Identifier]=Field(min_length=1,max_length=12)
    valid_since:float
    expires_at:float
    status:Literal['unknown','hypothesis','observed','conflicted']
    contradicts:list[Identifier]=Field(default_factory=list)


class SkillCard(Contract):
    skill_id:Identifier
    robot:Robot
    version:Digest
    kind:Literal['learned','scripted','geometric','manual','observation']
    source_ref:str
    profile:str
    policy_sha256:Digest|None=None
    processor_sha256:Digest|None=None
    calibration_sha256:Digest|None=None
    camera_sha256:Digest
    state_schema:dict
    action_schema:dict
    goals:list[Identifier]=Field(min_length=1)
    cameras:list[str]
    verified_scope:list[str]
    preconditions:list[str]
    postconditions:list[str]
    resources:list[Identifier]=Field(min_length=1)
    timeout_s:float=Field(gt=0,le=120)
    cancel_method:str
    verifier:str
    sim_admission:Literal['available','blocked','not_implemented']
    real_admission:Literal['available','blocked','not_implemented']
    limitation:str


class TaskNode(Contract):
    id:Identifier
    skill_id:Identifier
    skill_version:Digest
    robot:Robot
    goal_id:Identifier
    depends_on:list[Identifier]=Field(default_factory=list,max_length=8)
    deadline_s:float=Field(gt=0,le=120)


class TaskGraph(Contract):
    nodes:list[TaskNode]=Field(min_length=1,max_length=8)
    evidence_ids:list[Identifier]=Field(min_length=1,max_length=12)
    rationale:str=Field(min_length=1,max_length=600)

    @model_validator(mode='after')
    def acyclic(self):
        ids=[n.id for n in self.nodes]
        if len(ids)!=len(set(ids)):raise ValueError('Duplicate graph node')
        resolved=set()
        for _ in self.nodes:
            resolved.update(n.id for n in self.nodes if set(n.depends_on)<=resolved)
        if resolved!=set(ids):raise ValueError('Unknown dependency or graph cycle')
        return self


class CompatibilityRecord(Contract):
    recipient_skill_id:Identifier
    recipient_version:Digest
    policy_sha256:Digest|None
    camera_sha256:Digest
    calibration_sha256:Digest|None
    domain:Domain
    goal_id:Identifier
    checked_conditions:list[str]
    unknown_conditions:list[str]
    evidence_ids:list[Identifier]


class AssistanceCase(Contract):
    id:Identifier
    task:Identifier
    effect:Identifier
    domain:Domain
    donor:Robot
    recipient:Robot
    donor_episode:Identifier
    recipient_episode:Identifier
    reason:str=Field(max_length=600)
    before_evidence_ids:list[Identifier]=Field(min_length=1)
    compatibility:CompatibilityRecord
    assistance_type:Literal['observation','recovery','human','wait']
    expected_effect:str=Field(max_length=600)
    actions_provenance:Literal['human','policy','scripted','geometric','observation','safety']
    actual_effect:dict
    delay_s:float|None=Field(default=None,ge=0)
    operator_active_s:float|None=Field(default=None,ge=0)
    helper_outcome:Outcome='unknown'
    recipient_outcome:Outcome='unknown'
    task_outcome:Outcome='unknown'
    status:Literal['candidate','supported_in_scope','contradicted','stale']='candidate'


class ExperimentManifest(Contract):
    id:Identifier
    comparison:Literal['M0','M1','M2','M3']
    memory_sha256:Digest
    memory_ids:list[Identifier]
    skill_versions:dict[Identifier,Digest]
    scenes:list[Identifier]
    seeds:list[int]
    budgets:dict[str,float]
    split:Literal['train','validation','test','diagnostic']
    memory_updates:bool
    training_authorized:bool=False
    evaluator_version:Digest


class ToolCall(Contract):
    request_id:Identifier
    mission_id:Identifier
    generation:int=Field(ge=0)
    decision_id:Identifier
    role:Role
    tool:Identifier
    mode:Domain
    snapshot_ids:list[Identifier]=Field(max_length=12)
    expires_at:float
    arguments:dict=Field(default_factory=dict)
