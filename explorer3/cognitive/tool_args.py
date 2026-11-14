"""Closed schemas for model-visible tool arguments."""
from typing import Literal
from pydantic import Field
from .schemas import Contract,Identifier,TaskGraph,AssistanceCase

class Empty(Contract):pass
class Retrieve(Contract):
    task:Identifier
    effect:Identifier
    recipient_skill_id:Identifier
    goal_id:Identifier
    comparison:Literal['M0','M1','M2','M3']='M3'
class Node(Contract):node_id:Identifier
class Help(Contract):candidates:list[Literal['continue','observe','recovery','human']]=Field(min_length=1,max_length=3)
class Verify(Contract):evidence_ids:list[Identifier]=Field(min_length=1,max_length=12)
class Learning(Contract):
    profile:Identifier
    policy:Literal['act','smolvla','diffusion']
    episode_ids:list[Identifier]=Field(min_length=1,max_length=20000)
    steps:int=Field(ge=50,le=1_000_000)
    batch:int=Field(ge=1,le=32)
    seed:int=Field(ge=0,lt=2**31)
    device:Literal['cpu','cuda']
class Human(Contract):question:str=Field(min_length=1,max_length=600)
SCHEMAS={'get_snapshot':Empty,'list_skills':Empty,'retrieve_cases':Retrieve,'propose_graph':TaskGraph,
         'prepare_skill_job':Node,'propose_help':Help,'verify_postcondition':Verify,
         'add_memory_candidate':AssistanceCase,'prepare_learning_round':Learning,'request_human':Human}
