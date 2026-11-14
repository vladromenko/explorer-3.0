"""One pinned local inference worker. The only model tool proposes a closed plan."""
import hashlib
import importlib.metadata
import json
import resource
import socket
import sys
import time
from pathlib import Path
from typing import Literal
from pydantic import Field
from explorer3.paths import ROOT,DATA
from explorer3.resources import ResourceLease
from .schemas import Contract


class Proposal(Contract):
    choice:Literal['scan_then_dock','observe_then_dock','wait','human']
    reason:str=Field(min_length=1,max_length=600)


def verify_assets():
    lock=json.loads((ROOT/'configs/cognitive-assets.lock.json').read_text())
    root=DATA/'cache/cognitive/qwen3-vl-2b'
    for name,expected in lock['files'].items():
        sha=hashlib.sha256()
        with (root/name).open('rb') as source:
            for block in iter(lambda:source.read(4*1024*1024),b''):sha.update(block)
        if sha.hexdigest()!=expected:raise ValueError('Pinned model file changed: '+name)
    return root,lock


def propose(request,output):
    import torch
    from smolagents import Tool,ToolCallingAgent,TransformersModel
    config_path=ROOT/'configs/cognitive-model.json'
    if request.get('model_config_sha256') and hashlib.sha256(config_path.read_bytes()).hexdigest()!=request['model_config_sha256']:
        raise ValueError('Model configuration changed after experiment preparation')
    config=json.loads(config_path.read_text())
    for package,key in [('transformers','transformers_version'),('smolagents','smolagents_version')]:
        if importlib.metadata.version(package)!=config[key]:raise ValueError('Pinned cognitive runtime changed: '+package)
    accepted=[];attempts=[];network_attempts=[]
    allowed=request['allowed_choices']
    if not set(allowed)<=set(Proposal.model_fields['choice'].annotation.__args__):raise ValueError('Unknown allowed choice')
    def denied_connect(sock,address):
        if sock.family in (socket.AF_INET,socket.AF_INET6):
            network_attempts.append(str(address));raise PermissionError('Cognitive inference is offline')
        return original_connect(sock,address)
    original_connect=socket.socket.connect;socket.socket.connect=denied_connect
    class ProposePlan(Tool):
        name='final_answer'
        description='Propose one admitted simulation plan. Does not execute skills or grant permissions.'
        inputs={'choice':{'type':'string','description':'One of the allowed_choices in the task'},
                'reason':{'type':'string','description':'Brief reason based only on supplied observations and memory'}}
        output_type='object'
        def forward(self,choice,reason):
            value=Proposal(choice=choice,reason=reason)
            if value.choice not in allowed:raise ValueError('Choice is outside current admission')
            if accepted:raise ValueError('Only one proposal per decision')
            accepted.append(value.model_dump());return accepted[-1]
    class BoundedModel(TransformersModel):
        def generate(self,*args,**kwargs):
            if len(attempts)>=2:raise ValueError('One format repair exhausted')
            attempts.append({'started_at':time.time()})
            result=super().generate(*args,**kwargs)
            attempts[-1].update(text=result.content,tokens=result.token_usage.dict() if hasattr(result.token_usage,'dict') else str(result.token_usage))
            return result
        def _prepare_completion_args(self,*args,**kwargs):
            values=super()._prepare_completion_args(*args,**kwargs)
            if values['inputs'].shape[1]>config['max_input_tokens']:raise ValueError('Input token budget exceeded')
            # Text evidence only in this first path. Preserve a complete attention
            # mask instead of inferring padding from Qwen's EOS/PAD identity.
            values['attention_mask']=torch.ones_like(values['inputs'])
            values['max_time']=25.;return values
    started=time.monotonic()
    try:
        with ResourceLease('cognitive-model'):
            root,lock=verify_assets();verified=time.monotonic()
            model=BoundedModel(model_id=str(root),device_map='cuda',torch_dtype=torch.bfloat16,
                model_kwargs={'local_files_only':True,'attn_implementation':'sdpa'},max_new_tokens=config['max_new_tokens'],do_sample=False)
            loaded=time.monotonic()
            agent=ToolCallingAgent(tools=[ProposePlan()],model=model,max_steps=2,max_tool_threads=1,verbosity_level=0)
            task=('You propose a bounded simulation plan. Use final_answer exactly once with the structured choice and reason. Never claim success or invent observations. '
                  'Docking may only follow successful helper observation. The scene is fixed; arbitrary goals are unavailable. '
                  'A retrieved case is a hypothesis, not new evidence. No training or real robot access.\n'+json.dumps(request,ensure_ascii=False))
            agent.run(task)
            if not accepted:raise ValueError('Model did not provide a valid admitted tool proposal')
            result={'proposal':accepted[0],'backend':'smolagents.ToolCallingAgent','model':lock['repository'],
                'revision':lock['revision'],'steps':len(attempts),'attempts':attempts,'offline':True,
                'network_attempts':network_attempts,'weights_updated':False,'image_input':False,
                'input_source':'deterministic robot RGB measurements; no privileged outcome fields',
                'verify_s':verified-started,'load_s':loaded-verified,'inference_s':time.monotonic()-loaded,
                'total_s':time.monotonic()-started,'rss_peak_kb':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
                'cuda_peak_bytes':torch.cuda.max_memory_allocated()}
            Path(output).write_text(json.dumps(result,indent=2,ensure_ascii=False))
            return result
    finally:socket.socket.connect=original_connect


if __name__=='__main__':propose(json.loads(Path(sys.argv[1]).read_text()),sys.argv[2])
