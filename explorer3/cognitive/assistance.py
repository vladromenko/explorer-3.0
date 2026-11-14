"""Bounded assistance decisions reuse measured-history features and G ranking."""
import math
from explorer3 import research

SOURCES={'execution_monitor','policy_support_monitor','recorded_execution_history'}


def history_signals(rows,decision_ns,scales,*,delay_s=0.,progress=None,gain_artifact=None,domain='real'):
    features=research.features(rows,decision_ns,scales,delay_s,progress)
    result={'features':features,'gain':None,'gain_status':'unavailable','risk':None,
        'physical_features_measured':features['response_change'] is not None,
        'decision_timestamp_ns':decision_ns,'domain':domain}
    if gain_artifact is not None:
        from explorer3.gain_model import predict
        if gain_artifact.get('domain')!=domain:raise ValueError('Gain model domain mismatch')
        if domain=='real' and not gain_artifact.get('calibrated_for_humans'):
            raise ValueError('Scripted simulation gain is not a human intervention estimate')
        forecast=predict(gain_artifact,[{'features':features}])[0]
        result.update(**forecast,gain_status='available',risk=1-forecast['p_without'])
    return result


def select(options,observations):
    if not 1<=len(options)<=3 or len(set(options))!=len(options):raise ValueError('One to three distinct candidates required')
    if not set(options)<={'continue','observe','recovery','human'}:raise ValueError('Unknown help type')
    signals=[];conditions={};ids=[]
    for item in observations:
        ids.append(item.id)
        if item.status=='observed':
            for key,value in item.payload.get('conditions',{}).items():conditions.setdefault(key,[]).append(value)
            if item.source in SOURCES:signals.append(item.payload)
    verified={k:all(v is True for v in values) for k,values in conditions.items()}
    candidates=[];rejected=[]
    requirements={'continue':'recipient_can_continue','observe':'observation_skill_available',
                  'recovery':'recovery_compatible','human':None}
    for option in options:
        requirement=requirements[option]
        if requirement is not None and not verified.get(requirement,False):
            rejected.append({'candidate':option,'reason':'Not observed: '+requirement})
        else:candidates.append(option)
    # Never merge contradictory numerical histories by picking the latest one.
    signal=signals[0] if len(signals)==1 else {}
    gain=signal.get('gain') if signal.get('gain_status')=='available' else None
    risk=signal.get('risk');priority=None
    if gain is not None and (not isinstance(gain,(int,float)) or not math.isfinite(gain) or not -1<=gain<=1):
        raise ValueError('Invalid measured-model gain signal')
    if risk is not None and (not isinstance(risk,(int,float)) or not math.isfinite(risk) or not 0<=risk<=1):
        raise ValueError('Invalid risk signal')
    cost={k:signal.get('operator_cost',{}).get(k) for k in ('switch_seconds','correction_seconds')}
    if gain is not None and all(isinstance(cost.get(k),(int,float)) and not isinstance(cost[k],bool) and math.isfinite(cost[k]) and cost[k]>=0
                                for k in ('switch_seconds','correction_seconds')) and sum(cost.values())>0:
        ranked=research.prioritize([{'gain':gain,'request_time_ns':signal.get('decision_timestamp_ns',0),**cost}],method='gain')
        priority=ranked[0]['priority'] if ranked else 0.
    needs_help=signal.get('support_alarm') is True or (risk is not None and risk>=.5) or (gain is not None and gain>0)
    order=('observe','recovery','human','continue') if needs_help or not verified.get('recipient_can_continue') else ('continue','observe','human','recovery')
    selection=next((v for v in order if v in candidates),'wait')
    return {'candidates':candidates,'rejected':rejected,'selection':selection,'evidence_ids':ids,
        'baseline':'measured_conditions_v2','gain':gain,'gain_status':'available' if gain is not None else 'unavailable',
        'gain_per_operator_second':priority,'risk':risk,'history_signals':len(signals),
        'execution_started':False,'reason':'Verified continuation conditions and bounded monitor signals; no LLM confidence score'}
