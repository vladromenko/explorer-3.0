"""Small shared two-head estimator, explicit manual fitting, JSON-only inference."""
import json
from pathlib import Path
import numpy as np
from .research import matrix
from .gain_preparation import load,checksum


def predict(artifact,rows):
    x,names=matrix(rows,artifact['include_physical'])
    if names!=artifact['feature_names']:raise ValueError('Gain feature schema mismatch')
    probs=[]
    for head in artifact['heads']:
        mean=np.asarray(head['mean']);scale=np.asarray(head['scale']);coef=np.asarray(head['coef'])
        if mean.shape!=(x.shape[1],) or scale.shape!=mean.shape or coef.shape!=mean.shape:
            raise ValueError('Invalid gain model dimensions')
        if not np.isfinite(np.r_[mean,scale,coef,head['intercept']]).all() or (scale<=0).any():raise ValueError('Invalid gain weights')
        z=np.clip(((x-mean)/scale)@coef+head['intercept'],-500,500)
        probs.append(1/(1+np.exp(-z)))
    if len(probs)!=2:raise ValueError('Two gain heads required')
    return [{'p_without':float(a),'p_with':float(b),'gain':float(b-a)} for a,b in zip(*probs)]


def fit_plan(ident,output):
    """Called only by an explicitly launched training job; never by preparation."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    plan=load(ident);rows=[r for r in plan['rows'] if r['split']=='train']
    x,names=matrix(rows,plan['include_physical']);heads=[]
    weights=np.asarray([1/sum(other['scene_group']==row['scene_group'] for other in rows) for row in rows])
    for branch in ('continue','correction'):
        scaler=StandardScaler().fit(x,sample_weight=weights)
        model=LogisticRegression(max_iter=plan['max_iter'],random_state=plan['seed'])
        model.fit(scaler.transform(x),[r['outcomes'][branch] for r in rows],sample_weight=weights)
        if model.n_iter_.max()>=plan['max_iter']:raise ValueError('Gain fit did not converge within fixed budget')
        heads.append({'mean':scaler.mean_.tolist(),'scale':scaler.scale_.tolist(),
            'coef':model.coef_[0].tolist(),'intercept':float(model.intercept_[0])})
    artifact={'schema':'branch_gain_model_v1','plan_id':ident,'include_physical':plan['include_physical'],
        'feature_names':names,'heads':heads,'domain':'sim','supervision':'scripted_expert',
        'estimand':plan['estimand'],'protocol':plan['protocol'],'calibrated_for_humans':False,
        'training_scenes':sorted({r['scene_group'] for r in rows})}
    metrics={}
    for role in ('validation','test'):
        selected=[r for r in plan['rows'] if r['split']==role];pred=predict(artifact,selected)
        weights=[1/sum(other['scene_group']==r['scene_group'] for other in selected) for r in selected]
        metrics[role]={'scenes':len({r['scene_group'] for r in selected}),
            'brier_without':float(np.average([(p['p_without']-r['outcomes']['continue'])**2 for p,r in zip(pred,selected)],weights=weights)),
            'brier_with':float(np.average([(p['p_with']-r['outcomes']['correction'])**2 for p,r in zip(pred,selected)],weights=weights)),
            'gain_mse':float(np.average([(p['gain']-(r['outcomes']['correction']-r['outcomes']['continue']))**2 for p,r in zip(pred,selected)],weights=weights)),
            'not_policy_effect':True}
    output=Path(output);output.mkdir(parents=True,exist_ok=True)
    (output/'model.json').write_text(json.dumps(artifact,indent=2))
    result={'status':'completed','model':str(output/'model.json'),'model_sha256':checksum(artifact),
        'metrics':metrics,'plan_id':ident,'training_started':True,'physical_robot_access':False}
    (output/'result.json').write_text(json.dumps(result,indent=2));return result
