"""Independent verification: never imports the production aggregation functions."""
import argparse, json, hashlib
from pathlib import Path
import numpy as np
import pandas as pd

p=argparse.ArgumentParser(); p.add_argument('--output-dir',type=Path,required=True); p.add_argument('--replicate-dir',type=Path); a=p.parse_args(); root=a.output_dir
def read(name): return pd.read_csv(root/name,low_memory=False)
checks=[]
def close(x,y,label):
    np.testing.assert_allclose(x,y,rtol=1e-10,atol=1e-7,equal_nan=True); checks.append(label)
obs=read('inputs/observations.csv'); pred=read('inputs/oof_predictions.csv')
d=pred.merge(obs,on=['obs_id','outer_fold'],validate='many_to_one'); d['ae']=abs(d.predicted_flow-d.y_reference); d['se']=(d.predicted_flow-d.y_reference)**2
assert len(d)==57600 and set(obs.outer_fold)==set(range(5))
assert (obs.groupby('unordered_pair_id').distance_km.nunique()==1).all()
for q,g in obs.groupby('unordered_pair_id'):
    assert len(g)==48 and g.outer_fold.nunique()==1 and len(g[['origin_code','destination_code']].drop_duplicates())==2
    assert all('|'.join(sorted([str(i),str(j)]))==q for i,j in zip(g.origin_code,g.destination_code))
e=read('results/observation_errors.csv.gz').set_index(['model_id','obs_id']).sort_index(); raw=d.set_index(['model_id','obs_id']).sort_index()
close(e.AE,raw.ae,'observation AE'); close(e.SE,raw.se,'observation SE')
def scores(g):
    y=g.y_reference.to_numpy(); z=g.predicted_flow.to_numpy(); err=z-y
    return [np.mean(abs(err)),np.sqrt(np.dot(err,err)/len(err)),np.mean(2*abs(err)/(y+z+1e-9)),2*np.minimum(y,z).sum()/(y.sum()+z.sum()+1e-9),np.mean((g.predicted_log_flow-np.log1p(y))**2),100*err.sum()/y.sum()]
keys=['mae','rmse','smape','cpc','log_mse','total_bias_pct']
for r in read('results/metrics_reproduced.csv').itertuples():
    g=d[d.model_id.eq(r.model_id)]; g=g if r.scope=='pooled' else g[g.outer_fold.eq(int(r.scope))]
    close(scores(g),[getattr(r,k) for k in keys],'metrics '+r.model_id+str(r.scope))
for file in ['dyad_errors','directed_od_errors']:
    t=read('results/'+file+'.csv').groupby('model_id')[['SAE','SSE']].sum().sort_index()
    close(t,d.groupby('model_id')[['ae','se']].sum().sort_index(),file+' conservation')
c=read('results/observation_comparisons.csv.gz'); dc=read('results/dyad_comparisons.csv')
for (ca,re),g in c.groupby(['candidate','comparator']):
    aa=raw.loc[ca].loc[g.obs_id]; bb=raw.loc[re].loc[g.obs_id]
    for k,field in [('AE','ae'),('SE','se')]:
        close(g['improvement_'+k],bb[field].to_numpy()-aa[field].to_numpy(),'comparison sign '+ca+re+k)
        t=dc[dc.candidate.eq(ca)&dc.comparator.eq(re)].set_index('unordered_pair_id').sort_index()
        close(t['net_'+k+'_improvement'],g.groupby('unordered_pair_id')['improvement_'+k].sum().sort_index(),'dyad net '+k)
        close(t['net_'+k+'_gain'].sum()-t['net_'+k+'_loss'].sum(),g['improvement_'+k].sum(),'gain loss '+k)
        close(t['observation_'+k+'_positive_sum']+t['observation_'+k+'_negative_sum'],t['net_'+k+'_improvement'],'within dyad cancellation '+k)
for r in read('results/group_fold_metrics.csv').itertuples():
    g=d[d.model_id.eq(r.model_id)&d[r.group_type].eq(r.group_id)]; g=g if r.scope=='pooled' else g[g.outer_fold.eq(int(r.scope))]
    assert len(g)==r.n_obs
    close(scores(g) if len(g) else [np.nan]*6,[getattr(r,k) for k in keys],'group metrics')
for (ca,re,scope,kind),g in read('results/group_error_contributions.csv').groupby(['candidate','comparator','scope','group_type']):
    x=c[c.candidate.eq(ca)&c.comparator.eq(re)]; x=x if scope=='pooled' else x[x.outer_fold.eq(int(scope))]
    close([g.C_AE.sum(),g.C_SE.sum()],[x.improvement_AE.mean(),x.improvement_SE.mean()],'contribution sums')
cells=read('results/size_distance_cells.csv')
for (m,kind),unused in [( (m,k),None) for m in d.model_id.unique() for k in ['flow_decile','distance_quintile']]:
    for group,g in cells[cells.model_id.eq(m)].groupby(kind):
        x=d[d.model_id.eq(m)&d[kind].eq(group)]
        close([g.n_obs.sum(),g.reference_sum.sum(),g.SAE.sum(),g.SSE.sum()],[len(x),x.y_reference.sum(),x.ae.sum(),x.se.sum()],'cross cell margins')
for r in read('results/size_distance_comparisons.csv').itertuples():
    x=c[c.candidate.eq(r.candidate)&c.comparator.eq(r.comparator)&c.flow_decile.eq(r.flow_decile)&c.distance_quintile.eq(r.distance_quintile)]
    close([r.C_AE,r.C_SE],[x.improvement_AE.sum()/14400,x.improvement_SE.sum()/14400],'cross contributions')
ex=read('results/exclusion_sensitivity.csv')
for (ca,re,n),g in ex.groupby(['candidate','comparator','requested_n']):
    assert g.excluded_pair_ids.fillna('').nunique()==1 and g.n_obs.nunique()==1
    ids=str(g.excluded_pair_ids.iloc[0]).split(';') if n else []
    assert int(g.n_obs.iloc[0])==14400-48*len(ids)
    for r in g.itertuples():
        x=d[d.model_id.eq(r.model_id)&~d.unordered_pair_id.isin(ids)]
        close(scores(x),[getattr(r,k) for k in keys],'common exclusions')
curve=read('figures/diagnostic_01_source.csv')
for (ca,re,k),g in curve.groupby(['candidate','comparator','metric']):
    x=dc[dc.candidate.eq(ca)&dc.comparator.eq(re)]; x=x[x['net_'+k+'_improvement'] < -1e-8].sort_values(['net_'+k+'_loss','unordered_pair_id'],ascending=[False,True])
    assert list(g.unordered_pair_id)==list(x.unordered_pair_id)
    close(g.cumulative_share,x['net_'+k+'_loss'].cumsum()/x['net_'+k+'_loss'].sum(),'curve source')
heat=read('figures/diagnostic_02_source.csv'); gc=read('results/group_fold_comparisons.csv')
close(heat.improvement_mae,gc.improvement_mae,'heatmap source')
gm=read('results/group_fold_metrics.csv').set_index(['model_id','scope','group_type','group_id'])
for r in gc.itertuples():
    ca=gm.loc[(r.candidate,r.scope,r.group_type,r.group_id)]; re=gm.loc[(r.comparator,r.scope,r.group_type,r.group_id)]
    close([getattr(r,'improvement_'+k) for k in keys],[(ca[k]-re[k]) if k=='cpc' else (re[k]-ca[k]) for k in keys],'group comparison signs')
    # Recompute bias from fixed predictions, not the production comparison helper.
    part=d[d[r.group_type].eq(r.group_id)]
    if r.scope!='pooled': part=part[part.outer_fold.eq(int(r.scope))]
    candidate_bias=scores(part[part.model_id.eq(r.candidate)])[-1]
    comparator_bias=scores(part[part.model_id.eq(r.comparator)])[-1]
    close(r.signed_bias_difference_pp, comparator_bias-candidate_bias, 'signed bias difference')
    close(r.absolute_bias_improvement_pp, abs(comparator_bias)-abs(candidate_bias), 'absolute bias improvement')
    close(r.improvement_total_bias_pct, r.signed_bias_difference_pp, 'deprecated bias alias')
assert (gm.n_obs>0).all() and (cells.n_obs>0).all()
for r in read('results/loss_concentration.csv').itertuples():
    x=dc[dc.candidate.eq(r.candidate)&dc.comparator.eq(r.comparator)]; net=x['net_'+r.metric+'_improvement']; loss=(-net[net < -1e-8]).sort_values(ascending=False)
    close([r.loss_sum,r.gain_sum,r.net_improvement,r.top_loss_share],[loss.sum(),net[net>1e-8].sum(),net.sum(),loss.head(r.top_n_requested).sum()/loss.sum()],'loss concentration')
for r in read('results/error_concentration.csv').itertuples():
    x=d[d.model_id.eq(r.model_id)]; col='ae' if r.metric=='SAE' else 'se'
    if r.unit=='observation': ranked=x.sort_values(['ae','obs_id'],ascending=[False,True]); numerator=ranked.head(r.top_n)[col].sum()
    else: ranked=x.groupby('unordered_pair_id')[col].sum().sort_values(ascending=False); numerator=ranked.head(r.top_n).sum()
    close([r.selected_sum,r.total,r.share],[numerator,x[col].sum(),numerator/x[col].sum()],'own error concentration')
if (root/'results/offset_summary.csv').exists():
    for r in read('results/offset_summary.csv').itertuples():
        x=c[c.candidate.eq(r.candidate)&c.comparator.eq(r.comparator)]; x=x if r.scope=='pooled' else x[x.outer_fold.eq(int(r.scope))]
        mask=x.flow_decile.eq('D10') if r.focus=='D10' else x.distance_quintile.eq('Q1')
        close([r.rest_C_AE,r.focus_C_AE,r.total_C_AE],[x.loc[~mask,'improvement_AE'].sum()/len(x),x.loc[mask,'improvement_AE'].sum()/len(x),x.improvement_AE.mean()],'offset summary')
    for r in read('results/overlap_summary.csv').itertuples():
        x=c[c.candidate.eq(r.candidate)&c.comparator.eq(r.comparator)]
        mask=(x.flow_decile.eq('D10')&x.distance_quintile.eq('Q1')) if r.region=='D10_Q1' else ((x.flow_decile.eq('D10')&x.distance_quintile.ne('Q1')) if r.region=='D10_otherQ' else (x.flow_decile.ne('D10')&x.distance_quintile.eq('Q1')))
        z=x[mask]; close([r.n_obs,r.C_AE,r.C_SE,r.AE_observation_loss_share,r.SE_observation_loss_share],[len(z),z.improvement_AE.sum()/14400,z.improvement_SE.sum()/14400,abs(z.improvement_AE[z.improvement_AE<0]).sum()/abs(x.improvement_AE[x.improvement_AE<0]).sum(),abs(z.improvement_SE[z.improvement_SE<0]).sum()/abs(x.improvement_SE[x.improvement_SE<0]).sum()],'overlap summary')
for r in read('results/group_consistency_summary.csv').itertuples():
    x=gc[gc.candidate.eq(r.candidate)&gc.comparator.eq(r.comparator)&gc.group_type.eq(r.group_type)&gc.group_id.eq(r.group_id)&gc.scope.ne('pooled')]['improvement_'+r.metric].dropna()
    assert (len(x),int((x>1e-8).sum()),int((x < -1e-8).sum()),int((abs(x)<=1e-8).sum()))==(r.n_valid_folds,r.n_improved,r.n_worse,r.n_tied)
manifest=json.loads((root/'INPUT_MANIFEST.json').read_text()); source_checks=[]
for item in manifest['inputs']:
    path=Path(item['path'])
    if path.exists(): assert hashlib.sha256(path.read_bytes()).hexdigest()==item['sha256']; source_checks.append(str(path))
repro=[]
if a.replicate_dir:
    for folder in ['results','figures']:
        for file in sorted((root/folder).glob('*.csv*')):
            x=pd.read_csv(file,low_memory=False); y=pd.read_csv(a.replicate_dir/folder/file.name,low_memory=False)
            # Legacy results lack the two explicit bias columns; compare every old column.
            if file.name=='group_fold_comparisons.csv':
                added=['signed_bias_difference_pp','absolute_bias_improvement_pp']
                assert all(k in x for k in added)
                missing=[k for k in added if k not in y]
                x=x.drop(columns=missing)
            assert list(x.columns)==list(y.columns)
            # Deterministic output keys/order; sort all nonnumeric keys for comparison.
            sort=list(x.select_dtypes(exclude='number').columns)
            if sort: x=x.sort_values(sort,na_position='last').reset_index(drop=True); y=y.sort_values(sort,na_position='last').reset_index(drop=True)
            pd.testing.assert_frame_equal(x,y,check_exact=False,rtol=1e-10,atol=1e-7); repro.append(str(file.relative_to(root)))
report=dict(passed=True,independent_checks=len(checks),rtol=1e-10,atol=1e-7,source_files_hash_verified=source_checks,reproduced_csv_files=repro,row_counts={str(f.relative_to(root)):len(pd.read_csv(f,low_memory=False)) for sub in ['results','figures'] for f in sorted((root/sub).glob('*.csv*'))},visual_QA='pending')
(root/'validation_report.json').write_text(json.dumps(report,indent=2)); print(json.dumps(report,indent=2))
