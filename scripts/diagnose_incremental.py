"""Post hoc diagnostics from fixed OOF only. No fitting or prediction changes."""
import argparse, hashlib, json, platform, shutil, sys, time, os
os.environ.setdefault('MPLCONFIGDIR','/tmp/incremental-diagnostics-mpl')
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

MODELS=['G_BASE','G_NAME','G_COORD','G_POLY2']
PAIRS=[('G_NAME','G_BASE'),('G_COORD','G_BASE'),('G_NAME','G_POLY2'),('G_COORD','G_POLY2')]
METRICS=['mae','rmse','smape','cpc','log_mse','total_bias_pct']
TOL=1e-8  # absolute tolerance for zero improvement, in each reported metric's units
def bias_comparison(candidate, comparator):
    """Percentage-point differences; only absolute-bias improvement measures benefit."""
    signed = comparator - candidate
    return dict(signed_bias_difference_pp=signed,
                absolute_bias_improvement_pp=abs(comparator)-abs(candidate),
                improvement_total_bias_pct=signed)  # Deprecated compatibility alias.

def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
def read(p): return pd.read_csv(p,dtype={'origin_code':str,'destination_code':str,'obs_id':str,'unordered_pair_id':str})
def metric(d):
    if not len(d): return {k:np.nan for k in METRICS}
    y=d.y_reference.to_numpy(); p=d.predicted_flow.to_numpy(); e=p-y
    return dict(mae=np.abs(e).mean(),rmse=np.sqrt(np.mean(e*e)),smape=np.mean(2*np.abs(e)/(p+y+1e-9)),cpc=2*np.minimum(y,p).sum()/(y.sum()+p.sum()+1e-9),log_mse=np.mean((d.predicted_log_flow-np.log1p(y))**2),total_bias_pct=100*e.sum()/y.sum() if y.sum() else np.nan)
def frac(a,b): return float(a/b) if b else np.nan
def main():
    pa=argparse.ArgumentParser(); pa.add_argument('--input-dir',type=Path,required=True); pa.add_argument('--output-dir',type=Path,required=True); args=pa.parse_args()
    start=time.time(); src=args.input_dir.resolve(); out=args.output_dir.resolve()
    assert src!=out and not out.exists(),'Output must be new; existing results are protected'
    for name in ['inputs','results','figures','logs']: (out/name).mkdir(parents=True,exist_ok=True)
    def write(df,name): df.to_csv(out/name,index=False)
    mapping={'observations.csv':'data/observations.csv','oof_predictions.csv':'results/oof_predictions.csv','metrics_overall_v1.csv':'results/metrics_overall.csv','metrics_by_fold_v1.csv':'results/metrics_by_fold.csv','group_metrics_v1.csv':'results/group_metrics.csv','inner_splits.csv':'data/inner_splits.csv','selected_settings.csv':'results/selected_settings.csv'}
    inputs=[]
    for dest,original in mapping.items():
        p=src/dest if (src/dest).exists() else src/original
        assert p.exists(),p
        inputs.append(dict(path=str(p),sha256=sha(p),bytes=p.stat().st_size))
        shutil.copy2(p,out/'inputs'/dest)
    if (src/'CHECKSUMS.sha256').exists():
        for line in (src/'CHECKSUMS.sha256').read_text().splitlines():
            digest,name=line.split('  ',1); assert sha(src/name)==digest,name
    obs=read(out/'inputs/observations.csv'); pred=read(out/'inputs/oof_predictions.csv')
    assert len(obs)==14400 and obs.obs_id.nunique()==14400 and not obs.isna().any().any()
    assert obs[['origin_code','destination_code']].drop_duplicates().shape[0]==600 and obs.unordered_pair_id.nunique()==300
    assert set(obs.hour)==set(range(24)) and (obs.groupby('unordered_pair_id').size()==48).all()
    assert (obs.groupby('unordered_pair_id').outer_fold.nunique()==1).all()
    assert (obs.groupby(['origin_code','destination_code']).hour.nunique()==24).all()
    assert len(pred)==57600 and not pred.duplicated(['model_id','obs_id']).any() and set(pred.model_id)==set(MODELS)
    d=pred.merge(obs,on=['obs_id','outer_fold'],validate='many_to_one',how='left')
    assert d.y_reference.notna().all() and (d.groupby('model_id').size()==14400).all()
    assert np.isfinite(d[['predicted_flow','predicted_log_flow','y_reference']]).all().all()
    assert (d.predicted_flow>=0).all() and np.allclose(d.predicted_flow,np.maximum(0,np.expm1(d.predicted_log_flow)),rtol=1e-12,atol=1e-9)
    reproduced=[]; check_diffs=[]
    for m,p in d.groupby('model_id'):
        for scope,q in [('pooled',p)]+[(str(f),p[p.outer_fold.eq(f)]) for f in range(5)]:
            values=metric(q); reproduced.append(dict(model_id=m,scope=scope,**values))
            old=read(out/('inputs/metrics_overall_v1.csv' if scope=='pooled' else 'inputs/metrics_by_fold_v1.csv'))
            row=old[old.model_id.eq(m)]
            if scope!='pooled': row=row[row.outer_fold.eq(int(scope))]
            for k,v in values.items():
                diff=abs(v-row.iloc[0][k]); assert np.isclose(v,row.iloc[0][k],rtol=1e-10,atol=1e-8)
                check_diffs.append(diff)
    old=read(out/'inputs/group_metrics_v1.csv')
    for r in old.itertuples():
        p=d[d.model_id.eq(r.model_id)]
        p=p[p.flow_decile.eq(r.group_id)] if r.group_type=='flow_decile' else p[p.distance_quintile.eq(r.group_id)]
        if r.group_type=='distance_hour': p=p[p.hour.eq(int(r.hour))]
        values=metric(p)
        for k in ['mae','smape','total_bias_pct']:
            diff=abs(values[k]-getattr(r,k)); assert np.isclose(values[k],getattr(r,k),rtol=1e-10,atol=1e-8); check_diffs.append(diff)
    write(pd.DataFrame(reproduced),'results/metrics_reproduced.csv')
    d['e']=d.predicted_flow-d.y_reference; d['AE']=d.e.abs(); d['SE']=d.e**2
    d['sAPE']=2*d.AE/(d.predicted_flow+d.y_reference+1e-9); d['log_SE']=(d.predicted_log_flow-np.log1p(d.y_reference))**2
    write(d,'results/observation_errors.csv.gz')
    comps=[]
    for cand,ref in PAIRS:
        a=d[d.model_id.eq(cand)].copy(); b=d[d.model_id.eq(ref)].set_index('obs_id').loc[a.obs_id]
        a=a.drop(columns=['selected_alpha','base_only','model_id']); a['candidate']=cand; a['comparator']=ref
        for key in ['AE','SE','sAPE']:
            a['improvement_'+key]=b[key].to_numpy()-a[key].to_numpy()
            a['improvement_'+key+'_pct']=np.divide(100*a['improvement_'+key],b[key].to_numpy(),out=np.full(len(a),np.nan),where=b[key].to_numpy()!=0)
            a['improvement_'+key+'_pct_na_reason']=np.where(b[key].to_numpy()==0,'zero comparator error','')
        comps.append(a)
    c=pd.concat(comps,ignore_index=True); write(c,'results/observation_comparisons.csv.gz')
    def aggregate(part):
        peak=part.sort_values(['AE','obs_id'],ascending=[False,True]).iloc[0]
        return dict(outer_fold=int(part.outer_fold.iloc[0]),distance_km=float(part.distance_km.mean()),reference_total=part.y_reference.sum(),predicted_total=part.predicted_flow.sum(),mean_reference_flow=part.y_reference.mean(),SAE=part.AE.sum(),SSE=part.SE.sum(),signed_error_sum=part.e.sum(),n_overpredicted=int((part.e>0).sum()),max_AE=peak.AE,max_AE_obs_id=peak.obs_id,max_AE_origin=peak.origin_code,max_AE_destination=peak.destination_code,max_AE_hour=int(peak.hour),D10_count=int(part.flow_decile.eq('D10').sum()),n_obs=len(part))
    dyads=[]; directed=[]; ec=[]
    for (m,q),part in d.groupby(['model_id','unordered_pair_id']):
        i,j=q.split('|'); dyads.append(dict(model_id=m,unordered_pair_id=q,endpoint_1=i,endpoint_2=j,**aggregate(part)))
    for (m,i,j),part in d.groupby(['model_id','origin_code','destination_code']): directed.append(dict(model_id=m,origin_code=i,destination_code=j,unordered_pair_id=part.unordered_pair_id.iloc[0],**aggregate(part)))
    dy=pd.DataFrame(dyads); write(dy,'results/dyad_errors.csv'); write(pd.DataFrame(directed),'results/directed_od_errors.csv')
    for m in MODELS:
        part=d[d.model_id.eq(m)].sort_values(['AE','obs_id'],ascending=[False,True])
        for pct in [.01,.05]:
            n=int(np.ceil(len(part)*pct)); top=part.head(n)
            for key,source in [('SAE','AE'),('SSE','SE')]: ec.append(dict(model_id=m,unit='observation',ranking='AE',top_n=n,top_fraction=pct,metric=key,selected_sum=top[source].sum(),total=part[source].sum(),share=frac(top[source].sum(),part[source].sum())))
        for key in ['SAE','SSE']:
            part=dy[dy.model_id.eq(m)].sort_values([key,'unordered_pair_id'],ascending=[False,True])
            for n in [1,3,5,10]: ec.append(dict(model_id=m,unit='dyad',ranking=key,top_n=n,top_fraction=n/300,metric=key,selected_sum=part.head(n)[key].sum(),total=part[key].sum(),share=frac(part.head(n)[key].sum(),part[key].sum())))
    write(pd.DataFrame(ec),'results/error_concentration.csv')
    dc=[]
    for (cand,ref,q),part in c.groupby(['candidate','comparator','unordered_pair_id']):
        row=dict(candidate=cand,comparator=ref,unordered_pair_id=q,outer_fold=int(part.outer_fold.iloc[0]))
        for k in ['AE','SE','sAPE']:
            values=part['improvement_'+k]; net=values.sum()
            row.update({f'net_{k}_improvement':net,f'net_{k}_loss':max(-net,0),f'net_{k}_gain':max(net,0),f'observation_{k}_positive_sum':values.clip(lower=0).sum(),f'observation_{k}_negative_sum':values.clip(upper=0).sum()})
        dc.append(row)
    dc=pd.DataFrame(dc); write(dc,'results/dyad_comparisons.csv')
    lc=[]; curves=[]; selected=[]; exclusions=[]; sensitivities=[]
    for cand,ref in PAIRS:
        part=dc[dc.candidate.eq(cand)&dc.comparator.eq(ref)]
        for k in ['AE','SE']:
            net=part[f'net_{k}_improvement']; loss=part[net < -TOL].sort_values([f'net_{k}_loss','unordered_pair_id'],ascending=[False,True])
            total=loss[f'net_{k}_loss'].sum(); gain=part.loc[net>TOL,f'net_{k}_gain'].sum()
            for n in [1,3,5,10]: lc.append(dict(candidate=cand,comparator=ref,metric=k,top_n_requested=n,top_n_actual=min(n,len(loss)),loss_sum=total,gain_sum=gain,net_improvement=net.sum(),n_gain=int((net>TOL).sum()),n_loss=len(loss),n_tie=int((net.abs()<=TOL).sum()),top_loss_sum=loss.head(n)[f'net_{k}_loss'].sum(),top_loss_share=frac(loss.head(n)[f'net_{k}_loss'].sum(),total),na_reason='' if total else 'no loss dyads',tie_atol=TOL))
            running=0.
            for rank,r in enumerate(loss.itertuples(),1):
                running+=getattr(r,f'net_{k}_loss'); curves.append(dict(candidate=cand,comparator=ref,metric=k,rank=rank,unordered_pair_id=r.unordered_pair_id,loss=getattr(r,f'net_{k}_loss'),loss_total=total,n_loss_dyads=len(loss),cumulative_share=running/total))
                if rank<=3: selected.append(dict(candidate=cand,comparator=ref,criterion=k,rank=rank,unordered_pair_id=r.unordered_pair_id))
        loss=part[part.net_SE_improvement < -TOL].sort_values(['net_SE_loss','unordered_pair_id'],ascending=[False,True])
        for n in [0,1,3,5]:
            ids=loss.head(n).unordered_pair_id.tolist()
            for rank,q in enumerate(ids,1): exclusions.append(dict(candidate=cand,comparator=ref,requested_n=n,actual_n=len(ids),rank=rank,unordered_pair_id=q))
            for m in MODELS:
                p=d[d.model_id.eq(m)&~d.unordered_pair_id.isin(ids)]
                sensitivities.append(dict(candidate=cand,comparator=ref,requested_n=n,actual_n=len(ids),model_id=m,n_obs=len(p),n_pairs=p.unordered_pair_id.nunique(),reference_sum=p.y_reference.sum(),excluded_reference_fraction=1-p.y_reference.sum()/obs.y_reference.sum(),excluded_observation_fraction=1-len(p)/14400,excluded_pair_ids=';'.join(ids),**metric(p)))
    write(pd.DataFrame(lc),'results/loss_concentration.csv'); write(pd.DataFrame(curves),'figures/diagnostic_01_source.csv')
    chosen=pd.DataFrame(selected); write(chosen,'results/influential_selection.csv')
    details=d[d.unordered_pair_id.isin(chosen.unordered_pair_id.unique())].copy()
    reasons=chosen.groupby('unordered_pair_id').apply(lambda p:';'.join(f'{r.candidate}-vs-{r.comparator}:{r.criterion}:rank{r.rank}' for r in p.itertuples()),include_groups=False)
    details['selection_reason']=details.unordered_pair_id.map(reasons)
    write(details,'results/influential_dyad_hourly.csv'); write(pd.DataFrame(exclusions),'results/exclusion_sets.csv'); write(pd.DataFrame(sensitivities),'results/exclusion_sensitivity.csv')
    gm=[]; gc=[]; contribution=[]
    for kind,labels in [('flow_decile',[f'D{i}' for i in range(1,11)]),('distance_quintile',[f'Q{i}' for i in range(1,6)])]:
        for scope in ['pooled']+list(map(str,range(5))):
            dd=d if scope=='pooled' else d[d.outer_fold.eq(int(scope))]
            cc=c if scope=='pooled' else c[c.outer_fold.eq(int(scope))]
            nn=14400 if scope=='pooled' else 2880
            for m in MODELS:
                allm=dd[dd.model_id.eq(m)]
                for group in labels:
                    p=allm[allm[kind].eq(group)]
                    gm.append(dict(model_id=m,scope=scope,group_type=kind,group_id=group,n_obs=len(p),n_directed_OD=len(p[['origin_code','destination_code']].drop_duplicates()),n_unordered_pairs=p.unordered_pair_id.nunique(),reference_sum=p.y_reference.sum(),observation_share=len(p)/nn,reference_share=frac(p.y_reference.sum(),allm.y_reference.sum()),na_reason='' if len(p) else 'empty group',**metric(p)))
            for cand,ref in PAIRS:
                pair=cc[cc.candidate.eq(cand)&cc.comparator.eq(ref)]
                for group in labels:
                    p=pair[pair[kind].eq(group)]
                    ca=dd[dd.model_id.eq(cand)&dd[kind].eq(group)]; re=dd[dd.model_id.eq(ref)&dd[kind].eq(group)]
                    am,bm=metric(ca),metric(re)
                    row=dict(candidate=cand,comparator=ref,scope=scope,group_type=kind,group_id=group,n_obs=len(p),na_reason='' if len(p) else 'empty group')
                    for k in METRICS:
                        row['improvement_'+k]=(am[k]-bm[k]) if k=='cpc' else bm[k]-am[k]
                        if k in ['mae','rmse','smape','log_mse']: row['improvement_'+k+'_pct']=100*frac(bm[k]-am[k],bm[k])
                    row.update(bias_comparison(am['total_bias_pct'], bm['total_bias_pct']))
                    gc.append(row)
                    contribution.append(dict(candidate=cand,comparator=ref,scope=scope,group_type=kind,group_id=group,n_obs=len(p),denominator_N=nn,C_AE=p.improvement_AE.sum()/nn,C_SE=p.improvement_SE.sum()/nn,group_mean_AE_improvement=p.improvement_AE.mean(),group_mean_SE_improvement=p.improvement_SE.mean()))
    gm=pd.DataFrame(gm); gc=pd.DataFrame(gc); contribution=pd.DataFrame(contribution)
    write(gm,'results/group_fold_metrics.csv'); write(gc,'results/group_fold_comparisons.csv'); write(contribution,'results/group_error_contributions.csv')
    consistency=[]
    for (cand,ref,kind,g),p in gc[gc.scope.ne('pooled')].groupby(['candidate','comparator','group_type','group_id'],sort=False):
        for metric_name in ['mae','rmse','smape']:
            vals=p['improvement_'+metric_name].dropna(); ix=vals.idxmin() if len(vals) else None
            consistency.append(dict(candidate=cand,comparator=ref,group_type=kind,group_id=g,metric=metric_name,n_valid_folds=len(vals),n_improved=int((vals>TOL).sum()),n_worse=int((vals < -TOL).sum()),n_tied=int((vals.abs()<=TOL).sum()),minimum=vals.min(),median=vals.median(),maximum=vals.max(),worst_fold=p.loc[ix,'scope'] if ix is not None else '',tie_atol=TOL))
    write(pd.DataFrame(consistency),'results/group_consistency_summary.csv')
    cells=[]; cellcomp=[]
    for di in range(1,11):
        for qi in range(1,6):
            mask=d.flow_decile.eq(f'D{di}')&d.distance_quintile.eq(f'Q{qi}')
            for m in MODELS:
                p=d[mask&d.model_id.eq(m)]
                allm=d[d.model_id.eq(m)]
                cells.append(dict(model_id=m,flow_decile=f'D{di}',distance_quintile=f'Q{qi}',n_obs=len(p),n_unordered_pairs=p.unordered_pair_id.nunique(),reference_sum=p.y_reference.sum(),SAE=p.AE.sum(),SSE=p.SE.sum(),SAE_share=frac(p.AE.sum(),allm.AE.sum()),SSE_share=frac(p.SE.sum(),allm.SE.sum()),na_reason='' if len(p) else 'empty cell'))
            for cand,ref in PAIRS:
                allc=c[c.candidate.eq(cand)&c.comparator.eq(ref)]; p=allc[allc.flow_decile.eq(f'D{di}')&allc.distance_quintile.eq(f'Q{qi}')]
                cellcomp.append(dict(candidate=cand,comparator=ref,flow_decile=f'D{di}',distance_quintile=f'Q{qi}',n_obs=len(p),n_unordered_pairs=p.unordered_pair_id.nunique(),reference_sum=p.y_reference.sum(),improvement_mae=p.improvement_AE.mean(),C_AE=p.improvement_AE.sum()/14400,C_SE=p.improvement_SE.sum()/14400,observation_AE_loss_sum=(-p.improvement_AE.clip(upper=0)).sum(),observation_SE_loss_sum=(-p.improvement_SE.clip(upper=0)).sum(),observation_AE_loss_share=frac((-p.improvement_AE.clip(upper=0)).sum(),(-allc.improvement_AE.clip(upper=0)).sum()),observation_SE_loss_share=frac((-p.improvement_SE.clip(upper=0)).sum(),(-allc.improvement_SE.clip(upper=0)).sum()),na_reason='' if len(p) else 'empty cell'))
    write(pd.DataFrame(cells),'results/size_distance_cells.csv'); write(pd.DataFrame(cellcomp),'results/size_distance_comparisons.csv')
    write(gc[['candidate','comparator','scope','group_type','group_id','n_obs','improvement_mae']],'figures/diagnostic_02_source.csv')
    plt.rcParams.update({'font.size':9,'font.family':'DejaVu Sans'})
    curves=pd.DataFrame(curves); fig,axs=plt.subplots(1,2,figsize=(12,4.8),layout='constrained')
    for ax,k in zip(axs,['AE','SE']):
        for cand,ref in PAIRS:
            p=curves[curves.candidate.eq(cand)&curves.comparator.eq(ref)&curves.metric.eq(k)]
            if len(p):
                ax.plot(p['rank'],p.cumulative_share,label=f'{cand} vs {ref} (n={len(p)})')
                top=p[p['rank'].isin([1,3,5,10])]; ax.scatter(top['rank'],top.cumulative_share,s=14)
            else: ax.plot([],[],label=f'{cand} vs {ref}: no losses')
        for x in [1,3,5,10]: ax.axvline(x,color='gray',alpha=.3,linewidth=.6)
        ax.set_title('SAE net-loss concentration' if k=='AE' else 'SSE net-loss concentration'); ax.set_xlabel('Number of dyads, ranked separately per comparison'); ax.set_ylabel('Cumulative share of all net-loss dyads'); ax.set_ylim(0,1.03); ax.legend(fontsize=7)
    fig.suptitle('Post hoc diagnostic: markers at top 1, 3, 5, 10 dyads')
    for ext in ['png','pdf']: fig.savefig(out/f'figures/diagnostic_01_dyad_loss_concentration.{ext}',dpi=300)
    plt.close(fig)
    fig,axs=plt.subplots(2,4,figsize=(16,9),layout='constrained')
    for ri,(kind,labels) in enumerate([('flow_decile',[f'D{i}' for i in range(1,11)]),('distance_quintile',[f'Q{i}' for i in range(1,6)])]):
        limit=float(gc.loc[gc.group_type.eq(kind),'improvement_mae'].abs().max())
        for col,(cand,ref) in enumerate(PAIRS):
            p=gc[gc.candidate.eq(cand)&gc.comparator.eq(ref)&gc.group_type.eq(kind)]
            mat=p.pivot(index='group_id',columns='scope',values='improvement_mae').reindex(index=labels,columns=['pooled','0','1','2','3','4'])
            ax=axs[ri,col]; im=ax.imshow(mat,cmap='RdBu',vmin=-limit,vmax=limit,aspect='auto')
            ax.set_xticks(range(6),['All','F0','F1','F2','F3','F4']); ax.set_yticks(range(len(labels)),labels); ax.axvline(.5,color='black',lw=1.5)
            ax.set_title(f'{cand} vs {ref}\n'+('Flow decile' if ri==0 else 'Distance quintile'))
            for i in range(len(labels)):
                for j in range(6):
                    v=mat.iloc[i,j]; ax.text(j,i,'NA' if pd.isna(v) else f'{v:.1f}',ha='center',va='center',fontsize=7,color='white' if abs(v)>limit*.55 else 'black')
        cb=fig.colorbar(im,ax=list(axs[ri,:]),shrink=.85); cb.set_label('Comparator MAE − candidate MAE\nPositive = LLM improvement')
    fig.suptitle('Group MAE differences (not contribution); All = pooled. Shared scale within each row.')
    for ext in ['png','pdf']: fig.savefig(out/f'figures/diagnostic_02_group_fold_consistency.{ext}',dpi=300)
    plt.close(fig)
    co=pd.read_csv(out/'results/group_error_contributions.csv'); cc=pd.read_csv(out/'results/size_distance_comparisons.csv')
    offset=[]
    for (ca,re,scope,kind),g in co.groupby(['candidate','comparator','scope','group_type'],sort=False):
        focus='D10' if kind=='flow_decile' else 'Q1'; rest=g[g.group_id.ne(focus)]; f=g[g.group_id.eq(focus)].iloc[0]
        offset.append(dict(candidate=ca,comparator=re,scope=scope,focus=focus,rest_C_AE=rest.C_AE.sum(),focus_C_AE=f.C_AE,total_C_AE=g.C_AE.sum(),rest_C_SE=rest.C_SE.sum(),focus_C_SE=f.C_SE,total_C_SE=g.C_SE.sum()))
    offset=pd.DataFrame(offset); offset.to_csv(out/'results/offset_summary.csv',index=False)
    overlap=[]
    for label,select in [('D10_Q1',lambda x:x.flow_decile.eq('D10')&x.distance_quintile.eq('Q1')),('D10_otherQ',lambda x:x.flow_decile.eq('D10')&x.distance_quintile.ne('Q1')),('Q1_otherD',lambda x:x.flow_decile.ne('D10')&x.distance_quintile.eq('Q1'))]:
        for (ca,re),g in cc.groupby(['candidate','comparator'],sort=False):
            p=g[select(g)]; overlap.append(dict(candidate=ca,comparator=re,region=label,n_obs=p.n_obs.sum(),reference_sum=p.reference_sum.sum(),C_AE=p.C_AE.sum(),C_SE=p.C_SE.sum(),AE_observation_loss_share=p.observation_AE_loss_share.sum(),SE_observation_loss_share=p.observation_SE_loss_share.sum()))
    overlap=pd.DataFrame(overlap); overlap.to_csv(out/'results/overlap_summary.csv',index=False)
    for item in inputs: assert sha(Path(item['path']))==item['sha256']
    manifest=dict(inputs=inputs,command=sys.argv,python=platform.python_version(),numpy=np.__version__,pandas=pd.__version__,matplotlib=matplotlib.__version__,elapsed_seconds=time.time()-start,tie_atol=TOL,metric_reproduction_max_abs_difference=max(check_diffs),source_unchanged=True,post_hoc=True)
    (out/'INPUT_MANIFEST.json').write_text(json.dumps(manifest,indent=2,ensure_ascii=False))
    (out/'logs/execution.log').write_text(f'Completed fixed-OOF diagnostics in {time.time()-start:.3f} seconds. No fitting. Source hashes unchanged. Independent verification pending.\n')
    print(json.dumps(manifest,indent=2))

if __name__=='__main__': main()
