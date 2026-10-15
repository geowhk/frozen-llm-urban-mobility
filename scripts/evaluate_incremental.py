"""Recompute every reported diagnostic from the saved OOF; never fit models."""
import os
os.environ.setdefault('MPLCONFIGDIR', '/tmp/incremental-mpl')
from pathlib import Path
import argparse, shutil, json
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from urban_mobility.diagnostics import metrics, total_variation

ROOT=None
MODELS=['G_BASE','G_NAME','G_COORD','G_POLY2']
PAIRS=[('G_BASE','G_NAME'),('G_BASE','G_COORD'),('G_BASE','G_POLY2'),('G_POLY2','G_NAME'),('G_POLY2','G_COORD'),('G_NAME','G_COORD')]
ERRORS=['mae','rmse','smape','log_mse']

def score(part):
    y=part.y_reference.to_numpy(float); p=part.predicted_flow.to_numpy(float)
    return {**metrics(y,p),'log_mse':float(np.mean((part.predicted_log_flow.to_numpy()-np.log1p(y))**2)), 'total_bias_pct':float(100*(p.sum()-y.sum())/y.sum())}

def main():
    global ROOT
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input-dir',type=Path,required=True)
    parser.add_argument('--output-dir',type=Path,required=True)
    args=parser.parse_args(); src=args.input_dir.resolve(); ROOT=args.output_dir.resolve()
    if ROOT.exists(): parser.error('Output directory must not exist; input results are never overwritten.')
    for name in ['data','results','figures']: (ROOT/name).mkdir(parents=True,exist_ok=True)
    for name in ['data/observations.csv','data/inner_splits.csv','results/oof_predictions.csv','results/tuning_results.csv']:
        shutil.copy2(src/name,ROOT/name)
    (ROOT/'validation_report.json').write_text('{}')
    obs=pd.read_csv(ROOT/'data/observations.csv',dtype={'origin_code':str,'destination_code':str})
    pred=pd.read_csv(ROOT/'results/oof_predictions.csv',dtype={'selected_alpha':str})
    assert len(obs)==14400 and obs.obs_id.nunique()==14400
    assert len(pred)==57600 and not pred.duplicated(['obs_id','model_id']).any()
    assert set(pred.model_id)==set(MODELS)
    assert np.isfinite(pred[['predicted_flow','predicted_log_flow']]).all().all()
    assert (pred.predicted_flow>=0).all()
    df=pred.merge(obs,on=['obs_id','outer_fold'],validate='many_to_one',how='left')
    assert len(df)==57600 and df.y_reference.notna().all()
    assert (df.groupby(['model_id','outer_fold']).size()==2880).all()
    assert np.allclose(df.predicted_flow,np.maximum(0,np.expm1(df.predicted_log_flow)),rtol=1e-13,atol=1e-10)
    overall=pd.DataFrame([{'model_id':m,'n':len(p),**score(p)} for m,p in df.groupby('model_id',sort=False)])
    folds=pd.DataFrame([{'model_id':m,'outer_fold':f,'n':len(p),**score(p)} for (m,f),p in df.groupby(['model_id','outer_fold'],sort=False)])
    overall.to_csv(ROOT/'results/metrics_overall.csv',index=False); folds.to_csv(ROOT/'results/metrics_by_fold.csv',index=False)
    comps=[]; summary=[]
    for comparator,candidate in PAIRS:
        for scope,scores in [('pooled',overall)]+[(str(f),folds[folds.outer_fold.eq(f)]) for f in range(5)]:
            a=scores.set_index('model_id').loc[comparator]; b=scores.set_index('model_id').loc[candidate]
            for metric in ERRORS+['cpc','total_bias_pct']:
                comps.append(dict(comparator=comparator,candidate=candidate,scope=scope,metric=metric,comparator_value=a[metric],candidate_value=b[metric],candidate_minus_comparator=b[metric]-a[metric],error_reduction_pct=100*(a[metric]-b[metric])/a[metric] if metric in ERRORS else np.nan))
        for metric in ERRORS+['cpc','total_bias_pct']:
            a=folds[folds.model_id.eq(comparator)].set_index('outer_fold')[metric]
            b=folds[folds.model_id.eq(candidate)].set_index('outer_fold')[metric]
            better=(b>a) if metric=='cpc' else (b.abs()<a.abs()) if metric=='total_bias_pct' else (b<a)
            summary.append(dict(comparator=comparator,candidate=candidate,metric=metric,improved_folds=int(better.sum()),n_folds=5))
    comparisons=pd.DataFrame(comps); comparisons.to_csv(ROOT/'results/paired_comparisons.csv',index=False)
    fs=pd.DataFrame(summary); fs.to_csv(ROOT/'results/comparison_summary.csv',index=False)
    district=[]
    for model,part in df.groupby('model_id',sort=False):
        daily=part.groupby(['origin_code','destination_code'],as_index=False)[['y_reference','predicted_flow']].sum()
        assert len(daily)==600
        for side,key in [('outflow','origin_code'),('inflow','destination_code')]:
            for code,p in daily.groupby(key):
                y=p.y_reference.to_numpy(); yh=p.predicted_flow.to_numpy()
                rt,pt=y.sum(),yh.sum(); bias=100*(pt-rt)/rt if rt>0 else np.nan
                tv=total_variation(y,yh)
                district.append(dict(model_id=model,district_code=code,side=side,reference_total=rt,predicted_total=pt,signed_bias_pct=bias,absolute_bias_pct=abs(bias),TV=tv,undefined_reason='' if rt>0 and pt>0 else 'zero total'))
    district=pd.DataFrame(district); district.to_csv(ROOT/'results/district_metrics.csv',index=False)
    summaries=[]
    for (model,side),p in district.groupby(['model_id','side'],sort=False):
        row=dict(model_id=model,side=side,n_districts=len(p),district_total_spearman=float(spearmanr(p.reference_total,p.predicted_total).statistic),n_undefined=int(p.TV.isna().sum()))
        for metric in ['signed_bias_pct','absolute_bias_pct','TV']:
            for qname,q in [('q1',.25),('median',.5),('q3',.75)]: row[metric+'_'+qname]=p[metric].quantile(q)
            row[metric+'_IQR']=row[metric+'_q3']-row[metric+'_q1']
        for comparator in ['G_BASE','G_POLY2']:
            reference=district[district.model_id.eq(comparator)&district.side.eq(side)].set_index('district_code')
            candidate=p.set_index('district_code')
            for metric in ['absolute_bias_pct','TV']: row[metric+'_improved_vs_'+comparator]=int((candidate[metric]<reference[metric]).sum())
        summaries.append(row)
    ds=pd.DataFrame(summaries); ds.to_csv(ROOT/'results/district_summary.csv',index=False)
    groups=[]
    for kind,keys in [('flow_decile',['flow_decile']),('distance_quintile',['distance_quintile']),('distance_hour',['distance_quintile','hour'])]:
        for group,p in df.groupby(['model_id']+keys,sort=True):
            model,*vals=group
            groups.append(dict(model_id=model,group_type=kind,group_id=str(vals[0]),hour=int(vals[1]) if len(vals)>1 else '',n=len(p),**{k:v for k,v in score(p).items() if k in ['mae','smape','total_bias_pct']}))
    pd.DataFrame(groups).to_csv(ROOT/'results/group_metrics.csv',index=False)
    tune=pd.read_csv(ROOT/'results/tuning_results.csv',dtype={'alpha':str})
    assert len(tune)==210 and (tune.groupby(['model_id','outer_fold']).size()==14).all()
    selections=tune[tune.selected].copy()
    selections['base_only']=selections.alpha.eq('base_only')
    selections['lower_boundary']=selections.alpha.eq('1e-06')
    selections['upper_finite_boundary']=selections.alpha.eq('1000000.0')
    selections.to_csv(ROOT/'results/selected_settings.csv',index=False)
    for (m,f),t in tune.groupby(['model_id','outer_fold']):
        ties=t[np.isclose(t.validation_log_mse,t.validation_log_mse.min(),rtol=1e-10,atol=1e-12)]
        expected=max(ties.alpha,key=lambda a:np.inf if a=='base_only' else float(a))
        assert t.loc[t.selected,'alpha'].tolist()==[expected]
        assert pred.loc[pred.model_id.eq(m)&pred.outer_fold.eq(f),'selected_alpha'].unique().tolist()==[expected]
    fig1=pd.concat([overall.assign(scope='pooled'),folds.assign(scope=folds.outer_fold.astype(str))],ignore_index=True)
    fig1.to_csv(ROOT/'figures/diagnostic_01_source.csv',index=False)
    district.to_csv(ROOT/'figures/diagnostic_02_source.csv',index=False)
    ds.to_csv(ROOT/'figures/diagnostic_02_summary_source.csv',index=False)
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10})
    colors=['#595959','#3875aa','#c76a25','#608747']
    fig,axes=plt.subplots(2,2,figsize=(10,7),layout='constrained')
    for j,metric in enumerate(['mae','cpc']):
        values=overall.set_index('model_id').loc[MODELS,metric]
        axes[0,j].bar(MODELS,values,color=colors)
        axes[0,j].set_title('Pooled OOF '+metric.upper()); axes[0,j].set_ylabel('Flow units' if metric=='mae' else 'CPC (0–1)')
        for i,v in enumerate(values): axes[0,j].text(i,v,f'{v:.3f}',ha='center',va='bottom',fontsize=9)
        axes[0,j].set_ylim(0,values.max()*1.16)
        for m,c in zip(MODELS,colors):
            p=folds[folds.model_id.eq(m)]
            axes[1,j].plot(p.outer_fold,p[metric],marker='o',label=m,color=c)
        axes[1,j].set_title('Held-out folds: '+metric.upper()); axes[1,j].set_xlabel('Outer fold'); axes[1,j].set_ylabel('Flow units' if metric=='mae' else 'CPC (0–1)'); axes[1,j].set_xticks(range(5))
    axes[1,1].legend(fontsize=8,loc='best')
    fig.savefig(ROOT/'figures/diagnostic_01_incremental_performance.png',dpi=300); plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(11,5),layout='constrained')
    for ax,side in zip(axes,['outflow','inflow']):
        arrays=[district.loc[district.model_id.eq(m)&district.side.eq(side),'TV'].to_numpy() for m in MODELS]
        ax.boxplot(arrays,tick_labels=MODELS,showfliers=False)
        for i,(m,vals,c) in enumerate(zip(MODELS,arrays,colors),1):
            ax.scatter(i+np.linspace(-.12,.12,len(vals)),vals,s=16,alpha=.65,color=c)
            q1,med,q3=np.quantile(vals,[.25,.5,.75])
            ax.text(i,.98,f'{med:.3f}\n[{q1:.3f}, {q3:.3f}]',transform=ax.get_xaxis_transform(),ha='center',va='top',fontsize=8)
        ax.set_title(side.capitalize()+' allocation: 25 districts'); ax.set_ylabel('TV (smaller is better)'); ax.set_ylim(0,max(max(v) for v in arrays)*1.32)
        ax.set_xlabel('Labels: median [Q1, Q3]; boxes show IQR')
    fig.savefig(ROOT/'figures/diagnostic_02_allocation.png',dpi=300); plt.close(fig)
    # Independent formulas, not the imported metric helper.
    differences=[]
    for m,p in df.groupby('model_id'):
        y=p.y_reference.to_numpy(); h=p.predicted_flow.to_numpy(); e=np.abs(h-y)
        independent={'mae':sum(e)/len(e),'rmse':np.sqrt(sum(e*e)/len(e)),'smape':sum(2*e/(h+y+1e-9))/len(e),'cpc':2*sum(np.minimum(y,h))/(sum(y)+sum(h)+1e-9),'log_mse':sum((p.predicted_log_flow.to_numpy()-np.log1p(y))**2)/len(e),'total_bias_pct':100*(sum(h)-sum(y))/sum(y)}
        saved=overall.set_index('model_id').loc[m]
        for key,value in independent.items():
            delta=abs(value-saved[key]); assert np.isclose(value,saved[key],rtol=1e-12,atol=1e-10)
            differences.append(dict(model_id=m,metric=key,max_absolute_difference=delta))
    pd.DataFrame(differences).to_csv(ROOT/'results/independent_metric_check.csv',index=False)
    checks=json.loads((ROOT/'validation_report.json').read_text())
    checks['saved_oof_recalculation']={'passed':True,'rows':len(pred),'per_model':14400,'per_fold':2880,'independent_metric_max_abs_difference':max(r['max_absolute_difference'] for r in differences),'tuning_rows':len(tune),'district_rows':len(district),'group_rows':len(groups),'undefined_district_metrics':int(district.TV.isna().sum())}
    (ROOT/'validation_report.json').write_text(json.dumps(checks,indent=2,ensure_ascii=False,allow_nan=False))
    print(overall.to_string(index=False))

if __name__=='__main__': main()
