"""Prespecified incremental-value experiment. No LLM inference or legacy writes."""
import os
os.environ.setdefault('MPLCONFIGDIR', '/tmp/incremental-mpl')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '4')
import argparse, hashlib, json, platform, resource, sys, time, traceback
from pathlib import Path
import numpy as np
import pandas as pd
import scipy
from scipy.linalg import svd
from scipy.stats import spearmanr
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
import sklearn
from urban_mobility import core, splits, diagnostics as diag

OUT = None  # Set only by the command-line entry point.
ALPHAS = [10.0**k for k in range(-6, 7)]
MODELS = ['G_BASE', 'G_NAME', 'G_COORD', 'G_POLY2']
CHECKS = {}
SOLVER = []
START = time.time()

def save_json(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False))

def check(name, condition, detail=None):
    CHECKS[name] = {'passed': bool(condition), 'detail': detail}
    save_json(OUT/'validation_report.json', CHECKS)
    if not condition:
        raise AssertionError(name + ': ' + str(detail))

def log(message):
    print(time.strftime('%Y-%m-%d %H:%M:%S'), message, flush=True)
    save_json(OUT/'logs/status.json', {'status':'running', 'message':message, 'elapsed_seconds':time.time()-START, 'pid':os.getpid()})

def load_inputs(args):
    inp = args.inputs_dir.resolve()
    result = args.reference_results_dir.resolve()
    files = [inp/'mobility.parquet', inp/'prompt_source.parquet', inp/'folds.csv', inp/'inner_splits.csv', result/'oof_predictions_all_models.parquet', result/'bin_assignments.csv']
    df = pd.read_parquet(files[4])
    for c in ['orig','dest']: df[c] = df[c].astype(str)
    df['pair_id'] = df.orig+'|'+df.dest
    splits.validate_strict_data(df)
    folds, inner = pd.read_csv(files[2]), pd.read_csv(files[3])
    splits.validate_strict_splits(df.drop(columns='outer_fold'), folds, inner)
    check('saved_outer_exact', df.outer_fold.equals(df.dyad_id.map(folds.set_index('dyad_id').outer_fold)))
    mobility = pd.read_parquet(files[0]).rename(columns={'arrival_hour':'hour','flow':'y_gt'})
    prompt = pd.read_parquet(files[1]).rename(columns={'arrival_hour':'hour'})
    for frame in [mobility,prompt]:
        for c in ['orig','dest']: frame[c] = frame[c].astype(str)
    aligned = df[['orig','dest','hour']].merge(mobility,on=['orig','dest','hour'],validate='one_to_one')
    check('reference_exact', np.array_equal(df.y_gt,aligned.y_gt))
    check('distance_exact', np.array_equal(df.dist_km,aligned.dist_km))
    coords = ['orig_lat','orig_lon','dest_lat','dest_lon']
    pc = df[['orig','dest','hour']].merge(prompt,on=['orig','dest','hour'],validate='one_to_one')
    check('prompt_round3_coordinates', np.array_equal(df[coords].to_numpy(float),pc[coords].round(3).to_numpy(float)))
    bins = pd.read_csv(files[5])
    df = df.merge(bins[['query_id','flow_bin','distance_bin']],on='query_id',validate='one_to_one')
    features = {'G_POLY2':np.column_stack([pc[coords].round(3).to_numpy(float), np.sin(2*np.pi*df.hour/24),np.cos(2*np.pi*df.hour/24)])}
    for model,variant in [('G_NAME','original'),('G_COORD','geometry')]:
        folder = args.cache_dir.resolve()/variant
        for name in core.EXPECTED_CACHE_SHA256[variant]:
            p=folder/name; files.append(p)
            check(variant+'_'+name+'_hash',core.sha256_file(p)==core.EXPECTED_CACHE_SHA256[variant][name])
        idx=pd.read_parquet(folder/'forward_row_index.parquet')
        for c in ['orig','dest']: idx[c]=idx[c].astype(str)
        idx['cache_row']=np.arange(len(idx))
        match=df[['query_id','orig','dest','hour']].merge(idx[['query_id','orig','dest','hour','cache_row']],on=['query_id','orig','dest','hour'],validate='one_to_one',how='left')
        check(variant+'_key_coverage',match.cache_row.notna().all() and len(match)==14400)
        arr=np.load(folder/'forward_lasttoken_layer31.npy',mmap_mode='r')
        check(variant+'_shape_finite',arr.shape==(14400,4096) and np.isfinite(arr).all())
        features[model]=np.asarray(arr[match.cache_row.to_numpy(int)],dtype=np.float64)
    manifest=[]
    for p in files+[Path(__file__), Path(core.__file__), Path(splits.__file__), Path(diag.__file__)]:
        manifest.append({'path':str(p.resolve()),'size':p.stat().st_size,'sha256':core.sha256_file(p)})
    save_json(OUT/'INPUT_MANIFEST.json', {'files':manifest,'rows':14400,'cache_shape':[14400,4096],'period':'2024-11 weekday daily mean','coordinates':'historical prompt source rounded to 3 decimals','distance':'existing gravity centroid distance in km','hour':'arrival hour 0-23','cache_matching':'query_id, orig, dest, hour explicitly joined'})
    obs=df[['query_id','orig','dest','hour','dyad_id','y_gt','dist_km','outer_fold','flow_bin','distance_bin']].rename(columns={'query_id':'obs_id','orig':'origin_code','dest':'destination_code','dyad_id':'unordered_pair_id','y_gt':'y_reference','dist_km':'distance_km','flow_bin':'flow_decile','distance_bin':'distance_quintile'})
    obs.to_csv(OUT/'data/observations.csv',index=False)
    inner.to_csv(OUT/'data/inner_splits.csv',index=False)
    return df,inner,features

def gravity_features(train, rows):
    """Only training targets enter O,D; no total T and no fallback."""
    rel=train[['orig','dest']].drop_duplicates()
    ro=rel.groupby('orig').size()/24; rd=rel.groupby('dest').size()/24
    assert len(ro)==len(rd)==25 and (ro>0).all() and (rd>0).all()
    o=train.groupby(['orig','hour']).y_gt.sum()
    d=train.groupby(['dest','hour']).y_gt.sum()
    oo=np.array([o.loc[(i,t)]/ro.loc[i] for i,t in zip(rows.orig,rows.hour)])
    dd=np.array([d.loc[(j,t)]/rd.loc[j] for j,t in zip(rows.dest,rows.hour)])
    return np.column_stack([np.ones(len(rows)),np.log1p(oo),np.log1p(dd),np.log1p(rows.dist_km)])

def preprocess(xtrain, xeval, model):
    a=StandardScaler().fit(xtrain)
    h=a.transform(xtrain); he=a.transform(xeval)
    if model=='G_POLY2':
        poly=PolynomialFeatures(2,include_bias=False)
        h=poly.fit_transform(h); he=poly.transform(he)
        assert h.shape[1]==27
        b=StandardScaler().fit(h); h=b.transform(h); he=b.transform(he)
    return h,he

def stationarity(g,h,z,gamma,w,alpha,tag):
    residual=g@gamma+h@w-z
    ng=np.linalg.norm(g.T@residual)/(np.linalg.norm(g)*np.linalg.norm(residual)+np.finfo(float).eps)
    nh=np.linalg.norm(h.T@residual+alpha*w)/(np.linalg.norm(h)*np.linalg.norm(residual)+alpha*np.linalg.norm(w)+np.finfo(float).eps)
    row={'fit':tag,'alpha':alpha,'unpenalized_normalized_residual':float(ng),'penalized_normalized_residual':float(nh),'status':'pass' if max(ng,nh)<1e-8 else 'fail'}
    SOLVER.append(row)
    pd.DataFrame(SOLVER).to_csv(OUT/'logs/solver_diagnostics.csv',index=False)
    check(tag+'_stationarity',max(ng,nh)<1e-8,row)

def decomposition(g,h,z,tag):
    began=time.time()
    a=np.linalg.lstsq(g,z,rcond=None)[0]
    b=np.linalg.lstsq(g,h,rcond=None)[0]
    hp=h-g@b; zp=z-g@a
    log(tag+' SVD started '+str(hp.shape))
    u,s,vt=svd(hp,full_matrices=False,check_finite=False,lapack_driver='gesdd')
    uz=u.T@zp
    del u,hp
    info={'rank_G':int(np.linalg.matrix_rank(g)),'condition_G':float(np.linalg.cond(g)),'rank_H_perp':int((s>s[0]*max(h.shape)*np.finfo(float).eps).sum()),'smax':float(s[0]),'smin':float(s[-1]),'seconds':time.time()-began}
    CHECKS[tag+'_decomposition']=info
    log(tag+' SVD finished '+str(info))
    return a,b,s,vt,uz

def solve(g,h,z,dec,alpha,tag):
    a,b,s,vt,uz=dec
    w=vt.T@((s/(s*s+alpha))*uz)
    gamma=np.linalg.lstsq(g,z-h@w,rcond=None)[0]
    stationarity(g,h,z,gamma,w,alpha,tag)
    return gamma,w

def synthetic():
    rng=np.random.default_rng(20261012)
    g=np.column_stack([np.ones(80),rng.normal(size=(80,3))]); h=rng.normal(size=(80,9)); z=rng.normal(size=80)
    dec=decomposition(g,h,z,'synthetic')
    for alpha in [1e-6,1,1e6]:
        gamma,w=solve(g,h,z,dec,alpha,'synthetic_'+str(alpha))
        design=np.column_stack([g,h]); penalty=np.column_stack([np.zeros((9,4)),np.sqrt(alpha)*np.eye(9)])
        beta=np.linalg.lstsq(np.vstack([design,penalty]),np.r_[z,np.zeros(9)],rcond=None)[0]
        ours=np.r_[gamma,w]
        diff=float(np.max(np.abs(design@(ours-beta))))
        obj=lambda v:float(np.sum((design@v-z)**2)+alpha*np.sum(v[4:]**2))
        check('synthetic_augmented_'+str(alpha),diff<1e-8 and np.isclose(obj(ours),obj(beta),rtol=1e-10),{'prediction_max_diff':diff,'objective_difference':obj(ours)-obj(beta)})
    gamma,w=solve(g,h,z,dec,1e14,'synthetic_large_alpha')
    check('large_alpha_approaches_base',np.max(np.abs(g@gamma+h@w-g@dec[0]))<1e-8)

def reproduce(df):
    pred=np.zeros(len(df)); raw=pred.copy(); rows=[]
    for fold in range(5):
        test=df.outer_fold.eq(fold).to_numpy(); train=df.loc[~test]
        g=gravity_features(train,train); ge=gravity_features(train,df.loc[test])
        gamma=np.linalg.lstsq(g,np.log1p(train.y_gt),rcond=None)[0]
        raw[test]=ge@gamma; pred[test]=np.maximum(0,np.expm1(raw[test]))
        old=df.loc[test,'prediction_corrected_loglinear_gravity'].to_numpy()
        err=np.abs(pred[test]-old)
        row={'outer_fold':fold,'max_absolute_difference':float(err.max()),'max_relative_difference':float(np.max(err/np.maximum(np.abs(old),1e-15)))}; rows.append(row)
        check('gravity_reproduction_'+str(fold),np.allclose(pred[test],old,rtol=1e-9,atol=1e-7),row)
    pd.DataFrame(rows).to_csv(OUT/'results/gravity_reproduction.csv',index=False)
    return pred,raw

def run(df,inner,features,base,baselog):
    rows=[]; tuning=[]
    for model in MODELS:
        for fold in range(5):
            test=df.outer_fold.eq(fold).to_numpy(); outer=~test
            if model=='G_BASE':
                loghat=baselog[test]; selected='base_only'
            else:
                split=inner[inner.outer_fold.eq(fold)]
                tr=df.dyad_id.isin(split.loc[split.inner_subset.eq('train'),'dyad_id']).to_numpy()
                va=df.dyad_id.isin(split.loc[split.inner_subset.eq('validation'),'dyad_id']).to_numpy()
                check(f'{model}_{fold}_heldout_disjoint',not np.any(tr&va) and not np.any(tr&test) and tr.sum()==10080 and va.sum()==1440)
                g=gravity_features(df.loc[tr],df.loc[tr]); ge=gravity_features(df.loc[tr],df.loc[va]); z=np.log1p(df.loc[tr,'y_gt'].to_numpy())
                altered=df.loc[va].copy(); altered['y_gt']=123456789.
                check(f'{model}_{fold}_inner_target_invariance',np.array_equal(ge,gravity_features(df.loc[tr],altered)))
                h,he=preprocess(features[model][tr],features[model][va],model)
                dec=decomposition(g,h,z,f'{model}_{fold}_inner')
                zv=np.log1p(df.loc[va,'y_gt'].to_numpy())
                candidates=[{'alpha':'base_only','validation_log_mse':float(np.mean((ge@dec[0]-zv)**2)),'solver_status':'OLS'}]
                for alpha in ALPHAS:
                    gamma,w=solve(g,h,z,dec,alpha,f'{model}_{fold}_inner_{alpha}')
                    candidates.append({'alpha':alpha,'validation_log_mse':float(np.mean((ge@gamma+he@w-zv)**2)),'solver_status':'pass'})
                lowest=min(c['validation_log_mse'] for c in candidates)
                ties=[c for c in candidates if np.isclose(c['validation_log_mse'],lowest,rtol=1e-10,atol=1e-12)]
                best=max(ties,key=lambda c:float('inf') if c['alpha']=='base_only' else c['alpha']); selected=best['alpha']
                for c in candidates: tuning.append({'model_id':model,'outer_fold':fold,**c,'selected':c is best})
                pd.DataFrame(tuning).to_csv(OUT/'results/tuning_results.csv',index=False)
                del dec,h,he
                if selected=='base_only': loghat=baselog[test].copy()
                else:
                    g=gravity_features(df.loc[outer],df.loc[outer]); ge=gravity_features(df.loc[outer],df.loc[test]); z=np.log1p(df.loc[outer,'y_gt'].to_numpy())
                    altered=df.loc[test].copy(); altered['y_gt']=-999.
                    check(f'{model}_{fold}_outer_target_invariance',np.array_equal(ge,gravity_features(df.loc[outer],altered)))
                    h,he=preprocess(features[model][outer],features[model][test],model)
                    dec=decomposition(g,h,z,f'{model}_{fold}_outer')
                    gamma,w=solve(g,h,z,dec,float(selected),f'{model}_{fold}_outer_selected')
                    loghat=ge@gamma+he@w
                    check(f'{model}_{fold}_prediction_target_invariance',np.array_equal(loghat,gravity_features(df.loc[outer],altered)@gamma+he@w))
                    del dec,h,he
            with np.errstate(over='raise',invalid='raise'): pred=np.maximum(0,np.expm1(loghat))
            check(f'{model}_{fold}_finite',np.isfinite(pred).all() and np.isfinite(loghat).all())
            if selected=='base_only': check(f'{model}_{fold}_base_exact',np.array_equal(pred,base[test]))
            part=pd.DataFrame({'obs_id':df.loc[test,'query_id'],'model_id':model,'outer_fold':fold,'predicted_log_flow':loghat,'predicted_flow':pred,'selected_alpha':str(selected),'base_only':selected=='base_only'})
            rows.append(part); pd.concat(rows).to_csv(OUT/'results/oof_predictions_partial.csv',index=False)
            log(f'{model} fold {fold} complete; alpha={selected}')
    pd.concat(rows).to_csv(OUT/'results/oof_predictions.csv',index=False)

def main():
    global OUT
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ['inputs-dir','reference-results-dir','cache-dir','output-dir']:
        parser.add_argument('--'+name,type=Path,required=True)
    parser.add_argument('--preflight',action='store_true')
    args=parser.parse_args()
    OUT=args.output_dir.resolve()
    if OUT.exists(): parser.error('Output directory must not exist; existing runs are protected.')
    OUT.mkdir(parents=True)
    for folder in ['logs','data','results','figures']: (OUT/folder).mkdir(exist_ok=True)
    config={'models':MODELS,'alpha_grid':ALPHAS,'base_only_candidate':True,'tie_rtol':1e-10,'tie_atol':1e-12,'tie_order':'base_only then larger alpha','split_seed_existing':202511,'objective':'SSE + alpha ||w_H||^2; G unpenalized','solver':'float64 SVD of H residualized against G; no post-residualization scaling','stationarity_threshold':1e-8,'train_self_inclusion':True,'versions':{'python':platform.python_version(),'numpy':np.__version__,'scipy':scipy.__version__,'sklearn':sklearn.__version__,'pandas':pd.__version__}}
    # JSON is valid YAML and preserves exact scalar types without a new dependency.
    save_json(OUT/'resolved_config.yaml',config)
    try:
        df,inner,features=load_inputs(args); synthetic(); base,baselog=reproduce(df)
        log('PREFLIGHT PASSED')
        if not args.preflight:
            run(df,inner,features,base,baselog)
            save_json(OUT/'logs/status.json',{'status':'fitting_complete','elapsed_seconds':time.time()-START,'pid':os.getpid()})
        else: save_json(OUT/'logs/status.json',{'status':'preflight_complete','elapsed_seconds':time.time()-START})
    except Exception:
        save_json(OUT/'logs/status.json',{'status':'failed','traceback':traceback.format_exc(),'elapsed_seconds':time.time()-START})
        raise
    finally:
        save_json(OUT/'validation_report.json',CHECKS)
        (OUT/'environment.txt').write_text(json.dumps(config['versions'],indent=2)+'\n'+platform.platform()+f'\nelapsed_seconds={time.time()-START}\npeak_rss_platform_units={resource.getrusage(resource.RUSAGE_SELF).ru_maxrss}\n')

if __name__=='__main__': main()
