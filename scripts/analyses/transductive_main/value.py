from common import *
import json
from dataclasses import replace
import anndata as ad
import numpy as np
import pandas as pd
import v2_value_evaluate as v
from thinning import specs

def masked():
    rows = []
    original_filled = v.filled
    for dataset in ['Pancreas','Colon','CRISPRa']:
        frames = {m:[] for m in ['inductive','transductive']}
        for u in [u for u in units() if u.dataset == dataset]:
            frame = v.masked_unit_frame(u,['conditional'],OUT)
            data = ad.read_h5ad(u.corrupted)
            counts = data.layers['corrupted_counts']
            counts = counts.toarray() if hasattr(counts,'toarray') else np.asarray(counts)
            split = pd.read_parquet(u.splits).set_index('cell_id').loc[data.obs_names.astype(str),'split'].to_numpy()
            coords = pd.read_parquet(u.coordinates)
            keep = split[coords.cell_index.to_numpy()] == 'test'
            rr,cc = coords.cell_index.to_numpy()[keep],coords.gene_index.to_numpy()[keep]
            detection=(counts[np.isin(split,['development','validation'])]>0).mean(0)
            quintile=np.minimum((pd.Series(detection).rank(pct=True).to_numpy()*5).astype(int),4)
            frame['least_detected'] = quintile[cc] == 0
            frames['inductive'].append(frame)
            testrows,testcols=np.where((counts==0)&(split=='test')[:,None])
            scores=np.load(fusion_root(u)/'selectors'/u.key/'safe_fusion_transductive/test_scores.npy')
            rank=np.empty(len(scores),dtype=np.int64)
            rank[np.argsort(-scores,kind='stable')]=np.arange(len(scores))
            positions=np.searchsorted(testrows*counts.shape[1]+testcols,rr*counts.shape[1]+cc)
            trans=frame.copy()
            trans['value:conditional']=np.asarray(np.load(fusion_root(u)/'transductive'/u.key/'safe_fusion/mean.npy',mmap_mode='r')[rr,cc],dtype=np.float64)
            for fraction in v.FRACTIONS:
                trans[f'selected:{fraction}']=rank[positions]<max(1,int(round(fraction*len(scores))))
            frames['transductive'].append(trans)
        for model,parts in frames.items():
            frame=pd.concat(parts,ignore_index=True)
            stats=v.masked_statistics()
            stats['log_error_least_detected']=lambda f,x:(np.abs(v.log_value(f,x)-np.log1p(f['count'].to_numpy()))*f.least_detected.to_numpy(),f.least_detected.to_numpy().astype(float))
            rows += v.summarize(frame,stats,['conditional'],2000,1729,dict(benchmark='masked',dataset=dataset,model=model))
    result=pd.DataFrame(rows)
    old=pd.read_csv(E/'review_round3/value_v2_ablations/value/test/value_accuracy.csv')
    joined=old[(old.benchmark=='masked')&(old.value=='conditional')].merge(result[result.model=='inductive'],on=['dataset','statistic'],suffixes=('_old','_new'))
    error=max(float(np.max(np.abs(joined[c+'_old']-joined[c+'_new']))) for c in ['estimate','lower','upper'])
    assert len(joined)==24 and error<5.01e-6,(len(joined),error)
    dest=OUT/'value_accuracy'
    dest.mkdir(exist_ok=True)
    result.to_csv(dest/'masked.csv',index=False)
    (dest/'masked_parity.json').write_text(json.dumps(dict(rows=len(joined),max_difference=error),indent=2)+'\n')

def thinning():
    rows=[]
    frames={}
    for u,key,root,p in specs():
        for model in ['inductive','transductive']:
            for design in ['mask-trained','thinning-trained']:
                source=root/('mask_trained' if design=='mask-trained' else 'thinning_trained')/key
                modelroot=source if model=='inductive' else OUT/'thinning'/key/design
                inp=root/'mask_trained'/key/'input'
                data=ad.read_h5ad(inp/'hybrid.h5ad')
                splits=pd.read_parquet(u.splits).set_index('cell_id').loc[data.obs_names.astype(str),'split'].to_numpy()
                coord=pd.read_parquet(inp/'coordinates.parquet')
                keep=splits[coord.cell_index.to_numpy()]=='test'
                rr,cc=coord.cell_index.to_numpy()[keep],coord.gene_index.to_numpy()[keep]
                x=coord.original_value.to_numpy()[keep].astype(float)
                groups=data.obs[u.unit_column].astype(str).to_numpy()[rr]
                if u.dataset!='CRISPRa': groups=np.char.add('fold_'+u.key[-1]+':',groups.astype(str))
                frame=pd.DataFrame(dict(group=groups,count=x,expected_count=p*x,selected=v.filled(modelroot/'selector',.05,rr,cc)))
                frame['value:conditional']=np.asarray(np.load(modelroot/'safe_fusion/mean.npy',mmap_mode='r')[rr,cc],dtype=float)
                frames.setdefault((f'{u.dataset}, {int(p*100)}%',model,design),[]).append(frame)
    for (dataset,model,design),parts in frames.items():
        frame=pd.concat(parts,ignore_index=True)
        rows+=v.summarize(frame,v.thinning_statistics(),['conditional'],2000,1729,dict(benchmark='thinning',dataset=dataset,model=model,design=design))
    result=pd.DataFrame(rows)

    old=pd.read_csv(E/'review_round3/value_v2_ablations/value/test/value_accuracy.csv')
    old['dataset']=old.dataset.replace({'Norman, 50%':'CRISPRa, 50%'})
    original=result[(result.model=='inductive')&(result.design=='mask-trained')&(~result.dataset.str.startswith('Pancreas'))]
    joined=old[(old.benchmark=='thinning')&(old.value=='conditional')].merge(original,on=['dataset','statistic'],suffixes=('_old','_new'))
    error=max(float(np.max(np.abs(joined[c+'_old']-joined[c+'_new']))) for c in ['estimate','lower','upper'])
    assert len(joined)==18 and error<5.01e-6,(len(joined),error)
    result.to_csv(OUT/'value_accuracy/thinning.csv',index=False)
    (OUT/'value_accuracy/thinning_parity.json').write_text(json.dumps(dict(rows=len(joined),max_difference=error),indent=2)+'\n')

if __name__=='__main__':
    {'masked':masked,'thinning':thinning}[sys.argv[1]]()
