from tables import *
import re
text=(O/'original_supplementary_results.tex').read_text()
def rows(label):
    block=next(b for b in re.findall(r'\\begin\{table\}.*?\\end\{table\}',text,re.S) if '\\label{tab:'+label+'}' in b)
    out=[]
    for line in block.split('\\midrule',1)[1].split('\\bottomrule')[0].splitlines():
        if '&' not in line:continue
        line=re.sub(r'\\textbf\{([^{}]*)\}',r'\1',line).replace('$-$','-').replace('$','').replace('\\%','%')
        out.append([c.strip() for c in line.strip()[:-2].split('&')])
    return out

def one(f,**kw):
    for k,v in kw.items():f=f[f[k]==v]
    assert len(f)==1,(kw,len(f))
    return f.iloc[0]
checks=[]
def check(table,label,a,b):
    checks.append(dict(table=table,row=label,actual=a,printed=b,passed=a==b))
    if a!=b: print('PRINTED MISMATCH',table,label,a,b,flush=True)
values=['safe_fusion','safe_fusion_linear','autoencoder_fusion_3teachers','autoencoder_fusion_5teachers','scvi_inductive','graph_smooth','svd_impute','magic_inductive','gene_median']
m={name:pd.read_csv(O/'original/masked'/(name+'.csv')) for name in ['error_removed','log_error','log_error_strata']}
for row,value in zip(rows('value_accuracy'),values):
    actual=[]
    for metric,dec in [('error_removed',1),('log_error',3)]:
        for dataset in ['pancreas','colon','norman_crispra']:
            kw=dict(dataset=dataset,value=value)
            if metric=='error_removed':kw['fill_fraction']=.05
            actual.append(f'{one(m[metric],**kw)[metric]:.{dec}f}')
    check('S12',row[0],actual,row[1:])
for i,row in enumerate(rows('s_value')):
    actual=[]
    for dataset in ['colon','pancreas']:
        for value in ['safe_fusion','safe_fusion_linear','autoencoder_fusion_5teachers']:
            metric='error_removed' if i<3 else 'log_error_strata';field='error_removed' if i<3 else 'log_error';dec=1 if i<3 else 3
            kw=dict(dataset=dataset,value=value)
            if i<3:kw['fill_fraction']=[.01,.05,.1][i]
            else:kw.update(stratum_kind='true_count' if i<7 else 'detection_quintile',stratum=['1','2-3','4-10','>10','1'][i-3])
            actual.append(f'{one(m[metric],**kw)[field]:.{dec}f}')
    check('S13',row[0],actual,row[1:])
f=pd.read_csv(O/'original/recorded/recorded_zero_fills.csv')
for row,label in zip(rows('s_recorded'),['Pancreas','Colon','CRISPRa (Norman)','CRISPRi (Adamson)','Knockout (Dixit)','ECCITE-seq (Papalexi)','Zebrafish']):
    actual=[f'{one(f,dataset=label,method="Safe Fusion",fill_fraction=b).value_q50:.2f}' for b in [.01,.05,.10]]
    actual+=[f'{one(f,dataset=label,method="SVD",fill_fraction=.05).value_q50:.2f}']
    for method in ['Safe Fusion','SVD','All recorded zeros']:
        kw=dict(dataset=label,method=method)
        if method!='All recorded zeros':kw['fill_fraction']=.05
        actual.append(f'{100*one(f,**kw)["expected_above_2_share"]:.1f}')
    check('S14',row[0],actual,row[1:])
f=pd.read_csv(O/'original/thinning/thinning_positive_bias.csv');d=pd.read_csv(O/'original/detection_weighted/value_accuracy.csv')
for i,row in enumerate(rows('s_bias')):
    actual=[]
    for key,nk in zip(['colon_thinning_050','colon_thinning_025','pancreas_thinning_050','norman_thinning_050'],['Colon, 50%','Colon, 25%','Pancreas, 50%','Norman, 50%']):
        if i<6:
            r=one(f,dataset=key,design='mask-trained',positives='all positives' if i<3 else 'filled at 5%',recorded_count='all')
            field=['Boosted fused value','Linear combination','scVI teacher'][i%3]+':bias_vs_expected_thinned_count'
            actual.append(f'{r[field]:.2f} [{r[field+"_low"]:.2f}, {r[field+"_high"]:.2f}]')
        else:actual.append(f'{one(d,benchmark="thinning",dataset=nk,statistic="bias_filled_5pct",value="conditional_detection").estimate:.2f}')
    check('S15',row[0],actual,row[1:])
(O/'original/printed_checks.json').write_text(json.dumps(checks,indent=2)+'\n')
print('Original printed checks',sum(len(c['printed']) for c in checks),'mismatched rows',sum(not c['passed'] for c in checks))
assert all(c['passed'] for c in checks), 'Printed-value mismatch; see original/printed_checks.json'
