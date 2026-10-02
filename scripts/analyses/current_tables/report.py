from tables import *
import re,hashlib
text=(O/'original_supplementary_results.tex').read_text()
labels={'S12':'value_accuracy','S13':'s_value','S14':'s_recorded','S15':'s_bias','S26':'s_rule'}
blocks={k:next(b for b in re.findall(r'\\begin\{table\}.*?\\end\{table\}',text,re.S) if '\\label{tab:'+v+'}' in b) for k,v in labels.items()}

def clean(s):
    s=s.replace('\\%','%').replace('$-$','-').replace('$','')
    s=re.sub(r'\\textbf\{([^{}]*)\}',r'\1',s)
    return s.strip()

def printed(k):
    rows=[]; body=blocks[k].split('\\midrule',1)[1].split('\\bottomrule')[0]
    for line in body.splitlines():
        if '&' in line and line.strip().endswith('\\\\'):
            rows.append([clean(c) for c in line.strip()[:-2].split('&')])
    return rows
old={k:printed(k) for k in labels}
checks=[]

def check(k,label,got,want):
    passed=got==want
    checks.append(dict(table=k,row=label,actual=got,printed=want,passed=passed))
    if not passed: print('PRINTED MISMATCH',k,label,got,want,flush=True)

def only(df,**query):
    for col,value in query.items(): df=df[df[col]==value]
    assert len(df)==1,(query,len(df))
    return df.iloc[0]

def read(kind,part,name): return pd.read_csv(O/kind/part/name)

def compare(a,b): return f'{a} / {b}'

def md(headers,rows):
    return '\n'.join(['| '+' | '.join(headers)+' |','| '+' | '.join(['---']*len(headers))+' |']+['| '+' | '.join(map(str,r))+' |' for r in rows])

def tex(k,rows):
    block=blocks[k]

    source_lines=block.splitlines(); out=[]; row_index=0
    for line in source_lines:
        if row_index<len(old[k]) and '&' in line and clean(line.split('&')[0])==old[k][row_index][0]:
            cells=rows[row_index]
            line=' & '.join(c.replace('%',r'\%') for c in cells)+r' \\'
            row_index+=1
        out.append(line)
    (O/'tables'/f'{k}_comparison.tex').write_text('\n'.join(out)+'\n')

(O/'tables').mkdir(exist_ok=True)
values=['safe_fusion','safe_fusion_linear','autoencoder_fusion_3teachers','autoencoder_fusion_5teachers','scvi_inductive','graph_smooth','svd_impute','magic_inductive','gene_median']
datasets=['pancreas','colon','norman_crispra']
masked={kind:{name:read(kind,'masked',name+'.csv') for name in ['log_error','error_removed','log_error_strata']} for kind in ['original','current']}
s12=[]
for row,value in zip(old['S12'],values):
    result=[row[0]]; recomputed=[]
    for metric,dec in [('error_removed',1),('log_error',3)]:
        for dataset in datasets:
            q=dict(dataset=dataset,value=value)
            if metric=='error_removed':q['fill_fraction']=.05
            previous=f'{only(masked["original"][metric],**q)[metric]:.{dec}f}'
            recomputed.append(previous)
            fresh='NR' if value.startswith('autoencoder') else f'{only(masked["current"][metric],**q)[metric]:.{dec}f}'
            result.append(compare(previous,fresh))
    check('S12',row[0],recomputed,row[1:]);s12.append(result)

s13=[];s13norm=[]
for i,row in enumerate(old['S13']):
    result=[row[0]]; norm=[row[0]]; recomputed=[]
    for dataset in ['colon','pancreas','norman_crispra']:
        for value in ['safe_fusion','safe_fusion_linear','autoencoder_fusion_5teachers']:
            metric='error_removed' if i<3 else 'log_error_strata'; field='error_removed' if i<3 else 'log_error';dec=1 if i<3 else 3
            q=dict(dataset=dataset,value=value)
            if i<3:q['fill_fraction']=[.01,.05,.1][i]
            else:q.update(stratum_kind='true_count' if i<7 else 'detection_quintile',stratum=['1','2-3','4-10','>10','1'][i-3])
            previous=f'{only(masked["original"][metric],**q)[field]:.{dec}f}'
            fresh='NR' if value.startswith('autoencoder') else f'{only(masked["current"][metric],**q)[field]:.{dec}f}'
            if dataset=='norman_crispra': norm.append(compare(previous,fresh))
            else: recomputed.append(previous);result.append(compare(previous,fresh))
    check('S13',row[0],recomputed,row[1:]);s13.append(result);s13norm.append(norm)

recorded={kind:read(kind,'recorded','recorded_zero_fills.csv') for kind in ['original','current']}
s14=[];s14tex=[]
for row,label in zip(old['S14'],['Pancreas','Colon','CRISPRa (Norman)','CRISPRi (Adamson)','Knockout (Dixit)','ECCITE-seq (Papalexi)','Zebrafish']):
    vals={}
    for kind,frame in recorded.items():
        if label not in set(frame.dataset):continue
        vals[kind]=[f'{only(frame,dataset=label,method="Safe Fusion",fill_fraction=f).value_q50:.2f}' for f in [.01,.05,.1]]
        vals[kind]+=[f'{only(frame,dataset=label,method="SVD",fill_fraction=.05).value_q50:.2f}']
        for method in ['Safe Fusion','SVD','All recorded zeros']:
            q=dict(dataset=label,method=method)
            if method!='All recorded zeros':q['fill_fraction']=.05
            vals[kind].append(f'{100*only(frame,**q)["expected_above_2_share"]:.1f}')
    check('S14',row[0],vals['original'],row[1:])
    r=[row[0]]+[compare(a,b) for a,b in zip(vals['original'],vals.get('current',['NR']*7))]
    s14tex.append(r)
    if 'current' in vals:s14.append(r)

bias_old=read('original','thinning','thinning_positive_bias.csv')
bias_new=read('current','thinning','bias.csv')
detect_old=read('original','detection_weighted','value_accuracy.csv')
s15=[]
old_keys=['colon_thinning_050','colon_thinning_025','pancreas_thinning_050','norman_thinning_050']
new_keys=['Colon, 50%','Colon, 25%','Pancreas, 50%','CRISPRa, 50%']
labels_values=['Boosted fused value','Linear combination','scVI teacher']
method_values=['safe_fusion','safe_fusion_linear','scvi_inductive']
for i,row in enumerate(old['S15']):
    result=[row[0]]; recomputed=[]
    scope='all positives' if i<3 else 'filled at 5%'; value=method_values[i%3] if i<6 else 'conditional_detection'
    for ok,nk in zip(old_keys,new_keys):
        if i<6:
            r=only(bias_old,dataset=ok,design='mask-trained',positives=scope,recorded_count='all')
            field=labels_values[i%3]+':bias_vs_expected_thinned_count'
            previous=f'{r[field]:.2f} [{r[field+"_low"]:.2f}, {r[field+"_high"]:.2f}]'
        else:
            r=only(detect_old,benchmark='thinning',dataset=nk.replace('CRISPRa','Norman'),statistic='bias_filled_5pct',value='conditional_detection')
            previous=f'{r.estimate:.2f}'
        recomputed.append(previous)
        r=only(bias_new,dataset=nk,scope=scope,value=value)
        fresh=f'{r.estimate:.2f} [{r.lower:.2f}, {r.upper:.2f}]'
        result.append(compare(previous,fresh))
    check('S15',row[0],recomputed,row[1:]);s15.append(result)
s15all=['Detection-weighted, all thinned entries']
for nk in new_keys:
    r=only(bias_new,dataset=nk,scope='all positives',value='conditional_detection')
    previous=only(detect_old,benchmark='thinning',dataset=nk.replace('CRISPRa','Norman'),statistic='bias_all',value='conditional_detection')
    s15all.append(f'{previous.estimate:.2f} [{previous.lower:.2f}, {previous.upper:.2f}] / {r.estimate:.2f} [{r.lower:.2f}, {r.upper:.2f}]')


def rule_reports(kind,key):
    dirname='rule_rebuilt' if kind=='original' and key=='norman_crispra' else 'rule'
    return [json.loads((O/kind/dirname/design/key/'calibration_report.json').read_text()) for design in ['masked','recorded']]
def rule_values(reports):
    m,r=reports;a=m['detection_rule']
    return [100*a[f] for f in ['test_fill_fraction','test_masked_f1','best_fill_fraction_in_hindsight','best_masked_f1_in_hindsight']]+[100*r['detection_rule']['test_fill_fraction']]
for row,key in zip(old['S26'],['colon','pancreas_0','pancreas_1','pancreas_2','norman_crispra']):
    check('S26',row[0],[f'{v:.1f}' for v in rule_values(rule_reports('original',key))],row[1:])
s26=[];rule_rows=[]
current_reports={u.key:rule_reports('current',u.key) for u in units()}
def add_rule(label,vals,previous=None):
    fresh=[f'{v:.1f}' for v in vals]
    s26.append([label]+[compare(a,b) for a,b in zip(previous or ['—']*5,fresh)])
    rule_rows.append(dict(dataset=label,chosen=vals[0],f1=vals[1],best_fraction=vals[2],best_f1=vals[3],recorded_filled=vals[4]))
def pooled(keys,reports=current_reports):
    ms=[reports[k][0] for k in keys];rs=[reports[k][1] for k in keys]
    n=sum(m['test']['n_zeros'] for m in ms);positives=sum(m['test']['n_masked_positives'] for m in ms)
    chosen=sum(m['detection_rule']['test_n_selected'] for m in ms)
    hits=sum(m['detection_rule']['test_masked_recall']*m['test']['n_masked_positives'] for m in ms)
    grid=[]
    for i in range(1000):
        selected=sum(m['exact_budget_curve'][i]['n_selected'] for m in ms)
        true=sum(m['exact_budget_curve'][i]['n_true_positive'] for m in ms)
        grid.append((selected/n,2*true/(selected+positives)))
    best=max(grid,key=lambda x:x[1]);rn=sum(r['test']['n_zeros'] for r in rs);rc=sum(r['detection_rule']['test_n_selected'] for r in rs)
    return [100*chosen/n,200*hits/(chosen+positives),100*best[0],100*best[1],100*rc/rn]
add_rule('Colon, pooled',pooled(['colon_0','colon_1','colon_2']),old['S26'][0][1:])
for i in range(3):add_rule(f'Colon, fold {i+1}',rule_values(current_reports[f'colon_{i}']))
pan_keys=['pancreas_0','pancreas_1','pancreas_2']
pan_old={k:rule_reports('original',k) for k in pan_keys}
add_rule('Pancreas, pooled',pooled(pan_keys),[f'{v:.1f}' for v in pooled(pan_keys,pan_old)])
for i in range(3):add_rule(f'Pancreas, fold {i+1}',rule_values(current_reports[f'pancreas_{i}']),old['S26'][i+1][1:])
add_rule('Norman CRISPRa',rule_values(current_reports['norman_crispra']),old['S26'][4][1:])
pd.DataFrame(rule_rows).to_csv(O/'current/rule/summary.csv',index=False)
(O/'printed_value_checks.json').write_text(json.dumps(checks,indent=2)+'\n')
failures=[c for c in checks if not c['passed']]
assert not failures,failures

def bold_pairs(rows,columns,maximize):
    for col in columns:
        for side in [0,1]:
            candidates=[]
            for i,r in enumerate(rows):
                cell=r[col].split(" / ")[side]
                try: candidates.append((float(cell),i))
                except ValueError: pass
            best=(max if maximize else min)(v for v,i in candidates)
            for v,i in candidates:
                if v==best:
                    parts=rows[i][col].split(" / "); parts[side]="**"+parts[side]+"**"; rows[i][col]=" / ".join(parts)
bold_pairs(s12,range(1,4),True)
bold_pairs(s12,range(4,7),False)
for collection,offsets in [(s13,[1,4]),(s13norm,[1])]:
    for i,row in enumerate(collection):
        for offset in offsets:
            transposed=[["",row[j]] for j in range(offset,offset+3)]
            bold_pairs(transposed,[1],i<3)
            for j,r in zip(range(offset,offset+3),transposed):row[j]=r[1]

headers={
'S12':['Value','Pancreas error removed (%)','Colon error removed (%)','CRISPRa error removed (%)','Pancreas error','Colon error','CRISPRa error'],
'S13':['','Colon boosted','Colon linear','Colon autoencoder','Pancreas boosted','Pancreas linear','Pancreas autoencoder'],
'S14':['Dataset','Safe Fusion median 1%','Safe Fusion median 5%','Safe Fusion median 10%','SVD median 5%','Teacher mean above 2: Safe Fusion (%)','Teacher mean above 2: SVD (%)','Teacher mean above 2: all zeros (%)'],
'S15':['Value and entries','Colon, 50%','Colon, 25%','Pancreas, 50%','Norman, 50%'],
'S26':['Dataset','Chosen (%)','F1','Best (%)','Best F1','Recorded zeros filled (%)']}
tables={'S12':s12,'S13':s13,'S14':s14,'S15':s15,'S26':s26}
for k,rows in tables.items():
    (O/'tables'/f'{k}.md').write_text(md(headers[k],rows)+'\n')

for k,rows in [('S12',s12),('S13',s13),('S14',s14tex),('S15',s15)]:tex(k,rows)
p=O/'tables/S14_comparison.tex';p.write_text('\n'.join(x for x in p.read_text().splitlines() if not x.startswith('Dixit knockout &'))+'\n')
s26tex=blocks['S26'];start=s26tex.index('\\midrule')+len('\\midrule');end=s26tex.index('\\bottomrule')
s26tex=s26tex[:start]+'\n'+'\n'.join(r[0]+' & '+' & '.join(r[1:])+r' \\' for r in s26)+'\n'+s26tex[end:]
(O/'tables/S26_comparison.tex').write_text(s26tex)

for path in (O/'tables').glob('*.tex'):
    s=path.read_text().replace('for the inductive mode on the earlier pancreas and colon benchmarks','old inductive / current transductive values').replace('for the inductive mode on the earlier benchmarks','old inductive / current transductive values')
    s=s.replace('in the inductive mode on the earlier benchmarks','with inductive teachers (old) or transductive teachers (new)')
    s=s.replace('except for the detection-weighted value in the last row','except for the old detection-weighted value in the last row')
    s=re.sub(r'\*\*([^*]+)\*\*',lambda m:r'\textbf{'+m.group(1)+'}',s)

    rendered=[]
    for line in s.splitlines():
        if '&' in line and ' / ' in line and line.rstrip().endswith(r'\\'):
            cells=line.rstrip()[:-2].split('&')
            cells=[r'\shortstack{'+c.strip().replace(' / ',r' \\ ')+r'}' if ' / ' in c else c.strip() for c in cells]
            line=' & '.join(cells)+r' \\'
        rendered.append(line)
    s='\n'.join(rendered)+'\n'
    s=s.replace('\\caption{','\\caption{Each numeric cell shows old above new; NR means not rerun. ',1)
    path.write_text(s)

intro='''# Current supplementary tables

Completed S12, S13, S14, S15 and S26 for transductive Safe Fusion on the rebuilt pancreas folds, all three colon folds (34 held-out donors), and rebuilt Norman CRISPRa. Each comparison cell is **old inductive / new transductive**; old values are the printed supplement values, not the more recent inductive benchmarks. Changed benchmarks prevent interpreting these differences as isolated effects of transduction.

The original evaluators reproduced all 204 printed numeric cells in the five tables. The final audit passed 113 checks. Full-precision parity receipts are under `original/`; `printed_value_checks.json` checks rounding against the read-only supplement snapshot. Printed Norman values already use rebuilt inputs, despite the broad historical captions and task background; `slurm_norman_rebuilt_value.sh` supplies S12, and the saved `norman_rebuilt` detection-rule inputs supply S26. The original detection-weighted S15 row comes from the later value analysis and uses cross-fit colon. These provenance differences are retained explicitly.

Definitions, fitting cells, settings and seeds are unchanged. Where intervals are part of the evaluator, resampling uses 2,000 draws, seed 1729, over donors or perturbation targets; point-only source tables remain point-only. S12/S13 use fixed Safe Fusion selections for every candidate value. Selectors run on Slurm partition `cs`; all jobs use account `torch_pr_634_general`. No GPU is needed: existing teachers and boosted values are reused, and only linear values and calibration selectors are fitted.

Paths below are relative to `artifacts/paper_evidence/review_round4/current_tables/`. `tables/S*_comparison.tex` preserves the source tabular layouts; Markdown tables retain their column order and displayed precision. Extended Norman S13, detection-weighted-all S15, and pooled/per-fold S26 rows are labelled below.
'''
designs={
'S12':'''At the zeros selected by Safe Fusion at a 5% fill fraction, substitute each candidate value and calculate the percentage of the total masked-positive log error removed. Separately, average absolute log error over all held-out masked positives, pooling folds by entries exactly as `evaluate_value_accuracy.py` does.''',
'S13':'''Use the same fixed selections and error-removed definition at 1%, 5% and 10% fill fractions. Stratify all held-out masked positives by true count and by the bottom gene-detection quintile computed on development plus validation cells, using the original ranking and binning.''',
'S14':'''Use recorded counts of unmasked held-out cells, with teachers fitted to the hybrid input that retains training-cell masks. Report the pooled inserted-count medians and the fraction whose five-teacher mean exceeds two, using each method’s own fixed-budget selections and the original `evaluate_inserted_value.py` definitions.''',
'S15':'''Use the mask-trained models at entries lost to binomial thinning, with signed bias log(1 + value) − log(1 + p×original count). Pool all test thinning positives or the subset filled at 5%, and use the original donor/target percentile bootstrap; detection weighting multiplies the boosted value by the cross-fitted calibrated detection probability.''',
'S26':'''Refit the production selector and its five-fold isotonic calibration on fitting cells, then apply d = p/[p + 0.1(1 − p)] > 1/2 without using test labels to select the fraction. Report masked F1 and the best F1 on the original 1,000-point grid from 0.1% to 100%, then independently apply the rule to unmasked held-out recorded zeros.'''}
paths={
'S12':'`current/masked/{error_removed,log_error}.csv`; fits in `models/masked/`; source evaluator `scripts/evaluate_value_accuracy.py` and launcher `scripts/slurm_value_accuracy.sh`.',
'S13':'`current/masked/{error_removed,log_error_strata}.csv`; same evaluator and launcher as S12.',
'S14':'`current/recorded/recorded_zero_fills.csv`, `current/recorded/inputs.json`, `recorded_views/`; source `scripts/evaluate_inserted_value.py` and `scripts/slurm_inserted_value.sh`.',
'S15':'`current/thinning/bias.csv`; linear fits in `models/thinning/`; source `scripts/evaluate_inserted_value.py` (`thinning_positive_bias`), plus `scripts/v2_value_evaluate.py` for the original detection-weighted row.',
'S26':'`current/rule/summary.csv` and `current/rule/{masked,recorded}/<unit>/calibration_report.json`; source `scripts/calibrated_selective_fill.py` and `scripts/slurm_detection_rule.sh`.'}
commands={'S12':'masked','S13':'masked','S14':'extra recorded','S15':'extra thinning','S26':'extra rule'}
report=intro
for k in labels:
    report+=f'\n## {k} (`tab:{labels[k]}`)\n\n'+designs[k]+'\n\n'+md(headers[k],tables[k])+'\n\nPaths: '+paths[k]+'\n'
    if k=='S15':report+='\nExecution recovery: the slow pancreas thinning linear fit was retried unchanged and completed in 32 seconds. No bias-evaluator failure occurred.\n'
    if k=='S26':report+='\nFailures: none in the current calibration runs; original-source reproduction passed after selecting the rebuilt Norman inputs.\n'
    if k in ['S12','S13']:
        report+='\nNot rerun: both autoencoder variants. The unchanged five-teacher runner rejects contracts without `fitting_cell_proposals_exclude_own_counts`; transductive teachers correctly set this false. The three-teacher mode recomputes inductive teachers internally and exposes no transductive option. No metadata or runner semantics were changed to bypass these checks.\n'
    if k=='S13':report+='\nNorman extension (old / new; old values come from the reproduced source CSV, since S13 did not print Norman cells):\n\n'+md(['','Boosted','Linear','Autoencoder'],s13norm)+'\n'
    if k=='S14':report+='\nAdamson, Papalexi and zebrafish transductive inputs exist and are included. Dixit is omitted because no complete transductive deployment input/teacher/selector chain was found. SVD uses its transductive teacher on the same hybrid input; all five proposals are clipped at zero before taking their mean, as in the source evaluator.\n'
    if k=='S15':report+='\nRequested extension absent from the original table (old values from the reproduced source CSV):\n\n'+md(headers[k],[s15all])+'\n\nUnlike the printed detection-weighted row, all new detection-weighted estimates include donor/target bootstrap intervals. Colon 25% calibration is newly fitted; the saved 50% calibrations are reused after candidate and selection checks.\n'
    if k=='S26':report+='\nColon fold rows have no direct old-split counterpart. Pooled tissue rows aggregate selected counts and true positives; their hindsight optimum uses the same common fraction grid, while each fold retains its independently calibrated rule. The old pooled pancreas row is computed from the reproduced fold outputs and was not printed in S26.\n'
    array=' --array=0-16%5' if k=='S26' else ''
    report+=f'\nReproduce after prerequisites: `sbatch -A torch_pr_634_general -p cs{array} -o "$O/logs/reproduce-{k}-%A_%a.log" "$O/code/run.sh" {commands[k]}`.\n'
report+='''
## Reproduction and checks

Run from `the repository root`. Set `O=artifacts/paper_evidence/review_round4/current_tables`. The original artifacts referenced by the input manifests must remain available. These commands rerun within this output directory.

```bash
# Original-input reproduction (must pass before current evaluation).
sbatch -A torch_pr_634_general -p cs -o "$O/logs/reproduce-old-masked-%j.log" "$O/code/run.sh" original masked
sbatch -A torch_pr_634_general -p cs -o "$O/logs/reproduce-old-recorded-%j.log" "$O/code/run.sh" original recorded
sbatch -A torch_pr_634_general -p cs -o "$O/logs/reproduce-old-thinning-%j.log" "$O/code/run.sh" original thinning
sbatch -A torch_pr_634_general -p cs -o "$O/logs/reproduce-old-detection-%j.log" "$O/code/run.sh" extra old_detection
sbatch -A torch_pr_634_general -p cs --array=0-3,5-8%5 -o "$O/logs/reproduce-old-rule-%A_%a.log" "$O/code/old_rule.sh"
sbatch -A torch_pr_634_general -p cs --array=4,9 -o "$O/logs/reproduce-old-rule-rebuilt-%A_%a.log" "$O/code/old_rule_norman.sh"
# After the four original value/recorded/bias jobs finish, check their printed cells.
sbatch -A torch_pr_634_general -p cs -o "$O/logs/reproduce-old-printed-%j.log" "$O/code/run.sh" original_print
# Current linear fits: seven masked units, then ten mask-trained thinning units.
sbatch -A torch_pr_634_general -p cs --array=0-16%6 -o "$O/logs/reproduce-linear-%A_%a.log" "$O/code/run.sh" linear
# After the original rule jobs finish, fit current masked/recorded calibrations and colon 25% calibrations.
sbatch -A torch_pr_634_general -p cs --array=0-16%5 -o "$O/logs/reproduce-rule-%A_%a.log" "$O/code/run.sh" extra rule
# Wait for prerequisites with 600-second shell sleeps, then run per-table commands above.
# Final report and exact displayed-value checks:
sbatch -A torch_pr_634_general -p cs -o "$O/logs/reproduce-report-%j.log" "$O/code/run.sh" report
```

The masked input manifests are `review_round2/leakage_free/units_manifest.json` and `review_round3/colon_crossfit/units_manifest.json` relative to `artifacts/paper_evidence/`; current teacher/selector locations follow `review_round4/transductive_main/code/common.py`. Full invoked argument lists are in `commands.jsonl`; submitted jobs are in `jobs.tsv`; accounting and validation are in `accounting.tsv` and `audit.json`. Source-analysis READMEs are `review_round4/transductive_main/README.md`, `review_round4/transductive_downstream/README.md`, `review_round3/colon_crossfit/README.md`, and `review_round3/value_v2_ablations/README.md`; older source launchers retain the original commands where no round-2 README exists.

## Failures and remaining work

Autoencoder reruns are intentionally unavailable under the unchanged-runner requirement. Dixit S14 is omitted for missing transductive inputs. The initial pre-rebuild Norman source probe failed the printed-value check; locating and rerunning the actual rebuilt Norman source resolved it. A report submission referencing a completed/purged Slurm dependency was rejected and replaced, and two pending evaluation/report jobs were replaced to release S12/S13 from an unrelated thinning-fit dependency. One pancreas thinning linear fit was cancelled after over 54 minutes without a contract and retried unchanged; cluster policy rejected an explicit node exclusion, so the retry used normal placement. No requested supported table entry remains unfinished. `audit.json`, `execution_notes.json`, `dependency_recovery.txt` and `accounting.tsv` retain the details. Existing paper files, documentation, repository README, and scmp-*, cx-* and mm3-* jobs were not changed; no commit or push was made.
'''
(O/'README.md').write_text(report)
print(report)
