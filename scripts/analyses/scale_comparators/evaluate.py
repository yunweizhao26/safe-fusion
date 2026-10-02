#!/usr/bin/env python3

import argparse
import csv
import subprocess
import json
from pathlib import Path
import sys

ROOT = (Path.cwd() / 'artifacts/paper_evidence/review_round4/scale_comparators')
REPO = ROOT.parents[3]
sys.path.insert(0, str(REPO / 'scripts'))
import numpy as np
import pandas as pd
import scale_evaluate as evaluator

parser = argparse.ArgumentParser()
parser.add_argument('--size', type=int, required=True)
parser.add_argument('--reproduce', action='store_true')
parser.add_argument('--core-only', action='store_true')
parser.add_argument('--include-scgpt', action='store_true')
args = parser.parse_args()
mode = 'reproduction' if args.reproduce else 'evaluation'
view = ROOT / mode / f'cells_{args.size}'
view.mkdir(parents=True, exist_ok=True)
original_unit = REPO / 'artifacts/paper_evidence/review_round3/scale' / f'cells_{args.size}'
names = ('masked','selector','methods','baselines','transductive','without_magic') if args.reproduce else ('masked','selector')
for name in names:
    target = view / name
    if not target.is_symlink():
        target.symlink_to(original_unit / name, target_is_directory=True)

work = ROOT / mode / str(args.size)
work.mkdir(parents=True, exist_ok=True)
if args.reproduce:
    (work/'PASS.json').unlink(missing_ok=True)
link = work / f'cells_{args.size}'
if not link.exists():
    link.symlink_to(ROOT / mode / f'cells_{args.size}', target_is_directory=True)
if not args.reproduce:
    evaluator.SELECTORS = {'Safe Fusion': 'selector', 'Selector on scVI': 'selector_scvi',
                           'Selector on MAGIC': 'selector_magic'}
    evaluator.METHODS = {name: f'fits/{slug}' for name, slug in
                         [('DCA','dca'),('scGPT','scgpt'),('EnImpute','enimpute'),
                          ('scImpute','scimpute'),('scRecover','screcover')]}
    if args.core_only:
        evaluator.METHODS = {'DCA': 'fits/dca'}
        if args.include_scgpt:
            evaluator.METHODS['scGPT'] = 'fits/scgpt'

    latest = {stage: job for stage, size, job in csv.reader((ROOT/'jobs.tsv').open(), delimiter='\t') if int(size)==args.size}
    raw = subprocess.check_output(['sacct','-n','-X','-P','-j',','.join(latest.values()),'--format=JobID,State'], text=True)
    states = {parts[0]:parts[1] for line in raw.splitlines() if len(parts:=line.split('|'))>=2}
    for name, relative in list(evaluator.METHODS.items()):
        if states.get(latest.get(relative.split('/')[-1],'')) != 'COMPLETED':
            evaluator.METHODS[name] = 'not_completed/' + relative
    for name, relative in list(evaluator.SELECTORS.items()):
        if name!='Safe Fusion' and states.get(latest.get(relative,'')) != 'COMPLETED':
            evaluator.SELECTORS[name] = 'not_completed/' + relative
    evaluator.RANKED_AS_STORED.add('frozen_scgpt_masked_value_score')
    unit = ROOT / mode / f'cells_{args.size}'
    for name in ('selector_scvi', 'selector_magic', 'fits'):
        target = unit / name
        if not target.is_symlink():
            target.symlink_to(ROOT / f'cells_{args.size}' / name, target_is_directory=True)
sys.argv = ['scale_evaluate.py', '--root', str(work), '--sizes', str(args.size), '--seed', '1729', '--draws', '2000']
evaluator.main()
if args.reproduce:
    checks = {}
    original = REPO / 'artifacts/paper_evidence/review_round3/scale/results'
    for filename, keys in [('paired_differences.csv', ['cells','reference','method','statistic']),
                           ('method_summary.csv', ['cells','method','statistic']),
                           ('masked_f1_curves.csv', ['cells','method','fraction'])]:
        expected = pd.read_csv(original / filename)
        expected = expected[expected.cells == args.size].sort_values(keys).reset_index(drop=True)
        actual = pd.read_csv(work / 'results' / filename).sort_values(keys).reset_index(drop=True)
        pd.testing.assert_frame_equal(actual, expected, check_exact=False, rtol=0, atol=1e-12)
        numeric = actual.select_dtypes(include='number').columns
        checks[filename] = {'rows':len(actual), 'max_absolute_error':float(np.max(np.abs(actual[numeric].to_numpy()-expected[numeric].to_numpy())))}
    expected = pd.read_parquet(original/'masked_f1_unit_counts.parquet')
    expected = expected[expected.group == f'cells_{args.size}'].reset_index(drop=True)
    actual = pd.read_parquet(work/'results/masked_f1_unit_counts.parquet')
    pd.testing.assert_frame_equal(actual, expected, check_exact=True)
    checks['masked_f1_unit_counts.parquet'] = {'rows':len(actual), 'exact':True}
    (work/'PASS.json').write_text(json.dumps(checks,indent=2)+'\n')
    print('S28 REPRODUCTION PASS', args.size, flush=True)
