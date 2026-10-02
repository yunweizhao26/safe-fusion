from tables import *
keys=['colon','pancreas_0','pancreas_1','pancreas_2','norman_crispra']
printed=np.array([[13.3,44.8,4.8,57.3,10.9],[11.4,53.7,4.6,66.2,8.2],[11.8,52.9,4.6,65.3,8.7],[13.1,53.3,5.3,66.6,9.6],[2.2,37.5,.7,48.4,1.9]])
fields=['test_fill_fraction','test_masked_f1','best_fill_fraction_in_hindsight','best_masked_f1_in_hindsight']
rows=[]
for key,want in zip(keys,printed):
    local=O/'original'/('rule_rebuilt' if key=='norman_crispra' else 'rule')
    reports={kind:json.loads((local/kind/key/'calibration_report.json').read_text()) for kind in ['masked','recorded']}
    got=[100*reports['masked']['detection_rule'][f] for f in fields]+[100*reports['recorded']['detection_rule']['test_fill_fraction']]
    assert [f'{x:.1f}' for x in got]==[f'{x:.1f}' for x in want],(key,got,want)
    for kind,report in reports.items():
        oldpath=E/'detection_rule'/key if kind=='masked' else E/'downstream_deployment'/('pancreas/fold_'+key[-1] if key.startswith('pancreas') else key)/'detection_rule'
        if key=='norman_crispra':
            oldpath=E/'review_round2/norman_rebuilt'/('detection_rule/norman_crispra' if kind=='masked' else 'deployment/norman_crispra/detection_rule')
        old=json.loads((oldpath/'calibration_report.json').read_text())
        for f in fields if kind=='masked' else ['test_fill_fraction']:
            assert np.isclose(old['detection_rule'][f],report['detection_rule'][f],atol=1e-12,rtol=0),(key,kind,f)
    rows.append(dict(unit=key,printed_match=True,values=got))
(O/'original/rule/parity.json').write_text(json.dumps(rows,indent=2)+'\n')
