import hashlib
import json
from pathlib import Path

out = (Path.cwd() / 'artifacts/paper_evidence/review_round4/label_baselines')
paths = [out / 'screens/verification_receipt.json',
         out / 'sex/pancreas/gate.json', out / 'sex/colon/gate.json']
receipts = [json.loads(path.read_text()) for path in paths]
assert receipts[0]['saved_evaluators_match']
assert all(receipt['passed'] for receipt in receipts[1:])
assert sum(receipt['rows'] for receipt in receipts[1:]) == 48
result = {
    'passed': True,
    'scope': 'Saved evaluator results reproduce; manuscript differences remain explicitly reported.',
    'S16_paper_exact': receipts[0]['status'] == 'passed',
    'S16_paper_mismatches': receipts[0].get('paper_mismatches', []),
    'S18_paper_exact': True,
    'S16_numeric_cells': receipts[0]['paper_cells'],
    'S18_estimate_interval_triples': 48,
    'draws': 2000,
    'seed': 1729,
    'receipts': {str(path.relative_to(out)): hashlib.sha256(path.read_bytes()).hexdigest()
                 for path in paths},
}
(out / 'verification_passed.json').write_text(json.dumps(result, indent=2) + '\n')
print(json.dumps(result, indent=2))
