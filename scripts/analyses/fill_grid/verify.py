import csv
import hashlib
import json
import math
import re
import struct
from pathlib import Path

out = (Path.cwd() / 'artifacts/paper_evidence/review_round4/fill_grid')
def readcsv(path, **kwargs):
    with path.open() as f:
        return list(csv.DictReader(f, **kwargs))

rows = readcsv(out/'fill_grid_long.csv')
assert len(rows) == 392
keys = {(r['dataset'], r['endpoint'], r['method'], float(r['fill_pct'])) for r in rows}
assert len(keys) == 392
assert len({k[:2] for k in keys}) == 7
assert len({k[2] for k in keys}) == 8
assert {k[3] for k in keys} == {0.25, 0.5, 1, 2, 3, 5, 10}
for r in rows:
    for k in ['error_reduction_x100', 'ci_low_x100', 'ci_high_x100']:
        assert math.isfinite(float(r[k]))
    assert float(r['ci_low_x100']) <= float(r['ci_high_x100'])
assert all(r['exact_match'] == 'True' for r in readcsv(out/'table3_printed_parity.csv'))
checks = [r for p in out.glob('*/*/fill_checks.csv') for r in readcsv(p)]
assert len(checks) == 448
assert sum(r['saved_fill_bitwise_equal'] == 'True' for r in checks) == 192
assert all(r['saved_fill_bitwise_equal'] != 'False' for r in checks)
source_hashes = {}
for p in out.glob('*/*/source_sha256.json'):
    for source, digest in json.loads(p.read_text()).items():
        assert source_hashes.setdefault(source, digest) == digest, source
for line in (out/'source_sha256.txt').read_text().splitlines():
    digest, filename = line.split('  ', 1)
    assert hashlib.sha256(Path(filename).read_bytes()).hexdigest() == digest, filename
jobs = readcsv(out/'slurm_accounting.tsv', delimiter='|')
assert all(r['State'] == 'COMPLETED' and r['ExitCode'] == '0:0' for r in jobs)
assert len([r for r in jobs if '.' not in r['JobID']]) == 18
png = (out/'fill_grid.png').read_bytes()
assert png.startswith(b'\x89PNG\r\n\x1a\n')
width, height = struct.unpack('>II', png[16:24])
assert (width, height) == (3120, 1464)
pos = 8
while pos < len(png):
    length = struct.unpack('>I', png[pos:pos+4])[0]
    if png[pos+4:pos+8] == b'pHYs':
        xppm, yppm, unit = struct.unpack('>IIB', png[pos+8:pos+8+length])
        assert unit == 1 and abs(xppm*0.0254-240) < 0.01 and xppm == yppm
        break
    pos += length + 12
else:
    raise AssertionError('PNG lacks resolution metadata')
pdf = (out/'fill_grid.pdf').read_bytes()
assert pdf.startswith(b'%PDF-') and pdf.rstrip().endswith(b'%%EOF')
assert len(re.findall(rb'/Type\s*/Page\b', pdf)) == 1
assert b'/Count 1' in pdf and b'/MediaBox' in pdf and b'startxref' in pdf
receipt = dict(passed=True, rows=392, filled_matrices=448, saved_matrices_bitwise_equal=192, completed_jobs=18, sources_stable_between_gate_and_grid=len(source_hashes), source_code_unchanged=True, png_dimensions=[width, height], png_dpi=xppm*0.0254, pdf_pages=1, pdf_check='structure only; PNG visually inspected', ancillary_tool_failures=['pdfinfo not installed', '.venv-pdf/bin/python has a broken symlink'])
(out/'verification.json').write_text(json.dumps(receipt, indent=2)+'\n')
print(json.dumps(receipt, indent=2))
