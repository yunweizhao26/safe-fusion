
import sys
from pathlib import Path
from types import SimpleNamespace
import subprocess

import run


def evaluate_batches(cmd, check):
    methods = []
    common = []
    i = 0
    while i < len(cmd):
        if cmd[i] == '--method':
            methods.append(cmd[i + 1])
            i += 2
        else:
            common.append(cmd[i])
            i += 1
    at = common.index('--output-dir') + 1
    output = Path(common[at])
    output.mkdir(parents=True, exist_ok=False)
    frames = []
    reference = None
    for start in range(0, len(methods), 8):
        dest = output / f'batch_{start // 8:02d}'
        args = common.copy()
        args[at] = str(dest)
        args += [item for method in methods[start:start + 8] for item in ['--method', method]]
        run.dump(output / f'batch_{start // 8:02d}_command.json', args)
        subprocess.run(args, check=check)
        filename = 'donor_metrics.parquet' if (dest / 'donor_metrics.parquet').exists() else 'unit_metrics.parquet'
        frame = run.pd.read_parquet(dest / filename)
        baseline = frame[frame.method.isin(['corrupted_raw', 'reference_truth'])].reset_index(drop=True)
        if reference is None:
            reference = baseline
            frames.append(baseline)
        else:
            run.pd.testing.assert_frame_equal(baseline, reference, check_exact=True)
        frames.append(frame[~frame.method.isin(['corrupted_raw', 'reference_truth'])])
        print(f'{output}: batch {start // 8 + 1}/{(len(methods) + 7) // 8} complete', flush=True)
    run.pd.concat(frames, ignore_index=True).to_parquet(output / filename, index=False)


if __name__ == '__main__':

    run.subprocess = SimpleNamespace(run=evaluate_batches)
    run.unit(sys.argv[1], int(sys.argv[2]))
