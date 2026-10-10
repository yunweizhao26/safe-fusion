from pathlib import Path
import sys
root=Path.cwd();sys.path[:0]=[str(root/'scripts'),str(root/'src')]
source=root/'scripts/calibrated_selective_fill.py';code=source.read_text();needle='        test_detection = detection_probability(test_probability, rho)';assert code.count(needle)==1
code=code.replace(needle,needle+'\n        np.savez(output_root / "detection.npz", rows=test_rows, cols=test_cols, score=test_score, probability=test_probability.astype(np.float32), detection=detection_probability(test_probability.astype(np.float32),rho).astype(np.float32))')
exec(compile(code,str(source),'exec'),{'__name__':'__main__','__file__':str(source)})
