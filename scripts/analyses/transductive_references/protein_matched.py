import sys
from pathlib import Path
sys.path.insert(0,str(Path.cwd()/'scripts'))
import pandas as pd
import evaluate_protein_within_state as e
original=Path.cwd()/'artifacts/paper_evidence/papalexi_crossmodal/benchmark'
e.VALUE_RANKINGS={k:(str(original/v) if k != "fused_value" else v) for k,v in e.VALUE_RANKINGS.items()}
control=pd.read_parquet('artifacts/paper_evidence/review_round2/protein/global_fill_selector/selected_gene_scores.parquet')
score=e.cell_zero_score
e.cell_zero_score=lambda frame,scores,gene:score(frame,control,gene)
e.main()
