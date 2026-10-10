from pathlib import Path
import json
R=Path.cwd(); O=R/'artifacts/paper_evidence/review_round4/transductive_references'; S=Path('scripts/analyses/transductive_references')
def link(dst,src):
 dst.parent.mkdir(parents=True,exist_ok=True)
 if not dst.is_symlink() and not dst.exists(): dst.symlink_to(src)
def view(dst,src,overrides):
 dst.mkdir(parents=True,exist_ok=True)
 for child in src.iterdir():
  if child.name not in overrides: link(dst/child.name,child)
 for name,path in overrides.items(): link(dst/name,path)

for screen in ('norman_crispra','adamson_crispri','dixit_ko','papalexi_eccite'):
 src=R/'artifacts/paper_evidence/review_round2/leakage_free/norman_crispra' if screen=='norman_crispra' else R/'artifacts/external_perturbseq'/screen
 for name in ('corrupted.h5ad','coordinates.parquet','splits.parquet','prepared.h5ad'):
  if (src/name).exists(): link(O/'knockdown/masked'/screen/name,src/name)
 if screen!='dixit_ko':
  src=R/'artifacts/paper_evidence/review_round2/knockdown/norman_rebuilt/deployment' if screen=='norman_crispra' else R/'artifacts/paper_evidence/downstream_deployment'/screen
  dst=O/'knockdown/norman_rebuilt/deployment' if screen=='norman_crispra' else O/'knockdown/deployment'/screen
  for name in ('hybrid.h5ad','recorded.h5ad','empty_coordinates.parquet','coordinates.parquet','splits.parquet'): link(dst/name,src/name)
link(O/'knockdown/review_root/standard_imputers',R/'artifacts/paper_evidence/review_round2/knockdown/standard_imputers')
link(O/'knockdown/norman_rebuilt/standard_imputers',R/'artifacts/paper_evidence/review_round2/knockdown/norman_rebuilt/standard_imputers')
(O/'protein').mkdir(parents=True,exist_ok=True)
for name in ('corrupted.h5ad','coordinates.parquet','splits.parquet'):
 link(O/'protein'/name,R/'artifacts/paper_evidence/papalexi_crossmodal/benchmark'/name)
K=O/'knockdown'; old=R/'artifacts/paper_evidence/review_round2/knockdown'
for screen in ('adamson_crispri','papalexi_eccite','norman_crispra'):
 src=old/'norman_rebuilt/deployment' if screen=='norman_crispra' else R/'artifacts/paper_evidence/downstream_deployment'/screen
 new=K/'norman_rebuilt/deployment' if screen=='norman_crispra' else K/'deployment'/screen
 dest=K/'evaluation_inputs/norman/deployment' if screen=='norman_crispra' else K/'evaluation_inputs/deployment'/screen
 replacements={f'safe_fusion_{p}pct':new/f'safe_fusion_{p}pct' for p in range(1,11)}
 replacements.update({x:new/x for x in ('selector','selector_condition')})
 view(dest,src,replacements)
 for method,sub in [('safe_fusion','selector'),('safe_fusion_condition','selector_condition')]: link(K/'evaluation_inputs/review/selector_scores'/screen/method,new/sub)
link(K/'evaluation_inputs/review/standard_imputers',old/'standard_imputers')
link(K/'evaluation_inputs/norman/standard_imputers',old/'norman_rebuilt/standard_imputers')
for screen in ('norman_crispra','adamson_crispri','papalexi_eccite','dixit_ko'):
 src=old/'norman_rebuilt/masked' if screen=='norman_crispra' else R/'artifacts/external_perturbseq'/screen
 new=K/'masked'/screen
 view(K/'evaluation_inputs/masked'/screen,src,{x:new/x for x in ('selector_mlp_biology_range','selector_condition')})

view(O/'protein_cs',O/'protein',{x:O/'protein_cs'/x for x in ()})
for name in ('mlp_selector','global_fill_selector','evaluation','evaluation_cd274','evaluation_cd274_raw_counts','evaluation_pdl1_state','evaluation_within_state','figures'):
 path=O/'protein_cs'/name
 if path.is_symlink(): path.unlink()
for fold in range(3): link(O/f'sex_zeros/deployment_rebuilt/colon_{fold}',O/f'sex_zeros/deployment/colon_{fold}')
link(K/'evaluation_inputs/deployment/dixit_ko',R/'artifacts/paper_evidence/downstream_deployment/dixit_ko')
