from pipeline import *
import disease_control_common as common
index=int(sys.argv[1]); analysis=sys.argv[2]; d=D[index]; key=d['key']; tissue=key.split('_')[0]
assert d['design']=='deployment' and tissue in ('colon','pancreas')
labels={'Safe Fusion':'safe_fusion','Transductive conditional':'transductive','Transductive detection':'detection','SVD':'svd','MAGIC':'magic','scVI':'scvi'}
common.TISSUES[tissue]={**common.TISSUES[tissue],'units':(key,)}
common.unit_directory=lambda k:R/'deployment'/k
common.METHODS=tuple(labels)
common.fill_contract=lambda k,method,fraction:R/'deployment'/k/f'{labels[method]}_{int(round(100*fraction))}pct'
common.OUTPUT=R/'disease'/key
import r3_downstream_disease as disease

disease.contract=lambda k,method,fraction,root:common.fill_contract(k,method,fraction)
disease.method_list=lambda spec,root,comparators_only:list(labels)
root=R/'disease'/key
if analysis=='effects':
 sys.argv=['r3_downstream_disease.py','--tissue',tissue,'--root',str(root),'--draws','2000','--permutations','200','--seed','1729','--jobs','8']; disease.main()
elif analysis=='annotation':
 import r3_downstream_annotation as a
 if tissue=='pancreas':

  original=a.reference_mapping
  a.reference_mapping=lambda data,matrix,seed:original(data,matrix,seed+int(key[-1]))
 sys.argv=['r3_downstream_annotation.py','--tissue',tissue,'--root',str(root),'--draws','2000','--seed','1729']; a.main()
elif analysis=='modules':
 import disease_control_modules as m
 sys.argv=['disease_control_modules.py','--tissue',tissue,'--draws','2000','--seed','1729']; m.main()
else: raise ValueError(analysis)
