from common import *
import os
import fusion_value_selectors as f

i=int(os.environ['SLURM_ARRAY_TASK_ID'])
variants=[v for v in f.variants() if v.family in ['transductive_leave_one_out','transductive_architecture']]
u,v=units()[i//len(variants)],variants[i%len(variants)]
dest=OUT/'item3_selectors'/('colon' if u.dataset=='Colon' else 'pancreas_norman')
if not (dest/u.key/v.slug/'report.json').exists():
    sys.argv=['fusion_value_selectors.py','--units-manifest',str(CC/'units_manifest.json' if u.dataset=='Colon' else LF/'units_manifest.json'),
              '--unit-keys',u.key,'--unit',u.key,'--variants',v.name,'--transductive-root',str(fusion_root(u)/'transductive'),
              '--output-root',str(dest),'--seed','1729']
    f.main()
