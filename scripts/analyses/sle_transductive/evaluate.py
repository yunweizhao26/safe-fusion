import sys
from pathlib import Path
ROOT = Path.cwd()
sys.path.insert(0, str(ROOT/'scripts'))
import sle_common
case, script, *arguments = sys.argv[1:]
sle_common.CASE = Path(case).resolve()
sle_common.load_unit.__defaults__ = (sle_common.CASE,)
sys.argv = [script, *arguments]
import runpy
runpy.run_path(str(ROOT/'scripts'/script), run_name='__main__')
