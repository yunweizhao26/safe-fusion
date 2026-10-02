import json
import resource
import subprocess
import sys
import time
from pathlib import Path

started = time.monotonic()
process = subprocess.run(sys.argv[2:], check=False)
usage = resource.getrusage(resource.RUSAGE_CHILDREN)
Path(sys.argv[1]).write_text(json.dumps({
    'command': sys.argv[2:], 'wall_seconds': time.monotonic() - started,
    'peak_rss_kib': usage.ru_maxrss, 'user_seconds': usage.ru_utime,
    'system_seconds': usage.ru_stime, 'returncode': process.returncode,
}, indent=2) + '\n')
sys.exit(process.returncode)
