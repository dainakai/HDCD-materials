"""One-shot dependent finalization, after the resident numerical runner succeeds."""
from common import *
import subprocess
import time

while True:
    state=json.loads((ROOT/'campaign-state.json').read_text())
    if state['status']=='FAILED':
        write_json({'status':'upstream_failed','error':'Inspect campaign-state.json and the failed stage log.'},ROOT/'finalization-job.json')
        break
    if state['status']=='PASS':
        result=subprocess.run([str(CPU_PYTHON),str(SCRIPT/'finalize.py')])
        write_json({'status':'PASS' if result.returncode==0 else 'FAILED','returncode':result.returncode},ROOT/'finalization-job.json')
        break
    time.sleep(30)
