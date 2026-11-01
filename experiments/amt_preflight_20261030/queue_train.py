"""Wait for shared phase-3 features, then run predeclared comparisons."""
from common import *
import subprocess
import time

jobs=[('strict','15'),('legacy','15'),('strict','5'),('strict','30'),('strict','none')]
active={}; done=[]
# Five fits use 20 CPU threads on the host with 28 logical CPUs. The small
# attention models share two GPUs; tree fitting is the dominant cost.
max_parallel=5
while jobs or active:
    for slot,(process,log,name) in list(active.items()):
        if process.poll() is not None:
            log.close()
            if process.returncode:
                raise RuntimeError(f'{name} failed with exit {process.returncode}; inspect its log')
            done.append(name); del active[slot]
            emit(stage='comparison_finished',arm=name)
    ready=all(data_path(s,'images3.json').exists() for s in TRAIN)
    if ready:
        for slot in range(max_parallel):
            if slot in active or not jobs: continue
            gpu=slot%2
            rule,gate=jobs.pop(0);name=f'{rule}_phase3_z{gate}'
            log=(ROOT/f'logs/train-{name}.log').open('w')
            command=[sys.executable,str(Path(__file__).with_name('train_compare.py')),
                     '--phase','3','--z-mm',gate,'--rule',rule,'--device',f'cuda:{gpu}']
            process=subprocess.Popen(command,stdout=log,stderr=subprocess.STDOUT)
            active[slot]=(process,log,name)
            emit(stage='comparison_started',arm=name,pid=process.pid,gpu=gpu)
    if jobs or active:time.sleep(15)
write_json({'status':'PASS','completed':done},ROOT/'results/queue.json')
