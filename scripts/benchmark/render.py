"""Resume canonical Julia hologram generation in audited, bounded chunks."""
from common import *
import argparse
import fcntl
import os
import subprocess
import time
from PIL import Image


def run(seed, end=6000, chunk=100):
    data = GENERATION / f'seed{seed}'
    output = ROOT / f'runs/render/seed{seed}'
    output.mkdir(parents=True, exist_ok=True)
    lock = (output / '.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX)
    script = REPO / 'reference/generation/render_holograms.jl'
    expected_frames = json.loads((REPO/'reference/generation/render-sha256.json').read_text())[str(seed)]
    label_hashes = json.loads((REPO/'reference/generation/contract.json').read_text())['labels_sha256'][str(seed)]
    if sha256(data/'labels/particles.csv') != label_hashes['particles']:
        raise ValueError('Particle states differ from the fixed benchmark')
    identity = {'seed': seed, 'particles': sha256(data/'labels/particles.csv'),
                'parameters': sha256(data/'metadata/parameters.json'),
                'renderer': sha256(script), 'project': sha256(script.with_name('Project.toml')),
                'manifest': sha256(script.with_name('Manifest.toml'))}
    manifest = output / 'frames.json'
    saved = json.loads(manifest.read_text()) if manifest.exists() else {'identity': identity, 'frames': {}}
    assert saved['identity'] == identity
    def pair(i):
        return [data/f'01_bkgrem/{camera}H001S0011/{i+1:06d}.png' for camera in ['C002','C001']]
    def verify(i):
        result = {}
        for p in pair(i):
            with Image.open(p) as im:
                im.load()
                assert im.size == (1024,1024) and im.mode == 'L', (p, im.size, im.mode)
            result[str(p.relative_to(data))] = sha256(p)
        if result != expected_frames[str(i)]:
            raise ValueError(f'Hologram mismatch at seed {seed}, frame {i}. Use the pinned Julia/CUDA environment.')
        return result
    for key, hashes in saved['frames'].items():
        assert verify(int(key)) == hashes
    count = len(saved['frames'])
    assert set(saved['frames']) == {str(i) for i in range(count)}
    for start in range(count, end, chunk):
        stop = min(end, start + chunk)
        for i in range(start, stop):
            if any(p.exists() for p in pair(i)):
                raise RuntimeError(f'Unverified partial render at seed={seed}, frame={i}; inspect before resuming.')
        command = [JULIA,'--startup-file=no','--history-file=no',str(script),
                   f'--out-dir={data}','--device=1',f'--frame-start={start}',f'--frame-end={stop}',
                   f'--summary-name=render_{start}_{stop}.json']
        log = ROOT/f'logs/render-{seed}-{start}-{stop}.log'
        began = time.monotonic()
        emit(stage='render_start', seed=seed, start=start, end=stop, log=str(log))
        with log.open('w') as stream:
            subprocess.run(command, stdout=stream, stderr=subprocess.STDOUT, check=True,
                           env=os.environ.copy())
        for i in range(start,stop): saved['frames'][str(i)] = verify(i)
        write_json(saved, manifest)
        emit(stage='render_chunk_complete',seed=seed,frames=len(saved['frames']),seconds=time.monotonic()-began)
    if end == 6000:
        assert len(saved['frames']) == 6000
        write_json({'status':'PASS','frames':6000,'image_files':12000,'manifest':str(manifest),
                    'manifest_sha256':sha256(manifest),'identity':identity},output/'complete.json')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed',type=int,choices=SEEDS,required=True)
    parser.add_argument('--end',type=int,default=6000)
    parser.add_argument('--chunk',type=int,default=100)
    args=parser.parse_args()
    run(args.seed,args.end,args.chunk)
