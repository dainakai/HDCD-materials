#!/usr/bin/env python3
"""Generate, measure, train and evaluate all six numerical hologram sequences."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import shutil

REPO = Path(__file__).resolve().parents[1]
STAGES = ['labels', 'render', 'initialize', 'measurement', 'tracking', 'features', 'train', 'evaluate']


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', type=Path, default=REPO / 'runs/benchmark')
    p.add_argument('--from-stage', choices=STAGES, default='labels')
    p.add_argument('--through-stage', choices=STAGES, default='evaluate')
    p.add_argument('--gpu-python', default=sys.executable)
    p.add_argument('--julia', default='julia')
    p.add_argument('--gpu', default='0', help='Visible GPU index; independent runs may select different GPUs')
    p.add_argument('--cpu-jobs', type=int, default=1,
                   help='Concurrent CPU feature jobs (default: 1); increase when RAM permits')
    p.add_argument('--smoke', action='store_true', help='128-frame generation-to-feature integration check')
    a = p.parse_args()
    if a.cpu_jobs < 1:
        p.error('cpu-jobs must be a positive integer')
    for name in ['gpu_python', 'julia']:
        value = getattr(a, name)
        resolved = shutil.which(value) or (str(Path(value).expanduser().resolve()) if Path(value).is_file() else None)
        if resolved is None:
            p.error(f'Executable not found: {value}')
        setattr(a, name, str(Path(resolved).absolute()))
    if STAGES.index(a.from_stage) > STAGES.index(a.through_stage):
        p.error('from-stage must not follow through-stage')
    output = a.output.resolve()
    if output == REPO or any(output == REPO / n or REPO / n in output.parents
                            for n in ['src', 'data', 'data_sources', 'reference', 'scripts']):
        p.error('Choose a separate run directory')
    output.mkdir(parents=True, exist_ok=True)
    environment = {**os.environ, 'HDCD_RUN_ROOT': str(output), 'HDCD_GPU_PYTHON': a.gpu_python,
                   'HDCD_JULIA': a.julia, 'CUDA_VISIBLE_DEVICES': a.gpu, 'PYTHONHASHSEED': '0',
                   'PYTHONPATH': str(REPO / 'src'), 'CUBLAS_WORKSPACE_CONFIG': ':4096:8',
                   'OMP_NUM_THREADS': '4', 'OPENBLAS_NUM_THREADS': '4', 'MKL_NUM_THREADS': '4'}
    environment['PATH'] = str(Path(a.julia).expanduser().resolve().parent) + os.pathsep + environment['PATH']
    seeds = [260740] if a.smoke else list(range(260740, 260746))

    def call(script, *args, gpu=False):
        cmd = [a.gpu_python if gpu else sys.executable, str(REPO / 'scripts' / script), *map(str, args)]
        print('Running:', ' '.join(cmd), flush=True)
        try:
            subprocess.run(cmd, cwd=REPO, env=environment, check=True)
        except subprocess.CalledProcessError as error:
            (output / 'state.json').write_text(json.dumps({'status': 'FAIL', 'stage': stage,
                'failed_command': cmd, 'returncode': error.returncode, 'smoke': a.smoke}) + '\n')
            raise

    import fcntl
    with (output / '.pipeline.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        for stage in STAGES[STAGES.index(a.from_stage):STAGES.index(a.through_stage) + 1]:
            (output / 'state.json').write_text(json.dumps({'stage': stage, 'status': 'running', 'smoke': a.smoke}) + '\n')
            if stage == 'labels':
                for seed in seeds:
                    target = output / 'generation' / f'seed{seed}'
                    if (target / 'labels/verified.json').exists():
                        from generate_holograms import verify_labels
                        verify_labels(target, seed)
                    else:
                        call('generate_holograms.py', '--seed', seed, '--stage', 'labels', '--output', output / 'generation')
            elif stage == 'render':
                for seed in seeds:
                    call('benchmark/render.py', '--seed', seed, '--end', 128 if a.smoke else 6000, '--chunk', 128 if a.smoke else 1000)
            elif stage == 'initialize':
                call('benchmark/initialize.py', '--seeds', *seeds)
            elif a.smoke:
                call('benchmark/integration_smoke.py', gpu=True)
                break
            elif stage == 'measurement':
                for seed in seeds:
                    call('benchmark/minip_generate.py', '--seed', seed, gpu=True)
                    call('benchmark/measure.py', '--seed', seed, gpu=True)
            elif stage == 'tracking':
                for seed in seeds:
                    call('benchmark/tracking.py', '--seed', seed)
                    call('benchmark/minip_tracks.py', '--seed', seed)
            elif stage == 'features':
                for step in ['pool', 'static', 'score', 'images']:
                    for method in ['ours', 'depth_free', 'minip2d']:
                        image_gpu = step == 'images' and method == 'ours'
                        call('benchmark/run_stage.py', step, '--method', method, '--jobs', 1 if image_gpu else a.cpu_jobs,
                             '--device', 'cuda:0' if image_gpu else 'cpu', gpu=image_gpu)
            elif stage == 'train':
                for method in ['ours', 'depth_free', 'minip2d']:
                    call('benchmark/train.py', '--method', method, '--device', 'cuda:0', gpu=True)
            elif stage == 'evaluate':
                for method in ['ours', 'depth_free', 'minip2d']:
                    call('benchmark/evaluate.py', '--method', method, *(['--stages'] if method == 'ours' else []))
                call('verify_benchmark_results.py', '--run-root', output)
        (output / 'state.json').write_text(json.dumps({'status': 'PASS', 'through_stage': 'integration_smoke' if a.smoke and stage not in ['labels', 'render', 'initialize'] else stage, 'smoke': a.smoke}) + '\n')


if __name__ == '__main__':
    main()
