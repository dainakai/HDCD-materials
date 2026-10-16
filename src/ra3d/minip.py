"""Train and apply the depth-free variant and MinIP-based 2D comparison."""
from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import sklearn
import skops.io as sio
import torch
from PIL import Image
from safetensors.torch import load_file, save_file
from threadpoolctl import threadpool_limits

from . import baseline2d
from ._core import two_d_common as common
from ._core import two_d_relation as relation
from .config import DEVELOPMENT_SEEDS, HGB_PARAMS, IMAGE_FEATURES, KEY_COLUMNS, Units
from .io import read_table, require_columns, sha256, write_json, write_table
from .models import TRUSTED_SKLEARN_TYPES
from .training import TrainingCheckpoint, fit_classifier

FEATURES = list(common.TWO_D_FEATURES) + IMAGE_FEATURES
METHODS = {'depth-free': ('baseline2d', 'depth_free_minip'), 'minip-2d': ('minip2d', 'minip_2d')}
LABEL_COLUMNS = [
    'identity_candidate_frame', 'vanished_source_particle_id', 'survivor_source_particle_id',
    'spatial_positive', 'assigned_truth_index', 'identity_state', 'matched_truth_event_id',
    'identity_training_selected', 'identity_event_factor', 'training_source', 'relation_positive', 'train_label',
]


def score(models, frame):
    require_columns(frame, FEATURES, name='64-feature comparison input')
    matrix = common.feature_frame(frame, FEATURES)
    return np.mean([model.predict_proba(matrix)[:, 1] for model in models], axis=0)


def predict(frame, bundle, scene_id=0):
    """Rank and localize events without consulting ground-truth label columns."""
    frame = frame.drop(columns=LABEL_COLUMNS, errors='ignore').copy()
    return baseline2d.predict(frame, bundle, scene_id, base_scores=score(bundle['full_models'], frame))


def load(directory, device='cpu'):
    directory = Path(directory)
    config = json.loads((directory / 'model.json').read_text())
    if (config.get('schema_version') != 1 or config.get('method') not in {v[1] for v in METHODS.values()}
            or config.get('full_features') != FEATURES
            or config.get('stage1_features') != list(common.STAGE1_FEATURES)
            or config.get('static_features') != list(common.TWO_D_FEATURES)
            or config.get('relation_features') != list(relation.RELATION_FEATURES)):
        raise ValueError('Incompatible 64-feature comparison model')
    if config.get('sklearn_version') != sklearn.__version__:
        raise ValueError('Use the scikit-learn version recorded in model.json')
    for name in ['trees.skops', 'relation.safetensors', 'normalization.npz']:
        if sha256(directory / name) != config['files'][name]:
            raise ValueError(f'Model checksum mismatch: {name}')
    unknown = set(sio.get_untrusted_types(file=directory / 'trees.skops'))
    if not unknown <= TRUSTED_SKLEARN_TYPES:
        raise ValueError('Unsupported model types')
    bundle = sio.load(directory / 'trees.skops', trusted=sorted(unknown))
    attention = relation.RelationAttentionRanker(len(relation.RELATION_FEATURES))
    attention.load_state_dict(load_file(directory / 'relation.safetensors'))
    attention.to(device).eval()
    with np.load(directory / 'normalization.npz', allow_pickle=False) as norm:
        bundle.update(relation=attention, mean=norm['mean'], std=norm['std'])
    bundle['method'] = config['method']
    return bundle


class MinIPReader:
    """Read one-based grayscale PNGs with zero-based acquisition frame IDs."""
    def __init__(self, directory, method):
        if method not in METHODS:
            raise ValueError('Unknown MinIP comparison method')
        paths = sorted(Path(directory).glob('*.png'))
        if not paths or any(not p.stem.isdigit() or int(p.stem) < 1 for p in paths):
            raise ValueError('MinIP files must have numeric, one-based names such as 000001.png')
        frames = pd.DataFrame({'minip': [str(p) for p in paths]},
                              index=pd.Index([int(p.stem) - 1 for p in paths], name='frame'))
        if not frames.index.is_unique:
            raise ValueError('Duplicate MinIP frame numbers')
        self.dtype = np.float32 if method == 'depth-free' else np.float64
        self.acquisition = SimpleNamespace(frames=frames, units=Units(), wavelength_um=0.6328,
            phase_distance_um=21450., reconstruction_start_um=74900. if method == 'depth-free' else 64900.,
            padding_px=1536, phase_iterations=3, coefficients=None, mode='minip_' + method)

    def intensity(self, frame, z):
        if z != 0:
            raise ValueError('MinIP lookup requires the zero-depth placeholder')
        with Image.open(self.acquisition.frames.loc[frame, 'minip']) as image:
            if image.mode != 'L' or image.size != (1024, 1024):
                raise ValueError('Expected a 1024 by 1024 grayscale MinIP image')
            return np.asarray(image, dtype=self.dtype)


def infer(tracks_path, images, model, output, *, scene_id=0, static_budget=None, device='cpu'):
    """Generate candidates, trajectory/image features and events for a new sequence."""
    from ._core.two_d_budget import budget_select
    from ._core.features import add_derived_features
    from .features import generate_candidates, static_features, image_features
    from .tracks import read_tracks, materialize_tracks

    output = Path(output)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError('Choose a new output directory')
    output.mkdir(parents=True, exist_ok=True)
    table = read_tracks(tracks_path).copy()
    table['z'], table['vz'], table['postOutProb'] = 0., 0., 0.
    bundle = load(model, device)
    wide = common.add_2d_derived_features(generate_candidates(table))
    if wide.empty:
        result = pd.DataFrame(columns=KEY_COLUMNS + ['x_um', 'y_um', 'z_um', 'score'])
        write_table(result, output / 'predictions.parquet')
        return result
    end = table.groupby('track_id', sort=True).tail(1).set_index('track_id')
    edge = np.minimum.reduce([end.x.to_numpy(), end.y.to_numpy(), 1024. - end.x, 1024. - end.y])
    spatial = np.maximum(0., (5. - edge) / 5.)
    temporal = np.maximum(0., (5. - (int(table.frame.max()) - end.frame.to_numpy())) / 5.)
    boundary = pd.Series(np.maximum(spatial, temporal), index=end.index)
    wide['boundary_exit_score'] = wide.vanished_track_id.map(boundary)
    for column in ['z_um', 'vanished_z_um', 'survivor_z_um']:
        wide[column] = 0.
    scores = bundle['stage1'].predict_proba(common.feature_frame(wide, list(common.STAGE1_FEATURES)))[:, 1]
    if static_budget is None:
        ranked = common.attach_outside_rank(wide, scores, common.selected_gate_mask(wide))
        static_budget = int(common.shortlist_mask(ranked).sum())
    short, selection = budget_select(wide, scores, static_budget)
    static = common.add_2d_derived_features(static_features(
        short, table, materialize_tracks(table, output / 'tracks'), Units()))
    full = baseline2d.select_union(static, bundle['static'])
    requested = full.loc[full.static_global_top25.eq(1)].reset_index(drop=True)
    method = 'depth-free' if bundle['method'] == 'depth_free_minip' else 'minip-2d'
    image = image_features(requested, table, MinIPReader(images, method),
                           checkpoint_path=output / 'image_checkpoint.npz')
    full = full.drop(columns=IMAGE_FEATURES, errors='ignore').merge(image[KEY_COLUMNS + IMAGE_FEATURES],
        on=KEY_COLUMNS, how='left', sort=False, validate='one_to_one')
    full = add_derived_features(full)
    full['image_enriched'] = full.static_global_top25.astype(np.int8)
    write_table(static, output / 'static.parquet')
    write_table(full, output / 'full64.parquet')
    result = predict(full, bundle, scene_id)
    write_table(result, output / 'predictions.parquet')
    write_table(result, output / 'predictions.csv')
    write_json({'method': method, 'scene_id': scene_id, 'pool_rows': len(wide),
                'static_rows': len(static), 'full_rows': len(full), 'image_rows': len(image),
                'event_rows': len(result), 'selection': selection}, output / 'run.json')
    return result


def save(bundle, directory, method, audit):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    sio.dump({k: bundle[k] for k in ['stage1', 'static', 'full_models', 'residuals']}, directory / 'trees.skops')
    save_file({k: v.detach().cpu().contiguous() for k, v in bundle['relation'].state_dict().items()},
              directory / 'relation.safetensors')
    np.savez(directory / 'normalization.npz', mean=bundle['mean'], std=bundle['std'])
    write_json({
        'schema_version': 1, 'method': METHODS[method][1], 'sklearn_version': sklearn.__version__,
        'stage1_features': list(common.STAGE1_FEATURES), 'static_features': list(common.TWO_D_FEATURES),
        'full_features': FEATURES, 'relation_features': list(relation.RELATION_FEATURES),
        'files': {name: sha256(directory / name) for name in ['trees.skops', 'relation.safetensors', 'normalization.npz']},
        'audit': audit,
    }, directory / 'model.json')


def sample_training(frame, seed):
    """Keep both labels for a small execution check, with deterministic sampling."""
    positive = frame.index[frame.identity_training_selected.astype(bool)].to_numpy()
    negative = frame.index[frame.spatial_positive.eq(0)].to_numpy()
    rng = np.random.default_rng(seed)
    chosen = np.sort(np.r_[rng.choice(positive, min(128, len(positive)), replace=False),
                           rng.choice(negative, min(384, len(negative)), replace=False)])
    return frame.loc[chosen].reset_index(drop=True)


def train(root, output, method, *, device='cpu', cpu_threads=4, smoke=False, progress=print):
    root, output = Path(root), Path(output)
    if (output / 'model.json').exists():
        raise FileExistsError('Completed model exists; choose a new output directory')
    directory, identity_name = METHODS[method]
    output.mkdir(parents=True, exist_ok=True)
    frames, truth, manifest = {}, {}, {}
    for seed in DEVELOPMENT_SEEDS:
        path = root / directory / 'train' / f'seed{seed}' / 'full.parquet'
        full = read_table(path)
        if not full.synth_seed.eq(seed).all():
            raise ValueError(f'Wrong training scene in {path}')
        frames[seed] = sample_training(full, seed) if smoke else full
        manifest[str(path.relative_to(root))] = sha256(path)
        path = root / 'labels/train' / f'seed{seed}' / 'collision_events.parquet'
        truth[seed] = read_table(path)
        manifest[str(path.relative_to(root))] = sha256(path)
        for stage in ['gate', 'static']:
            path = root / directory / 'train' / f'seed{seed}' / (stage + '.parquet')
            manifest[str(path.relative_to(root))] = sha256(path)
    identity = {'method': identity_name, 'training_seeds': list(DEVELOPMENT_SEEDS), 'inputs': manifest,
                'full_features': FEATURES, 'device': device, 'base_oof_recomputed': True, 'smoke': smoke}
    checkpoint = TrainingCheckpoint(output / 'checkpoints', identity)
    emit = progress or (lambda value: None)
    parameters = {**HGB_PARAMS, 'max_iter': 5} if smoke else None
    audit = {**identity, 'test_seed_used_in_fit': False, 'attention_epochs': 12}
    torch.set_num_threads(cpu_threads)
    with threadpool_limits(limits=cpu_threads):
        def stage_frames(filename):
            for seed in DEVELOPMENT_SEEDS:
                table = read_table(root / directory / 'train' / f'seed{seed}' / filename)
                yield sample_training(table, seed) if smoke else table

        bundle = {}
        for stage, filename, features in [('stage1', 'gate.parquet', list(common.STAGE1_FEATURES)),
                                          ('static', 'static.parquet', list(common.TWO_D_FEATURES))]:
            bundle[stage], audit[stage] = checkpoint.tree(stage,
                lambda: fit_classifier(stage_frames(filename), features, parameters=parameters), emit)
        contexts = []
        for held in DEVELOPMENT_SEEDS:
            models = []
            for state in [269723, 269724, 269725]:
                model, _ = checkpoint.tree(f'outer_{held}_{state}',
                    lambda: fit_classifier((frames[s] for s in DEVELOPMENT_SEEDS if s != held),
                                           FEATURES, random_state=state, parameters=parameters), emit)
                models.append(model)
            base = score(models, frames[held])
            np.save(output / f'oof_{held}.npy', base)
            contexts.append(relation.context_frame(frames[held], base))
            del models
            gc.collect()
        attention, mean, std, audit['relation_fit'] = relation.fit_attention(contexts, device)
        bundle.update(relation=attention, mean=mean, std=std)
        del contexts
        gc.collect()
        bundle['full_models'] = []
        for state in [269723, 269724, 269725]:
            model, _ = checkpoint.tree(f'full_{state}',
                lambda: fit_classifier(frames.values(), FEATURES, random_state=state, parameters=parameters), emit)
            bundle['full_models'].append(model)
        bundle['residuals'], audit['residuals'] = checkpoint.tree('residuals',
            lambda: relation.fit_residual_models(frames, DEVELOPMENT_SEEDS, truth), emit)
        save(bundle, output, method, audit)
    write_json(audit, output / 'training_audit.json')
    return bundle, audit


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command', required=True)
    fitting = commands.add_parser('train', help='Fit all collision stages from the five training scenes')
    fitting.add_argument('--data', type=Path, default=Path('data'))
    fitting.add_argument('--method', choices=METHODS, required=True)
    fitting.add_argument('--smoke', action='store_true', help='512 rows/scene and short tree fits; execution check only')
    inference = commands.add_parser('predict', help='Apply a trained model to a full 64-feature table')
    inference.add_argument('--features', type=Path, required=True)
    inference.add_argument('--model', type=Path, required=True)
    inference.add_argument('--scene-id', type=int, default=260745)
    processing = commands.add_parser('infer', help='Compute features and events from new trajectories and MinIP images')
    processing.add_argument('--tracks', type=Path, required=True)
    processing.add_argument('--minip-dir', type=Path, required=True)
    processing.add_argument('--model', type=Path, required=True)
    processing.add_argument('--scene-id', type=int, default=0)
    processing.add_argument('--static-budget', type=int)
    for command in [fitting, inference, processing]:
        command.add_argument('--output', type=Path, required=True)
        command.add_argument('--device', default='cpu')
        command.add_argument('--cpu-threads', type=int, default=4)
    args = parser.parse_args()
    torch.set_num_threads(args.cpu_threads)
    with threadpool_limits(limits=args.cpu_threads):
        if args.command == 'infer':
            if args.static_budget is not None and args.static_budget <= 0:
                parser.error('--static-budget must be positive')
            result = infer(args.tracks, args.minip_dir, args.model, args.output, scene_id=args.scene_id,
                           static_budget=args.static_budget, device=args.device)
            print(f'Saved {len(result)} event estimates to {args.output}')
        elif args.command == 'predict':
            if args.output.exists():
                parser.error('Choose a new output path')
            result = predict(read_table(args.features), load(args.model, args.device), args.scene_id)
            write_table(result, args.output)
            print(f'Saved {len(result)} event estimates to {args.output}')
        else:
            _, audit = train(args.data, args.output, args.method, device=args.device,
                             cpu_threads=args.cpu_threads, smoke=args.smoke,
                             progress=lambda value: print(json.dumps(value), flush=True))
            if args.smoke:
                directory = METHODS[args.method][0]
                test = read_table(args.data / directory / 'validation/seed260745/full.parquet').head(2048)
                prediction = predict(test, load(args.output, args.device), 260745)
                if prediction.empty or not np.isfinite(prediction[['score', 'x_um', 'y_um']]).all().all():
                    raise AssertionError('Reloaded model did not produce finite events')
                write_json({'status': 'PASS', 'method': args.method, 'training_seeds': audit['training_seeds'],
                            'test_seed_used_in_fit': False, 'base_oof_recomputed': True, 'model_reloaded': True,
                            'input_rows': len(test), 'finite_events': len(prediction), 'purpose': 'training execution check'},
                           args.output / 'smoke_report.json')
            print('Model saved:', args.output / 'model.json', flush=True)


if __name__ == '__main__':
    main()
