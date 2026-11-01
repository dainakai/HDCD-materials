"""Fit paired collision models on a fixed upstream candidate/feature bank."""
from common import *
import argparse
import importlib.util
import time
import torch
from threadpoolctl import threadpool_limits
from ra3d.training import fit_staged
from ra3d.models import load_model
from ra3d.cascade import predict_full
from ra3d.evaluation import event_curve
from ra3d.strict_labels import attach_identity_labels
from ra3d._core.policy import adjusted_score


def load_legacy(name):
    spec = importlib.util.spec_from_file_location('ra3d._core.preflight_legacy_' + name,
                                                 REPO / f'src/ra3d/_core/{name}.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def make_labels(frame, seed, gate, rule):
    truth = read_table(truth_path(seed))
    mapping = read_table(tracks_path(seed, 'mapping.parquet'))
    if rule == 'strict':
        out, audit, _ = attach_identity_labels(frame, truth, mapping, z_tolerance_um=gate)
    else:
        legacy = load_legacy('labels')
        projected = frame.copy()
        projected['z_um'] -= 100.
        out, audit, _ = legacy.attach_identity_labels(projected, truth, mapping)
        out['z_um'] = frame['z_um'].to_numpy()
    out['synth_seed'] = seed
    return out, audit


def top20_labels(contexts, gate, audits):
    result = []
    for context in contexts:
        seed = int(context.synth_seed.iloc[0]); scores = context.score.to_numpy()
        indices = []
        for local in context.groupby('vanished_track_id', sort=False).indices.values():
            indices.extend(local[np.argsort(-scores[local], kind='stable')[:20]].tolist())
        labeled, audit = make_labels(context.iloc[indices].reset_index(drop=True), seed, gate, 'strict')
        modified = context.copy()
        for col in ['identity_training_selected', 'spatial_positive', 'matched_truth_event_id']:
            modified.iloc[indices, modified.columns.get_loc(col)] = labeled[col].to_numpy()
        audits.append({'stage': 'relation_top20', 'seed': seed, **audit})
        result.append(modified)
    return result


def evaluate(bundle, full, name):
    out = ROOT / 'results' / name
    out.mkdir(parents=True, exist_ok=True)
    truth = read_table(truth_path(260745))
    bundle['recipe']['source_policy']['lineage_rank_order'] = 'before'
    prediction, audit = predict_full(full, bundle, scene_id=260745)
    rows = []
    for rank in ['before', 'after']:
        pred = prediction.copy()
        if rank == 'after':
            ordered = pred.sort_values(['_policy_score', 'candidate_frame_raw', '_source_index'],
                                       ascending=[False, True, True], kind='stable')
            new_rank = ordered.groupby('vanished_track_id', sort=False).cumcount()
            pred.loc[ordered.index, '_lineage_rank'] = new_rank.to_numpy()
            pred['score'] = adjusted_score(pred._policy_score, pred._lineage_rank, 1.)
        pred = pred.sort_values('score', ascending=False, kind='stable').reset_index(drop=True)
        membership = ['_source_index', 'candidate_frame', 'x_um', 'y_um', 'z_um']
        pd.testing.assert_frame_equal(pred[membership].sort_values('_source_index').reset_index(drop=True),
                 prediction[membership].sort_values('_source_index').reset_index(drop=True), check_exact=True)
        write_table(pred, out / f'predictions_{rank}.parquet')
        for gate in [2500, 5000, 10000, 15000, 30000, None]:
            curve, metrics = event_curve({260745: pred}, {260745: truth},
                      projected_xy=gate is None, z_tolerance_um=15000 if gate is None else gate)
            rows.append({'arm': name, 'rank': rank, 'evaluation_z_um': gate, **summarize(metrics)})
            if gate is None:
                write_json(metrics, out / f'metrics_{rank}.json')
                write_table(curve, out / f'pr_{rank}.csv')
    write_table(pd.DataFrame(rows), out / 'sensitivity.csv')
    write_json({'status': 'PASS', 'rank_changes_only_scores': True, 'selection': audit,
                'rank_changed_survivors': int((prediction._lineage_rank !=
                    prediction.sort_values(['_policy_score', 'candidate_frame_raw', '_source_index'],
                       ascending=[False, True, True], kind='stable').groupby('vanished_track_id', sort=False)
                       .cumcount().reindex(prediction.index)).sum())}, out / 'verification.json')
    emit(stage='evaluation_complete', arm=name, primary=[r for r in rows if r['evaluation_z_um'] is None])


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--phase', type=int, choices=[3, 12], required=True)
    parser.add_argument('--z-mm', choices=['5', '15', '30', 'none'], required=True)
    parser.add_argument('--rule', choices=['strict', 'legacy'], default='strict')
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--threads', type=int, default=4)
    args = parser.parse_args()
    gate = None if args.z_mm == 'none' else float(args.z_mm)*1000
    name = f'{args.rule}_phase{args.phase}_z{args.z_mm}'
    output = ROOT / 'models' / name
    started = time.monotonic(); audits = []
    torch.set_num_threads(args.threads)
    if args.rule == 'legacy':
        assert gate == 15000
        import ra3d.training as training
        training.fit_relation = load_legacy('relation').fit_model
    if not (output / 'model.json').exists():
        stages = {s: {} for s in ['stage1', 'static', 'full', 'truth']}
        sources = {}
        for seed in TRAIN:
            for stage, filename in [('stage1', 'stage1_training.parquet'), ('static', 'static77.parquet'),
                                    ('full', 'paper92.parquet')]:
                path = data_path(seed, 'paper92_phase3.parquet') if stage == 'full' and args.phase == 3 else feature_path(seed, filename)
                data = read_table(path)
                labeled, audit = make_labels(data, seed, gate, args.rule)
                stages[stage][seed] = labeled
                audits.append({'stage': stage, 'seed': seed, **audit})
                sources[str(path)] = sha256(path)
            stages['truth'][seed] = read_table(truth_path(seed))
        sources['protocol'] = sha256(ROOT / 'config/protocol.json')
        sources['engine_strict_labels'] = sha256(ROOT / 'engine/ra3d/strict_labels.py')
        sources['train_script'] = sha256(Path(__file__))
        output.mkdir(parents=True, exist_ok=True)
        write_json(audits, output / 'labels.json')
        bundle, audit = fit_staged(stages, output, device=args.device, cpu_threads=args.threads,
            source_manifest=sources, progress=lambda v: emit(arm=name, **v),
            relation_context_transform=(lambda c: top20_labels(c, gate, audits)) if args.rule == 'strict' else None)
        write_json(audits, output / 'labels.json')
        del stages
    else:
        bundle = load_model(output, device=args.device)
    # Test image extraction can proceed independently of training. Only the
    # evaluation depends on its completed, verified feature table.
    if args.phase == 3 and not data_path(260745, 'images3.json').exists():
        emit(stage='waiting_for_test_features', arm=name)
        while not data_path(260745, 'images3.json').exists():
            time.sleep(5)
    test_path = data_path(260745, 'paper92_phase3.parquet') if args.phase == 3 else feature_path(260745, 'paper92.parquet')
    with threadpool_limits(limits=args.threads):
        evaluate(bundle, read_table(test_path), name)
    write_json({'status': 'PASS', 'arm': name, 'seconds': time.monotonic()-started,
                'train_seeds': TRAIN, 'test_seed': 260745, 'test_used_in_fitting': False,
                'fixed_upstream_tables': True, 'full_pipeline_regenerated': False,
                'phase_iterations': args.phase, 'training_z_gate_um': gate, 'label_rule': args.rule}, output / 'complete.json')
    emit(status='PASS', arm=name, seconds=time.monotonic()-started)
