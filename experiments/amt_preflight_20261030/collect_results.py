"""Collect the completed controlled comparisons into portable review artifacts."""
from common import *
import shutil
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ARMS = ['legacy_phase12_z15', 'legacy_phase3_z15', 'strict_phase12_z15',
        'strict_phase3_z15', 'strict_phase3_z5', 'strict_phase3_z30', 'strict_phase3_znone']
destination = Path(__file__).parent / 'results'
destination.mkdir(exist_ok=True)
(destination / 'figures').mkdir(exist_ok=True)
verification = json.loads((ROOT / 'results/verification.json').read_text())
assert verification['status'] == 'PASS' and set(verification['arms']) == set(ARMS)

primary = []; sensitivity = []; labels = []; completion = []
for arm in ARMS:
    source = ROOT / 'results' / arm
    complete = json.loads((ROOT / 'models' / arm / 'complete.json').read_text())
    assert complete['status'] == 'PASS'
    completion.append(complete)
    provenance = destination / 'model_provenance' / arm
    provenance.mkdir(parents=True, exist_ok=True)
    for filename in ['complete.json', 'labels.json', 'model.json', 'training_audit.json']:
        shutil.copy2(ROOT / 'models' / arm / filename, provenance / filename)
    for rank in ['before', 'after']:
        metrics = json.loads((source / f'metrics_{rank}.json').read_text())
        primary.append({'arm': arm, 'phase': complete['phase_iterations'],
                        'rule': complete['label_rule'], 'training_z_mm':
                        None if complete['training_z_gate_um'] is None else complete['training_z_gate_um']/1000,
                        'rank': rank, **summarize(metrics)})
    sensitivity.append(read_table(source / 'sensitivity.csv'))
    for item in json.loads((ROOT / 'models' / arm / 'labels.json').read_text()):
        labels.append({'arm': arm, **{k: item[k] for k in [
            'stage', 'seed', 'candidate_rows', 'spatial_positive_rows',
            'identity_training_positive_rows', 'neutral_spatial_rows',
            'identity_consistent_events', 'identity_fallback_events'] if k in item}})
primary = pd.DataFrame(primary)
write_table(primary, destination / 'primary_metrics.csv')
write_table(pd.concat(sensitivity, ignore_index=True), destination / 'evaluation_z_sensitivity.csv')
label_table=pd.DataFrame(labels)
write_table(label_table, destination / 'training_label_counts.csv')
gate_labels=label_table.loc[label_table.stage.eq('full') & label_table.arm.str.startswith('strict_phase3_')]
gate_labels=gate_labels.groupby('arm',sort=True)[[
    'candidate_rows','spatial_positive_rows','identity_training_positive_rows',
    'neutral_spatial_rows','identity_consistent_events','identity_fallback_events']].sum().reset_index()
write_table(gate_labels,destination/'training_gate_label_summary.csv')
write_json(completion, destination / 'completed_arms.json')

feature_rows = []; image_audits = []
for seed in SEEDS:
    table = read_table(data_path(seed, 'feature_differences.csv'))
    table.insert(0, 'seed', seed)
    feature_rows.append(table)
    image_audits.append(json.loads(data_path(seed, 'images3.json').read_text()))
feature_differences=pd.concat(feature_rows, ignore_index=True)
write_table(feature_differences, destination / 'image_feature_differences.csv')
write_table(feature_differences.groupby('feature')[['valid','changed','missing_phase12','missing_phase3']].sum().reset_index(),
            destination/'image_feature_summary.csv')
write_json(image_audits, destination / 'image_audits.json')
(destination/'acquisition').mkdir(exist_ok=True)
for seed in SEEDS:
    shutil.copy2(ROOT/f'config/seed{seed}-collision3.yaml',destination/f'acquisition/seed{seed}-collision3.yaml')

for name in ['geometry.csv', 'geometry.json', 'revised_generation.json',
             'revised_generation_all.csv', 'revised_generation_all.json',
             'fig7_input_identity.json', 'fig7_shared_tracks.json',
             'R4_existing_controls.csv', 'R4_paired_block_bootstrap.csv',
             'R7.json', 'R7_scene_counts.csv', 'R7_scene_1sd.csv',
             'R7_diameter_pair_counts.csv', 'verification.json']:
    shutil.copy2(ROOT / 'results' / name, destination / name)
for name in ['fig7_shared_test.png', 'figure14_manuscript.png', 'figure15_manuscript.png']:
    shutil.copy2(ROOT / 'figures' / name, destination / 'figures' / name)
shutil.copy2(ROOT / 'config/train_compare_phase12_snapshot.py',
             destination / 'model_provenance/train_compare_phase12_snapshot.py')
shutil.copy2(Path(__file__).with_name('train_compare.py'),
             destination / 'model_provenance/train_compare_phase3_snapshot.py')

plt.rcParams.update({'font.size': 10, 'axes.spines.top': False, 'axes.spines.right': False})
fig, axes = plt.subplots(2, 2, figsize=(13, 9), constrained_layout=True)
fig.suptitle('Controlled comparison: fixed 74.9 mm measurements and candidate selection\n'
             'Five training sequences; held-out seed 260745; original 102.4 mm truth', fontsize=13)
base = primary.loc[primary['rank'].eq('before')].set_index('arm')

ax = axes[0, 0]
for rule, color, label in [('legacy', '#c97848', 'Legacy labels'),
                            ('strict', '#304d6d', 'Consistent identity labels')]:
    values = [base.loc[f'{rule}_phase{phase}_z15', 'event_ap'] for phase in [12, 3]]
    ax.plot([0, 1], values, marker='o', label=label, color=color)
    for i, value in enumerate(values):
        ax.annotate(f'{value:.4f}', (i, value), xytext=(0, -18 if rule=='strict' and i==1 else 8), textcoords='offset points', ha='center')
minimum = base.loc[['legacy_phase12_z15', 'legacy_phase3_z15',
                    'strict_phase12_z15', 'strict_phase3_z15'], 'event_ap'].min()
ax.set_ylim(minimum-.0012, base.event_ap.max()+.0016)
ax.set_xticks([0, 1], ['12 iterations', '3 iterations']); ax.set_xlim(-.2, 1.2)
ax.set_ylabel('Event AP'); ax.set_title('(a) Image phase retrieval (training Z gate 15 mm)')
ax.legend(loc='lower right')

gates = ['5', '15', '30', 'none']
selected = base.loc[[f'strict_phase3_z{g}' for g in gates]]
ax = axes[0, 1]
ax.plot(np.arange(4), selected.event_ap, marker='o', color='#304d6d')
for i, value in enumerate(selected.event_ap): ax.annotate(f'{value:.4f}', (i, value), xytext=(0, 8), textcoords='offset points', ha='center')
ax.set_xticks(np.arange(4), ['5', '15', '30', 'No Z gate'])
ax.set_xlabel('Training Z gate [mm]'); ax.set_ylabel('Event AP')
ax.set_title('(b) Consistent identity labels, 3 iterations')
ax.margins(y=.3)

ax = axes[1, 0]
for target, color in [(.8, '#304d6d'), (.9, '#c97848')]:
    ax.plot(np.arange(4), selected[f'fp_r{target}'], marker='o', color=color, label=f'Recall >= {target}')
    for i,value in enumerate(selected[f'fp_r{target}']):
        ax.annotate(str(int(value)),(i,value),xytext=(0,8),textcoords='offset points',ha='center')
ax.set_xticks(np.arange(4), ['5', '15', '30', 'No Z gate'])
ax.set_xlabel('Training Z gate [mm]'); ax.set_ylabel('False positive events')
ax.set_yscale('log'); ax.set_yticks([30,50,100,300,600]); ax.set_yticklabels(['30','50','100','300','600']); ax.minorticks_off()
ax.set_ylim(25,750); ax.set_title('(c) Common XY/time test matching, no Z gate'); ax.legend()

ax = axes[1, 1]
bootstrap = read_table(destination / 'R4_paired_block_bootstrap.csv').set_index('arm').loc[ARMS]
# Read the named interval fields rather than relying on CSV column order.
delta = bootstrap['guarded_delta_ap'].to_numpy()*100
lower = bootstrap['paired_delta_ci025'].to_numpy()*100
upper = bootstrap['paired_delta_ci975'].to_numpy()*100
yy=np.arange(len(ARMS))
ax.hlines(yy,lower,upper,color='#304d6d'); ax.scatter(delta,yy,color='#304d6d',s=22)
ax.plot(lower,yy,'|',color='#304d6d'); ax.plot(upper,yy,'|',color='#304d6d')
ax.axvline(0, color='0.6', linewidth=1)
ax.set_yticks(np.arange(len(ARMS)), [a.replace('_phase', ' / ').replace('_z', ' iter / Z ') for a in ARMS], fontsize=8)
ax.invert_yaxis(); ax.set_xlabel('AP difference: rank after minus before [percentage points]')
ax.set_title('(d) R4: paired temporal-block bootstrap, 95% interval')
fig.savefig(destination / 'figures/controlled_comparison.png', dpi=180)
plt.close(fig)

def markdown_table(headers, rows):
    return '\n'.join(['| '+' | '.join(headers)+' |',
                      '| '+' | '.join(['---']*len(headers))+' |']+
                     ['| '+' | '.join(map(str,row))+' |' for row in rows])

def label(arm):
    rule,phase,gate=arm.split('_')
    return f'{"旧規則" if rule=="legacy" else "ID優先統一"}・{phase[5:]}回・Z{gate[1:]}'

phase_rows=[]
for arm in ARMS[:4]:
    r=base.loc[arm]
    phase_rows.append([label(arm),f'{r.event_ap:.6f}',int(r['fp_r0.8']),int(r['fp_r0.9']),f'{r.maximum_recall:.4f}'])
rank_rows=[]
for arm in ARMS:
    a=primary.loc[primary.arm.eq(arm)&primary['rank'].eq('before')].iloc[0]
    b=primary.loc[primary.arm.eq(arm)&primary['rank'].eq('after')].iloc[0]
    rank_rows.append([label(arm),f'{a.event_ap:.6f}',f'{b.event_ap:.6f}',f'{b.event_ap-a.event_ap:+.6f}',
                      f'{int(a["fp_r0.8"])} → {int(b["fp_r0.8"])}',f'{int(a["fp_r0.9"])} → {int(b["fp_r0.9"])}'])
gate_rows=[]
for gate in gates:
    r=base.loc[f'strict_phase3_z{gate}']
    gate_rows.append(['なし' if gate=='none' else f'±{gate} mm',f'{r.event_ap:.6f}',
                      int(r['fp_r0.8']),int(r['fp_r0.9']),f'{r.maximum_recall:.4f}'])
tables='### 位相回復回数の比較\n\n'+markdown_table(
    ['条件','AP','FP @ recall ≥ 0.8','FP @ recall ≥ 0.9','最大recall'],phase_rows)
tables+='\n\n順位は抑制前に固定し、学習Z許容差は±15 mmとした。\n'
tables+='\n### R4：順位付けを抑制前から抑制後へ変えた比較\n\n'+markdown_table(
    ['条件','抑制前 AP','抑制後 AP','ΔAP（後−前）','FP @ 0.8（前→後）','FP @ 0.9（前→後）'],rank_rows)
tables+='\n\n各行で生存候補と補正後の座標は同一で、順位に応じたスコアだけが変わる。\n'
tables+='\n### R5：統一規則・3回での学習Z許容差\n\n'+markdown_table(
    ['学習Z許容差','AP','FP @ recall ≥ 0.8','FP @ recall ≥ 0.9','最大recall'],gate_rows)
tables+='\n\n順位は抑制前に固定した。全条件のテスト評価ではZゲートを使用していない。\n'
(destination/'tables.md').write_text(tables)

write_json({'status': 'PASS', 'files': {str(p.relative_to(destination)): sha256(p)
           for p in sorted(destination.rglob('*')) if p.is_file() and p.name != 'manifest.json'},
           'source_protocol_sha256': sha256(ROOT / 'config/protocol.json'),
           'revision_profile_sha256': sha256(Path(__file__).with_name('revision-profile.json')),
           'experiment_source_sha256': {p.name:sha256(p) for p in sorted(Path(__file__).parent.glob('*.py'))},
           'engine_sha256': json.loads((ROOT / 'config/engine-sha256.json').read_text()),
           'scope': 'Preflight controls on existing holograms, not a regenerated release.'}, destination / 'manifest.json')
emit(status='PASS', output=str(destination), models=len(ARMS))
