"""Publish measured comparison figures, including current submission assembly."""
from common import *
import importlib.util
import subprocess
import sys
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

PROJECT=ROOT.parents[1];DATA=PROJECT/'HDCD-materials';AUDIT=ROOT.parent
sys.path.append(str(DATA/'scripts'))
# The host's installed ReportLab is read-only; the local venv supplies pypdf.
# Append after the pinned environment so its numerical dependencies win.
sys.path.append('/usr/lib/python3/dist-packages')
from figure_support.delivery import render
from ra3d._core.matching import candidate_truth_adjacency

METHODS=['ours','depth_free','minip2d']
NAMES={'ours':'Ours','depth_free':'Depth-free','minip2d':'MinIP-based 2D'}
COLORS={'ours':'#0072B2','depth_free':'#D55E00','minip2d':'#009E73'}
OUT=ROOT/'figures';OUT.mkdir(exist_ok=True)

def main():
    summaries=[];curves={};metrics={};predictions={};pr=[];targets=[];composition=[]
    truth=events(260745)
    for method in METHODS:
        folder=ROOT/f'results/{method}'
        summaries.append(json.loads((folder/'summary.json').read_text()))
        curves[method]=pd.read_csv(folder/'precision_recall.csv',float_precision='round_trip')
        metrics[method]=json.loads((folder/'metrics.json').read_text())
        predictions[method]=pd.read_parquet(folder/'predictions.parquet')
        label='minip_2d' if method=='minip2d' else method
        curve=curves[method].copy();curve['cohort']='terminal';curve['method']=label;pr.append(curve)
        for target in [.5,.6,.7,.8,.85,.9,.95]:
            matching=curve.loc[curve.recall>=target]
            if not matching.empty:targets.append({'cohort':'terminal','method':label,'target_recall':target,**matching.iloc[0].to_dict()})
        point=curve.loc[curve.recall>=.8].iloc[0]
        selected=predictions[method].loc[predictions[method].score>=point.threshold].reset_index(drop=True).copy()
        selected['z_um']=0.;projected=truth.copy();projected['z_um']=0.
        eligible=candidate_truth_adjacency(selected,projected)
        for kind,count in [('duplicate_or_competition',len(eligible)-int(point.matched_events)),
                           ('no_frame_xy_truth',len(selected)-len(eligible))]:
            composition.append({'cohort':'terminal','method':label,'category':kind,'count':count})
    table=pd.DataFrame(summaries);write_table(table,ROOT/'results/table3.csv')
    panels=ROOT/'results/figure08';panels.mkdir(exist_ok=True)
    for name,frame in [('event_pr',pd.concat(pr)),('event_summary',pd.DataFrame([
         {'cohort':'terminal','method':'minip_2d' if m=='minip2d' else m,**metrics[m]} for m in METHODS])),
         ('target_recall',pd.DataFrame(targets)),('fp_composition',pd.DataFrame(composition))]:
        write_table(frame,panels/f'{name}.csv')
    subprocess.run([str(DATA/'.venv-figures311/bin/python'),str(SCRIPT/'render_figure8.py')],check=True)
    spec=importlib.util.spec_from_file_location('submission_assembly',PROJECT/'CollisionPaper/本番フォーマット/prepare_figures.py')
    assembly=importlib.util.module_from_spec(spec);spec.loader.exec_module(assembly);assembly.register_label_font()
    manifest=json.loads((PROJECT/'CollisionPaper/本番フォーマット/figure_sources/manifest.json').read_text())
    entry=next(x for x in manifest if x['output']=='fig08.pdf')
    assembly.combine([panels/Path(n).name for n in entry['sources']],OUT/'figure08.pdf',entry['columns'],entry.get('width_fraction',.98))
    render(OUT/'figure08.pdf',OUT/'figure08.png','/usr/bin/pdftoppm',150)
    plt.rcParams.update({'font.size':11,'axes.spines.top':False,'axes.spines.right':False,
                        'pdf.fonttype':42,'legend.frameon':False})
    published=json.loads((AUDIT/'results/published-inference/verification.json').read_text())['terminal']
    comparison=[]
    for method,dim in zip(METHODS,['3d','2d','minip2d']):
        cases={'Published':published[dim]['metrics'],
            '74.9 mm only':json.loads((AUDIT/f'results/range_only/{method}/metrics.json').read_text()),
            'Oct 29 combined changes':json.loads((AUDIT/f'aligned/results/{method}/metrics.json').read_text()),
            'Revised 102.3 mm / phase 3':metrics[method]}
        for case,m in cases.items():
            row={'method':method,'case':case,'ap':m['event_ap'],'max_recall':m['maximum_recall'],'max_f1':m['maximum_f1']}
            for target in [.8,.9]:
                point=m[f'operating_at_recall_{target:.2f}']
                row[f'fp_{target}']=None if point is None else int(point['false_positive_events'])
                row[f'precision_{target}']=None if point is None else point['precision']
            comparison.append(row)
    compare=pd.DataFrame(comparison);write_table(compare,ROOT/'results/scenario-comparison.csv')
    fig,axes=plt.subplots(1,3,figsize=(16,6))
    cases=['Published','74.9 mm only','Oct 29 combined changes','Revised 102.3 mm / phase 3'];colors=['#B2BBC4','#69AFBF','#C6996C','#284E70']
    x=np.arange(3);width=.20
    for ax,column,title in zip(axes,['ap','fp_0.8','fp_0.9'],['(a) Event AP','(b) False positives at recall 0.8','(c) False positives at recall 0.9']):
        for i,(case,color) in enumerate(zip(cases,colors)):
            vals=compare.loc[compare['case'].eq(case)].set_index('method').loc[METHODS,column].to_numpy()
            bars=ax.bar(x+(i-1.5)*width,vals,width,color=color,label=case)
            ax.bar_label(bars,fmt='%.3f' if column=='ap' else '%.0f',padding=4,fontsize=9,rotation=90)
        ax.set_xticks(x,[NAMES[m].replace('-based ','\n') for m in METHODS]);ax.set_title(title);ax.margins(y=.18)
        if column=='ap':ax.set_ylim(0,1.12)
    fig.suptitle('Synthetic benchmark: revised generation and common manuscript conditions',fontsize=15)
    handles,labels=axes[0].get_legend_handles_labels()
    fig.legend(handles,labels,ncol=4,loc='upper center',bbox_to_anchor=(.5,.92),fontsize=10)
    fig.text(.5,.015,'All scenarios: test seed 260745, 2,000 truth events, common time/XY matching.\n'
             'Combined-condition comparisons do not isolate one factor; revised generation changes the truth events.',ha='center',fontsize=9)
    fig.tight_layout(rect=(0,.07,1,.84));fig.savefig(OUT/'comparison.png',dpi=170);fig.savefig(OUT/'comparison.pdf');plt.close(fig)
    # Candidate labels are recomputed after each stage's new selection.
    labelrows=[]
    for method in METHODS:
        for seed in TRAIN:
            for stage in ['pool','static','images']:
                record=json.loads(path(seed,method,stage+'.json').read_text())
                labelrows.append({'method':method,'seed':seed,'stage':stage,**record['labels']})
    frame=pd.DataFrame(labelrows)
    write_table(frame,ROOT/'results/training-labels.csv')
    write_json({'status':'PASS','methods':METHODS,'figure08_assembly':'Current submission prepare_figures.py',
                'figure08_axis_adjustment':json.loads((panels/'axis-audit.json').read_text()),
                'protocol_sha256':sha256(ROOT/'config/protocol.json'),
                'figure_hashes':{p.name:sha256(p) for p in OUT.glob('*.pdf')}},ROOT/'results/figures.json')

if __name__=='__main__':main()
