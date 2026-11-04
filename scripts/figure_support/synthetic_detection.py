"""Prepare Figure 8 from event predictions and common matching rules."""
from pathlib import Path
import json,subprocess,sys
import pandas as pd
from ra3d.io import write_table
from ra3d._core.matching import candidate_truth_adjacency
from figure_support import composition
REPO=Path(__file__).resolve().parents[2]
METHODS=['ours','depth_free','minip2d']

def main(output,python):
    summaries=[];curves={};metrics={};predictions={};pr=[];targets=[];fp_composition=[]
    truth=pd.read_parquet(REPO/'data/test/collision_events.parquet')
    for method in METHODS:
        folder=REPO/f'reference/evaluation/{method}'
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
            fp_composition.append({'cohort':'terminal','method':label,'category':kind,'count':count})
    table=pd.DataFrame(summaries);write_table(table,output/'table4.csv')
    panels=output/'figure8-panels';panels.mkdir(exist_ok=True)
    for name,frame in [('event_pr',pd.concat(pr)),('event_summary',pd.DataFrame([
         {'cohort':'terminal','method':'minip_2d' if m=='minip2d' else m,**metrics[m]} for m in METHODS])),
         ('target_recall',pd.DataFrame(targets)),('fp_composition',pd.DataFrame(fp_composition))]:
        write_table(frame,panels/f'{name}.csv')
    subprocess.run([str(python),str(REPO/'scripts/render_figure8.py'),'--output',str(panels)],check=True)
    composition.register_label_font()
    entry=next(e for e in json.loads((REPO/'reference/figure-layouts.json').read_text()) if e['output']=='fig08.pdf')
    composition.combine([panels/n for n in entry['sources']],output/'figures/fig08.pdf',entry['columns'],.98)
