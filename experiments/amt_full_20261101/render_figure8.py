"""Use the public figure style without clipping newly measured FP counts."""
from bootstrap import ROOT
import sys
sys.path.append(str(ROOT.parents[1]/'HDCD-materials/scripts'))
import matplotlib as mpl
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import json
from figure_support.fonts import register_fonts
from figure_support import collision_plot as plot

assert mpl.__version__=='3.11.1'
output=ROOT/'results/figure08'
mpl.rcdefaults();register_fonts();plot.configure_matplotlib();plot.OUTPUT=output
axis_audit={}

def save_figure(fig,stem):
    stem={'05_validation_topn_precision_recall':'02_validation_topn_precision_recall',
          '02_validation_false_positive_burden':'03_validation_false_positive_burden'}.get(stem,stem)
    ax=fig.axes[0]
    if stem=='03_validation_false_positive_burden':
        values=np.concatenate([np.asarray(line.get_ydata(),dtype=float) for line in ax.lines])
        values=values[np.isfinite(values)]
        if len(values):
            low,high=float(values.min()),float(values.max())
            assert low>=0
            if low==0:
                ax.set_yscale('symlog',linthresh=1,linscale=1)
                ax.set_ylim(-.08,max(4000,high*1.3))
                ax.set_ylabel('Number of false positives\n(linear 0–1; logarithmic above 1)')
            else:ax.set_ylim(min(.8,low*.8),max(4000,high*1.3))
            axis_audit.update({'scale':ax.get_yscale(),'limits':list(ax.get_ylim()),
                              'measured_minimum':low,'measured_maximum':high,'includes_zero':low==0})
    if stem=='04_validation_false_positive_composition':
        ax.set_yticklabels(['Ours','Depth-free\nvariant','MinIP-based 2D'])
    if stem=='02_validation_topn_precision_recall':
        for text in ax.get_legend().get_texts():text.set_fontsize(12.)
    fig.savefig(output/f'{stem}.png',dpi=plot.PNG_DPI)
    fig.savefig(output/f'{stem}.pdf');plt.close(fig)

plot.save_figure=save_figure
curves=pd.read_csv(output/'event_pr.csv',float_precision='round_trip')
plot.plot_precision_recall(curves,pd.read_csv(output/'event_summary.csv',float_precision='round_trip'))
plot.plot_topn(curves)
plot.plot_fp_burden(pd.read_csv(output/'target_recall.csv',float_precision='round_trip'))
plot.plot_fp_composition(pd.read_csv(output/'fp_composition.csv'))
(output/'axis-audit.json').write_text(json.dumps(axis_audit,indent=2)+'\n')
