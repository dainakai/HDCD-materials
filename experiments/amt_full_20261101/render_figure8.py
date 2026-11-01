"""Use the public figure style with a lower FP limit that includes one FP."""
from bootstrap import ROOT
import sys
sys.path.append(str(ROOT.parents[1]/'HDCD-materials/scripts'))
import matplotlib as mpl
import matplotlib.pyplot as plt
import pandas as pd
from figure_support.fonts import register_fonts
from figure_support import collision_plot as plot

assert mpl.__version__=='3.11.1'
output=ROOT/'results/figure08'
mpl.rcdefaults();register_fonts();plot.configure_matplotlib();plot.OUTPUT=output

def save_figure(fig,stem):
    stem={'05_validation_topn_precision_recall':'02_validation_topn_precision_recall',
          '02_validation_false_positive_burden':'03_validation_false_positive_burden'}.get(stem,stem)
    ax=fig.axes[0]
    if stem=='03_validation_false_positive_burden':
        # The new Ours result is one FP at recall 0.5. The public lower limit
        # of 1.6 would silently clip that measured point.
        ax.set_ylim(.8,4000)
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
