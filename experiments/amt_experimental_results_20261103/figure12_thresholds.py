"""Show observed, discrete review budgets at the requested precision/coverage levels."""
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import sys
from common import *
sys.path.insert(0,str(REPO/'scripts'))
from figure_support import review_budget as original


def thresholds(curve):
    assert curve.top_n_per_scene.tolist() == list(range(1, 81))
    # All sequences contribute the same integer N. Never invent a pooled-score
    # budget or interpolate a fractional review as an observed operating point.
    last_above = curve.loc[curve.top_n_precision.ge(.5)].iloc[-1]
    next_below = curve.loc[curve.top_n_per_scene.eq(last_above.top_n_per_scene + 1)].iloc[0]
    first_95 = curve.loc[curve.coverage_of_reviewed_confirmed.ge(.95)].iloc[0]
    def point(row):
        return dict(per_sequence=int(row.top_n_per_scene), total_reviews=int(row.candidates_reviewed),
                    confirmed=int(row.cumulative_confirmed_collisions),
                    precision=float(row.top_n_precision), coverage=float(row.coverage_of_reviewed_confirmed))
    return dict(precision_target=.5, coverage_target=.95,
                exact_precision_50_exists=bool(np.isclose(curve.top_n_precision,.5,rtol=0,atol=1e-12).any()),
                last_precision_at_least_50=point(last_above), next_below_50=point(next_below),
                first_coverage_at_least_95=point(first_95),
                policy='Equal integer top-N reviews within each of 14 sequences; all labels in precision denominator; coverage denominator 186 confirmed in top80.')


def main():
    curve=pd.read_csv(DATA/'review_budget.csv')
    result=thresholds(curve)
    assert not result['exact_precision_50_exists']
    assert [result[k]['per_sequence'] for k in ['last_precision_at_least_50','next_below_50','first_coverage_at_least_95']]==[24,25,34]
    save(DATA/'figure12_thresholds.json',result)
    operating=curve.loc[curve.top_n_per_scene.isin([20,24,25,30,33,34,60,80])].rename(columns={
        'top_n_per_scene':'reviews_per_scene','candidates_reviewed':'total_reviews',
        'cumulative_confirmed_collisions':'confirmed_tp','coverage_of_reviewed_confirmed':'confirmed_tp_coverage'})
    operating.to_csv(DATA/'review_operating_points.csv',index=False)
    plt.rcdefaults()
    original.configure_plotting()
    plt.rcParams.update({'pdf.fonttype':42,'ps.fonttype':42})
    fig=plt.figure(figsize=(9.2,5.4));ax=fig.add_subplot(1,1,1)
    fig.subplots_adjust(left=.105,right=.975,top=.80,bottom=.15)
    n=curve.top_n_per_scene.to_numpy(int)
    ax.step(np.r_[0,n],np.r_[0,100*curve.coverage_of_reviewed_confirmed],where='post',
            color=original.ORANGE,lw=2.2,label='Coverage of 186 confirmed collisions')
    ax.plot(n,100*curve.top_n_precision,color=original.BLUE,lw=2.2,label='Top-N precision')
    for key in ['last_precision_at_least_50','first_coverage_at_least_95']:
        point=result[key];rank=point['per_sequence'];precision=point['precision']*100
        ax.axvline(rank,color=original.GRAY,lw=1.5,linestyle='-',zorder=0)
        # Only the review count and precision annotate each of the two guides.
        ax.annotate(f'N = {rank}\nPrecision {precision:.1f}%',xy=(rank,precision),
                    xytext=(-8,-12) if rank==24 else (8,9),textcoords='offset points',
                    ha='right' if rank==24 else 'left',va='top' if rank==24 else 'bottom',fontsize=11,color=original.BLUE)
    ax.set(xlim=(0,80),ylim=(0,104),xlabel='Candidates reviewed per scene',ylabel='Coverage and precision [%]')
    ax.set_xticks(np.arange(0,81,10));ax.yaxis.set_major_locator(mticker.MultipleLocator(20.))
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=100.));original.style_axis(ax)
    upper=ax.secondary_xaxis('top',functions=(lambda v:v*14,lambda v:v/14))
    upper.set_xlabel('Total candidates reviewed across 14 scenes')
    upper.set_xticks(np.arange(0,81,20)*14);upper.tick_params(direction='out',width=.8)
    handles,legends=ax.get_legend_handles_labels()
    fig.legend(handles,legends,frameon=False,loc='upper center',bbox_to_anchor=(.54,.965),ncols=2)
    save(DATA/'figure12_display.json',dict(original_style_source='scripts/figure_support/review_budget.py',
        figure_size_inches=[9.2,5.4],axes_position=[.105,.15,.870,.65],
        vertical_guides=[24,34],annotations=['N = 24 / Precision 51.5%','N = 34 / Precision 37.2%'],
        unchanged_curve_data_sha256=sha(DATA/'review_budget.csv'),horizontal_guides=0,shaded_bands=0))
    for suffix in ['pdf','png']:fig.savefig(FIGURES/f'fig12.{suffix}',dpi=300,facecolor='white')
    plt.close(fig)
    print(result)


if __name__=='__main__':main()
