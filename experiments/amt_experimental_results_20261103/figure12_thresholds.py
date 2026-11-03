"""Show observed, discrete review budgets at the requested precision/coverage levels."""
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from common import *


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
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':11.5,'pdf.fonttype':42,'ps.fonttype':42})
    blue,orange='#236899','#c66b22'
    fig=plt.figure(figsize=(10.4,6.7))
    ax=fig.add_axes([.085,.13,.88,.57])
    n=curve.top_n_per_scene
    ax.step(n,100*curve.coverage_of_reviewed_confirmed,where='post',color=orange,lw=2.1,label='Confirmed-event coverage')
    ax.plot(n,100*curve.top_n_precision,color=blue,lw=2.1,label='Review precision')
    ax.axvspan(24,25,color=blue,alpha=.13,lw=0)
    ax.hlines(50,0,25,colors=blue,linestyles=':',lw=1.1)
    ax.vlines(34,0,100*177/186,colors=orange,linestyles='--',lw=1.1)
    ax.hlines(95,0,34,colors=orange,linestyles=':',lw=1.1)
    ax.scatter([24,25],[100*173/336,100*173/350],s=45,c=blue,edgecolors='white',linewidths=.8,zorder=5)
    ax.scatter([34],[100*177/186],s=65,c=orange,edgecolors='white',linewidths=.8,zorder=5)
    ax.annotate('24–25',(24.5,50),xytext=(28,61),color=blue,fontsize=10.5,
                arrowprops={'arrowstyle':'-','color':blue,'lw':.8})
    ax.annotate('34',(34,95.1613),xytext=(38,85),color=orange,fontsize=11,
                arrowprops={'arrowstyle':'-','color':orange,'lw':.8})
    ax.set(xlim=(0,80),ylim=(0,105),xlabel='Candidates reviewed per sequence, N',ylabel='Coverage or precision [%]')
    ax.set_xticks(range(0,81,10));ax.set_yticks(range(0,101,20))
    ax.grid(axis='y',color='#dce2e6',lw=.6);ax.set_axisbelow(True)
    ax.spines[['top','right']].set_visible(False)
    ax.legend(loc='lower left',bbox_to_anchor=(.02,.035),frameon=True,facecolor='white',edgecolor='#dce2e6')
    upper=ax.secondary_xaxis('top',functions=(lambda v:v*14,lambda v:v/14))
    upper.set_xlabel('Total reviews across 14 sequences',labelpad=9)
    upper.set_xticks(np.arange(0,1121,140))
    fig.text(.085,.955,'Precision = 50%: boundary between N = 24 and 25',color=blue,weight='bold',fontsize=11.4,va='top')
    fig.text(.085,.918,'N = 24: 336 reviews, precision 51.49%\nN = 25: 350 reviews, precision 49.43%\nBoth: 173 / 186 events, coverage 93.01%',fontsize=10.6,linespacing=1.5,va='top')
    fig.text(.565,.955,'Coverage ≥ 95%: first attained at N = 34',color=orange,weight='bold',fontsize=11.4,va='top')
    fig.text(.565,.918,'476 reviews, 177 / 186 confirmed events\nCoverage 95.16%; precision 37.18%\nN = 33 remains below 95% coverage (94.62%).',fontsize=10.6,linespacing=1.5,va='top')
    fig.text(.085,.035,'Coverage is conditional on the 186 confirmed events in the reviewed top-80 cohort.',fontsize=9.8,color='#465561')
    for suffix in ['pdf','png']:fig.savefig(FIGURES/f'fig12.{suffix}',dpi=300,facecolor='white')
    plt.close(fig)
    print(result)


if __name__=='__main__':main()
