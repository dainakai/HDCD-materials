"""Consolidate existing verified figures and results; never rerun scientific fits."""
import html
import shutil
import pandas as pd
from pypdf import PdfWriter
from common import *


def main():
    from build_report import table,figure_card
    synthetic=WORK/'audit/revision-paper-final-20261103/package'
    assert read(synthetic.parent/'verification.json')['status']=='PASS'
    target=PACKAGE/'synthetic-reference';target.mkdir(exist_ok=True)
    copied={}
    for source in [*(synthetic/'figures'/f'fig{n:02d}.{ext}' for n in [7,8,9] for ext in ['pdf','png']),
                   *(synthetic/'tables'/f for f in ['table04-metrics.csv','table05-stages.csv','version-comparison.csv']),
                   synthetic/'data/figure07-summary.json',synthetic/'data/figure09-cases.json']:
        shutil.copy2(source,target/source.name)
        assert sha(source)==sha(target/source.name)
        copied[source.name]=sha(source)
    f7=read(synthetic/'data/figure07-summary.json')
    methods=pd.read_csv(synthetic/'tables/table04-metrics.csv')
    stages=pd.read_csv(synthetic/'tables/table05-stages.csv')
    versions=pd.read_csv(synthetic/'tables/version-comparison.csv')
    result=read(DATA/'results.json');thresholds=read(DATA/'figure12_thresholds.json')
    base=read(synthetic.parent/'publication.json')['report_url'].removesuffix('/synthetic.html')
    names={'ours':'Ours','depth_free':'Depth-free','minip2d':'MinIP2D'}
    method_table=table(['手法','AP','最大F1','Recall 80%のPrecision / FP','Recall 90%のPrecision / FP'],
        [[names[r.method],f'{r.event_ap:.6f}',f'{r.maximum_f1:.6f}',
          f'{r.precision_at_recall_0p8*100:.2f}% / {r.false_positives_at_recall_0p8}',
          f'{r.precision_at_recall_0p9*100:.2f}% / {r.false_positives_at_recall_0p9}'] for r in methods.itertuples()])
    stage_names={'trajectory':'軌跡','image':'＋画像特徴','relation':'＋関係モデル','suppression':'＋座標による重複抑制'}
    stage_table=table(['段階','AP','最大F1','FP80','FP90'],[[stage_names[r.stage],f'{r.event_ap:.6f}',f'{r.maximum_f1:.6f}',
        r.false_positives_at_recall_0p8,r.false_positives_at_recall_0p9] for r in stages.itertuples()])
    comparison=[]
    for method in ['ours','depth_free','minip2d']:
        old=versions.loc[versions.method.eq(method)&versions['case'].eq('Published')].iloc[0]
        new=versions.loc[versions.method.eq(method)&versions['case'].eq('Revised 102.3 mm / phase 3')].iloc[0]
        comparison.append([names[method],f'{old.ap:.6f} → {new.ap:.6f}',f'{old.max_f1:.6f} → {new.max_f1:.6f}',
                           f'{int(old["fp_0.8"])} → {int(new["fp_0.8"])}',f'{int(old["fp_0.9"])} → {int(new["fp_0.9"])}'])
    comparison_table=table(['手法','AP：公開版 → 新版','最大F1','FP80','FP90'],comparison)
    budget_table=table(['条件','各系列の件数','総レビュー数','Precision','Coverage'],
        [[label,p['per_sequence'],p['total_reviews'],f'{p["precision"]*100:.2f}%',f'{p["coverage"]*100:.2f}%']
         for label,p in [('Precisionが50%以上の最大件数',thresholds['last_precision_at_least_50']),
                         ('その次の件数',thresholds['next_below_50']),
                         ('Coverageが95%以上になる最小件数',thresholds['first_coverage_at_least_95'])]])
    fig7note=(f'衝突検出と同じテスト系列seed 260745・6,000フレームを使用。粒子APは{f7["particle_ap"]:.6f}。'
        f'XY位置誤差の中央値は{f7["xy_p50_p90_p95_um"][0]:.2f} µm、絶対Z誤差の中央値は{f7["absolute_z_p50_p90_p95_um"][0]:.2f} µmです。'
        'PR曲線の低confidence追加検出と、(b–f)の計測・追跡用データは用途が異なりますが、異なる試験系列ではありません。')
    synthetic_notes={7:fig7note,
        8:'同じ新版のテスト系列による衝突検出評価です。OursのAP・最大F1・Recall 80%/90%でのPrecisionは比較2法を上回ります。最大Recallは比較法の方が高く、全指標で優れるとは書けません。',
        9:'確定例は(a) N1、(b) D4です。N1は154＋242→261 µmの真の衝突でスコア0.998955→0.997927。D4は同じ真値粒子7364を2つの観測IDが追う例で0.943667→0.624751です。N1は最終候補に残り、D4は除外されます。表示スコアは関係モデル前の値で、最終除外を画像特徴だけの効果とは解釈しません。'}
    synthetic_cards=''.join(f'''<article class="card" id="fig{n:02d}"><div class="figure-title"><span>FIGURE {n}</span><h3>{title}</h3></div>
<p>{html.escape(synthetic_notes[n])}</p><figure><a href="synthetic-reference/fig{n:02d}.pdf"><img loading="lazy" src="synthetic-reference/fig{n:02d}.png" alt="確定版Fig.{n}"></a></figure>
<div class="buttons"><a class="button" href="synthetic-reference/fig{n:02d}.pdf">PDF</a><a class="button" href="synthetic-reference/fig{n:02d}.png">PNG</a></div></article>'''
        for n,title in [(7,'同じテスト系列での粒子計測'),(8,'合成ホログラムの衝突検出性能'),(9,'確定したN1・D4の事例')])
    inventory=table(['対象','今回の状態','主要結果・本文の注意点'],[
        ['Fig.3','従来の処理手順の図を維持','今回の統計更新の対象外。'],
        ['Fig.4–5','新版の実験計測で作成済み','72,047軌跡。合成データ生成に使用した旧実験統計は固定したまま。'],
        ['Fig.7','テスト系列の出典確認済み',f'粒子AP {f7["particle_ap"]:.6f}。'],
        ['Fig.8 / Table 4','全量評価と照合済み','Ours AP 0.922169、最大F1 0.888366、FP80 51、FP90 486。'],
        ['Fig.9','N1・D4で著者確定','元の図のデザインを使用。画像・例示の更新で全量性能値は不変。'],
        ['Table 5','段階別評価と照合済み','画像特徴段階のAPとFP90は改善するが、FP80は177→183。'],
        ['Fig.10','03・08で著者確定','元の配置を復元。画像枠内の視野のみ03は1.33倍、08は1.07倍。'],
        ['Fig.11','1,108件の二値判定を集計','確認衝突186、非衝突922。曖昧12件を除く。'],
        ['Fig.12','元デザイン＋2本の縦線','N=24：P 51.49%、C 93.01%。N=34：P 37.18%、C 95.16%。'],
        ['Fig.13','全除外理由を監査済み','粒径184 TP / 908 FP、速度179 TP / 822 FP。外れ値は除去しない。'],
        ['Fig.14','全記録時間と公称体積で再計算','観測186、予測222.41 ± 14.91、観測/予測83.63%。'],
        ['Fig.15','両親粒径が有効な185件を集計','総186件のうち1件は親粒径欠測。31区間ペア。']])
    exp_compare=table(['指標','公開版','新版'],[
        ['確認衝突 / 非衝突 / 曖昧','185 / 896 / 39','186 / 922 / 12'],
        ['上位30件：Coverage / Precision','95.68% / 42.14%','94.09% / 41.67%'],
        ['粒径残差RMSE','19.836 µm','19.899 µm'],['速度残差RMSE','0.04263 m/s','0.07939 m/s'],
        ['重力予測（同じ時間・体積で比較）','210.119件','222.410件'],
        ['125–150 µm同士：観測 / 予測','18 / 約9.5件','7 / 9.60件']])
    report=f'''<!doctype html><html lang="ja"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>確定図表と結果の一覧｜合成・実験ホログラム</title><link rel="stylesheet" href="report.css"></head><body>
<header><div class="eyebrow">AMT · FIGURES AND RESULTS · 2026-11-03</div><h1>合成・実験ホログラムの<br>図表と結果のまとめ</h1><p>Fig.4–5・7–15とTable 4–5を、現在の確定版からまとめました。Fig.10は03・08、Fig.9はN1・D4です。Fig.10の配置とFig.12のデザインを戻し、依頼された表示範囲と2本の線だけを調整しました。</p>
<div class="buttons"><a class="button primary" href="all-figures.pdf">11図をまとめて見る PDF</a><a class="button" href="experimental-replacements.zip">実験節の差し替え一式</a><a class="button" href="{base}/paper-replacements.zip">合成節の差し替え一式</a><a class="button" href="index.html">実験節の詳細</a><a class="button" href="{base}/synthetic.html">合成節の詳細</a></div></header>
<nav><div><a href="#inventory">図表一覧</a><a href="#synthetic-results">合成の結果</a><a href="#experimental-results">実験の結果</a><a href="#fig10">Fig.10</a><a href="#fig12">Fig.12</a><a href="#integration">本文差し替え</a></div></nav><main>
<section id="inventory"><h2>現在の図表と確定内容</h2>{inventory}</section>
<section id="synthetic-results"><h2>合成ホログラム：Table 4・5</h2><p>生成深さ0–102.3 mm、計測の伝搬距離74.9–177.2 mm、1,024面、すべての位相回復3回の版です。R4は抑制前の順位を保持し、R5の学習対応は親・娘のID整合性と15 mmのZゲートを含む規則で統一されています。パディングと鉛直速度摂動は維持しています。</p>
<h3>Table 4：最終性能</h3>{method_table}<p><a href="synthetic-reference/table04-metrics.csv">全精度のCSV</a></p>
<h3>公開版からの変化</h3>{comparison_table}<p>真値・画像・教師規則・モデルを合わせて更新しているため、この差を位相回復3回だけの効果には帰属できません。前端だけを直した中間版との比較では、OursのFP90は430→486です。</p>
<h3>Table 5：段階ごとの変化</h3>{stage_table}<p>画像特徴段階ではAPが上がりFP90が減る一方、FP80は177→183、最大F1は0.849082→0.848550です。「全段階で全指標が改善」とは記述できません。候補集合と位置補正に使う特徴も変わる段階比較です。</p><p><a href="synthetic-reference/table05-stages.csv">全精度のCSV</a></p></section>
{synthetic_cards}
<section id="experimental-results"><h2>実験ホログラム：14系列・各80件</h2><p>1,120件の保存済みレビューから、確認衝突186件、非衝突922件、曖昧12件を集計しました。旧版と新版は同じ撮影系列ですが、各版で順位とレビュー集合が異なります。</p>{exp_compare}
<h3>Fig.12の2つの位置</h3>{budget_table}<p>50%ちょうどになる整数件数はありません。図は50%以上を保てる最大件数24と、Coverageが初めて95%以上になる34を示します。Coverageの分母は上位80件から確認した186件で、実験全体の絶対再現率ではありません。</p>
<h3>Fig.13の除外と残差</h3><p>粒径診断から除かれる28件は確認衝突2・非衝突14・曖昧12。速度診断から除かれる119件は7・100・12です。必要な計測窓の欠測や、速度算出に必要な3点未満の観測が理由です。全119候補の観測フレームと個別理由を監査しました。外れ値除去は行っていません。</p>
<p>確認衝突の速度RMSEは0.07939 m/sへ増加し、最大残差の1件が二乗誤差の64.3%を占めます。<a href="fig13-exclusions.html">各サンプルの除外理由</a>と<a href="index.html#physical-audit">速度外れ値の感度確認</a>を別途掲載しています。</p></section>
{''.join(figure_card(n) for n in range(10,16))}
<section id="population"><h2>実験集団の更新：Fig.4–5</h2><p>今回の再計測は17,876,795観測、平均116.97粒子/フレーム、平均LWC 15.40 g/m³です。速度分布は72,047軌跡。これらは新版の実験統計であり、合成データを作った旧経験分布は固定済みです。</p></section>
{figure_card(4)}{figure_card(5)}
<section id="integration"><h2>論文に反映する内容</h2><p>図・表・数値の差し替えに加え、再構成距離、教師対応規則、順位の固定順、テスト系列の利用履歴、APの定義、段階比較の解釈を本文でそろえる必要があります。実験節では、上位30件のCoverageが95%未満であること、速度残差の増加、粒径ペアの局所傾向を更新してください。</p>
<p>これらを反映した英文本文・キャプション・Abstract/Conclusion用の断片を各差し替えZIPに収録しています。論文本体は編集していません。原稿に挿入した後の全文コンパイルと最終配置確認は残っています。</p>
<p><a href="text/experimental_section.tex">実験節の英文案</a> · <a href="{base}/paper/latex/results-synthetic.tex">合成節の英文案</a> · <a href="data/consolidated_results_provenance.json">一覧の出典と図の同一性</a></p></section>
<footer>生成: {stamp()} · Quick Tunnelの停止でリンクは利用できなくなり、作り直すとURLが変わります。</footer></main></body></html>'''
    (PACKAGE/'summary.html').write_text(report)
    writer=PdfWriter()
    for n in [4,5,7,8,9,10,11,12,13,14,15]:
        path=target/f'fig{n:02d}.pdf' if n in [7,8,9] else FIGURES/f'fig{n:02d}.pdf'
        writer.append(path,outline_item=f'Figure {n}')
    writer.write(PACKAGE/'all-figures.pdf')
    save(DATA/'consolidated_results_provenance.json',dict(created_at=stamp(),
        synthetic_package='audit/revision-paper-final-20261103/package',synthetic_copied_files=copied,
        synthetic_performance_unchanged=True,experimental_results_sha256=sha(DATA/'results.json'),
        figure10_selected_options=['03','08'],figure9_selected_options=['N1','D4'],
        figures=[4,5,7,8,9,10,11,12,13,14,15],tables=[4,5],manuscript_edited=False))
    print('Consolidated 11 figures and Tables 4–5 with verified synthetic/experimental results.')


if __name__=='__main__':main()
