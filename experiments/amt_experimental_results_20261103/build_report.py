"""Create the author package and mobile report without changing the manuscript."""
import html
import shutil
import subprocess
import zipfile
import hashlib
import numpy as np
import pandas as pd
from concurrent.futures import ThreadPoolExecutor
from common import *
from texts import CAPTIONS,LABELS,section,ABSTRACT,CONCLUSION,POPULATION

TITLES={4:'再処理した粒子集団の統計',5:'再処理した粒径・速度分布',10:'代表的な実験衝突の再構成',11:'レビュー判定と最終スコア',
    12:'レビュー件数・カバー率・適合率',13:'体積保存・運動量保存との整合性',14:'系列別・全体の衝突率',15:'親粒径ペア別の衝突数'}
NOTES={4:'追加の差し替え候補です。合成データを作った元の統計と、今回再処理した実験統計を本文で区別してください。',
5:'72,047軌跡を集計しました。鉛直流の補正値0.26 m/sは変更していません。図の表示範囲外の軌跡は元の集計から削除していません。',
10:'旧Fig.10と同じ物理的な衝突を、今回の軌跡・粒径・位相回復3回で再描画しました。新しい候補IDと表示時刻を記録し、表示上の時刻原点と解析の候補時刻を区別しました。',
11:'曖昧12件を除いた1,108件。確認衝突186件、非衝突922件です。高スコアでも非衝突が残るという解釈は維持できます。',
12:'Precision 50%の境界は各系列24件（51.49%）と25件（49.43%）の間で、Coverageはどちらも93.01%です。Coverageが初めて95%を超えるのは34件（95.16%、Precision 37.18%）。40件の縦線は削除しました。',
13:'粒径は確認衝突184件・非衝突908件、速度は179件・822件。欠測による除外のみで、外れ値は残しています。速度RMSEの増加を本文でも明記します。',
14:'全記録時間と102.3 mmの公称体積で再計算。観測186件、予測222.41 ± 14.91件、比は83.6%。誤差棒は固定した密度・カーネルに対するPoisson標準偏差です。',
15:'親粒径が両方そろう185件を31区間ペアに割り当てました。親粒径が1つ欠ける1件は総数186件に含めますが、このマップには入れません。'}

def table(headers,rows):
    return '<div class="table-wrap"><table><thead><tr>'+''.join('<th>'+html.escape(str(h))+'</th>' for h in headers)+'</tr></thead><tbody>'+''.join('<tr>'+''.join('<td>'+html.escape(str(v))+'</td>' for v in row)+'</tr>' for row in rows)+'</tbody></table></div>'

def figure_card(n):
    caption=html.escape(CAPTIONS[n])
    extra={10:'<p><a href="fig10-candidates.html">衝突例の候補10件をGIF・軌跡・静止画で比較する</a>（各例のPDFとPNG付き）</p>',13:'<p><a href="fig13-exclusions.html">各サンプルの除外理由を読む</a>（全119件、元の観測フレーム付き）</p>'}.get(n,'')
    old=(f'<details><summary>旧論文の Fig.{n} と見比べる</summary><p class="muted">旧論文の掲載図。旧レビュー・旧計測値を変更せず表示しています。</p><img class="baseline" loading="lazy" src="baseline/fig{n:02d}.png" alt="旧版 Fig.{n}"></details>' if n>=10 else '')
    return f'''<article class="card" id="fig{n:02d}"><div class="figure-title"><span>FIGURE {n}</span><h3>{TITLES[n]}</h3></div>
<p>{NOTES[n]}</p>{extra}<figure><a href="figures/fig{n:02d}.pdf"><img loading="lazy" class="{'case-image' if n==10 else ''}" src="preview/fig{n:02d}.png" alt="新版 Fig.{n}：{TITLES[n]}"></a></figure>
<div class="buttons"><a class="button" href="figures/fig{n:02d}.pdf">PDF</a><a class="button" href="figures/fig{n:02d}.png">PNG</a><a class="button" href="text/caption_fig{n:02d}.tex">キャプション TeX</a></div>
<details><summary>英文キャプション</summary><pre>{caption}</pre></details>{old}</article>'''

def write_texts():
    target=PACKAGE/'text';target.mkdir(exist_ok=True)
    for n,caption in CAPTIONS.items():(target/f'caption_fig{n:02d}.tex').write_text(caption+'\n')
    (target/'experimental_section.tex').write_text(section())
    (target/'abstract_experimental_replacement.tex').write_text(ABSTRACT+'\n')
    (target/'conclusion_experimental_replacement.tex').write_text(CONCLUSION+'\n')
    (target/'population_provenance_replacement.tex').write_text(POPULATION+'\n')
    manuscript=WORK/'CollisionPaper/本番フォーマット/template.tex'
    content=manuscript.read_text()
    anchors=[{'id':'experimental-section','start':r'\subsection{Demonstration on experimental holograms}', 'end':r'\conclusions',
        'file':'experimental_section.tex','operation':'Replace from the subsection heading up to, but excluding, the conclusions heading.'},
        {'id':'abstract','start':'In 14 experimental hologram sequences recorded in a vertical wind tunnel,',
         'end':'a reference prediction from a gravitational collision kernel.','file':'abstract_experimental_replacement.tex','operation':'Replace the three experimental-result sentences in the abstract; retain the surrounding synthetic-result sentences.'},
        {'id':'conclusion','start':'Application of the proposed method to 14 experimental hologram sequences yielded',
         'file':'conclusion_experimental_replacement.tex','operation':'Replace the experimental-result paragraph only; retain limitations and the synthetic evaluation.'},
        {'id':'population-provenance','start':'These statistics are used to generate synthetic holograms',
         'file':'population_provenance_replacement.tex','operation':'If replacing Figures 4–5, replace the population-introduction paragraph and retain the separate smoothing-method sentence. Clearly distinguish updated measurements from fixed synthetic inputs.'}]
    for a in anchors:assert a['start'] in content,a
    save(target/'replacement_anchors.json',{'manuscript_sha256_at_build':sha(manuscript),'anchors':anchors,
        'manual_integration_only':True,'synthetic_replacements':'Use the already delivered numerical-hologram package separately; this package does not replace Figures 7–9 or synthetic metrics.'})
    # A compact optional table; it is not assigned an existing manuscript table number.
    rows=pd.read_csv(DATA/'review_operating_points.csv')
    lines=[r'\begin{tabular}{rrrrr}',r'\hline',r'$N$ per sequence & Reviews & Confirmed & Coverage (\%) & Precision (\%) \\',r'\hline']
    for r in rows.itertuples():lines.append(f'{int(r.reviews_per_scene)} & {int(r.total_reviews)} & {int(r.confirmed_tp)} & {100*r.confirmed_tp_coverage:.1f} & {100*r.top_n_precision:.1f} '+r'\\')
    lines += [r'\hline',r'\end{tabular}']
    (target/'optional_review_budget_table.tex').write_text('\n'.join(lines)+'\n')

def main():
    s=read(DATA/'results.json');frozen=read(PRIVATE/'freeze.json')
    assert s['review']['confirmed']==186 and s['review']['minimum_n_for_95pct']==34
    assert s['pair_map_audit']['in_pair_map']==185 and s['gravity']['occupied_pair_bins']==31
    write_texts()
    shutil.copy2(Path(__file__).with_name('report.css'),PACKAGE/'report.css')
    (PACKAGE/'baseline').mkdir(exist_ok=True)
    def baseline(n):
        path=PACKAGE/'baseline'/f'fig{n:02d}.png'
        if not path.exists():subprocess.run(['/usr/bin/pdftoppm','-singlefile','-scale-to','1600','-png',str(WORK/f'CollisionPaper/本番フォーマット/figures/fig{n:02d}.pdf'),str(path.with_suffix(''))],check=True)
    with ThreadPoolExecutor(max_workers=3) as pool:list(pool.map(baseline,range(10,16)))
    compare=table(['指標','公開版','今回の版'],[
        ['確認衝突 / 非衝突 / 曖昧','185 / 896 / 39','186 / 922 / 12'],
        ['全レビューに対する適合率','16.52%','16.61%'],
        ['上位30件：確認衝突数','177 / 420','175 / 420'],
        ['上位30件：条件付きカバー率','95.68%','94.09%'],
        ['上位30件：適合率','42.14%','41.67%'],
        ['粒径整合性：有効な確認衝突','185','184'],['速度整合性：有効な確認衝突','180','179'],
        ['粒径残差RMSE [µm]','19.836','19.899'],['速度残差RMSE [m/s]','0.04263','0.07939'],
        ['割り当て可能な粒径ペア数','185','185（総数186のうち1件は欠測）']])
    budget=pd.read_csv(DATA/'review_operating_points.csv')
    bt=table(['各系列 N','総レビュー数','確認衝突','カバー率','適合率'],[[int(r.reviews_per_scene),int(r.total_reviews),int(r.confirmed_tp),f'{100*r.confirmed_tp_coverage:.2f}%',f'{100*r.top_n_precision:.2f}%'] for r in budget.itertuples()])
    gravity=pd.read_csv(DATA/'public_vs_revision_gravity.csv')
    gt=table(['条件','確認数','予測数 ± SD','観測 / 予測','観測率 [m⁻³s⁻¹]'],[[name,int(r.confirmed),f'{r.expected:.3f} ± {r.standard_deviation:.3f}',f'{100*r.observed_to_predicted:.2f}%',f'{r.observed_rate_m3_s:,.1f}'] for name,r in zip(['公開版の保存図','旧計測＋今回と同じ時間・体積・SD','今回の計測＋今回の条件'],gravity.itertuples())])
    scenes=pd.read_csv(DATA/'review_counts_by_scene.csv')
    st=table(['系列','原データID','確認衝突','非衝突','曖昧'],[[int(r.scene_index),f'S{int(r.scene_num):04d}',int(r.collision),int(r.non_collision),int(r.ambiguous)] for r in scenes.itertuples()])
    sens=pd.read_csv(DATA/'velocity_window_sensitivity.csv');sens=sens.loc[sens.role.eq('TP')]
    vt=table(['版','各窓の最小点数','有効数','RMSE [m/s]','絶対残差中央値 [m/s]'],[['公開版' if r.version=='public' else '今回',int(r.min_points_per_window),int(r.n),f'{r.rmse:.5f}',f'{r.median_absolute_error:.5f}'] for r in sens.itertuples()])
    download_rows=''.join(f'<li><a href="data/{p.name}">{p.name}</a></li>' for p in sorted(DATA.glob('*.csv')))
    report=f'''<!doctype html><html lang="ja"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>実験節の差し替え図表・結果 | AMT phase3</title><link rel="stylesheet" href="report.css"></head><body>
<header><div class="eyebrow">AMT · EXPERIMENTAL RESULTS · 2026-11-03</div><h1>実験節の図表と結果を<br>新版のレビューから再作成</h1><p>14系列・各80件のレビューを固定し、同じ新版の予測・軌跡・粒子計測から集計しました。Fig.10–15、英文本文、キャプション、数値表を差し替え用に用意しています。</p>
<div class="buttons"><a class="button" href="fig10-candidates.html">Fig.10候補10例</a><a class="button" href="fig13-exclusions.html">Fig.13全除外例の理由</a><a class="button primary" href="experimental-replacements.zip">差し替え一式 ZIP</a><a class="button" href="experimental_figures.pdf">図をまとめて読む PDF</a><a class="button" href="text/experimental_section.tex">実験節の英文 TeX</a></div>
<div class="stats"><div class="stat"><strong>1,120 / 1,120</strong><span>保存済みレビュー</span></div><div class="stat"><strong>186</strong><span>確認衝突（旧版185）</span></div><div class="stat"><strong>94.1%</strong><span>上位30件の条件付きカバー率</span></div><div class="stat"><strong>83.6%</strong><span>確認衝突数 / 重力予測</span></div></div></header>
<nav><div><a href="#findings">結果と変更点</a><a href="#figures">Fig.10–15</a><a href="#comparison">旧版比較</a><a href="#wording">本文差し替え</a><a href="#audit">検証・根拠</a><a href="#downloads">数値表</a></div></nav><main>
<section id="findings"><h2>図と数値だけでなく、3か所の解釈も更新します</h2><div class="notice"><strong>「掲載図と性能値を置き換えるだけ」で済む状態ではありません。</strong><p>差し替え用の英文には、以下の解釈の変更を反映済みです。論文本体は編集していません。</p></div>
<ol><li><strong>上位30件で95%を回収、とは書けません。</strong> 今回は175/186件＝94.1%。95%を超える最小件数は34件（Precision 37.18%）です。Precisionを50%以上に保つ最大の整数件数は24件で、Coverageは93.01%です。</li>
<li><strong>物理整合性には大きい外れが残ります。</strong> 速度のRMSEは0.04263から0.07939 m/sへ増加しました。「非衝突の方が両関係で広く散らばる」という一律の記述は避け、中央値・平均絶対残差とRMSEを分けて記載します。</li>
<li><strong>粒径ペアの局所的な説明が変わります。</strong> 125–150 µm同士は、旧版18件対予測9.5件から、今回7件対9.60件へ変化しました。総数186件と、粒径ペアを割り当てられる185件も区別します。</li></ol>
<h3>レビュー件数の選択</h3><p>各系列で同じ整数件数をレビューするため、Precisionがちょうど50%になる点はありません。24件ずつ・計336件では51.49%、25件ずつ・計350件では49.43%です。34件ずつ・計476件でCoverageが95.16%に達します。</p>{bt}<p class="muted">曖昧な候補もレビュー作業量と適合率の分母に含めます。カバー率の分母は今回確認した186件です。80位より下や候補化されなかった真の衝突を含む絶対再現率ではありません。</p></section>
<div id="figures"></div>{''.join(figure_card(n) for n in range(10,16))}
<section id="comparison"><h2>公開版から何が変わったか</h2><p>同じ14本の撮影系列を再処理し、新しいモデルの順位に従って選ばれた上位80件を今回レビューしました。下表は各版のレビュー集合の比較です。順位を使った旧判定の転記や、異なる候補を同一イベントとみなす対応付けはしていません。</p>{compare}
<p>上位10件では確認数113→118、上位20件では161→169と増えていますが、上位30件では177→175です。順位全体にわたり一律に改善したとはいえません。候補集合・処理条件・GPU数値差・今回の判定が変わるため、特定の変更だけの因果効果とも解釈できません。</p>
<h3>重力モデル比較は、計測値の変更とR7の条件統一を分ける</h3>{gt}<p>時間・体積・誤差棒だけを今回の本文条件にそろえると旧計測で予測210.119件です。今回の再計測では222.410件となり、さらに5.85%増加しました。予測の増加をPoisson化だけで説明することはできません。</p>
<details><summary>系列ごとのレビュー内訳</summary>{st}</details></section>
<section id="physical-audit"><h2>欠測と速度の外れを明示する</h2><p><a href="fig13-exclusions.html">全119候補の個別除外理由と元軌跡の観測フレーム</a>を別ページにまとめました。粒径の除外28件は確認衝突2・非衝突14・曖昧12、速度の除外119件は確認衝突7・非衝突100・曖昧12です。速度APIで点数0とされる中には実観測が1点ある場合もあり、実際の点数を区別しました。</p><p>粒径の診断は、親2粒子と衝突後の粒径がそろう184件。速度は各窓に3点以上ある179件です。確認衝突のラベル自体は変更していません。</p>
<p><strong>S0015の34位</strong>は親粒径が1つ欠けています。特徴量の「小さい親／大きい親」は片方が欠けると残った1つの値で作られるため、そのまま粒径ペア集計に使うと誤った対角セルに入ります。今回は親2つの有限・正の値を明示的に確認し、この1件をペアマップから除外しました。総衝突数からは除外していません。<strong>S0022の27位</strong>は衝突後粒径がなく、粒径整合性から除外していますが、親2粒径があるためペア集計には入ります。</p>
<p>最大の速度残差は<strong>S0025の8位</strong>で、予測0.717 m/sに対して測定1.569 m/s、衝突後の観測は3点です。この1件が確認衝突の速度残差二乗和の64.3%、上位2件が82.2%を占めます。短い軌跡窓がRMSEを強く左右することは確認できますが、レビューラベルの誤りと断定はしていません。</p>
<h3>窓内の必要点数を変えた補足確認</h3>{vt}<p class="muted">主図は公開コードと同じ3点の条件を維持。5点・10点の結果は感度確認です。追加の閾値で外れを取り除いた値を主結果にはしていません。</p></section>
<section id="wording"><h2>論文へ移すファイル</h2><ol><li><code>figures/fig10.pdf</code>～<code>fig15.pdf</code>を同名で差し替えます。図番号と参照ラベルを維持しました。</li><li><a href="text/experimental_section.tex">experimental_section.tex</a>を「Demonstration on experimental holograms」の節全体と入れ替えます。図の挿入文とキャプションも含みます。</li><li>要旨の実験結果3文は<a href="text/abstract_experimental_replacement.tex">要旨用の英文</a>、結論の実験結果段落は<a href="text/conclusion_experimental_replacement.tex">結論用の英文</a>で更新します。</li><li>既に渡した合成ホログラムの差し替え一式（Fig.7–9、表、数値）は別に適用します。今回のファイルはその内容を上書きしません。</li></ol>
<p><a href="text/replacement_anchors.json">置換箇所と元原稿ハッシュ</a>、<a href="text/optional_review_budget_table.tex">任意のレビュー予算表 TeX</a>も付属します。実験結果節には既存の番号付き表がないため、追加表に勝手な論文表番号は振っていません。</p>
<details><summary>英文本文を読む</summary><pre>{html.escape(section())}</pre></details>
<h3>Fig.4–5も再処理版を用意しました</h3><p>実験の粒子集団を紹介する図として、新しい測定値から再作成しています。ただし「この図の統計から今回の合成データを生成した」と書くと履歴と異なります。合成データの元の実験分布と鉛直速度摂動は固定したままです。差し替える場合は<a href="text/population_provenance_replacement.tex">データの由来を区別する英文</a>も併用してください。Fig.3は手順説明図なので今回の数値更新対象に含めていません。</p>
<details><summary>補助のFig.4–5を見る</summary>{figure_card(4)}{figure_card(5)}</details></section>
<section id="audit"><h2>今回の検証</h2><div class="notice good"><strong>レビューと計算の対応を固定</strong><p>全14系列の実行完了を確認し、SQLiteの整合したコピーに1,120判定と1,121履歴を保存しました。候補ID・順位・モデル・予測ファイル・GIFハッシュを全件突合しています。</p></div>
<ul><li>各系列の粒子計測・平滑化軌跡のSHA-256を、処理完了時の記録と照合しました。</li><li>親・衝突後の粒径と窓境界を、今回の軌跡から再計算し、全候補の推論時特徴量と一致することを確認しました。</li><li>重力予測は既存の物理APIによる再計算、および対称な細分ビン行列の直接積分とも照合しました。</li><li>実験レビューをモデル学習には使っていません。既存Fig.10の表示用10フレームに加え、追加候補10例の50フレームを固定済みコード・位相回復3回・ローカルGPUで再構成しました。</li><li>粒径ペアは185件、全衝突数は186件。窓不足・欠測を数値表に残し、残差による外れ値除去をしていません。</li></ul>
<p><a href="verification.json">検証結果 JSON</a> / <a href="provenance.json">版・出典の記録</a> / <a href="data/results.json">全結果 JSON</a></p>
<p class="muted">モデル SHA-256: <code>{MODEL_SHA}</code><br>最後の判定保存: {html.escape(frozen['last_review_at'])}<br>解析用スナップショット: {html.escape(frozen['created_at'])}</p></section>
<section id="downloads"><h2>数値表・ダウンロード</h2><p>CSVは図の集計値・有効標本数・旧新版比較・補足確認を収録しています。レビューDB、アクセスキー、元ホログラム、モデル本体は公開パッケージに含めていません。</p><ul>{download_rows}</ul><div class="buttons"><a class="button primary" href="experimental-replacements.zip">一式をダウンロード</a><a class="button" href="experimental_figures.pdf">8図のPDF</a></div></section>
<footer>処理版: revision/amt-102p3mm-phase3 · 生成: {stamp()}<br>このページはローカル配信＋Cloudflare Quick Tunnelです。トンネル停止でURLが無効になり、作り直すとURLが変わります。</footer></main></body></html>'''
    (PACKAGE/'index.html').write_text(report)
    save(PACKAGE/'provenance.json',{'created_at':stamp(),'branch':'revision/amt-102p3mm-phase3','model_sha256':MODEL_SHA,
        'review_snapshot_sha256':frozen['snapshot_sha256'],'review_join_sha256':frozen['joined_sha256'],'review_last_updated_at':frozen['last_review_at'],
        'scene_count':14,'reviews_per_scene':80,'phase_retrieval_iterations':3,'relative_depth_mm':[0,102.3],
        'absolute_reconstruction_mm':[74.9,177.2],'density_frames_per_scene':FRAMES,'volume_cm3':VOLUME_CM3,
        'sources':[{'scene':r['scene'],'kind':Path(r['path']).name,'sha256':r['sha256']} for r in frozen['sources']],
        'analysis_code_sha256':{p.name:sha(p) for p in sorted(Path(__file__).parent.glob('*.py'))},
        'human_labels_unchanged':True,'synthetic_training_distribution_unchanged':True,'original_manuscript_edited':False})
    (PACKAGE/'README.txt').write_text('AMT experimental replacement package — 2026-11-03\n\nOpen index.html for the Japanese report.\nMain replacements: figures/fig10.pdf through fig15.pdf.\nSupplementary revised population plots: fig04.pdf and fig05.pdf; use the provenance wording provided.\nReplace the experimental subsection using text/experimental_section.tex. Update the abstract and conclusion separately.\nNo manuscript, original images, model, or live review labels were modified.\nReview DB, access keys and full raw data are deliberately excluded.\nCSV files carry raw scene IDs; manuscript scene indices run 1–14 in acquisition order.\nPrimary counts: 186 confirmed; 185 assignable parent pairs; 184 valid diameter diagnostics; 179 valid velocity diagnostics.\nCoverage is conditional on the reviewed top-80 cohort, not absolute recall.\nSee verification.json and provenance.json for checks and hashes.\n')
    print('Built HTML, captions, full replacement subsection and integration anchors.')

if __name__=='__main__':main()
