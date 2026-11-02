#!/usr/bin/env python3
"""Package completed synthetic results. Never fit models or write the manuscript.

Run with the repository .venv; bundled pdfplumber/reportlab are read-only additions.
"""
from __future__ import annotations

import csv
import hashlib
import html
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import zipfile

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
WORK = REPO.parent
SOURCE = REPO / 'experiments/amt_full_20261101/results'
FULL = WORK / 'audit/revision-full-20261101'
OUT = WORK / 'audit/revision-paper-replacements-20261102'
PACKAGE = OUT / 'package'
PUBLIC = WORK / 'audit/revision-review-20261101/public'
PAPER = WORK / 'CollisionPaper/本番フォーマット/template.tex'
REFINED = WORK / 'audit/revision-figure9-refinement-20261102'
BUNDLE = Path('/home/dai-server/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/lib/python3.12/site-packages')
sys.path.append(str(BUNDLE))
os.environ.setdefault('MPLCONFIGDIR', str(OUT / 'cache/matplotlib'))

from texts import CAPTIONS, FIGURE_LABELS, TABLE4_CAPTION, TABLE5_CAPTION, figure9_caption, method_edits, result_paragraphs


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load(path):
    return json.loads(Path(path).read_text())


def write(relative, value):
    path = PACKAGE / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    if isinstance(value, (dict, list)):
        value = json.dumps(value, indent=2, ensure_ascii=False) + '\n'
    path.write_text(value, encoding='utf-8')
    return path


def cp(source, relative):
    target = PACKAGE / relative
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source, target)
    return target


def figure_tex(number):
    width = '1.0' if number == 9 else '0.98'
    return (r'\begin{figure}[tbp]' + '\n  \\centering\n'
            + rf'  \includegraphics[width={width}\linewidth,height=0.78\textheight,keepaspectratio]{{figures/fig{number:02d}.pdf}}' + '\n'
            + '  \\caption{' + CAPTIONS[number] + '}\n'
            + '  \\label{' + FIGURE_LABELS[number] + '}\n\\end{figure}\n')


def table_tex(n):
    keys = ['event_ap', 'maximum_f1', 'precision_at_recall_0p8', 'precision_at_recall_0p9', 'candidate_pool_recall']
    names = ['Average precision', r'Maximum $F_1$', r'Precision at recall $\geq0.8$', r'Precision at recall $\geq0.9$', 'Candidate-pool recall']
    lines = [r'\begin{table}[tbp]', r'  \centering', '  \\caption{' + TABLE4_CAPTION + '}',
             r'  \label{tab:projected_xy_metric_summary}', r'  \begin{tabular}{lrrr}',
             r'    \tophline', r'    Metric & Ours & Depth-free variant & MinIP-based 2D \\', r'    \middlehline']
    for name, key in zip(names, keys):
        o = n['methods']['ours'][key]
        values = [f'{o:.3f}']
        for method in ['depth_free', 'minip2d']:
            v = n['methods'][method][key]
            values.append(f'{v:.3f} (${{{v-o:+.3f}}}$)')
        lines.append('    ' + ' & '.join([name] + values) + r' \\')
    lines += [r'    \bottomhline', r'  \end{tabular}', r'\end{table}']
    t4 = '\n'.join(lines) + '\n'
    lines = [r'\begin{table}[tbp]', r'  \centering', r'  \small', '  \\caption{' + TABLE5_CAPTION + '}',
             r'  \label{tab:projected_xy_stage_ablation}', r'  \resizebox{\linewidth}{!}{', r'    \begin{tabular}{lrrrrr}',
             r'      \tophline', r'      Stage & AP $\uparrow$ & $P$ at $R\geq0.8$ $\uparrow$ & FP at $R\geq0.8$ $\downarrow$ & $P$ at $R\geq0.9$ $\uparrow$ & FP at $R\geq0.9$ $\downarrow$ \\', r'      \middlehline']
    for name, row in zip(['Trajectory features only', '+ image features', '+ relation-attention', '+ coordinate suppression'], n['table4']):
        values = [f"{row['event_ap']:.3f}", f"{row['precision_at_recall_0p8']:.3f}", f"{row['false_positives_at_recall_0p8']:,}",
                  f"{row['precision_at_recall_0p9']:.3f}", f"{row['false_positives_at_recall_0p9']:,}"]
        if row['stage'] == 'suppression':
            values = [r'\textbf{' + x + '}' for x in values]
        lines.append('      ' + ' & '.join([name] + values) + r' \\')
    lines += [r'      \bottomhline', r'    \end{tabular}', r'  }', r'\end{table}']
    return t4, '\n'.join(lines) + '\n'


def render_fig9(n):
    """Package new illustrative cases drawn with the recovered original script."""
    selection = load(REFINED/'selection.json')
    verification = load(REFINED/'verification.json')
    assert selection['status'] == verification['status'] == 'PASS'
    assert sha(REFINED/'fig09.pdf') == selection['final_pdf_sha256'] == verification['final_pdf_sha256']
    assert sha(REFINED/'selection.json') == verification['selection_sha256']
    assert n['figure9_cases'] == selection['cases']
    for name in ['fig09.pdf', 'fig09.png', 'crop-contrast-comparison.png']:
        cp(REFINED/name, 'figures/'+name)
    subprocess.run(['/usr/bin/pdftoppm','-singlefile','-scale-to','2400','-png',
                    str(PAPER.parent/'figures/fig09.pdf'),str(OUT/'original-fig09')],check=True,
                   stdout=subprocess.DEVNULL)
    cp(OUT/'original-fig09.png', 'figures/fig09-original-reference.png')
    cp(REFINED/'previous-smaller-cases/fig09.png','figures/fig09-previous-smaller.png')
    cp(REFINED/'selection.json', 'provenance/fig09-layout.json')
    cp(REFINED/'verification.json', 'provenance/fig09-verification.json')
    cp(REFINED/'search-protocol.json', 'provenance/fig09-search-protocol.json')
    cp(REFINED/'identity-audit.json', 'provenance/fig09-identity-audit.json')
    cp(REFINED/'original-comparison.json', 'provenance/fig09-original-comparison.json')
    cp(Path(selection['candidate_search_directory'])/'C01.png', 'figures/fig09-original-slot-current.png')


def source_span(source, start, end):
    assert source.count(start) == 1, (start, source.count(start))
    a = source.index(start)
    b = source.index(end, a) + len(end)
    return {'start_line': source.count('\n', 0, a)+1, 'end_line': source.count('\n', 0, b)+1,
            'start_anchor': start, 'end_anchor': end,
            'original_text_sha256': hashlib.sha256(source[a:b].encode()).hexdigest()}


def make_text(n, source):
    mapping = []
    for number in [7, 8, 9]:
        path = f'latex/fig{number:02d}.tex'
        write(path, figure_tex(number))
        label = FIGURE_LABELS[number]
        anchor = '\\label{' + label + '}'
        lineno = source.count('\n', 0, source.index(anchor)) + 1
        mapping.append({'id': f'fig{number:02d}', 'title': f'Fig.{number}', 'label': label, 'source_line': lineno,
                        'file': path, 'asset': f'figures/fig{number:02d}.pdf', 'action': 'Replace the complete figure environment, or replace the PDF and caption separately.'})
    t4, t5 = table_tex(n)
    for number, label, tex in [(4, 'tab:projected_xy_metric_summary', t4), (5, 'tab:projected_xy_stage_ablation', t5)]:
        path = f'latex/table{number:02d}.tex'; write(path, tex)
        mapping.append({'id': f'table{number:02d}', 'title': f'Table {number}', 'label': label,
                        'source_line': source.count('\n', 0, source.index('\\label{' + label + '}'))+1,
                        'file': path, 'action': 'Replace the complete table environment. These are Tables 4 and 5 in this manuscript, despite historical export filenames table3 and table4.'})
    for edit in method_edits():
        path = 'latex/methods/' + edit['id'] + '.tex'
        write(path, edit['replacement'] + '\n')
        mapping.append({'id': edit['id'], 'title': edit['title'], 'file': path, 'note': edit['note'],
                        **source_span(source, edit['start'], edit['end'])})
    parts = result_paragraphs(n)
    result_text = '\n\n'.join([parts[0], r'\input{latex/table04.tex}', parts[1], r'\input{latex/fig08.tex}',
                               parts[2], r'\input{latex/table05.tex}', parts[3], r'\input{latex/fig09.tex}', parts[4]]) + '\n'
    write('latex/results-synthetic.tex', result_text)
    start = r'\subsection{Method validation with synthetic holograms}'
    end = 'Future work should improve the transferability of the synthetic evaluation so that it can provide reliable estimates of performance, especially missed-event rates, on experimental data.'
    mapping.append({'id': 'synthetic-results', 'title': '数値検証の節全体', 'file': 'latex/results-synthetic.tex',
                    'note': '上のFig.8・9とTable 4・5をinputする統合案。個別差し替えとの二重挿入はしない。',
                    **source_span(source, start, end)})
    m = n['methods']; o, d, p = (m[k] for k in ['ours','depth_free','minip2d'])
    abstract = f"Average precision was {o['event_ap']:.3f} for the proposed method, {d['event_ap']:.3f} for the depth-free variant, and {p['event_ap']:.3f} for the MinIP-based 2D variant; precision at a recall of 0.8 was {o['precision_at_recall_0p8']:.3f}, {d['precision_at_recall_0p8']:.3f}, and {p['precision_at_recall_0p8']:.3f}, respectively."
    original = 'Average precision was 0.917 for the proposed method, 0.901 for the depth-free variant, and 0.858 for the MinIP-based 2D variant; precision at a recall of 0.8 was 0.960, 0.895, and 0.813, respectively.'
    write('latex/abstract-synthetic.tex', abstract+'\n')
    mapping.append({'id': 'abstract', 'title': 'Abstractの合成評価値', 'file': 'latex/abstract-synthetic.tex', **source_span(source, original, original)})
    conclusion = f"On the synthetic test sequence, the proposed method achieved an AP of {o['event_ap']:.3f}, compared with {d['event_ap']:.3f} for the depth-free variant and {p['event_ap']:.3f} for MinIP-based 2D.\nPrecision at recall 0.8 was {o['precision_at_recall_0p8']:.3f}, {d['precision_at_recall_0p8']:.3f}, and {p['precision_at_recall_0p8']:.3f}, respectively, with corresponding false-positive counts of 51, 93, and 289.\n"
    write('latex/conclusion-synthetic.tex', conclusion)
    mapping.append({'id': 'conclusion', 'title': 'Conclusionに再掲された合成評価値', 'file': 'latex/conclusion-synthetic.tex',
                    **source_span(source, 'On the synthetic test sequence, the proposed method achieved an average precision of 0.917,',
                                   'Precision at a recall of 0.8 was 0.960, compared with 0.813 and 0.895 for the two baselines, respectively, and the corresponding false-positive counts were reduced from 368 and 188 to 67.')})
    original = 'A cumulative ablation study of the proposed method further showed that adding image features, relationships among candidates, and duplicate suppression successively improved performance.'
    replacement = 'The scoring-stage comparison showed higher AP for the final combination, while the image-feature stage slightly increased false positives at recall 0.8; the comparison also includes changes in candidate selection and localization-feature availability.\n'
    write('latex/conclusion-stages.tex', replacement)
    mapping.append({'id': 'conclusion-stages', 'title': 'Conclusionの単調改善の主張', 'file': 'latex/conclusion-stages.tex', **source_span(source, original, original)})
    write('replacement-map.json', {'manuscript_sha256': sha(PAPER), 'manuscript_relative_path': 'CollisionPaper/本番フォーマット/template.tex',
                                  'source_manuscript_unchanged': True, 'items': mapping})
    return mapping


def table_html(headers, rows):
    esc = html.escape
    return '<div class="table-scroll" tabindex="0" role="region" aria-label="左右にスクロールできる表"><table><thead><tr>' + ''.join('<th scope="col">'+esc(str(h))+'</th>' for h in headers) + '</tr></thead><tbody>' + ''.join('<tr>'+''.join('<td>'+esc(str(v))+'</td>' for v in row)+'</tr>' for row in rows)+'</tbody></table></div>'


def readme(mapping, n):
    lines = ['# 合成ホログラム評価：論文差し替え用セット', '',
             '対象：2026-11-02確定の102.3 mm・全位相回復3回の評価。ブランチ revision/amt-102p3mm-phase3。',
             '原稿は変更していません。以下は著者が差し替えるためのファイルです。', '',
             '## 差し替え方', '',
             '1. figures/fig07.pdf、fig08.pdf、fig09.pdfを原稿の同名ファイルへコピーします。PNGは閲覧用です。',
             '2. latex/fig07.tex、fig08.tex、fig09.texにある図環境・キャプションを使います。元のlabelは維持しています。',
             '3. latex/table04.tex、table05.texを該当する表環境と差し替えます。現在の原稿ではTable 4と5です。旧出力のtable3/table4という名前で判断しないでください。',
             '4. latex/results-synthetic.texは「Method validation with synthetic holograms」の節全体の差し替え案です。この案はlatex/内の図表をinputします。手順2・3で個別に挿入した図表を、同じ節に重ねて入れないでください。',
             '5. latex/methods/の各断片とAbstract・Conclusion用の断片を、replacement-map.jsonの対象箇所と照合します。数値だけの差し替えでは本文と整合しません。',
             '6. 最後にご自身の原稿でコンパイルし、図表の配置と参照を確認してください。スニペットは既存のunit、tophline、middlehline、bottomhline、graphicxの定義を利用します。', '',
             '結果の節をinputで使う場合は、パッケージのfigures/とlatex/をTeXメインファイルと同じ階層に置いてください。',
             '図表のPDFは検査済みですが、原稿を編集しての全文コンパイルは行っていません。', '',
             '## 図表と本文の対応', '', '| 対象 | 原稿の行／ラベル | ファイル |', '| --- | --- | --- |']
    for item in mapping:
        target = item.get('label', str(item.get('start_line', item.get('source_line'))))
        lines.append(f"| {item['title']} | `{target}` | `{item['file']}` |")
    lines += ['', '行番号はreplacement-map.json記録の原稿ハッシュに対する位置です。編集後はラベルと開始・終了文で照合してください。', '',
              '## 読み替えが必要な点', '',
              '- OursはAP・最大F1・80%/90%再現率での適合率が両比較法を上回ります。候補集合recallと最大最終recallは比較法の方が高い値です。',
              '- 画像特徴段階でAPは0.892678から0.897689に改善し、FP90は1,186から1,004に減りますが、FP80は177から183に増えます。「全段階で全指標が改善」は使えません。',
              '- Table 5は候補集合と位置補正に渡せる特徴も変わります。スコアだけの要因実験とは解釈しません。最終行で追加されるのは座標による重複抑制です。',
              '- Fig.9は著者の依頼により事例を選び直しました。128×128画素、各クロップの1–99パーセンタイルによる線形コントラスト調整、元の作図スクリプトと投稿用ラベル処理を使用しています。画像は固定済みの新系列から位相回復3回で再構成しました。学習・推論スコア・全量性能評価は変更していません。',
              '- Fig.9は286＋285→360 µmの真の衝突と、真値径276 µmの共有粒子アーティファクトです。スコアはそれぞれ0.958486→0.957430と0.908647→0.491434です。前版のクロップ位置の誤りを修正し、対象中心の128画素を切り出し、撮影範囲外だけを元図と同じ黒表示にしました。表示時刻は−1.00、0、＋0.25 msです。赤丸・青四角は実観測に限り、焦点は継続軌跡の推定Zです。事例は結果を見た後の例示選択であり、平均効果の評価ではありません。',
              '- 元図のイベント1165・frame 3859も追跡しました。同じseed構成ですが、生成奥行き変更に伴う再生成後は位置と粒子IDが異なります。元図と同じ画像に新スコアを付けた図ではありません。元の候補ID、同じ計画枠の新旧真値、表示事例の照合結果はprovenance/fig09-original-comparison.jsonとfig09-identity-audit.jsonにあります。',
              '- アーティファクトは二つの軌跡IDが同一粒子を捉えた候補です。その粒子は2フレーム後に別の粒子と衝突しますが、候補位置から約481 µm離れており共通評価ゲート350 µmの外です。「周辺に物理的な衝突が一切ない例」とは説明しません。',
              '- Fig.7(a)はconfidenceを動かしたPR曲線、(b–f)は衝突用の計測／追跡系統の集計です。低confidenceの追加検出まで同一ファイルとは説明しません。',
              '- 生成座標0–102.3 mmと、カメラからの距離74.9–177.2 mmを区別します。',
              '- APはprecision包絡線の積分です。scikit-learnの非補間APと同じ定義とは書きません。',
              '- 学習には使っていないテストseedですが、旧生成データによる実装確認・感度解析で使用履歴があります。「一度も参照していない独立テスト」は主張しません。',
              '- 公開版との比較では真値・画像・教師規則・モデルなどを合わせて変更しています。全体の改善を位相回復3回だけの効果には帰属できません。',
              '- 前端だけを修正した中間版との比較では、OursのFP90は430から486へ増加しています。公開版550からは減少しています。', '',
              '## 残る実験データの更新', '',
              '実験14系列の再処理・各系列上位80件のレビューは別の実行です。このセットの合成評価値はその結果を待たず使用できます。',
              'ただし、数値検証の節以前に置かれている実験由来のFig.3–5、実験の結果・Abstract・Conclusionの実験値はこのセットの更新対象ではありません。',
              '合成生成に用いた旧実験統計は固定済みであり、新しい実験統計へ後から置き換えません。新Fig.4–5との版の関係を本文で明示します。', '',
              '## 再現性とファイル', '',
              'tables/は全精度のCSV、data/は図の集計データ、latex/は英語の挿入案です。',
              'provenance/には保存済み評価の検証記録、図の出典・処理差分を収めています。',
              'manifest.jsonは本セット各ファイルのSHA-256、SHA256SUMSはコマンド用の照合リストです。',
              '元の研究画像、モデル重み、実験レビュー回答、認証キーは含めていません。', '']
    write('README.md', '\n'.join(lines))


def figure9_notes(n):
    true, artifact = n['figure9_cases']
    event = true['truth_event']
    original = load(REFINED/'original-comparison.json')
    slot = original['same_planned_slot_1165']
    a, b = slot['original'], slot['current']
    slot_table = table_html(['イベント1165 / frame 3859','元論文の系列','今回の再生成系列'], [
        ['親ID → 娘ID',f"{int(a['parent1_id'])}, {int(a['parent2_id'])} → {int(a['child_id'])}",f"{int(b['parent1_id'])}, {int(b['parent2_id'])} → {int(b['child_id'])}"],
        ['位置 x, y, z (mm)',', '.join(f"{a[k]/1000:.3f}" for k in ['x_um','y_um','z_um']),', '.join(f"{b[k]/1000:.3f}" for k in ['x_um','y_um','z_um'])],
        ['真値径 (µm)',f"{a['parent1_diameter_um']:.0f} + {a['parent2_diameter_um']:.0f} → {a['child_diameter_um']:.0f}",f"{b['parent1_diameter_um']:.0f} + {b['parent2_diameter_um']:.0f} → {b['child_diameter_um']:.0f}"],
    ])
    return f'''<p><strong>対象中心のクロップへ修正しました。</strong>真の衝突は {event['parent1_diameter_um']:.0f}＋{event['parent2_diameter_um']:.0f}→{event['child_diameter_um']:.0f} µm。アーティファクトの真値径は {artifact['shared_truth_diameter_um']:.0f} µmで、trajectory scoreは{artifact['trajectory_score']:.6f}、画像特徴追加後は{artifact['image_feature_score']:.6f}です。0.9以上から{artifact['trajectory_score']-artifact['image_feature_score']:.3f}低下します。</p>
<p>元図と同じ128×128画素（1.28 mm四方）、赤丸・青四角と短い軌跡、右側の横棒、フォント・パネル配置を使っています。前版で加えた「枠を撮影範囲内へずらす処理」が粒子を端へ寄せていたため、削除しました。元図と同じく対象中心で切り出し、撮影範囲外は黒表示にします。記録された画素だけで1–99パーセンタイルの線形コントラスト調整を行っています。</p>
<p>6枚の画像と完成PDFを目視確認し、実観測9点を正解粒子と照合しました。時刻は<strong>−1.00、0、＋0.25 ms</strong>です。粒子が撮影範囲から出る前の実観測を使うため、元図の＋1.00 msから変更しました。粒子画像の倍率・画素・推論結果は変更していません。</p>
<details><summary>元論文と異なるサンプルになった理由と、元の事例の追跡結果</summary>
<p>粒径・速度分布の元CSVは新旧でバイト単位で同じで、学習・テストに使うseedの構成も維持しています。ただし、生成体積を0–102.3 mmへ変更し、学習5系列とテスト1系列の真値・ホログラムを再生成して、粒子計測と追跡も再実行しています。学習用の個々の画像まで同じではありません。HoloD3を共通にしても入力画像と軌跡IDは一致しません。元図の真イベントは候補12515_12410_3859（計画イベント1165）、アーティファクトは17742_17721_5331でした。元のアーティファクトは0.890→0.336で、今回指定のtrajectory score 0.9以上には達していません。</p>
{slot_table}
<p>同じ計画イベント1165の今回の実現（候補12486_12575_3859）も再構成して比較しました。粒子径は近いものの、位置と軌跡は異なり、合体後にも赤の観測が残ります。今回は二粒子から一粒子への変化が見やすく、さらに大きいイベント1815を採用しました。「同じseedの同じ計画枠」と「同じ画像・同じ粒子の実現」を区別しています。</p>
<figure><img src="paper/figures/fig09-original-slot-current.png" alt="元図と同じ計画イベント1165を今回の再生成データで再構成した比較候補。採用図ではありません" loading="lazy"><figcaption>比較用：今回のイベント1165。差し替えには上の採用PDFを使ってください。</figcaption></figure>
<figure><a href="paper/figures/fig09-original-reference.png"><img src="paper/figures/fig09-original-reference.png" alt="CollisionPaperにある元論文のFig.9" loading="lazy"></a><figcaption>元論文のFig.9。元作図スクリプトと投稿用ラベル処理を今回も使用しています。</figcaption></figure>
<p>CollisionPaperのmainと一致するコミット2a5d3f4を確認。旧作業コピーから回収した作図スクリプトは保存済みスナップショットとバイト単位で一致しています。</p>
<a href="paper/provenance/fig09-original-comparison.json" download>新旧事例の対応記録</a></details>
<details><summary>アーティファクトの正解対応と、選択上の注意</summary>
<p>終了軌跡8518と継続軌跡8507は、表示した−1.00 msと0 msの両方で、同じ真値粒子3605の投影円内にあります。Z誤差は最大約66 µmです。一対一の対応表は継続軌跡にしか真値を割り当てないため、両IDがその対応表で確認できたとは記述していません。画像特徴は衝突前9フレームすべてで共有成分を示します。</p>
<p>この粒子は2フレーム後に別の粒子と衝突します。ただし、そのイベントは候補位置から約481 µm離れ、共通評価の350 µmゲート外です。ここで示すアーティファクトは「同一粒子に付いた二つのIDを組にした候補」であり、周囲に衝突が一切ないという意味ではありません。真イベント1815は最終候補に残り、アーティファクトは除外されています。表示スコアはrelation-attention前です。</p>
<p>真イベント1815の厳密な学習ラベルはidentity_unknownです。終了軌跡に一対一対応がないためで、今回の図では表示点を独立に正解の親7607・7608、娘7683と照合しました。学習ラベルや評価規則は変更していません。選択は結果を見た後の例示であり、画像特徴の平均効果を示すものではありません。</p>
<a href="paper/provenance/fig09-identity-audit.json" download>表示点と正解粒子の照合記録</a> · <a href="paper/provenance/fig09-layout.json" download>事例・描画条件・出典</a></details>
<details><summary>128画素クロップとコントラスト調整の比較</summary><figure><a href="paper/figures/crop-contrast-comparison.png"><img src="paper/figures/crop-contrast-comparison.png" alt="同じ採用事例を256画素、128画素固定0–1、128画素1–99パーセンタイルで比較" loading="lazy"></a><figcaption>左：256画素・1–99％。中央：128画素・固定0–1。右：採用した128画素・1–99％。表示処理を特徴抽出や推論へ戻していません。</figcaption></figure></details>'''


def html_page(n, mapping):
    esc = html.escape
    method_names = {'ours':'Ours（提案法）','depth_free':'Depth-free','minip2d':'MinIP-based 2D'}
    metrics = [[method_names[k], f"{v['event_ap']:.6f}", f"{v['maximum_f1']:.6f}", f"{v['precision_at_recall_0p8']:.6f}",
                v['false_positives_at_recall_0p8'], f"{v['precision_at_recall_0p9']:.6f}", f"{v['false_positives_at_recall_0p9']:,}"] for k,v in n['methods'].items()]
    metric_table = table_html(['手法','AP','最大F1','適合率@80%','FP80','適合率@90%','FP90'], metrics)
    stage_names = ['軌跡特徴','＋画像特徴','＋relation-attention','＋座標による重複抑制']
    stage_table = table_html(['段階','AP','適合率@80%','FP80','適合率@90%','FP90'],
        [[name,f"{r['event_ap']:.6f}",f"{r['precision_at_recall_0p8']:.6f}",r['false_positives_at_recall_0p8'],
          f"{r['precision_at_recall_0p9']:.6f}",f"{r['false_positives_at_recall_0p9']:,}"] for name,r in zip(stage_names,n['table4'])])
    old = list(csv.DictReader((PACKAGE/'tables/version-comparison.csv').open()))
    comparison = []
    for method in method_names:
        before = next(r for r in old if r['method']==method and r['case']=='Published')
        after = n['methods'][method]
        comparison.append([method_names[method],f"{float(before['ap']):.6f} → {after['event_ap']:.6f}",
                           f"{before['fp_0.8']} → {after['false_positives_at_recall_0p8']}",f"{before['fp_0.9']} → {after['false_positives_at_recall_0p9']}"])
    comparison_table = table_html(['手法','AP（公開版→今回）','FP80','FP90'],comparison)
    cards=[]
    fig_notes = {7:'衝突評価と同じテスト系列の粒子計測・追跡。XY誤差の中央値2.783 µm、Z誤差の中央値28.480 µm。Z誤差の95パーセンタイルは948.589 µmです。',
                 8:'同じ2,000真値イベントに対する3手法の比較。再現率80%では、提案法は1,651候補を調べて1,600イベントに対応し、誤検出51件です。',
                 9:f"元の作図スクリプトで新しい2事例を描画。真イベントは{n['figure9_cases'][0]['trajectory_score']:.6f}→{n['figure9_cases'][0]['image_feature_score']:.6f}、共有粒子アーティファクトは{n['figure9_cases'][1]['trajectory_score']:.6f}→{n['figure9_cases'][1]['image_feature_score']:.6f}。結果を見た後に選んだ例示です。"}
    for num in [7,8,9]:
        path=f'figures/fig{num:02d}'
        snippet=(PACKAGE/f'latex/fig{num:02d}.tex').read_text()
        extra = ''
        if num == 9:
            extra = figure9_notes(n)
        cards.append(f'''<section id="fig{num}"><div class="section-heading"><span>FIGURE {num:02d}</span><h2>Fig.{num} · {['粒子計測と追跡','衝突検出性能と確認件数','画像特徴の事例'][num-7]}</h2></div>
<p>{fig_notes[num]}</p><figure><a href="paper/{path}.png" aria-label="Fig.{num}を拡大"><img src="paper/{path}.png" alt="Fig.{num}の差し替え図" loading="lazy"></a><figcaption>図をタップすると拡大できます。論文への挿入にはPDFを使用してください。</figcaption></figure>
<div class="downloads"><a href="paper/{path}.pdf" download>PDFを保存</a><a href="paper/{path}.png" download>PNGを保存</a><a href="paper/latex/fig{num:02d}.tex" download>図環境・キャプション .tex</a></div>
{extra}<details><summary>英語キャプションとLaTeXを表示</summary><button class="copy" data-copy="tex-fig{num}" type="button">LaTeXをコピー</button><pre id="tex-fig{num}">{esc(snippet)}</pre></details></section>''')
    maps=[]
    for item in mapping:
        location = item.get('label') or f"{item.get('start_line')}–{item.get('end_line')}行"
        maps.append(f'<tr><td>{esc(item["title"])}</td><td><code>{esc(location)}</code></td><td><a href="paper/{item["file"]}" download>{esc(Path(item["file"]).name)}</a></td></tr>')
    snippets=[]
    for name,title in [('results-synthetic','数値検証の節全体（英語案）'),('abstract-synthetic','Abstractの合成評価値'),('conclusion-synthetic','Conclusionの合成評価値')]:
        value=(PACKAGE/f'latex/{name}.tex').read_text()
        snippets.append(f'<details><summary>{title}</summary><button type="button" class="copy" data-copy="{name}">LaTeXをコピー</button><pre id="{name}">{esc(value)}</pre></details>')
    bytes_=sum(p.stat().st_size for p in PACKAGE.rglob('*') if p.is_file())
    return f'''<!doctype html><html lang="ja"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1"><meta name="robots" content="noindex,nofollow"><title>合成ホログラム評価 · 論文差し替え用セット</title><link rel="stylesheet" href="paper-assets/paper.css"><script src="paper-assets/paper.js" defer></script></head>
<body><a href="#main" class="skip">本文へ</a><header><a href="index.html">HDCD · 研究報告</a><a href="paper-replacements.zip" download>一括ダウンロード ↓</a></header>
<main id="main"><div class="hero"><p class="eyebrow">SYNTHETIC VALIDATION / 2026.11.02</p><h1>合成ホログラム評価<br>論文の差し替え用セット</h1><p class="lead">Fig.7–9、Table 4・5、元数値、英語キャプションと本文案をまとめました。<br>確定した全量評価を使用し、原稿への差し替えは行っていません。</p>
<div class="summary"><div><span>提案法のAP</span><strong>0.9222</strong></div><div><span>誤検出 · 再現率80%</span><strong>51<small>件</small></strong></div><div><span>誤検出 · 再現率90%</span><strong>486<small>件</small></strong></div></div>
<div class="downloads"><a class="primary" href="paper-replacements.zip" download>差し替えセットを保存 (.zip)</a><a href="paper/README.md" download>差し替え手順 (.md)</a><button id="print" type="button">印刷</button></div>
<p class="micro">生成奥行き102.3 mm · 全位相回復3回 · 評価seed 260745 · 6,000フレーム / 真値2,000イベント</p></div>
<nav class="jump" aria-label="ページ内の移動"><a href="#results">結果</a><a href="#fig7">Fig.7</a><a href="#fig8">Fig.8</a><a href="#fig9">Fig.9</a><a href="#tables">表</a><a href="#changes">本文の変更</a><a href="#versions">公開版との差</a></nav>
<section id="results"><h2>確定した性能評価値</h2>{metric_table}<p>提案法は上表のAP・最大F1・指定再現率での適合率が最良です。一方、候補集合recallは0.9765 / 0.9855 / 0.9900、最終最大recallは0.9675 / 0.9715 / 0.9715です。いずれもOurs / Depth-free / MinIP-based 2Dの順です。</p><p class="note">FP80・FP90は、それぞれ再現率80%・90%以上に到達する最も高い閾値での誤検出件数。同点はまとめて含めます。評価は時間±5フレーム・XY 350 µm・Z条件なしの最大一対一対応です。</p></section>
{''.join(cards)}
<section id="tables"><h2>Table 4・5 · LaTeX表と全精度CSV</h2><p>原稿ではTable 3は手法構成の比較表です。今回の性能表と段階比較表は、それぞれ<strong>Table 4・5</strong>に対応します。ラベルは維持しています。</p>
<div class="downloads"><a href="paper/latex/table04.tex" download>Table 4 .tex</a><a href="paper/tables/table04-metrics.csv" download>Table 4 .csv</a><a href="paper/latex/table05.tex" download>Table 5 .tex</a><a href="paper/tables/table05-stages.csv" download>Table 5 .csv</a></div>
<h3>Table 5の段階比較</h3>{stage_table}<p class="note">画像特徴段階でFP80は<strong>177→183件</strong>、最大F1は<strong>0.849082→0.848550</strong>と悪化しています。APとFP90は改善しました。候補集合と位置補正に使える特徴も変わる比較です。</p></section>
<section id="changes"><h2>図表と数値に合わせて変える本文</h2><p><strong>数値だけの差し替えでは整合しません。</strong>下記の箇所について、実測値と実装に対応した英語案を同梱しました。全文の挿入案と個別の図表を二重に挿入しないよう、READMEの手順を確認してください。</p>
<ol><li>Fig.7がFig.8と同じテスト系列・計測／追跡系統の評価であること。</li><li>R4の順位は抑制前に固定。生き残った候補を並べ直さないこと。</li><li>R5のt−1の親・t＋1の娘、neutral、学習Zゲート±15 mmと評価Z条件なし。</li><li>APのprecision包絡線による定義と同点処理。</li><li>全段階・全指標での単調改善という記述の修正。</li><li>テストseedの過去の利用履歴と、合成生成に用いた実験統計の版。</li><li>Fig.9の事例スコア、Abstract・Conclusionに再掲された合成評価値。</li></ol>
<div class="table-scroll" tabindex="0" role="region" aria-label="本文変更箇所"><table><thead><tr><th>差し替え対象</th><th>原稿のラベル／行</th><th>ファイル</th></tr></thead><tbody>{''.join(maps)}</tbody></table></div>{''.join(snippets)}
<p class="note">数値検証の節以前にある実験由来のFig.3–5と、その後の実験結果は再処理・レビュー後の更新対象です。今回のセットで論文全体が完成したという意味ではありません。合成評価の結果と図表は確定しています。</p></section>
<section id="versions"><h2>公開版からの結果の変化</h2>{comparison_table}<p>公開版はv1.0.1。生成深さ102.4→102.3 mm、主計測の前端64.9→74.9 mm、焦点画像特徴の位相回復12→3回、物理Z原点と教師規則をそろえ、画像から再生成・再学習しています。Depth-free用MinIPの前端は公開版から74.9 mmでした。</p><p>真値とモデルも変わるため、上表は変更一式の比較です。位相回復3回だけの効果ではありません。前端だけを修正した中間版との比較では、OursのFP90は430→486件と増えています。</p><div class="downloads"><a href="paper/tables/version-comparison.csv" download>中間版も含む比較 .csv</a><a href="index.html">R4・R5・位相回復の対照実験を見る</a></div></section>
<section id="provenance"><h2>保存物と確認範囲</h2><p>Fig.7・8は確定済みのPDFと同一です。Fig.9は見やすさを確認して事例を選び直し、固定済み新系列の画像を表示用に再構成しました。元の論文の作図関数・投稿用ラベル処理で仕上げています。フォントは原作図と同じLiberation Sansで、パネル説明には投稿用スクリプトが指定するArial互換フォントを使っています。モデル、候補スコア、性能評価値は変更していません。</p><p>保存済みの全34出力ハッシュ、各数値と図の元データ、原稿の参照ラベルを照合しました。元の原稿は変更していません。原稿に統合しての全文コンパイルは著者による差し替え後の確認事項です。</p><div class="downloads"><a href="paper/manifest.json" download>出典・ハッシュ</a><a href="paper/SHA256SUMS" download>SHA256SUMS</a><a href="paper/replacement-map.json" download>差し替え対応 .json</a></div><p class="micro">元研究画像・モデル重み・実験レビュー回答はこのセットに含めていません。展開後の容量：約{bytes_/1e6:.1f} MB。</p></section>
<footer>Quick Tunnelはサーバーとトンネルの稼働中に利用できます。再作成するとURLが変わります。<a href="#main">先頭へ ↑</a></footer></main><div id="copy-status" role="status" aria-live="polite"></div></body></html>'''


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    PACKAGE.mkdir(exist_ok=True)
    original_paper = sha(PAPER)
    source_manifest = load(SOURCE/'manifest.json')
    checks = source_manifest['numerical_outputs_manifest']
    assert source_manifest['status']=='PASS' and len(checks)==34
    for path, digest in checks.items():
        assert sha(SOURCE/path)==digest, path
    n = load(SOURCE/'manuscript/numeric-replacements.json')
    selection = load(REFINED/'selection.json')
    n['figure9_cases'] = selection['cases']
    CAPTIONS[9] = figure9_caption(n)
    for number in [7,8]:
        for suffix in ['pdf','png']:
            cp(SOURCE/f'figures/figure{number:02d}.{suffix}', f'figures/fig{number:02d}.{suffix}')
    render_fig9(n)
    copies = {
        'results/table3.csv':'tables/table04-metrics.csv',
        'results/table4/table4.csv':'tables/table05-stages.csv',
        'results/scenario-comparison.csv':'tables/version-comparison.csv',
        'manuscript/claim-checks.json':'provenance/claim-checks.json',
        'results/figure07/summary.json':'data/figure07-summary.json',
        'results/table4/protocol.json':'provenance/table05-protocol.json',
    }
    for src,dst in copies.items(): cp(SOURCE/src,dst)
    write('data/numeric-results.json', n)
    write('data/figure09-cases.json', {k:selection[k] for k in ['status','cases','selection','focus','crop_size_px','crop_position','display_normalization']})
    extra_sources={}
    for name in ['event_pr.csv','event_summary.csv','target_recall.csv','fp_composition.csv']:
        source = FULL/'results/figure08'/name
        cp(source,'data/figure08-'+name); extra_sources[str(source.relative_to(WORK))]=sha(source)
    for method in ['ours','depth_free','minip2d']:
        for name in ['metrics','independent_checks','rank_sensitivity']:
            cp(SOURCE/f'results/{method}/{name}.json',f'provenance/{method}-{name}.json')
    mapping = make_text(n, PAPER.read_text())
    readme(mapping,n)
    write('provenance/evaluation-source.json', {'status':'PASS', 'evaluation_source':'experiments/amt_full_20261101/results',
          'source_export_manifest_sha256':sha(SOURCE/'manifest.json'),'verified_output_hashes':checks,
          'additional_plot_data_sources':extra_sources,'training_seeds':list(range(260740,260745)),
          'test_seed':260745,'frames_per_sequence':6000,'truth_events_per_sequence':2000,
          'generation_depth_um':[0,102300],'primary_propagation_distance_um':[74900,177200],
          'particle_reconstruction_planes':1024,'phase_retrieval_iterations':3,
          'prior_test_seed_use':'Implementation and sensitivity checks on earlier generated data; excluded from model fitting.'})
    files = {str(p.relative_to(PACKAGE)):sha(p) for p in sorted(PACKAGE.rglob('*')) if p.is_file() and p.name not in {'manifest.json','SHA256SUMS'}}
    write('manifest.json', {'schema_version':1,'status':'PASS','edition':'2026-11-02 synthetic paper replacement package, centered large particles and high-score artifact',
          'source_branch':'revision/amt-102p3mm-phase3','source_head':subprocess.check_output(['git','rev-parse','HEAD'],cwd=REPO,text=True).strip(),
          'manuscript_sha256':original_paper,'files':files,'file_count_excluding_manifests':len(files),
          'manuscript_modified':False,'full_manuscript_compilation':'Not performed: manuscript replacement is reserved for the author.'})
    sums = {**files,'manifest.json':sha(PACKAGE/'manifest.json')}
    write('SHA256SUMS',''.join(f'{v}  {k}\n' for k,v in sorted(sums.items())))
    zip_path=OUT/'paper-replacements.zip'
    with zipfile.ZipFile(zip_path,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=7) as z:
        for p in sorted(PACKAGE.rglob('*')):
            if p.is_file(): z.write(p, 'synthetic-paper-replacements/'+str(p.relative_to(PACKAGE)))
    shutil.copytree(PACKAGE,PUBLIC/'paper',dirs_exist_ok=True)
    (PUBLIC/'paper-assets').mkdir(exist_ok=True)
    for name in ['paper.css','paper.js']: shutil.copyfile(HERE/name,PUBLIC/'paper-assets'/name)
    shutil.copyfile(zip_path,PUBLIC/'paper-replacements.zip')
    (PUBLIC/'synthetic.html').write_text(html_page(n,mapping))
    assert sha(PAPER)==original_paper
    receipt={'status':'BUILT_PENDING_VISUAL_AND_HTTPS_CHECKS','package':str(PACKAGE),'zip':str(zip_path),
             'zip_sha256':sha(zip_path),'file_count':len(files)+2,'source_outputs_verified':len(checks),
             'manuscript_before_sha256':original_paper,'manuscript_after_sha256':sha(PAPER)}
    (OUT/'build-receipt.json').write_text(json.dumps(receipt,indent=2)+'\n')
    print(json.dumps(receipt,indent=2))


if __name__=='__main__': main()
