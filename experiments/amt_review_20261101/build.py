"""Build a self-contained, explicitly scoped HTML report from verified local evidence."""
from __future__ import annotations

import csv
import html
import json
import re
import shutil
from pathlib import Path
from markdown_it import MarkdownIt

from prepare import OUT, PREFLIGHT, REPO, RESULTS, WORKSPACE, read, save, sha

HERE = Path(__file__).resolve().parent
PUBLIC = OUT / "public"
MD = MarkdownIt("commonmark", {"html": True}).enable("table")


def esc(value):
    return html.escape(str(value))


def csv_rows(path):
    return list(csv.DictReader(Path(path).open()))


def table(headers, rows):
    return '<table><thead><tr>' + ''.join(f'<th scope="col">{esc(x)}</th>' for x in headers) + '</tr></thead><tbody>' + ''.join(
        '<tr>' + ''.join(f'<td>{esc(x)}</td>' for x in row) + '</tr>' for row in rows) + '</tbody></table>'


def copy(source, relative):
    destination = PUBLIC / relative
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source, destination)


def style_body(body):
    body = body.replace('<table>', '<div class="table-wrap" tabindex="0" role="region" aria-label="表。幅が広い場合は横にスクロールできます"><table>')
    body = body.replace('</table>', '</table></div>')
    body = re.sub(r'<p><img src="([^"]+)" alt="([^"]*)"\s*/?></p>',
                  r'<figure><a href="\1" target="_blank" rel="noopener" aria-label="図を拡大：\2"><img src="\1" alt="\2" loading="lazy"></a><figcaption>\2 <span>タップして拡大</span></figcaption></figure>', body)
    titles = []
    def heading(match):
        index = len(titles) + 1
        title = match.group(1)
        titles.append((f'section-{index}', title))
        return f'<h2 id="section-{index}"><span class="section-number">{index:02d}</span>{title}</h2>'
    return re.sub(r'<h2>(.*?)</h2>', heading, body), titles


def page(body, title, subtitle, current, *, full=False):
    body, headings = style_body(body)
    links = ''.join(f'<a href="#{key}">{label}</a>' for key, label in headings)
    cover = '' if full else '''<div class="verdict"><span class="pill">今回の判定</span><h2>新条件の全量評価を実行中。<br>生成から3手法の再学習へ。</h2><p>実験レビューを始める前に、数値検証からモデルまでを固定するための報告です。</p><div class="facts"><div><strong>7<span>条件</span></strong><small>固定入力の比較が完了</small></div><div><strong>6<span>系列</span></strong><small>新しい正解データを生成済み</small></div><div><strong><b id="full-render-count">処理中</b><span> / 36,000対</span></strong><small>新ホログラムの検証済み数</small></div></div></div>'''
    return f'''<!doctype html>
<html lang="ja"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1"><meta name="color-scheme" content="light"><meta name="robots" content="noindex,nofollow"><title>{esc(title)} | HDCD研究報告</title><link rel="stylesheet" href="assets/report.css"><script src="assets/report.js" defer></script></head>
<body><a class="skip" href="#content">本文へ移動</a><header class="top"><a class="brand" href="index.html">HDCD <span>RESEARCH NOTE</span></a><div class="top-links"><a href="index.html" {'aria-current="page"' if not full else ''}>今回の確認</a><a href="preflight.html" {'aria-current="page"' if full else ''}>詳細報告</a><button type="button" id="print-report">印刷 / PDF</button></div></header>
<div class="shell"><aside><div class="toc-label">この報告書の内容</div><nav aria-label="目次">{links}</nav><div class="aside-note">数値検証を先に確定<br>その後に実験レビューへ</div></aside>
<main id="content"><div class="title-block"><div class="eyebrow">MATERIALS REVISION · 2026.11.01</div><h1>{esc(title)}</h1><p class="subtitle">{esc(subtitle)}</p><div class="provenance">固定入力対照 <code>c7110c8</code> · 新版の全量計算は進行状況欄で更新</div></div>{cover}
<details class="mobile-toc"><summary>目次を開く</summary><nav aria-label="モバイル目次">{links}</nav></details>
<article>{body}</article><footer>確認日時：{esc(current)}<br>Quick Tunnelは実行中のみ利用できます。再作成するとURLが変わります。<a href="#content">先頭へ戻る ↑</a></footer></main></div></body></html>'''


def main():
    PUBLIC.mkdir(parents=True, exist_ok=True)
    for name in ['report.css', 'report.js']:
        copy(HERE / name, 'assets/' + name)
    for name in ['controlled_comparison.png', 'fig7_shared_test.png', 'figure14_manuscript.png', 'figure15_manuscript.png']:
        copy(RESULTS / 'figures' / name, 'assets/' + name)
    for name in ['readiness.json', 'controlled_deltas.csv', 'experimental_inventory.csv', 'synthetic_render_inventory.csv']:
        copy(OUT / 'evidence' / name, 'evidence/' + name)
    for name in ['primary_metrics.csv', 'verification.json', 'revised_generation_all.json']:
        copy(RESULTS / name, 'evidence/' + name)
    copy(WORKSPACE / 'audit/aligned/results/scenario-comparison.csv', 'evidence/scenario-comparison.csv')
    metrics = csv_rows(RESULTS / 'primary_metrics.csv')
    save(PUBLIC / 'evidence/comparison.json', metrics)
    historical = csv_rows(WORKSPACE / 'audit/aligned/results/scenario-comparison.csv')
    versions = []
    for case, name, conditions in [
        ('Published', '公開版 v1.0.1', '奥行102.4 mm、前端64.9 mm。計測3回／画像特徴12回。'),
        ('74.9 mm only', '前端のみ修正版', '同じ生成真値、前端74.9 mm。保存済み3手法を再推論。'),
        ('Aligned + adjacent IDs', '10月29日の本文整合実験', '全12回、padding 2048、Zゲートなし、抑制後順位。上流から再学習。'),
    ]:
        values = {x['method']: x for x in historical if x['case'] == case}
        versions.append([name, conditions, *(f"{float(values[m]['ap']):.6f}" for m in ['ours', 'depth_free', 'minip2d'])])
    versions += [
        ['今回の固定入力対照', '旧102.4 mm画像。計測3回／padding 1536、画像特徴3回／padding 2048。ID統一、Z±15、抑制前順位。', '0.922500', '今回未学習', '今回未学習'],
        ['目標の最終版', '新102.3 mm画像。全3回。候補選択から3手法を新規処理。', '未確定', '未確定', '未確定']]
    deltas = csv_rows(OUT / 'evidence/controlled_deltas.csv')
    delta_rows = [[r['change'], f"{float(r['ap_from']):.6f} → {float(r['ap_to']):.6f}", f"{float(r['delta_ap']):+.6f}",
                   f"{r['fp80_from']} → {r['fp80_to']}", f"{r['fp90_from']} → {r['fp90_to']}"] for r in deltas]
    experiments = csv_rows(OUT / 'evidence/experimental_inventory.csv')
    experimental_rows = [[r['scene_num'], r['collision'], r['non_collision'], r['ambiguous'],
                          f"{float(r['historical_front_um'])/1000:.1f} → 74.9", '12 → 3',
                          f"{r['raw_C001']} / {r['raw_C002']}", f"{r['retained_C001']} / {r['retained_C002']}"] for r in experiments]
    widget = '''<div class="compare"><div class="compare-title">条件を選んで比較</div><div class="filters" role="group" aria-label="比較する変更"><button data-comparison="phase" aria-pressed="true">位相回復</button><button data-comparison="rule" aria-pressed="false">教師規則</button><button data-comparison="rank" aria-pressed="false">順位</button><button data-comparison="z" aria-pressed="false">Zゲート</button></div><div id="comparison-output" aria-live="polite"><p>上の表に全比較を掲載しています。切替表示を読み込み中です。</p></div></div>'''
    content = (HERE / 'report.md').read_text()
    content = content.replace('{{VERSION_TABLE}}', table(['版', '処理条件と範囲', 'Ours AP', 'Depth-free AP', 'MinIP 2D AP'], versions))
    content = content.replace('{{DELTA_TABLE}}', table(['変更', 'AP（前→後）', 'ΔAP', 'FP80（前→後）', 'FP90（前→後）'], delta_rows))
    content = content.replace('{{EXPERIMENT_TABLE}}', table(['系列', '旧衝突', '旧非衝突', '旧困難', '前端 mm（旧→指定）', '回数（旧→指定）', '原画像数 C1 / C2', '保持画像 C1 / C2'], experimental_rows))
    content = content.replace('{{COMPARE_WIDGET}}', widget)
    checked = read(OUT / 'evidence/readiness.json')['checked_at']
    (PUBLIC / 'index.html').write_text(page(MD.render(content), '新版への変更と、論文差し替えの準備状況',
        '旧版との比較、数値ホログラム検証の到達点、実験レビューを始める条件。', checked))

    # Publish only resources directly linked by the existing report, not its directory.
    previous = REPO / 'docs/revision-preflight-20261030.md'
    source = previous.read_text()
    for match in list(re.finditer(r'!?\[[^\]]*\]\(([^)]+)\)', source)):
        link = match.group(1)
        resolved = (previous.parent / link).resolve()
        if not resolved.is_relative_to(REPO):
            raise ValueError(f'Unexpected report resource: {link}')
        if resolved.suffix == '.md':
            destination = 'preflight-methods.html'
            methods = MD.render(resolved.read_text()).replace('../../docs/revision-preflight-20261030.md', 'preflight.html')
            (PUBLIC / destination).write_text(page(methods, '対照実験の再実行方法', '前回の固定入力比較の手順と保存物。', checked, full=True))
        elif resolved.suffix in {'.json', '.csv', '.png'}:
            destination = ('assets/' if resolved.suffix == '.png' else 'evidence/') + resolved.name
            copy(resolved, destination)
        else:
            raise ValueError(f'Unhandled resource: {link}')
        source = source.replace('(' + link + ')', '(' + destination + ')')
    intro = '<p class="source-note">前回の報告書全文です。今回追記した論文差し替え判定と実験準備は<a href="index.html">今回の確認</a>でご覧ください。</p>'
    (PUBLIC / 'preflight.html').write_text(page(intro + MD.render(source), '新版マテリアルに向けた事前検証',
        '実験実施：2026年10月30日。独立検証：11月1日。', checked, full=True))
    # Private manifest also serves as the HTTP allowlist. No public directory listing.
    files = {str(p.relative_to(PUBLIC)): sha(p) for p in sorted(PUBLIC.rglob('*')) if p.is_file()}
    save(OUT / 'public-manifest.json', {'files': files, 'total_bytes': sum((PUBLIC / p).stat().st_size for p in files)})
    print(json.dumps({'public_files': len(files), 'public_bytes': sum((PUBLIC / p).stat().st_size for p in files), 'index': str(PUBLIC / 'index.html')}))


if __name__ == '__main__':
    main()
