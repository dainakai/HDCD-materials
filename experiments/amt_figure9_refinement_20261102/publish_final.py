#!/usr/bin/env python3
"""Mark N1/D4 confirmed across the review pages and refresh the scoped allowlist."""
import hashlib
import json
from pathlib import Path
import re
import shutil

REPO=Path(__file__).resolve().parents[2]
WORK=REPO.parent
PREVIEW=WORK/'audit/revision-review-20261101'
PUBLIC=PREVIEW/'public'
FINAL=WORK/'audit/revision-figure9-final-20261103'
PACKAGE=WORK/'audit/revision-paper-final-20261103'


def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def main():
    selection=json.loads((FINAL/'selection.json').read_text())
    check=json.loads((FINAL/'verification.json').read_text())
    package_check=json.loads((PACKAGE/'verification.json').read_text())
    assert selection['author_confirmed'] and [r['review_id'] for r in selection['cases']]==['N1','D4']
    assert check['status']==package_check['status']=='PASS' and check['visual_review']=='PASS'
    assert sha(PUBLIC/'paper/figures/fig09.pdf')==selection['final_pdf_sha256']
    assert sha(PUBLIC/'paper-replacements.zip')==package_check['zip_sha256']
    # Keep the conventional repository outputs aligned with the final package.
    # Earlier artifacts remain in their dated private directories and Git history.
    for name in ['fig09.pdf','fig09.png','selection.json','identity-audit.json','search-protocol.json','original-comparison.json','verification.json']:
        shutil.copyfile(FINAL/name,REPO/'experiments/amt_figure9_refinement_20261102/results'/name)
    page=PUBLIC/'fig09-artifact-options.html';html=page.read_text()
    html=re.sub(r'<title>.*?</title>','<title>Fig.9 · N1 / D4 採用確定</title>',html,count=1)
    html=re.sub(r'<h1>.*?</h1>','<h1>Fig.9：(a) N1、(b) D4で確定</h1>',html,count=1)
    banner='''<section id="fig9-final-delivery"><h2>確定したFig.9</h2><p>(a) N1は−0.25・0・＋0.25 ms、(b) D4は−0.50・0・＋0.50 msです。元のレイアウトで組み合わせ、図中のスコアと本文・キャプション案も更新しました。</p><a href="paper/figures/fig09.png"><img class="confirmed" src="paper/figures/fig09.png" alt="採用が確定したN1とD4のFig.9"></a><p><a href="paper/figures/fig09.pdf" download>Fig.9 PDF</a> · <a href="paper/latex/fig09.tex" download>キャプション・図環境</a> · <a href="synthetic.html">合成評価のまとめ</a> · <a href="paper-replacements.zip" download>図表・本文の差し替えセット</a></p><p>以下は候補比較と診断の記録です。D4が採用され、D2は採用していません。</p></section>'''
    if 'id="fig9-final-delivery"' in html:
        html=re.sub(r'<section id="fig9-final-delivery">.*?</section>',banner,html,count=1,flags=re.S)
    else:html=html.replace('</h1>','</h1>'+banner,1)
    html=html.replace('第一候補は<strong>D4（321 µm、0.944→0.625）</strong>です。','<strong>D4（321 µm、0.944→0.625）の採用が確定しました。</strong>')
    html=html.replace('D4 · 第一候補：','D4 · 採用確定：').replace('D2 · 比較候補：','D2 · 未採用の比較例：')
    html=html.replace('確認画像・照合記録を保存','候補比較時点の画像・照合記録を保存')
    html=html.replace('(b)はまだ選定段階で、Fig.9全体と論文差し替えZIPの更新前です。','(b)はD4で確定し、Fig.9全体と論文差し替えZIPを更新済みです。')
    page.write_text(html)
    changed=[page]
    notice='<p id="fig9-current-selection"><strong>採用確定：</strong>(a) N1の−0.25・0・＋0.25 ms、(b) D4の−0.50・0・＋0.50 ms。<a href="synthetic.html#fig9">完成したFig.9と差し替え用セット</a>をご覧ください。この下は以前の候補比較の記録です。</p>'
    for name in ['fig09-interior-a.html','fig09-clean-a.html','fig09-frames.html']:
        p=PUBLIC/name;text=p.read_text()
        text,n=re.subn(r'<p id="fig9-current-selection">.*?</p>',notice,text,count=1,flags=re.S)
        assert n==1;p.write_text(text);changed.append(p)
    # Preserve the private pre-selection review as immutable evidence, and give
    # the current public JSON an explicit final-decision pointer.
    review=json.loads((WORK/'audit/revision-figure9-shared-particle-20261103/review.json').read_text())
    review.update(author_selected_b=True,selected_b='D4',final_selection='paper/provenance/fig09-layout.json',
        review_snapshot_note='The archived candidate-review ZIP is the pre-selection comparison. The current author package is paper-replacements.zip.')
    for c in review['cases']:c['status']='author_confirmed' if c['case_id']=='D4' else 'not_selected'
    p=PUBLIC/'fig09-shared-particle/review.json';p.write_text(json.dumps(review,ensure_ascii=False,indent=2)+'\n');changed.append(p)
    current=json.loads((PREVIEW/'public-manifest.json').read_text())
    changed += [PUBLIC/'synthetic.html',PUBLIC/'paper-replacements.zip',PUBLIC/'paper-assets/paper.css',PUBLIC/'paper-assets/paper.js']
    changed += [PUBLIC/'paper'/p.relative_to(PACKAGE/'package') for p in (PACKAGE/'package').rglob('*') if p.is_file()]
    for p in changed:current['files'][str(p.relative_to(PUBLIC))]=sha(p)
    current['total_bytes']=sum((PUBLIC/p).stat().st_size for p in current['files'])
    temp=PREVIEW/'public-manifest.fig09-final.tmp';temp.write_text(json.dumps(current,indent=2)+'\n');temp.replace(PREVIEW/'public-manifest.json')
    receipt=dict(status='PASS',author_confirmed=['N1','D4'],published_paths=sorted(set(str(p.relative_to(PUBLIC)) for p in changed)),
        final_selection_sha256=sha(FINAL/'selection.json'),final_pdf_sha256=selection['final_pdf_sha256'],
        package_zip_sha256=package_check['zip_sha256'],manuscript_unchanged=True,
        previous_package_preserved='audit/revision-paper-replacements-20261102/paper-replacements.zip')
    (FINAL/'publication-inputs.json').write_text(json.dumps(receipt,indent=2)+'\n')
    print(json.dumps({'status':'PASS','published_files':len(receipt['published_paths']),'confirmed':['N1','D4']}))


if __name__=='__main__':main()
