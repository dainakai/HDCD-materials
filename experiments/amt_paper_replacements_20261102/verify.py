#!/usr/bin/env python3
"""Verify delivered scientific files, source provenance, and replacement locations."""
import csv
import hashlib
import json
from pathlib import Path
import re
import subprocess
import zipfile

from build import OUT, PACKAGE, PAPER, REPO, SOURCE, load, sha
from pypdf import PdfReader


def balanced_tex(text):
    # Ignore TeX comments and escaped braces; the replacement files contain no verbatim.
    text='\n'.join(re.split(r'(?<!\\)%',line)[0] for line in text.splitlines())
    level=0
    for m in re.finditer(r'(?<!\\)[{}]',text):
        level += 1 if m.group()=='{' else -1
        assert level>=0, 'Unexpected closing brace'
    assert level==0, 'Unbalanced TeX braces'
    assert re.findall(r'\\begin\{([^}]+)\}',text)==list(reversed(re.findall(r'\\end\{([^}]+)\}',text))) or '\\begin' not in text


def image_hashes(path):
    return {hashlib.sha256(image.data).hexdigest() for image in PdfReader(path).pages[0].images}


def main():
    manifest=load(PACKAGE/'manifest.json')
    assert sha(PAPER)==manifest['manuscript_sha256']
    for name,expected in manifest['files'].items():
        assert sha(PACKAGE/name)==expected, name
    sums={}
    for line in (PACKAGE/'SHA256SUMS').read_text().splitlines():
        digest,name=line.split('  ',1)
        assert sha(PACKAGE/name)==digest, name
        sums[name]=digest
    expected_names={str(p.relative_to(PACKAGE)) for p in PACKAGE.rglob('*') if p.is_file()}
    assert expected_names==set(sums)|{'SHA256SUMS'}
    with zipfile.ZipFile(OUT/'paper-replacements.zip') as archive:
        assert archive.testzip() is None
        names={name.removeprefix('synthetic-paper-replacements/') for name in archive.namelist()}
        assert names==expected_names
        for name in expected_names:
            assert archive.read('synthetic-paper-replacements/'+name)==(PACKAGE/name).read_bytes(),name
    for number in [7,8]:
        assert sha(PACKAGE/f'figures/fig{number:02d}.pdf')==sha(SOURCE/f'figures/figure{number:02d}.pdf')
    original_images=image_hashes(SOURCE/'figures/figure09.pdf')
    new_images=image_hashes(PACKAGE/'figures/fig09.pdf')
    assert len(original_images)==6 and new_images==original_images
    fonts=subprocess.check_output(['/usr/bin/pdffonts',str(PACKAGE/'figures/fig09.pdf')],text=True)
    assert all(' yes ' in row for row in fonts.splitlines()[2:]),fonts
    source_text=PAPER.read_text()
    mapping=load(PACKAGE/'replacement-map.json')['items']
    for item in mapping:
        assert (PACKAGE/item['file']).is_file()
        if 'original_text_sha256' in item:
            a=source_text.index(item['start_anchor'])
            b=source_text.index(item['end_anchor'],a)+len(item['end_anchor'])
            assert hashlib.sha256(source_text[a:b].encode()).hexdigest()==item['original_text_sha256']
        if 'label' in item:
            assert source_text.count('\\label{'+item['label']+'}')==1
            assert (PACKAGE/item['file']).read_text().count('\\label{'+item['label']+'}')==1
    for name in PACKAGE.glob('latex/**/*.tex'):
        balanced_tex(name.read_text())
        for target in re.findall(r'\\input\{([^}]+)\}',name.read_text()):
            assert (PACKAGE/target).is_file(),target
    data=load(PACKAGE/'data/numeric-results.json')
    table_rows=list(csv.DictReader((PACKAGE/'tables/table04-metrics.csv').open()))
    for row in table_rows:
        saved=data['methods'][row['method']]
        for key,value in row.items():
            if key!='method': assert float(value)==saved[key], (row['method'],key)
    composition=list(csv.DictReader((PACKAGE/'data/figure08-fp_composition.csv').open()))
    for method,source_method in [('ours','ours'),('depth_free','depth_free'),('minip2d','minip_2d')]:
        assert sum(int(r['count']) for r in composition if r['method']==source_method)==data['methods'][method]['false_positives_at_recall_0p8']
    stage_rows=list(csv.DictReader((PACKAGE/'tables/table05-stages.csv').open()))
    assert len(stage_rows)==4
    for row,saved in zip(stage_rows,data['table4']):
        assert row['stage']==saved['stage']
        for key in ['event_ap','maximum_f1','false_positives_at_recall_0p8','false_positives_at_recall_0p9']:
            assert float(row[key])==saved[key]
    table4=(PACKAGE/'latex/table04.tex').read_text()
    table5=(PACKAGE/'latex/table05.tex').read_text()
    assert '\\citep{scikit-learn}' not in table4
    assert 'coordinate' in table5 and 'candidate selection' in table5
    results=(PACKAGE/'latex/results-synthetic.tex').read_text()
    for old in ['0.917','0.901','0.858','266 to 216','0.890 to 0.336','successively improved performance across all']:
        assert old not in results, old
    result={'status':'PASS','package_files':len(expected_names),'archive_entries_verified':len(expected_names),
            'source_figures_7_8_bit_identical':True,'figure9_original_image_payloads_preserved':len(original_images),
            'figure9_fonts_embedded':True,'replacement_locations_verified':len(mapping),
            'metrics_and_fp_composition_consistent':True,'tex_braces_environments_inputs_checked':True,
            'full_manuscript_compilation':False,'manuscript_unchanged':True,
            'zip_sha256':sha(OUT/'paper-replacements.zip')}
    (OUT/'verification.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))


if __name__=='__main__': main()
