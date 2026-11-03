#!/usr/bin/env python3
"""Read back scoped Quick Tunnel assets and verify their exact local bytes."""
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import hashlib
import io
import json
import os
from pathlib import Path
from urllib.request import urlopen
import zipfile

REPO=Path(__file__).resolve().parents[2]
PREVIEW=REPO.parent/'audit/revision-review-20261101'
OUT=REPO.parent/'audit/revision-figure9-shared-particle-20261103'
PUBLIC=PREVIEW/'public'


def sha(data):return hashlib.sha256(data).hexdigest()


def main():
    inputs=json.loads((OUT/'publication-inputs.json').read_text())
    proc=json.loads((PREVIEW/'preview-processes.json').read_text())
    base=proc['url'].rstrip('/')
    assert base=='https://messenger-recognition-forgotten-valium.trycloudflare.com'
    for name in ['server_pid','tunnel_pid']:os.kill(int(proc[name]),0)
    paths=inputs['published_paths']+['fig09-artifact-options/comparison.json',
        'fig09-artifact-options/caption-options.txt','fig09-artifact-options/review-assets.zip',
        'fig09-confirmed-a/fig09-a-confirmed.pdf']
    paths=list(dict.fromkeys(paths))
    def check(path):
        local=(PUBLIC/path).read_bytes()
        with urlopen(base+'/'+path+'?review=shared-particle-20261103',timeout=30) as response:
            data=response.read();status=response.status
        assert status==200 and sha(data)==sha(local),path
        return dict(path=path,status=status,bytes=len(data),sha256=sha(data)),data
    with ThreadPoolExecutor(max_workers=4) as workers:
        results=list(workers.map(check,paths))
    remote={r['path']:data for r,data in results}
    z=zipfile.ZipFile(io.BytesIO(remote['fig09-shared-particle/review-assets.zip']))
    expected=inputs['zip_members'];assert set(z.namelist())==set(expected)
    assert all(sha(z.read(name))==digest for name,digest in expected.items())
    old=zipfile.ZipFile(io.BytesIO(remote['fig09-artifact-options/review-assets.zip']))
    assert 'WITHDRAWN.txt' in old.namelist()
    assert json.loads(remote['fig09-artifact-options/comparison.json'])['status']=='WITHDRAWN'
    review=json.loads(remote['fig09-shared-particle/review.json'])
    assert [r['case_id'] for r in review['cases']]==['D4','D2']
    assert not review['author_selected_b']
    html=remote['fig09-artifact-options.html'].decode()
    assert '第一候補はB01' not in html and '前回のB01・B02は撤回' in html
    assert '真値の輪郭' in html and '＋0.25 msでは青が欠測' in html
    assert 'grid-template-columns:1fr' in html and 'width=device-width' in html
    for path,expected_hash in inputs['protected_files'].items():
        assert sha(Path(path).read_bytes())==expected_hash
    visual=dict(status='inspected',original_manuscript_fig09_render_inspected=True,
        contact_sheets_inspected=['D1','D2','D3','D4'],
        published_cases=['D4','D2'],all_published_track_frames_inspected=True,
        truth_outline_views_inspected=['D4 frame 5337','D2 frame 5508'],
        all_four_diagnostic_images_inspected=True,
        three_panel_previews_inspected=['D4-proposed-three.png','D2-proposed-three.png'],
        browser_render_test=False,
        web_checks='HTTPS page/assets/JSON/download byte checks and local HTML resource/anchor checks. Responsive CSS inspected; no browser-render claim.')
    result=dict(status='PASS',verified_at=datetime.now(timezone.utc).isoformat(),
        url=base+'/fig09-artifact-options.html',https_files_checked=len(results),zip_members_checked=len(expected),
        responses=[r for r,data in results],visual_review=visual,
        numerical_outputs_unchanged=inputs['numerical_outputs_unchanged'],
        author_confirmed_a_unchanged=True,old_b_recommendations_and_downloads_withdrawn=True,
        server={key:proc[key] for key in ['port','server_pid','tunnel_pid','server_log','tunnel_log']},
        stop_command=f'python3 {PREVIEW}/stop_preview.py',
        tunnel_lifetime='Stops with the process; recreating it changes the URL.')
    (OUT/'verification.json').write_text(json.dumps(result,indent=2,ensure_ascii=False)+'\n')
    # A compact versioned record does not expose host process details.
    compact={k:v for k,v in result.items() if k not in ['responses','server','stop_command']}
    target=REPO/'experiments/amt_figure9_refinement_20261102/results/shared-particle/verification.json'
    target.write_text(json.dumps(compact,indent=2,ensure_ascii=False)+'\n')
    print(json.dumps({k:result[k] for k in ['status','url','https_files_checked','zip_members_checked',
        'numerical_outputs_unchanged','author_confirmed_a_unchanged','old_b_recommendations_and_downloads_withdrawn']},indent=2))


if __name__=='__main__':main()
