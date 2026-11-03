#!/usr/bin/env python3
"""Check the existing authorized Quick Tunnel against the local delivery hashes."""
import concurrent.futures
from datetime import datetime
import hashlib
import json
import os
from urllib.request import Request, urlopen
from urllib.error import HTTPError

from build import OUT, PUBLIC, WORK, REFINED, load, sha


def main():
    receipt=load(WORK/'audit/revision-review-20261101/preview-processes.json')
    for key in ['server_pid','tunnel_pid']:os.kill(int(receipt[key]),0)
    base=receipt['url'].rstrip('/')
    names=['synthetic.html','paper-replacements.zip','paper-assets/paper.css','paper-assets/paper.js']
    names += [str(p.relative_to(PUBLIC)) for p in sorted((PUBLIC/'paper').rglob('*')) if p.is_file()]
    names += load(REFINED/'publication-inputs.json')['published_paths']
    names = list(dict.fromkeys(names))
    def verify(name):
        with urlopen(Request(base+'/'+name,headers={'User-Agent':'HDCD-publication-verifier/1.0'}),timeout=40) as r:
            assert r.status==200, (name,r.status)
            content=r.read()
        digest=hashlib.sha256(content).hexdigest()
        assert digest==sha(PUBLIC/name),name
        return {'path':name,'bytes':len(content),'sha256':digest,'status':200}
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
        checks=list(executor.map(verify,names))
    for path in ['/.git/config','/reviews.sqlite3','/input-inventory.json']:
        try:
            with urlopen(base+path,timeout=15) as r:
                raise AssertionError((path,r.status))
        except HTTPError as exc:
            assert exc.code==404,(path,exc.code)
    result={'status':'PASS','verified_at':datetime.now().astimezone().isoformat(),
            'report_url':base+'/synthetic.html','zip_url':base+'/paper-replacements.zip',
            'https_files_verified':len(checks),'checks':checks,
            'browser_rendering':'Not verified: local browser use is disabled in Work settings.',
            'server_port':receipt['port'],'server_pid':receipt['server_pid'],'tunnel_pid':receipt['tunnel_pid'],
            'server_log':receipt['server_log'],'tunnel_log':receipt['tunnel_log'],
            'process_receipt':'audit/revision-review-20261101/preview-processes.json',
            'stop_command':f'python3 {WORK}/audit/revision-review-20261101/stop_preview.py',
            'scope':'Existing report allowlist plus the synthetic replacement package; no source research images, weights or review answers.'}
    (OUT/'publication.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k!='checks'},indent=2))


if __name__=='__main__': main()
