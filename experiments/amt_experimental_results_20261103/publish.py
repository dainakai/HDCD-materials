"""Publish only this report's deliverables through the existing manifest-only server."""
import os
import shutil
import zipfile
from common import *

def main():
    assert read(PACKAGE/'verification.json')['status']=='PASS'
    final_files={str(p.relative_to(PACKAGE)):sha(p) for p in sorted(PACKAGE.rglob('*')) if p.is_file() and p.name not in ['experimental-replacements.zip','file-manifest.json']}
    save(PACKAGE/'file-manifest.json',{'files':final_files})
    archive=PACKAGE/'experimental-replacements.zip'
    with zipfile.ZipFile(archive.with_suffix('.partial'),'w',compression=zipfile.ZIP_DEFLATED,compresslevel=6) as z:
        for name in [*final_files,'file-manifest.json']:z.write(PACKAGE/name,name)
    archive.with_suffix('.partial').replace(archive)
    with zipfile.ZipFile(archive) as z:
        assert z.testzip() is None
        for name,digest in final_files.items():assert __import__('hashlib').sha256(z.read(name)).hexdigest()==digest
    preview=WORK/'audit/revision-review-20261101'
    prefix='experimental-results-20261103'
    target=preview/'public'/prefix
    target.mkdir(exist_ok=True)
    additions={}
    for source in sorted(PACKAGE.rglob('*')):
        if not source.is_file():continue
        relative=source.relative_to(PACKAGE)
        destination=target/relative;destination.parent.mkdir(parents=True,exist_ok=True)
        temporary=destination.with_suffix(destination.suffix+'.partial')
        shutil.copy2(source,temporary);temporary.replace(destination)
        additions[f'{prefix}/{relative}']=sha(destination)
    # Merge the latest allowlist. Never rebuild or delete concurrent Fig.9 or other report assets.
    manifest_path=preview/'public-manifest.json'
    for attempt in range(5):
        before=manifest_path.read_bytes();current=__import__('json').loads(before)
        current['files'].update(additions)
        current['total_bytes']=sum((preview/'public'/name).stat().st_size for name in current['files'])
        temporary=manifest_path.with_name('public-manifest.experimental-results.partial')
        save(temporary,current)
        if manifest_path.read_bytes()==before:
            temporary.replace(manifest_path);break
    else:raise RuntimeError('Concurrent manifest updates; no existing entries replaced.')
    preview_receipt=read(preview/'preview-processes.json')
    base=preview_receipt['url'].rstrip('/')
    save(ROOT/'publication.json',{'created_at':stamp(),'url':base+'/'+prefix+'/index.html',
        'zip_url':base+'/'+prefix+'/experimental-replacements.zip','pdf_url':base+'/'+prefix+'/experimental_figures.pdf',
        'port':preview_receipt['port'],'server_pid':preview_receipt['server_pid'],'tunnel_pid':preview_receipt['tunnel_pid'],
        'server_log':preview_receipt['server_log'],'tunnel_log':preview_receipt['tunnel_log'],
        'archive_sha256':sha(archive),'archive_bytes':archive.stat().st_size,'published_files':len(additions),
        'stop_method':'Use audit/revision-review-20261101/stop_preview.py only when all shared previews may be stopped; this report did not start another server or tunnel.',
        'scope':'Existing report/tunnel reused. Only this package prefix was copied; previous entries preserved.'})
    print(base+'/'+prefix+'/index.html')
    print(f'ZIP: {archive.stat().st_size:,} bytes; {len(additions)} public deliverables')

if __name__=='__main__':main()
