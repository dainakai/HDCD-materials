"""Materialize this campaign's own truth tables and absolute image manifests."""
from common import *
import yaml
import shutil


def verify_upstream():
    snapshot=json.loads((SCRIPT/'production-source.json').read_text())
    original=Path(snapshot['source'])
    destination=ROOT/'upstream/production'
    for name,digest in snapshot['files'].items():
        target=destination/name
        if not target.exists():
            target.parent.mkdir(parents=True,exist_ok=True)
            shutil.copy2(original/name,target)
        assert sha256(target)==digest, f'Production source or measurement weights changed: {target}'
    reference=REPO/'reference/generation'
    for name in ['render_holograms.jl','Project.toml','Manifest.toml']:
        target=ROOT/'renderer'/name
        if not target.exists():
            target.parent.mkdir(parents=True,exist_ok=True)
            shutil.copy2(reference/name,target)
        assert sha256(target)==sha256(reference/name)


def main():
    verify_upstream()
    records=[]
    for seed in SEEDS:
        source=GENERATION/f'seed{seed}'
        for name in ['particles','collision_events']:
            src=source/f'labels/{name}.csv';dst=truth_path(seed,name+'.parquet')
            marker=dst.with_suffix('.json')
            digest=sha256(src)
            if marker.exists():
                saved=json.loads(marker.read_text())
                assert saved['source_sha256']==digest and sha256(dst)==saved['sha256']
            else:
                frame=pd.read_csv(src)
                if name=='particles':
                    assert frame.frame.nunique()==6000
                    inside=frame.loc[frame.center_in_volume.eq(1)]
                    assert len(inside)==1200000 and inside.z_um.between(0,102300).all()
                else:
                    assert len(frame)==2000 and frame.z_um.between(0,102300).all()
                write_table(frame,dst)
                write_json({'source':str(src),'source_sha256':digest,'sha256':sha256(dst),'rows':len(frame)},marker)
            records.append(json.loads(marker.read_text()))
        frames=pd.read_csv(source/'frames.csv')
        for col in ['primary','secondary']:
            frames[col]=frames[col].map(lambda v:str(source/v))
        manifest=ROOT/f'config/seed{seed}-frames.csv';write_table(frames,manifest)
        config=yaml.safe_load((source/'acquisition.yaml').read_text())
        config['frames']=manifest.name
        (ROOT/f'config/seed{seed}-collision.yaml').write_text(yaml.safe_dump(config,sort_keys=False))
        config=yaml.safe_load((source/'holod3-acquisition.yaml').read_text())
        config['name']=f'amt-full-102p3-phase3-seed{seed}'
        config['description']='Full revised numerical sequence. Original production measurement pipeline; endpoint-inclusive MinIP.'
        config['frames']={'primary_holograms':str(source/'01_bkgrem/C002H001S0011'),
                          'secondary_holograms':str(source/'01_bkgrem/C001H001S0011'),
                          'minip':str(minip_dir(seed))}
        (ROOT/f'config/seed{seed}-measurement.yaml').write_text(yaml.safe_dump(config,sort_keys=False))
        emit(stage='inputs_initialized',seed=seed)
    write_json({'status':'PASS','sources':records},ROOT/'config/truth-manifest.json')


if __name__=='__main__':main()
