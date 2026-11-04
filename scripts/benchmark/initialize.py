"""Prepare truth tables and synchronized image manifests."""
from common import *
import yaml
import shutil


def verify_upstream():
    import sys
    sys.path.insert(0, str(REPO / 'scripts'))
    from prepare_data import prepare
    prepare('measurement')
    contract = json.loads((REPO / 'reference/measurement/manifest.json').read_text())
    for name, expected in contract['files'].items():
        if sha256(REPO / 'reference/measurement' / name) != expected:
            raise ValueError(f'Measurement source checksum mismatch: {name}')
    engine = json.loads((REPO / 'reference/engine-manifest.json').read_text())
    for name, expected in engine['files'].items():
        if sha256(REPO / name) != expected:
            raise ValueError(f'Collision engine source checksum mismatch: {name}')
    write_json(json.loads((REPO / 'reference/benchmark.json').read_text()), ROOT / 'config/protocol.json')


def main(seeds=SEEDS):
    verify_upstream()
    from generate_holograms import verify_labels
    records=[]
    for seed in seeds:
        source=GENERATION/f'seed{seed}'
        verify_labels(source, seed)
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
        config['description']='Numerical hologram sequence with an endpoint-inclusive MinIP.'
        config['frames']={'primary_holograms':str(source/'01_bkgrem/C002H001S0011'),
                          'secondary_holograms':str(source/'01_bkgrem/C001H001S0011'),
                          'minip':str(minip_dir(seed))}
        (ROOT/f'config/seed{seed}-measurement.yaml').write_text(yaml.safe_dump(config,sort_keys=False))
        emit(stage='inputs_initialized',seed=seed)
    write_json({'status':'PASS','sources':records},ROOT/'config/truth-manifest.json')


if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser()
    parser.add_argument('--seeds',type=int,nargs='+',default=SEEDS)
    main(parser.parse_args().seeds)
