"""Reprocess an experimental scene without fitting or consulting review labels."""
from common import *
import argparse
import csv
import fcntl
import shutil
import subprocess
import time
import numpy as np
import pandas as pd
import torch
from threadpoolctl import threadpool_limits
from ra3d.io import read_table, write_table, legacy_candidate_csv, json_value
from ra3d.tracks import read_tracks
from ra3d.acquisition import Acquisition, Reconstructor
from ra3d.models import load_model
from ra3d.config import Units, STAGE1_FEATURES, IMAGE_FEATURES
from ra3d.features import generate_candidates, static_features, image_features
from ra3d._core.features import feature_frame
from ra3d.cascade import shortlist, score_static, image_mask, scoring_union, predict_full


def command(args, log):
    emit(stage='command', command=list(map(str, args)), log=str(log))
    log.parent.mkdir(parents=True, exist_ok=True)
    with log.open('a') as stream:
        subprocess.run(list(map(str, args)), cwd=SOURCE, check=True, stdout=stream,
                       stderr=subprocess.STDOUT, env={**os.environ, 'CUDA_VISIBLE_DEVICES':'0',
                         'PATH':'/home/dai-server/.juliaup/bin:'+os.environ['PATH']})


def background(n, root, limit):
    outputs = {}
    for camera in ['C001', 'C002']:
        target = root / f'prepared/01_bkgrem/{camera}H001S{n:04d}'
        expected = limit or FRAMES
        marker = target / 'verified.json'
        if marker.exists():
            record = read(marker)
            assert record['frames'] == expected
            assert sha(target / 'frames.csv') == record['manifest_sha256']
            outputs[camera] = record
            continue
        if target.exists() and any(target.iterdir()):
            raise RuntimeError(f'Partial background output preserved; inspect before retry: {target}')
        source = RAW / f'00_img/{camera}H001S{n:04d}'
        command([GPU_PYTHON, SOURCE/'src/preprocessing/backrem_streaming.py', '--input-dir',source,
                 '--output-dir',target,'--suffix','.bmp','--limit',limit,'--window',129,'--anchor-stride',32,
                 '--batch-size',16,'--save-workers',4,'--median-backend','auto','--device','cuda:0'],
                 root/f'logs/background-{camera}.log')
        images = sorted(target.glob('*.png'))
        assert [int(p.stem) for p in images] == list(range(1, expected+1))
        summary = read(target/'streaming_summary.json')
        assert summary['frames'] == expected
        with (target/'frames.csv').open('w',newline='') as stream:
            writer=csv.writer(stream);writer.writerow(['raw_index','name','sha256'])
            for p in images:writer.writerow([int(p.stem),p.name,sha(p)])
        record = {'frames':expected,'manifest_sha256':sha(target/'frames.csv'),
                  'summary_sha256':sha(target/'streaming_summary.json'),'source':str(source)}
        save(marker,record);outputs[camera]=record
    return outputs


def minip(n, root, limit):
    sys.path.insert(0,str(SOURCE))
    from src.preprocessing import proc
    # Import the exact terminal-plane projection already checked in the numerical campaign.
    import importlib.util
    spec=importlib.util.spec_from_file_location('verified_minip', REPO/'experiments/amt_full_20261101/minip_generate.py')
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    projection=module.projection
    contract=read(root/'input/contract.json')
    destination=root/'minip';destination.mkdir(exist_ok=True)
    pairs=proc.matched_pairs(root/'prepared','C002','C001',proc.scene_name(n),True,None,None,limit)
    assert len(pairs)==(limit or FRAMES)
    device=torch.device('cuda:0')
    setup=proc.build_setup(root/'prepared',int(contract['calibration_scene'][-4:]),device,
                           1024,1536,2,contract['glassplate_to_volume_um'])
    assert setup.z0_um==74900. and setup.slices==512
    terminal=torch.fft.ifftshift(proc.transfer(-100.,1536,.6328,proc.transfer_sqrt_arr(1536,.6328,10.,device)))
    checked=False
    with torch.inference_mode():
        for k,pair in enumerate(pairs):
            path=destination/pair.name;marker=path.with_suffix('.json')
            inputs={camera:sha(p) for camera,p in [('primary',pair.camera1_path),('secondary',pair.camera2_path)]}
            if marker.exists():
                record=read(marker);assert record['inputs']==inputs and sha(path)==record['sha256'];continue
            if path.exists():raise RuntimeError(f'Unverified projection preserved: {path}')
            p=torch.from_numpy(proc.load_gray2float(pair.camera1_path)).to(device)
            s=torch.from_numpy(proc.quadratic_distortion_correction(proc.load_gray2float(pair.camera2_path),setup.coeffs)).to(device)
            holo=proc.phase_retrieval_holo(p,s,setup.d_pr,setup.d_pr_inv,3,'torch')
            result,old=projection(holo,setup,terminal)
            if not checked:
                reference=proc.xy_min_projection(holo,setup,3)
                assert torch.equal(old,reference) and bool((result<=reference).all())
                save(destination/'endpoint-check.json',{'status':'PASS','historical_512_bit_equal':True,
                     'terminal_plane_1023_included':True,'first_image':pair.name});checked=True
            arr=result.cpu().numpy();assert arr.shape==(1024,1024) and np.isfinite(arr).all()
            proc.save_png(path,arr);save(marker,{'inputs':inputs,'sha256':sha(path)})
            if k<2 or (k+1)%100==0:emit(stage='minip',scene=n,frames=k+1,total=len(pairs))
    return {'frames':len(pairs),'phase_iterations':3,'front_um':74900,'last_um':177200,'planes':513}


def measurement(n, root, limit):
    contract=read(root/'input/contract.json');out=root/'measurement'
    if out.exists() and any(out.iterdir()):raise RuntimeError(f'Partial measurement preserved: {out}')
    weights={'--yolo-weights':'models/yolo/best.pt','--depth-checkpoint':'models/depth/depth_compare_robust_v1.pt',
             '--depth-fallback-checkpoint':'models/depth/depth_compare_best.pt','--slice-diam-weights':'models/slice_diam/slice_diammodel_best.pt'}
    args=[GPU_PYTHON,SOURCE/'src/pipeline/run_pipeline_fused.py','--data-dir',root/'prepared',
          '--image-dir',root/'minip','--run-dir',out,'--limit',limit,'--device','cuda:0','--scenenum',n,
          '--calibration-scene',contract['calibration_scene'],'--glassplate-to-volume-um',contract['glassplate_to_volume_um'],
          '--depth-padlen',1536]
    for flag,name in weights.items():args.extend([flag,SOURCE/name])
    command(args,root/'logs/measurement.log')
    frame=pd.read_csv(out/'particles_3d.csv',usecols=['frame','file','slice','z_um','depth_um'])
    stats=pd.read_csv(out/'frame_stats.csv')
    assert stats.frame.tolist()==list(range(limit or FRAMES))
    assert np.array_equal(frame.frame,frame.file.map(lambda x:int(Path(x).stem)-1))
    assert frame.frame.between(0,(limit or FRAMES)-1).all()
    assert np.allclose(frame.z_um,frame['slice']*100.,atol=1e-8,rtol=0)
    assert np.allclose(frame.depth_um,74900.+(frame['slice']-1)*100.,atol=1e-8,rtol=0)
    metrics=read(out/'fused_depth_slice_metrics.json');assert metrics['priter']==3
    return {'frames':len(stats),'particle_rows':len(frame),'particles_sha256':sha(out/'particles_3d.csv'),
            'phase_iterations':3,'front_um':74900,'last_um':177200,
            'weights':{name:sha(SOURCE/name) for name in weights.values()}}


def tracking(n, root, limit):
    out=root/'tracking';linked=out/'linking';out.mkdir(exist_ok=True)
    if not (linked/'zsmooth_tuning_summary.json').exists():
        args=[GPU_PYTHON,SOURCE/'src/tracking/link_best_tiebreak_z010.py',
              '--pipeline-run-dir',root/'measurement','--out-dir',linked]
        if (linked/'depth3d_link_diagnostics.json').exists():
            args=['/home/dai-server/.juliaup/bin/julia',SOURCE/'src/tracking/laptrack/zsmooth_new.jl',
                  f'--run-dir={linked}','--selection=auto','--force']
        command(args,root/'logs/tracking.log')
    raw=read_tracks(linked/'trajectory3d');parts=[]
    for file in sorted((linked/'smoothedtrajs').glob('path_*.csv')):
        item=pd.read_csv(file);item['track_id']=int(file.stem.split('_')[-1]);parts.append(item)
    smooth=pd.concat(parts,ignore_index=True).sort_values(['track_id','frame'],kind='stable').reset_index(drop=True)
    assert raw[['track_id','frame']].equals(smooth[['track_id','frame']])
    assert raw.frame.between(0,(limit or FRAMES)-1).all()
    write_table(raw,out/'raw.parquet');write_table(smooth,out/'smoothed.parquet')
    return {'rows':len(raw),'tracks':int(raw.track_id.nunique()),
            'outputs':{name:sha(out/name) for name in ['raw.parquet','smoothed.parquet']}}


def inference(n, root, limit):
    out=root/'inference';out.mkdir(exist_ok=True)
    bundle=load_model(MODEL,device='cuda:0');raw=read_tracks(root/'tracking/raw.parquet')
    tracks=read_tracks(root/'tracking/smoothed.parquet')
    def progress(value):emit(scene=n,**value)
    if not (out/'static_scored.parquet').exists():
        wide=generate_candidates(raw)
        write_table(wide,out/'wide.parquet')
        scores=bundle['stage1_model'].predict_proba(feature_frame(wide,STAGE1_FEATURES))[:,1]
        short=legacy_candidate_csv(shortlist(wide,scores,bundle['recipe']['outside_fraction']))
        write_table(short,out/'shortlist.parquet')
        static=static_features(short,tracks,root/'tracking/linking/smoothedtrajs',Units(),progress)
        scored=score_static(static,bundle['static_model']);write_table(scored,out/'static_scored.parquet')
    scored=read_table(out/'static_scored.parquet')
    requested=scored.loc[image_mask(scored,bundle['recipe']['image_fraction'])].reset_index(drop=True)
    recon=Reconstructor(Acquisition.load(root/'input/acquisition.yaml'),'cuda:0')
    if not (out/'image_features.parquet').exists():
        images=image_features(requested,tracks,recon,progress=progress,checkpoint_path=out/'image_checkpoint.npz')
        write_table(images,out/'image_features.parquet')
    images=read_table(out/'image_features.parquet')
    full=scoring_union(scored,images,bundle['recipe']);write_table(full,out/'paper92.parquet')
    predictions,audit=predict_full(full,bundle,scene_id=n)
    assert np.array_equal(predictions._lineage_rank,predictions._lineage_rank_before_suppression)
    write_table(predictions,out/'predictions.parquet')
    if not limit:assert len(predictions)>=80
    top=predictions.head(80).copy();top['review_rank']=np.arange(1,len(top)+1)
    top['scene_num']=n;top['run_id']=RUN_ID;top['model_sha256']=sha(MODEL/'model.json')
    top['prediction_sha256']=sha(out/'predictions.parquet')
    # Preserve candidate-level physical features for the later, separately reviewed analysis.
    selected=full.iloc[top._source_index.to_numpy(int)].reset_index(drop=True)
    for name in selected:
        if name not in top and name not in ['synth_seed']:top[name]=selected[name].to_numpy()
    write_table(top,out/'top80.parquet')
    return {'predictions':len(predictions),'review_candidates':len(top),'model_sha256':sha(MODEL/'model.json'),
            'prediction_sha256':sha(out/'predictions.parquet'),'top80_sha256':sha(out/'top80.parquet'),
            'input_tracks_sha256':sha(root/'tracking/smoothed.parquet'),
            'acquisition_sha256':sha(root/'input/acquisition.yaml'),'audit':audit}


def assets(n, root, limit):
    from ra3d.review import render_event_movie
    output=root/'review';output.mkdir(exist_ok=True)
    tracks=read_tracks(root/'tracking/smoothed.parquet')
    top=read_table(root/'inference/top80.parquet')
    recon=Reconstructor(Acquisition.load(root/'input/acquisition.yaml'),'cuda:0')
    candidates=[]
    for _,event in top.iterrows():
        key=hashlib.sha256(str(event.candidate_id).encode()).hexdigest()
        image=output/'media'/f'{key}.gif';marker=image.with_suffix('.json')
        if marker.exists():assert read(marker)['sha256']==sha(image)
        else:
            result=render_event_movie(event,tracks,recon,image)
            save(marker,{**result,'sha256':sha(image)})
        detail={'candidate_id':str(event.candidate_id),'scene_num':n,'review_rank':int(event.review_rank),
                'candidate_frame_raw':int(event.candidate_frame_raw),'score':float(event.score),
                'vanished_track_id':int(event.vanished_track_id),'survivor_track_id':int(event.survivor_track_id),
                'asset':str(image.relative_to(ROOT)),'asset_sha256':sha(image),'tracks':[]}
        for role,column in [('vanished','vanished_track_id'),('survivor','survivor_track_id')]:
            points=tracks.loc[tracks.track_id.eq(int(event[column])) & tracks.frame.between(event.candidate_frame_raw-30,event.candidate_frame_raw+30)]
            detail['tracks'].append({'role':role,'points':json_value(points[['frame','x','y','z','diameter_px']].to_dict('records'))})
        candidates.append(detail)
        emit(stage='review_assets',scene=n,candidates=len(candidates),total=len(top))
    receipt={'status':'PASS','scene_num':n,'run_id':RUN_ID,'budget':80,'candidates':candidates,
             'model_sha256':sha(MODEL/'model.json'),'prediction_sha256':sha(root/'inference/predictions.parquet'),
             'top80_sha256':sha(root/'inference/top80.parquet'),'smoke_only':bool(limit)}
    save(output/'manifest.json',receipt)
    return {'candidates':len(candidates),'manifest_sha256':sha(output/'manifest.json'),'smoke_only':bool(limit)}


STAGES=['background','minip','measurement','tracking','inference','assets']


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('stage',choices=STAGES)
    parser.add_argument('--scene',type=int,required=True,choices=SCENES)
    parser.add_argument('--smoke',type=int,default=0)
    args=parser.parse_args();verify_engine();root=scene(args.scene)
    if args.smoke:
        root=ROOT/f'smoke/{args.smoke}/scene{args.scene:04d}'
        if not (root/'input').exists():
            shutil.copytree(scene(args.scene)/'input',root/'input')
            shutil.copytree(scene(args.scene)/'prepared/02_calibration',root/'prepared/02_calibration')
            shutil.copy2(scene(args.scene)/'prepared/variables.yaml',root/'prepared/variables.yaml')
            frame=pd.read_csv(root/'input/frames.csv').iloc[:args.smoke].copy()
            for col in ['primary','secondary']:frame[col]=frame[col].str.replace(str(scene(args.scene)),str(root),regex=False)
            frame.to_csv(root/'input/frames.csv',index=False)
            # The calibration file is an immutable local copy with the same coefficients.
    lock=(root/'.stage.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    marker=root/f'stages/{args.stage}.json'
    if marker.exists():
        assert read(marker)['status']=='PASS';emit(stage=args.stage,scene=args.scene,status='ALREADY_COMPLETE');raise SystemExit(0)
    previous=STAGES.index(args.stage)
    if previous:assert read(root/f'stages/{STAGES[previous-1]}.json')['status']=='PASS'
    torch.set_num_threads(4);started=time.monotonic()
    with threadpool_limits(limits=4):
        result=globals()[args.stage](args.scene,root,args.smoke)
    save(marker,{'status':'PASS','scene':args.scene,'stage':args.stage,'smoke_frames':args.smoke,
                 'finished_at':stamp(),'seconds':time.monotonic()-started,'result':result,
                 'process_source_sha256':sha(Path(__file__)),
                 'input_contract_sha256':sha(root/'input/contract.json')})
    emit(stage=args.stage,scene=args.scene,status='PASS',seconds=time.monotonic()-started)
