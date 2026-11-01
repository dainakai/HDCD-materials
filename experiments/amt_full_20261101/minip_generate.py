"""Retain the original 512-plane MinIP arithmetic and append its terminal plane."""
from common import *
import argparse
import fcntl
import sys
import time
import torch
SOURCE = ROOT/'upstream/production'
sys.path.insert(0,str(SOURCE))
from src.preprocessing import proc


def projection(holo, setup, terminal_transfer):
    offset=(setup.padlen-setup.datlen)//2
    frequency=proc.fft2(proc.mean_pad_complex(holo,setup.padlen),'torch')*setup.d_tf_unshifted
    minimum=None
    for i in range(setup.slices+1):
        if i:
            frequency=frequency*(terminal_transfer if i==setup.slices else setup.d_slice_unshifted)
        field=proc.ifft2(frequency,'torch')
        intensity=(field.real.square()+field.imag.square()).clamp_(0.,1.)
        central=intensity[offset:offset+setup.datlen,offset:offset+setup.datlen]
        if i==setup.slices: historical=minimum.clone()
        minimum=central if minimum is None else torch.minimum(minimum,central)
    return minimum,historical


def run(seed,limit,device):
    source=GENERATION/f'seed{seed}'
    destination=minip_dir(seed);destination.mkdir(parents=True,exist_ok=True)
    lock=(destination/'.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX)
    pairs=proc.matched_pairs(source,'C002','C001',proc.scene_name(11),True,None,None,limit)
    assert len(pairs)==(limit or 6000)
    dev=torch.device(device)
    setup=proc.build_setup(source,10,dev,1024,1536,2,20000.)
    assert setup.slices==512 and setup.z0_um==74900. and setup.effective_dz_um==200.
    wavlen=setup.variables['lambda']
    terminal=torch.fft.ifftshift(proc.transfer(-100.,1536,wavlen,proc.transfer_sqrt_arr(1536,wavlen,10.,dev)))
    indices=list(range(0,1024,2))+[1023]
    identity={'algorithm':'original proc.py, direct 200 um recurrence, then direct 100 um terminal propagation',
              'code_sha256':sha256(Path(__file__)),
              'source_files':{n:sha256(SOURCE/n) for n in ['src/preprocessing/proc.py','src/detection/depth_model_estimate.py']},
              'slice_indices':indices,'padding':1536,'phase_iterations':3,'front_um':74900.}
    identity_file=destination/'identity.json'
    if identity_file.exists(): assert json.loads(identity_file.read_text())==identity
    else: write_json(identity,identity_file)
    began=time.monotonic();count=0
    with torch.inference_mode():
        for pair in pairs:
            path=destination/pair.name;marker=path.with_suffix('.json')
            inputs={'primary':sha256(pair.camera1_path),'secondary':sha256(pair.camera2_path)}
            if marker.exists():
                saved=json.loads(marker.read_text());assert saved['inputs']==inputs and sha256(path)==saved['sha256']
                continue
            if path.exists(): raise RuntimeError(f'Unverified projection: {path}')
            primary=torch.from_numpy(proc.load_gray2float(pair.camera1_path)).to(dev)
            secondary=torch.from_numpy(proc.quadratic_distortion_correction(proc.load_gray2float(pair.camera2_path),setup.coeffs)).to(dev)
            holo=proc.phase_retrieval_holo(primary,secondary,setup.d_pr,setup.d_pr_inv,3,'torch')
            result,old=projection(holo,setup,terminal)
            if count==0:
                reference=proc.xy_min_projection(holo,setup,3)
                assert torch.equal(old,reference),'The existing 512-plane calculation changed.'
                assert bool((result<=reference).all())
                write_json({'status':'PASS','original_512_planes_bit_equal':True,'terminal_plane_included':True,
                            'first_compared_image':pair.name,'changed_float_pixels':int((result<reference).sum()),
                            'mean_reduction':float((reference-result).mean())},destination/'endpoint-check.json')
            arr=result.cpu().numpy();assert np.isfinite(arr).all() and arr.shape==(1024,1024)
            proc.save_png(path,arr)
            write_json({'inputs':inputs,'sha256':sha256(path)},marker);count+=1
            if count<=2 or count%100==0:emit(stage='minip',seed=seed,frame=pair.name,newly_rendered=count,seconds=time.monotonic()-began)
    write_json({'status':'PASS','frames':len(pairs),'newly_rendered':count,'seconds':time.monotonic()-began,
                'identity':identity},destination/('complete.json' if not limit else f'smoke-{limit}.json'))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--seed',type=int,required=True)
    p.add_argument('--limit',type=int,default=0);p.add_argument('--device',default='cuda:0');a=p.parse_args()
    run(a.seed,a.limit,a.device)
