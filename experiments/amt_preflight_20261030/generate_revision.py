"""Generate revised numerical truth, with an explicit new identity and optics."""
from pathlib import Path
import argparse
import json
import os
import sys
import yaml

REPO = Path(__file__).resolve().parents[2]
ROOT = REPO.parent / 'audit/revision-preflight-20261030'
for name in ['MPLCONFIGDIR','XDG_CACHE_HOME','TMPDIR']:
    destination=ROOT/('tmp' if name=='TMPDIR' else 'cache')/name.lower()
    destination.mkdir(parents=True,exist_ok=True)
    os.environ[name]=str(destination)
sys.path.insert(0, str(REPO / 'src'))
os.environ['PYTHONDONTWRITEBYTECODE'] = '1'
from ra3d.simulation import generate

def write_profiles(output, seed):
    # This is the one processing profile to use for future rendered images.
    # Stage-specific padding is deliberately preserved.
    frame_csv = output / 'frames.csv'
    with frame_csv.open('w') as f:
        f.write('frame,primary,secondary\n')
        for frame in range(6000):
            f.write(f'{frame},01_bkgrem/C002H001S0011/{frame+1:06d}.png,01_bkgrem/C001H001S0011/{frame+1:06d}.png\n')
    for name, padding in [('measurement-acquisition.yaml', 1536), ('acquisition.yaml', 2048)]:
        data = {'schema_version':1, 'frames':'frames.csv',
                'units':{'pixel_pitch_um':10.,'slice_spacing_um':100.,'fps':4000.,'image_size_px':1024},
                'optics':{'wavelength_um':.6328,'phase_distance_um':21450.,'reconstruction_start_um':74900.,
                          'phase_iterations':3,'padding_px':padding,'mode':'dual_phase_retrieval'}}
        (output / name).write_text(yaml.safe_dump(data))
    upstream = {
        'schema_version':1, 'name':f'amt-preflight-seed-{seed}',
        'description':'Planned revised numerical sequence; render paired holograms before processing.',
        'mode':'dual_phase_retrieval',
        'frames':{'primary_holograms':'01_bkgrem/C002H001S0011',
                  'secondary_holograms':'01_bkgrem/C001H001S0011'},
        'optics':{'wavelength_um':.6328,'pixel_pitch_um':10.,'image_size_px':1024,
                  'reconstruction_start_um':74900.,'slice_spacing_um':100.,
                  'slice_count':1024,'phase_retrieval_distance_um':21450.},
        'reconstruction':{'phase_retrieval_iterations':3,'fft_padding_side':1536,'minip_slice_step':2},
        'calibration':{'secondary_distortion_coefficients':None},
        'transforms':{'primary':[],'secondary':[],'minip':[]}}
    (output / 'holod3-acquisition.yaml').write_text(yaml.safe_dump(upstream,sort_keys=False))
    (output / 'revision-profile.json').write_text(json.dumps({
        'version':'amt-preflight-20261030','truth_depth_um':102300,'reconstruction_planes':1024,
        'reconstruction_first_um':74900,'reconstruction_last_um':177200,'phase_iterations':3,
        'measurement_padding_px':1536,'image_padding_px':2048,'small_particle_y_jitter':True,
        'historical_variables_yaml':'Provenance-only legacy metadata; use the explicit acquisition profiles.',
        'minip_endpoint_note':'Historical step 2 samples through 177100 um; explicitly include 177200 um before claiming full endpoint coverage.',
        'holograms_rendered':False},indent=2)+'\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, required=True)
    parser.add_argument('--profiles-only', action='store_true', help='Refresh only this experiment\'s planned processing profiles')
    args = parser.parse_args()
    output = ROOT / f'generation/seed{args.seed}'
    if args.profiles_only:
        metadata=json.loads((output/'metadata/parameters.json').read_text())
        assert metadata['depth_um']==102300 and metadata['slices']==1024
        assert metadata['theme']=='amt_revision_1023mm_3phase_preflight'
    else:
        if (output / 'metadata/generated_summary.json').exists():
            raise FileExistsError(output)
        contract = json.loads((REPO / 'reference/generation/contract.json').read_text())
        arguments = [v.replace('{root}', str(REPO)) for v in contract['label_arguments']]
        arguments[arguments.index('--theme')+1] = 'amt_revision_1023mm_3phase_preflight'
        sys.argv = [__file__, '--seed', str(args.seed), '--project-root', str(REPO),
                    '--out-dir', str(output), '--volume-depth-um', '102300', *arguments]
        generate.main()
    write_profiles(output, args.seed)
