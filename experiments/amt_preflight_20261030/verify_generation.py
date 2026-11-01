"""Check complete regenerated truth sequences without rendering images."""
from common import *
import argparse


def verify(seed):
    directory=ROOT/f'generation/seed{seed}'
    particles=pd.read_csv(directory/'labels/particles.csv',usecols=['frame','particle_id','z_um','center_in_volume'])
    events=pd.read_csv(directory/'labels/collision_events.csv')
    inside=particles.loc[particles.center_in_volume.eq(1)]
    assert len(inside)==1200000 and inside.groupby('frame').size().eq(200).all()
    assert inside.frame.nunique()==6000
    assert particles.z_um.between(0,102300,inclusive='left').all()
    assert len(events)==2000 and events.z_um.between(0,102300).all()
    old=read_table(truth_path(seed)).set_index('collision_event_id')
    changed=np.any(abs(events.set_index('collision_event_id')[['frame','x_um','y_um','z_um']]-old[['frame','x_um','y_um','z_um']])>1e-6,axis=1)
    summary={'status':'PASS','seed':seed,'frames':6000,'particles_per_frame':200,'events':len(events),
             'in_volume_particle_points':len(inside),'all_particle_points':len(particles),
             'outside_nominal_xy_points':int(particles.center_in_volume.eq(0).sum()),
             'all_z_min_um':float(particles.z_um.min()),'all_z_max_um':float(particles.z_um.max()),
             'event_z_min_um':float(events.z_um.min()),'event_z_max_um':float(events.z_um.max()),
             'all_particle_centers_and_events_within_revised_grid':True,'events_with_changed_time_or_position':int(changed.sum()),
             'holograms_rendered':False,'processing_completed':'numerical truth only',
             'source_csv_sha256':sha256(directory/'labels/particles.csv'),
             'events_csv_sha256':sha256(directory/'labels/collision_events.csv')}
    write_json(summary,ROOT/f'results/revised_generation_seed{seed}.json')
    if seed==260745:
        write_json(summary,ROOT/'results/revised_generation.json')
    emit(**summary)
    return summary


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--seeds',type=int,nargs='+',default=[260745])
    args=parser.parse_args()
    rows=[verify(seed) for seed in args.seeds]
    if set(args.seeds)==set(SEEDS):
        write_table(pd.DataFrame(rows),ROOT/'results/revised_generation_all.csv')
        write_json({'status':'PASS','sequences':rows,'all_six_complete':True,
                    'total_in_volume_points':sum(r['in_volume_particle_points'] for r in rows),
                    'total_all_particle_points':sum(r['all_particle_points'] for r in rows),
                    'total_events':sum(r['events'] for r in rows),
                    'total_out_of_range_particle_centers':0,'total_out_of_range_events':0,
                    'holograms_rendered':False},ROOT/'results/revised_generation_all.json')
