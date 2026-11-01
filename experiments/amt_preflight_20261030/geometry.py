"""Audit truth coverage for every scene without changing the recorded truth."""
from common import *

rows = []; sources = []; exceptions = []
for seed in SEEDS:
    path = truth_path(seed, 'particles.parquet')
    particles = pd.read_parquet(path, columns=['frame', 'particle_id', 'z_um', 'center_in_volume'])
    truth = read_table(truth_path(seed))
    inside = particles.loc[particles.center_in_volume.eq(1)]
    outside = particles.loc[particles.center_in_volume.ne(1)]
    assert inside.groupby('frame').size().eq(200).all()
    extra = inside.loc[inside.z_um.gt(102300)].copy(); extra.insert(0, 'seed', seed)
    exceptions.append(extra)
    row = {'seed': seed, 'frames': inside.frame.nunique(), 'all_particle_points': len(particles),
           'in_volume_points': len(inside), 'outside_nominal_xy_points': len(outside),
           'particle_z_min_um': float(inside.z_um.min()), 'particle_z_max_um': float(inside.z_um.max()),
           'outside_xy_z_min_um': float(outside.z_um.min()), 'outside_xy_z_max_um': float(outside.z_um.max()),
           'old_grid_missed_particle_points': int(inside.z_um.gt(92300).sum()),
           'corrected_grid_missed_particle_points': len(extra),
           'events': len(truth), 'event_z_min_um': float(truth.z_um.min()), 'event_z_max_um': float(truth.z_um.max()),
           'old_grid_missed_events': int(truth.z_um.gt(92300).sum()),
           'corrected_grid_missed_events': int((truth.z_um.lt(0) | truth.z_um.gt(102300)).sum())}
    rows.append(row)
    sources.extend([{'seed': seed, 'role': role, 'path': str(p), 'sha256': sha256(p)}
                    for role, p in [('particles', path), ('events', truth_path(seed))]])
    emit(**row)
write_table(pd.DataFrame(rows), ROOT / 'results/geometry.csv')
write_table(pd.concat(exceptions, ignore_index=True), ROOT / 'results/particles_beyond_102300um.csv')
write_json({'status': 'PASS', 'sources': sources,
            'coordinate_rule': 'camera_distance_um=74900+truth_z_um',
            'new_grid_camera_planes_um': [74900, 177200], 'planes': 1024, 'step_um': 100,
            'old_grid_camera_planes_um': [64900, 167200], 'old_grid_in_truth_coordinates_um': [-10000, 92300]},
           ROOT / 'results/geometry.json')
