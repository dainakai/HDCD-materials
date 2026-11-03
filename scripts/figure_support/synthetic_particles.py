"""Render particle precision-recall and error panels from the test measurements."""
from pathlib import Path
import json,math,shutil
import numpy as np
import pandas as pd
from figure_support import particle_plot as plot, particle_density as density, trajectory_length as length
from figure_support.fonts import register_fonts
from figure_support import composition
REPO=Path(__file__).resolve().parents[2]

def main(source,folder,output):
    folder.mkdir(parents=True,exist_ok=True)
    curve={k:v.to_numpy() for k,v in pd.read_parquet(source/'particle_pr_curve.parquet').items()}
    curve['average_precision']=json.loads((source/'summary.json').read_text())['particle_ap']
    errors=pd.read_parquet(source/'matched_particle_errors.parquet')
    for name in ['number_density.csv','truth_trajectory_summary.csv','measured_trajectory_summary.csv']:
        shutil.copy2(source/name,folder/name)
    register_fonts('1.07.4');plot.configure_plotting();plot.OUTPUT_DIR=folder
    plot.plot_pr(curve)
    xy=errors['xy_um'];z=np.abs(errors['z_um'])/1000.;diameter=errors['diameter_relative']*100.
    plot.plot_positive_histogram(xy,np.percentile(xy,[50,90,95]),xlim=(0,max(40,math.ceil(float(np.percentile(xy,99))/10)*10)),bins=140,
        xlabel=r'In-plane center error, $\sqrt{\Delta x^2+\Delta y^2}$ [$\mu$m]',unit=r'$\mu$m',facecolor='#4C78A8',edgecolor='#2F4B66',
        filename='xy_center_error_smoothed_hist_seed260745.pdf',decimals=2,major_tick=10)
    plot.plot_positive_histogram(z,np.percentile(z,[50,90,95]),xlim=(0,5),bins=125,
        xlabel=r'Axial absolute error, $|\Delta z|$ [mm]',unit='mm',facecolor='#568B7B',edgecolor='#315E53',
        filename='axial_error_smoothed_hist_seed260745.pdf',decimals=3,major_tick=1)
    plot.plot_diameter_histogram(diameter,np.percentile(diameter,[2.5,5,50,95,97.5]))
    for prefix,name in enumerate(['particle_detection_pr_curve_seed260745','xy_center_error_smoothed_hist_seed260745',
                                  'axial_error_smoothed_hist_seed260745','diameter_relative_error_hist_seed260745'],1):
        (folder/f'{name}.pdf').replace(folder/f'{prefix:02d}_{name}.pdf')
    density.SOURCE=folder/'number_density.csv';density.OUTPUT=folder;density.main()
    length.SOURCE=folder;length.OUTPUT=folder;length.main()
    composition.register_label_font()
    entry=next(e for e in json.loads((REPO/'reference/figure-layouts.json').read_text()) if e['output']=='fig07.pdf')
    composition.combine([folder/n for n in entry['sources']],output/'fig07.pdf',entry['columns'],.98)
