#!/usr/bin/env python3
"""Overlay the original paper's actual track observations on the existing gallery."""
import json
import shutil
import zipfile

from frame_gallery import OUT, FINAL, PREVIEW, PUBLIC, REPO, ROOT, sha, contact_sheet, page, original
import numpy as np
import pandas as pd
from PIL import Image
import matplotlib.pyplot as plt
from ra3d._core.track_images import point_at


def render_overlay(panel, vanished, survivor):
    frame,x0,y0=panel['frame'],panel['x0'],panel['y0']
    # Match the original 76.32-point image-panel width for marker/line proportions.
    fig=plt.figure(figsize=(1.06,1.06),dpi=256/1.06,facecolor=(0,0,0,0))
    ax=fig.add_axes([0,0,1,1],facecolor=(0,0,0,0))
    ax.set_xlim(-.5,127.5);ax.set_ylim(127.5,-.5);ax.set_axis_off()
    evidence=[]
    for role,track,color,marker in [
        ('terminating',vanished,original.TERMINATING_COLOR,'o'),
        ('continuing',survivor,original.CONTINUING_COLOR,'s'),
    ]:
        point=point_at(track,frame,nearest=False)
        if point is None:
            evidence.append(dict(role=role,present=False))
            continue
        assert -.5<=point.x-x0<=127.5 and -.5<=point.y-y0<=127.5
        original.overlay_track_tail(ax,track,frame,x0,y0,color)
        original.overlay_track_marker(ax,point,x0,y0,color=color,marker=marker)
        marker_artist=ax.collections[-1]
        assert np.allclose(marker_artist.get_offsets(),[[point.x-x0,point.y-y0]])
        tail=track.loc[track.frame.between(frame-4,frame)].sort_values('frame')
        assert np.allclose(ax.lines[-1].get_xydata(),tail[['x','y']].to_numpy()-[x0,y0])
        evidence.append(dict(role=role,present=True,track_id=int(track.track_id.iloc[0]),
            marker=marker,color=color,crop_xy_px=[point.x-x0,point.y-y0],
            tail_frames=tail.frame.astype(int).tolist(),tail_crop_xy_px=(tail[['x','y']].to_numpy()-[x0,y0]).tolist()))
    fig.canvas.draw()
    overlay=np.asarray(fig.canvas.buffer_rgba()).copy()
    plt.close(fig)
    assert overlay.shape==(256,256,4)
    with Image.open(OUT/'images'/panel['filename']) as source:
        assert source.size==(128,128)
        base=source.resize((256,256),Image.Resampling.NEAREST).convert('RGBA')
    result=Image.alpha_composite(base,Image.fromarray(overlay)).convert('RGB')
    # Unmarked pixels are exactly the existing image, including original contrast.
    unmarked=overlay[:,:,3]==0
    assert np.array_equal(np.asarray(result)[unmarked],np.asarray(base)[:,:,:3][unmarked])
    target=OUT/'tracks'/panel['filename'];result.save(target)
    return dict(label=panel['label'],frame=frame,source_png_sha256=panel['png_sha256'],
        overlay_png_sha256=sha(target),observations=evidence,
        unmarked_pixels_unchanged=True,overlay_size_px=256)


def main():
    (OUT/'tracks').mkdir(exist_ok=True)
    raw_manifest=json.loads((OUT/'manifest.json').read_text())
    selection=json.loads((FINAL/'selection.json').read_text())
    assert sha(FINAL/'selection.json')==raw_manifest['source_selection_sha256']
    protected=raw_manifest['protected_outputs_unchanged']
    assert protected=={path:sha(path) for path in protected}
    tracks=pd.read_parquet(ROOT/'runs/tracking/seed260745/smoothed.parquet')
    evidence=[]
    for case,chosen in zip(raw_manifest['cases'],selection['cases']):
        assert case['review_id']==chosen['review_id']
        vanished=tracks.loc[tracks.track_id.eq(chosen['vanished_track_id'])].sort_values('frame')
        survivor=tracks.loc[tracks.track_id.eq(chosen['survivor_track_id'])].sort_values('frame')
        for panel in case['panels']:
            assert sha(OUT/'images'/panel['filename'])==panel['png_sha256']
            evidence.append(render_overlay(panel,vanished,survivor))
        letter=case['letter']
        contact_sheet(f"({letter}) "+('True collision' if letter=='a' else 'Shared-particle artifact'),
            case['panels'],OUT/f'{letter}-tracks-all-frames.png',with_tracks=True)
    assert protected=={path:sha(path) for path in protected}
    result=dict(status='PASS',panels=evidence,source_manifest_sha256=sha(OUT/'manifest.json'),
        track_sha256=sha(ROOT/'runs/tracking/seed260745/smoothed.parquet'),
        original_plot_sha256=sha(REPO/'experiments/amt_figure9_refinement_20261102/source/original_plot.py'),
        raw_gallery_images_unchanged=True,current_paper_outputs_unchanged=True,
        marker_count=sum(o['present'] for p in evidence for o in p['observations']))
    (OUT/'tracks-manifest.json').write_text(json.dumps(result,indent=2)+'\n')
    with zipfile.ZipFile(OUT/'images-with-tracks.zip','w',compression=zipfile.ZIP_DEFLATED) as z:
        for p in sorted((OUT/'tracks').glob('*.png')):z.write(p,'tracks/'+p.name)
        for letter in ['a','b']:z.write(OUT/f'{letter}-tracks-all-frames.png',f'{letter}-tracks-all-frames.png')
    (OUT/'fig09-frames.html').write_text(page(raw_manifest['cases'],with_tracks=True))
    public_paths=[]
    for p in sorted((OUT/'tracks').glob('*.png')):
        target=PUBLIC/'fig09-frames/tracks'/p.name
        target.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(p,target);public_paths.append(target)
    for name in ['a-tracks-all-frames.png','b-tracks-all-frames.png','images-with-tracks.zip']:
        target=PUBLIC/'fig09-frames'/name;shutil.copyfile(OUT/name,target);public_paths.append(target)
    target=PUBLIC/'fig09-frames.html';shutil.copyfile(OUT/'fig09-frames.html',target);public_paths.append(target)
    receipt=json.loads((PREVIEW/'public-manifest.json').read_text())
    receipt['files'].update({str(p.relative_to(PUBLIC)):sha(p) for p in public_paths})
    receipt['total_bytes']=sum((PUBLIC/name).stat().st_size for name in receipt['files'])
    temp=PREVIEW/'public-manifest.frames-tracks.tmp'
    temp.write_text(json.dumps(receipt,indent=2)+'\n');temp.replace(PREVIEW/'public-manifest.json')
    print(json.dumps({'status':'PASS','panels':len(evidence),'actual_markers':result['marker_count'],
        'published_files':len(public_paths),'paper_figure_unchanged':True},indent=2))


if __name__=='__main__':main()
