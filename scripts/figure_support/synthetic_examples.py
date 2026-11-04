from __future__ import annotations
import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any
import matplotlib
from matplotlib import font_manager
import matplotlib.patheffects as path_effects
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd
DATASET_NAME = 'hdcd_102p3mm_phase3_seed260745'
SCENE = 'C000H001S0011'
SCENE_KEY = f'single:{SCENE}'
FPS = 4000.0
PIXEL_SIZE_UM = 10.0
ARIAL_COMPATIBLE_FONT = Path(__file__).resolve().parents[2] / 'reference/figure_fonts/LiberationSans-Regular.ttf'
TERMINATING_COLOR = '#D84A3A'
CONTINUING_COLOR = '#2C6FB7'
TRAJECTORY_COLOR = '#6B747C'
IMAGE_SCORE_COLOR = '#009E73'
INK = '#263238'
MUTED_INK = '#536168'

@dataclass(frozen=True)
class CaseSpec:
    key: str
    title: str
    vanished_track_id: int
    survivor_track_id: int
    candidate_frame: int
    display_frames: tuple[int, int, int]
    time_reference_frame: int
    truth_event_id: int | None = None
    expected_shared_source_id: int | None = None

    @property
    def candidate_key(self) -> str:
        return f'{self.vanished_track_id}_{self.survivor_track_id}_{self.candidate_frame}'
TRUE_CASE = CaseSpec('true_collision','Ground-truth collision',12754,12764,3958,(3957,3958,3959),3958,1204)
ARTIFACT_CASE = CaseSpec('shared_particle_artifact','Shared-particle artifact',17945,18031,5337,(5335,5337,5339),5337,None,7364)
CASES=(TRUE_CASE,ARTIFACT_CASE)

def configure_plotting() -> None:
    if not ARIAL_COMPATIBLE_FONT.exists():
        raise FileNotFoundError(f'Arial-compatible font not found: {ARIAL_COMPATIBLE_FONT}')
    font_manager.fontManager.addfont(str(ARIAL_COMPATIBLE_FONT))
    plt.rcParams.update({'font.family': 'Liberation Sans', 'font.sans-serif': ['Liberation Sans'], 'font.size': 8.5, 'axes.labelsize': 8.5, 'axes.titlesize': 8.8, 'legend.fontsize': 7.7, 'xtick.labelsize': 7.5, 'ytick.labelsize': 7.5, 'axes.spines.top': False, 'axes.spines.right': False, 'axes.unicode_minus': True, 'pdf.use14corefonts': False, 'pdf.fonttype': 42, 'ps.fonttype': 42})

def normalize_valid_crop(crop: np.ndarray, valid: np.ndarray) -> np.ndarray:
    """Normalize recorded pixels only, leaving padded pixels exactly black."""
    values = crop[valid]
    if values.size == 0:
        return np.zeros_like(crop, dtype=np.uint8)
    lo, hi = np.percentile(values, [1.0, 99.0])
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        lo, hi = (float(np.min(values)), float(np.max(values)))
    normalized = np.zeros_like(crop, dtype=np.uint8)
    if hi > lo:
        scaled = np.rint(np.clip((values - lo) / (hi - lo), 0.0, 1.0) * 255.0).astype(np.uint8)
        normalized[valid] = scaled
    return normalized

def overlay_track_tail(ax: plt.Axes, track: pd.DataFrame, frame: int, x0: float, y0: float, color: str) -> None:
    tail = track[track['frame'].between(frame - 4, frame)].sort_values('frame')
    if tail.empty:
        return
    ax.plot(tail['x'].to_numpy(float) - x0, tail['y'].to_numpy(float) - y0, color=color, linewidth=1.3, alpha=1.0, solid_capstyle='round', zorder=5)

def overlay_track_marker(ax: plt.Axes, point: Any | None, x0: float, y0: float, *, color: str, marker: str) -> None:
    if point is None:
        return
    x = float(point.x) - x0
    y = float(point.y) - y0
    ax.scatter([x], [y], s=27, marker=marker, facecolors=color, edgecolors='none', linewidths=0.0, zorder=7)

def add_scale_bar(ax: plt.Axes, view_size: int) -> None:
    scale_bar_px = 500.0 / PIXEL_SIZE_UM
    x0 = 8.0
    y = float(view_size - 10)
    bar = ax.plot([x0, x0 + scale_bar_px], [y, y], color='white', linewidth=2.7, solid_capstyle='butt', zorder=8)[0]
    bar.set_path_effects([path_effects.Stroke(linewidth=4.2, foreground=INK), path_effects.Normal()])
    label = ax.text(x0 + scale_bar_px / 2.0, y - 4.5, '500 µm', ha='center', va='bottom', color='white', fontsize=6.3, zorder=9)
    label.set_path_effects([path_effects.Stroke(linewidth=2.1, foreground=INK), path_effects.Normal()])

def time_title(case: CaseSpec, elapsed_ms: float) -> str:
    elapsed = f'{elapsed_ms:+.2f}'.replace('-', '−')
    if elapsed_ms < 0.0:
        return f'pre  {elapsed} ms'
    if elapsed_ms > 0.0:
        return f'post  {elapsed} ms'
    reference = 'collision' if case.key == TRUE_CASE.key else 'candidate'
    return f'{reference}  0.00 ms'

def add_image_panel(ax: plt.Axes, panel: dict[str, Any], *, case: CaseSpec, vanished: pd.DataFrame, survivor: pd.DataFrame, view_size: int, show_scale_bar: bool) -> None:
    ax.imshow(panel['image'], cmap='gray', vmin=0, vmax=255, interpolation='nearest')
    if panel['vanished_point'] is not None:
        overlay_track_tail(ax, vanished, int(panel['frame']), float(panel['x0']), float(panel['y0']), TERMINATING_COLOR)
    if panel['survivor_point'] is not None:
        overlay_track_tail(ax, survivor, int(panel['frame']), float(panel['x0']), float(panel['y0']), CONTINUING_COLOR)
    overlay_track_marker(ax, panel['vanished_point'], float(panel['x0']), float(panel['y0']), color=TERMINATING_COLOR, marker='o')
    overlay_track_marker(ax, panel['survivor_point'], float(panel['x0']), float(panel['y0']), color=CONTINUING_COLOR, marker='s')
    if show_scale_bar:
        add_scale_bar(ax, view_size)
    ax.set_xlim(-0.5, view_size - 0.5)
    ax.set_ylim(view_size - 0.5, -0.5)
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(0.6)
        spine.set_color(INK)
    ax.set_title(time_title(case, float(panel['elapsed_ms'])), pad=2.5, color=INK, fontsize=8.0, fontweight='normal')

def add_score_panel(ax: plt.Axes, candidate_rows: dict[str, pd.Series]) -> None:
    trajectory = np.array([float(candidate_rows[TRUE_CASE.key]['static_score']), float(candidate_rows[ARTIFACT_CASE.key]['static_score'])])
    image_score = np.array([float(candidate_rows[TRUE_CASE.key]['full_score']), float(candidate_rows[ARTIFACT_CASE.key]['full_score'])])
    group_y = np.array([1.0, 0.0])
    offset = 0.105
    height = 0.14
    trajectory_y = group_y + offset
    image_y = group_y - offset
    trajectory_bars = ax.barh(trajectory_y, trajectory, height=height, color=TRAJECTORY_COLOR, edgecolor='none', label='trajectory score', zorder=3)
    image_bars = ax.barh(image_y, image_score, height=height, color=IMAGE_SCORE_COLOR, edgecolor='none', label='+ image features', zorder=3)
    for bars, values, label_color in ((trajectory_bars, trajectory, TRAJECTORY_COLOR), (image_bars, image_score, IMAGE_SCORE_COLOR)):
        for bar, value in zip(bars, values, strict=True):
            ax.text(float(value) + 0.014, bar.get_y() + bar.get_height() / 2.0, f'{value:.3f}', ha='left', va='center', color=label_color, fontsize=7.4, zorder=4)
    for y, label in ((1.0, 'True collision  (retained)'), (0.0, 'Shared-particle artifact  (rejected)')):
        ax.text(0.045, y + 0.3, label, ha='left', va='center', color=INK, fontsize=7.8)
    ax.set_xlim(0.0, 1.1)
    ax.set_ylim(-0.48, 1.48)
    ax.set_yticks([])
    ax.set_xticks([0.0, 0.25, 0.5, 0.75, 1.0])
    ax.set_xlabel('Candidate score')
    ax.grid(False)
    ax.axvline(0.0, color='#111111', linewidth=0.85, zorder=2, clip_on=False)
    ax.tick_params(axis='x', colors=MUTED_INK, length=2.5, width=0.55)
    ax.spines['left'].set_visible(False)
    ax.spines['bottom'].set_color(INK)
    ax.spines['bottom'].set_linewidth(0.65)
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, 1.13), ncol=2, frameon=False, handlelength=1.25, handletextpad=0.45, borderaxespad=0.0, columnspacing=0.85, fontsize=7.2)

def build_figure(reconstructed: dict[str, tuple[list[dict[str, Any]], pd.DataFrame, pd.DataFrame]], candidate_rows: dict[str, pd.Series], *, view_size: int) -> plt.Figure:
    fig = plt.figure(figsize=(7.25, 4.3), facecolor='white')
    grid = fig.add_gridspec(2, 4, width_ratios=(1.0, 1.0, 1.0, 1.62), left=0.025, right=0.985, bottom=0.205, top=0.885, wspace=0.1, hspace=0.78)
    for row_index, case in enumerate(CASES):
        panels, vanished, survivor = reconstructed[case.key]
        axes = [fig.add_subplot(grid[row_index, column]) for column in range(3)]
        for column, (ax, panel) in enumerate(zip(axes, panels, strict=True)):
            add_image_panel(ax, panel, case=case, vanished=vanished, survivor=survivor, view_size=view_size, show_scale_bar=row_index == 0 and column == 0)
        subcaption = '(a) Ground-truth collision: 212 + 250 → 293 µm' if row_index == 0 else '(b) Shared-particle artifact: two track IDs share one true particle'
        axes[1].text(0.5, -0.23, subcaption, transform=axes[1].transAxes, ha='center', va='top', color=INK, fontsize=8.0)
    score_ax = fig.add_subplot(grid[:, 3])
    add_score_panel(score_ax, candidate_rows)
    score_ax.text(0.5, -0.145, '(c) Scores with and without image features', transform=score_ax.transAxes, ha='center', va='top', color=INK, fontsize=8.0)
    legend_handles = [Line2D([0], [0], color=TERMINATING_COLOR, linewidth=1.4, marker='o', markersize=4.6, markerfacecolor=TERMINATING_COLOR, markeredgecolor='none', markeredgewidth=0.0, label='terminating track'), Line2D([0], [0], color=CONTINUING_COLOR, linewidth=1.4, marker='s', markersize=4.5, markerfacecolor=CONTINUING_COLOR, markeredgecolor='none', markeredgewidth=0.0, label='continuing track')]
    fig.legend(handles=legend_handles, loc='upper center', bbox_to_anchor=(0.305, 0.995), ncol=2, frameon=False, handlelength=1.7, columnspacing=1.25)
    return fig


def main(output):
    from ra3d._core.track_images import point_at
    from figure_support import composition
    source=Path(__file__).resolve().parents[2]/'data/synthetic_examples'
    tracks=pd.read_parquet(source/'tracks.parquet')
    cases=json.loads((source/'cases.json').read_text())
    reconstructed={};scores={}
    for spec,case in zip(CASES,cases,strict=True):
        v=tracks.loc[tracks.track_id.eq(spec.vanished_track_id)].sort_values('frame')
        s=tracks.loc[tracks.track_id.eq(spec.survivor_track_id)].sort_values('frame')
        panels=[]
        for frame,name in zip(case['display_frames'],case['images']):
            with np.load(source/name) as data:
                panels.append(dict(frame=frame,elapsed_ms=(frame-case['candidate_frame'])/4,
                    image=normalize_valid_crop(data['raw'],data['valid']),x0=float(data['x0']),y0=float(data['y0']),
                    vanished_point=point_at(v,frame,nearest=False),survivor_point=point_at(s,frame,nearest=False)))
        reconstructed[spec.key]=(panels,v,s)
        scores[spec.key]=pd.Series(dict(static_score=case['trajectory_score'],full_score=case['image_feature_score']))
    plt.rcdefaults();configure_plotting()
    fig=build_figure(reconstructed,scores,view_size=128)
    for ax in fig.axes[:6]:ax.images[0].set_interpolation('none')
    subtitle='Ground-truth collision: 154 + 242 → 261 µm'
    for ax in fig.axes:
        for text in ax.texts:
            if text.get_text().startswith('(a) Ground-truth collision:'):text.set_text('(a) '+subtitle)
    intermediate=output/'figure9-source.pdf'
    fig.savefig(intermediate,bbox_inches='tight',pad_inches=.025,facecolor='white');plt.close(fig)
    entry=next(e for e in json.loads((source.parents[1]/'reference/figure-layouts.json').read_text()) if e['output']=='fig09.pdf')
    composition.register_label_font()
    composition.relabel(str(intermediate.resolve()),output/'figures/fig09.pdf',entry['panel_centres'],1.,entry['subtitles'])
