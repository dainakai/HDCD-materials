# Acquisition, reconstruction and coordinate settings

[Repository home](../README.md) · [Data dictionary](data.md)

The following settings identify the supplied synthetic benchmark and the separate numerical inputs used for Figure 7. Use the acquisition configuration for the processing stage being run.

## Synthetic data and collision benchmark

| Setting | Value |
| --- | --- |
| Training seeds | 260740–260744 |
| Held-out test seed | 260745 |
| Frames / true events per seed | 6,000 / 2,000 |
| Frame rate | 4,000 Hz |
| Hologram size | 1,024 × 1,024 pixels |
| Pixel pitch | 10 µm |
| Wavelength | 0.6328 µm |
| Camera separation | 21,450 µm |
| Generated volume front, measured from the primary camera | 74,900 µm |
| Generated volume dimensions | 10.24 × 10.24 × 102.4 mm |
| Reconstruction first / last planes | 64,900 / 167,200 µm |
| Axial step / plane count | 100 µm / 1,024 |
| Particle measurement: phase iterations / FFT padding | 3 / 1,536 pixels |
| Collision image features: phase iterations / FFT padding | 12 / 2,048 pixels |
| MinIP slice step | 2 |
| Detector confidence / NMS IoU | 0.10 / 0.15 |
| Reference measurement depth and diameter backend | TensorRT, FP16 |

The benchmark settings apply to all five training sequences and the held-out test. The generator writes `holod3-acquisition.yaml` for particle measurement and `acquisition.yaml` for collision image features. Both share the reconstruction grid and use the stage-specific iteration and padding values above. These settings describe the recorded inputs; upstream measurement weights are outside this distribution.

The test particle simulation uses the original fixed generation recipe. The renderer's propagation distances belong to the image-generation geometry. [Generation details](simulation.md) include the separate camera distances and the deterministic-input contract.

## Comparison image processing

| Setting | Depth-free variant | MinIP-based 2D |
| --- | --- | --- |
| Measurement/trajectory input | Ours measurements and trajectories | Independent MinIP measurements and trajectories |
| Image-feature input | MinIP | MinIP |
| MinIP first / last sampled plane | 74,900 / 177,100 µm | 64,900 / 167,100 µm |
| MinIP sampled spacing / planes | 200 µm / 512 | 200 µm / 512 |
| Phase iterations / FFT padding | 3 / 1,536 pixels | 3 / 1,536 pixels |
| Particle diameter | Ours focused-slice estimate | MinIP contrast-area equivalent-circle diameter |
| Linker depth weight / tie-break penalty | Ours linking | 0 / 0 |
| XY smoothing | Ours smoothed trajectories | Savitzky–Golay, window 7, order 2, legacy mode |

These settings apply to all five training sequences and the independent test sequence. [Method comparison](baseline2d.md#method-comparison) describes the retained evaluation inputs.

## Coordinates

For one-based plane index `s`, the camera-to-focus distance is:

```text
focus_distance_um = reconstruction_start_um + (s - 1) * 100
```

The stored collision-candidate coordinate is `z_um = s * 100`. The fitted residual regressors produce the final event time and position. [Units and identifiers](data.md#units-and-identifiers) describe every conversion.

## Figure-specific inputs

| Figure panels | Included input | Processing conditions |
| --- | --- | --- |
| 7(a) | `data/paper_figures/fig07/low_confidence_detections.parquet` | Confidence sweep from 0.001; compare to complete in-volume truth |
| 7(b–d) | `data/paper_figures/fig07/paired_tracks_20mm.parquet` | Reconstruction first plane 74,900 µm; 1,024 planes at 100 µm; 3 phase iterations, 1,536-pixel padding; fixed linked-track membership and paired raw/smoothed measurements |
| 7(e,f) | Density and trajectory-summary files in `data/paper_figures/fig07/` | The supplied benchmark's measured density and trajectory-length summaries |
| 8 and Tables 3–4 | Test candidates, features and predictions under `data/` | The synthetic collision benchmark settings above |

The Figure 7 error calculation uses the paired file directly, matches particles by frame and raw XY, and calculates errors from the paired smoothed measurements. The [figure guide](figures.md#4-what-is-recalculated-for-each-figure) gives filters, units and expected sample counts.

## Experimental and supplementary inputs

