# Experiment configurations

Every experiment is fully described by one JSON file. `scripts/train.py` reads
it and writes a verbatim copy to `config_usada.json` in the output folder, so
each result can be traced back to the exact settings that produced it.

Folders are grouped by modality and then by the experiment they belong to.

## Video

| Folder | Files | Experiment |
| --- | --- | --- |
| `video/exp1_bases/` | 2 | Monomial vs. Chebyshev temporal basis, all else held fixed |
| `video/exp2_capacidad/` | 8 | Number of Gaussians and polynomial degrees |
| `video/exp3_perdida/` | 8 | Loss ablation: baseline, `l1_mse`, `edge`, `temporal`, `motion`, `combo` |
| `video/exp3_agregacion_frames/` | 6 | Exponent `q` used to combine per-frame errors, plus the pure `max` variant |
| `video/exp3_agregacion_pixel/` | 3 | Exponent applied at pixel level (`exponente_pixel`) |
| `video/escalamiento_gaussianas/` | 3 | 25k / 50k / 75k Gaussians at a fixed 800 epochs |
| `video/escalamiento_epocas/` | 4 | 400 / 800 / 1200 / 1600 epochs at a fixed 150k Gaussians |
| `video/final/` | 1 | Final video configuration: `motion` loss, 200k Gaussians, 1200 epochs |
| `video/pruebas/` | 1 | One-off run kept for traceability; not part of any reported result |

The two scaling folders sweep different axes and use different clips, so they
are not a single series: `escalamiento_gaussianas` varies capacity on one clip,
`escalamiento_epocas` varies training length on another.

`video/final/ganador_motion_200k_1200ep.json` is the configuration behind the
reconstruction, temporal interpolation and model reduction results.

## Audio

| Folder | Files | Experiment |
| --- | --- | --- |
| `audio/gabor/` | 24 | Gabor atom model and its ablations: loss, initialisation, number of atoms, mono vs. stereo, L/R vs. Mid-Side |
| `audio/audio_only/` | 3 | Long stereo Mid-Side runs used for the final audio results |

## Audiovisual

| Folder | Files | Experiment |
| --- | --- | --- |
| `audiovisual/thriller_10s_1ep/` | 3 | Full pipeline on a 10 s segment: `video.json`, `audio.json`, `pipeline.json` |
| `audiovisual/rockyourbody_10s_1ep/` | 3 | Same pipeline on a second segment |

Each audiovisual experiment needs its three files: `pipeline.json` is the
master configuration and points to the video and audio ones.

## Running an experiment

```bash
python scripts/train.py --config configs/video/final/ganador_motion_200k_1200ep.json
```

`scripts/pipeline/run_tests_secuencial.py` runs the loss ablation phases in
order, and `jobs/*.sbatch` contains the exact Slurm scripts used on the Khipu
HPC cluster.

## Input data

Configs reference clips by name under `data/clips/<clip>/`. The source video
and audio files are not distributed with this repository; see the main README
for how to obtain them and regenerate the clips.
