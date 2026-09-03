# MBB-GS — Moving Gaussians

**Multimedia representation with 2D Gaussian Splatting.**

A video is usually stored as a sequence of independent frames. That format is
convenient for playback, but it says nothing about *how* the visual content
changes over time. MBB-GS represents a video instead as a fixed population of
2D Gaussians whose attributes are continuous functions of time, so the video
becomes a signal that can be evaluated at any instant — including instants that
were never recorded.

Jose Leandro Machaca Soloaga · Mauro Ianfranco Bobadilla Castillo
Advisor: Eric Biagioli — Universidad de Ingeniería y Tecnología (UTEC)

---

## The idea

A single set of $N$ Gaussians is kept for the entire clip. Each attribute $p$ of
each Gaussian $i$ — position, opacity, colour, scale, rotation and depth — is a
Chebyshev polynomial in normalised time $\tau(t) \in [-1, 1]$:

$$p_i(t) = a_{i,0}^{(p)} + \sum_{k=1}^{K_p} a_{i,k}^{(p)} \, T_k(\tau(t))$$

The model stores coefficients, not frames. To render frame $t$ the polynomials
are evaluated, and the resulting Gaussians are fed to a differentiable CUDA
rasteriser:

$$t \longrightarrow \tau(t) \longrightarrow G(t) \longrightarrow \hat{I}(t)$$

Chebyshev is the main basis: it is bounded on $[-1, 1]$ and well conditioned at
high degree, which matters because position and opacity use degrees around 100
and 80. A monomial basis is kept for the ablation that motivates that choice.

Two properties follow directly from this design. Temporal interpolation is free —
evaluating the trained model between two original frames needs no second network
— and the model can be pruned and quantised after training, because each
Gaussian owns an explicit, inspectable set of coefficients.

---

## Results

Reconstruction quality on the loss ablation (600 frames, 720p, 100k Gaussians,
800 epochs). Weighting the error towards moving regions wins on every metric:

| Loss | PSNR | PSNR min | SSIM | LPIPS | Temporal PSNR |
| --- | ---: | ---: | ---: | ---: | ---: |
| Baseline (L1 + DSSIM) | 31.83 | 28.10 | 0.962 | 0.0594 | 35.17 |
| Edge | 31.88 | 28.05 | 0.963 | 0.0574 | 35.23 |
| Temporal | 31.93 | 28.32 | 0.963 | 0.0588 | 35.32 |
| **Motion** | **33.24** | **30.32** | **0.965** | **0.0483** | **36.88** |

Temporal basis, under otherwise identical settings (8k Gaussians, 360 epochs).
The monomial basis degrades badly at high degree:

| Basis | PSNR | PSNR min | SSIM | Temporal PSNR |
| --- | ---: | ---: | ---: | ---: |
| Monomial | 23.16 | 20.19 | 0.750 | 30.61 |
| **Chebyshev** | **29.95** | **27.70** | **0.926** | **34.04** |

Model reduction, starting from 150k Gaussians on 750 frames at 720p: adaptive
pruning to 120k Gaussians followed by UINT16 quantisation takes the model from
about 236 MB to 85 MB, at roughly 63 dB PSNR against the full reconstruction.

Audio, as a complementary extension: 9 s of stereo at 44.1 kHz with 160k Gabor
atoms in Mid-Side reaches 27.78 dB SNR and 27.77 dB SI-SDR.

Full tables, settings and figures are in the [wiki](wiki/Results.md).

---

## Repository layout

```
src/gs2d_video/      video model: temporal bases, Gaussian model, losses,
                     CUDA rasteriser, training loop, metrics, I/O
src/gs2d_gabor/      audio model: Gabor atoms, losses, CUDA rendering
cuda/                CUDA extensions, compiled per environment
scripts/             command line entry points, grouped by role
configs/             one JSON per experiment, grouped by paper experiment
jobs/                Slurm scripts used on the Khipu HPC cluster
tests/               test suite
wiki/                documentation source
```

`scripts/` is organised by what each tool does — `datos`, `reconstruccion`,
`compresion`, `metricas`, `visualizacion`, `pipeline`, `audio` — with
`train.py` at the top as the single entry point for training. Each folder has
its own README.

---

## Installation

Requires Python ≥ 3.10 and, for training, an NVIDIA GPU with CUDA.

```bash
git clone TODO_REPO_URL
cd MBB-GS

conda create -n mbb-gs python=3.10 -y
conda activate mbb-gs

pip install -r requirements.txt
pip install -e .
```

The editable install is what makes `import gs2d_video` work from anywhere; the
scripts rely on it.

Check that PyTorch sees the GPU:

```bash
python -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available())"
```

### CUDA extensions

Compiled binaries are not versioned, so they must be built in each environment:

```bash
cd cuda/raster_cuda    && python setup.py build_ext --inplace && cd ../..
cd cuda/gabor_audio_cuda && python setup.py build_ext --inplace && cd ../..
```

The second one is only needed for the audio extension. See
[Installation](wiki/Installation.md) for architecture flags, Windows notes and
the `RASTER_BATCH_SIZE` build option.

The exact environment used on the Khipu cluster for the reported numbers is
frozen in `requirements-tesis-khipu.txt`.

---

## Reproducing an experiment

Clips are read as PNG sequences from `data/clips/<clip>/`:

```bash
python scripts/data/extraer_clips_720p.py \
    --video mi_video.mp4 --nombre_clip mi_clip \
    --inicio_seg 10 --duracion_seg 20
```

Train, using the final video configuration:

```bash
python scripts/train.py --config configs/video/final/ganador_motion_200k_1200ep.json
```

Everything lands in `outputs/<nombre_experimento>/`: rendered frames,
checkpoints, logs, `metricas.json`, `metricas_por_frame.csv`, and
`config_usada.json`, a verbatim copy of the configuration that produced the
run.

For large models, render from the checkpoint afterwards instead of inside
training:

```bash
python scripts/reconstruction/regenerar_clip_desde_checkpoint_streaming.py \
    --checkpoint outputs/mi_experimento/checkpoints/checkpoint_final.pt \
    --salida outputs/mi_experimento/frames_renderizados \
    --device cuda
```

Temporal interpolation — the same model evaluated between the original frames:

```bash
python scripts/reconstruction/regenerar_fps_interpolado.py \
    --checkpoint outputs/mi_experimento/checkpoints/checkpoint_final.pt \
    --salida outputs/mi_experimento/frames_60fps \
    --fps_origen 30 --fps_salida 60 --device cuda
```

`configs/README.md` maps every configuration folder to the experiment it
belongs to.

---

## Tests

```bash
pytest -q
```

The suite runs on CPU and needs no trained model. It covers the Chebyshev and
monomial bases, model evaluation and metadata, checkpoint loading for both
bases, the temporal basis used when pruning, and the analytic Gabor gradients
against PyTorch autograd.

---

## Data and results

The source videos and audio are commercial recordings and are **not**
redistributed here. The configurations record which segment each experiment
used, and `scripts/data/` regenerates the clips from your own copy.

Rendered videos, checkpoints and the full per-experiment outputs are published
separately: TODO_DRIVE_URL

The lightweight per-experiment records — metrics, per-frame CSVs and the exact
configuration used — are included in this repository, so the reported numbers
can be checked without retraining anything.

---

## Documentation

The [wiki](wiki/Home.md) covers the model and temporal representation, the CUDA
rasteriser, training and the configuration reference, loss functions,
reconstruction and interpolation, metrics, pruning and quantisation, the Gabor
audio extension, the audiovisual pipeline, reproducibility and troubleshooting.

---

## Citation

If you use this code, please cite the thesis (see `CITATION.cff`).

## License

Released under the MIT License; see `LICENSE`. The CUDA rasteriser was written
for this project, taking the general approach of 3D Gaussian Splatting as a
conceptual reference; no code was copied from it.

## Acknowledgements

Computational work was supported by the Khipu HPC cluster at UTEC.
