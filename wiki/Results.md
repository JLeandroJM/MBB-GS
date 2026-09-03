# Results

The experiments below are the ones reported in the thesis. They contextualise
the design decisions in the rest of the wiki; they are not a quality guarantee
for arbitrary video, since results depend on content, duration, resolution,
model capacity and training length.

Each experiment's configuration is in `configs/`, mapped folder by folder in
`configs/README.md`.

## Temporal basis: Chebyshev vs. monomial

8000 Gaussians, 360 epochs, same degrees, same clip, same loss. The only
difference is the basis.

| Basis | PSNR | PSNR min | SSIM | Temporal PSNR | PSNR std |
| --- | ---: | ---: | ---: | ---: | ---: |
| Monomial | 23.16 | 20.19 | 0.750 | 30.61 | 2.61 |
| **Chebyshev** | **29.95** | **27.70** | **0.926** | **34.04** | **1.09** |

Nearly 7 dB, and less than half the variance across frames. This is the
experiment that justifies making Chebyshev the main path: with degrees around
100, a monomial basis is too ill-conditioned to fit.

Configs: `configs/video/exp1_bases/`

## Loss ablation

600 frames, 720p, 100k Gaussians, 800 epochs. Everything fixed except the loss.

| Loss | PSNR | PSNR min | PSNR p5 | SSIM | LPIPS | Temporal PSNR |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Baseline | 31.83 | 28.10 | 29.35 | 0.962 | 0.0594 | 35.17 |
| Edge | 31.88 | 28.05 | 29.30 | 0.963 | 0.0574 | 35.23 |
| Temporal | 31.93 | 28.32 | 29.42 | 0.963 | 0.0588 | 35.32 |
| **Motion** | **33.24** | **30.32** | **31.12** | **0.965** | **0.0483** | **36.88** |

`motion` wins on every metric, and the gap on PSNR min and p5 is larger than on
the mean — weighting moving regions lifts the *worst* frames most, which is what
you would expect if the failures were concentrated in dynamic content.

The `l1_mse` and `combo` variants were part of the study but their runs did not
finish because of failures in the execution environment, so they are not
reported.

Configs: `configs/video/exp3_perdida/`

## Aggregation across frames

Same setup, varying the exponent $q$ that combines per-frame errors.

| q | PSNR | PSNR min | SSIM | LPIPS | Temporal PSNR |
| ---: | ---: | ---: | ---: | ---: | ---: |
| **1** | **33.24** | **30.32** | **0.965** | **0.048** | **36.90** |
| 2 | 29.77 | 27.43 | 0.942 | 0.099 | 33.55 |
| 4 | 21.19 | 16.84 | 0.859 | 0.336 | 30.33 |
| 8 | 14.70 | 13.37 | 0.517 | 0.821 | 30.03 |

Quality falls monotonically. Concentrating updates on the hardest frames
unbalances the gradients reaching the high-order temporal coefficients, which
need evidence from many frames. The uniform average is the right choice, and
`exponente_frame` should be left at 1.

Configs: `configs/video/exp3_agregacion_frames/`

## Temporal interpolation

90k Gaussians, 180 original frames, 720p, 2200 epochs, Chebyshev, `motion` loss.

| Metric | Value |
| --- | ---: |
| PSNR | 40.97 dB |
| PSNR min | 36.83 dB |
| SSIM | 0.985 |
| LPIPS | 0.028 |
| Temporal PSNR | 43.78 dB |

The same model was then evaluated at intermediate instants to produce a higher
frame rate, with the caveats in
[Reconstruction and Interpolation](Reconstruction-and-Interpolation).

## Model reduction

150k initial Gaussians, 750 frames, 720p, 1600 epochs, degrees
100/80/30/12/6/4, `motion` loss.

Full reconstruction:

| Metric | Value |
| --- | ---: |
| PSNR | 35.94 dB |
| PSNR min | 27.40 dB |
| PSNR p5 | 30.71 dB |
| SSIM | 0.970 |
| LPIPS | 0.048 |
| Temporal PSNR | 39.59 dB |

Then adaptive pruning at 20 % and UINT16 quantization:

| | Gaussians | Size | PSNR vs. full reconstruction |
| --- | ---: | ---: | ---: |
| Full | 150 000 | ~236 MB | — |
| Pruned 20 % + uint16 | 120 000 | ~85 MB | ~63 dB |

A 2.8× size reduction at a difference that is effectively invisible.

Configs: `configs/video/escalamiento_epocas/`, `configs/video/final/`

## Gabor audio

9 s of stereo at 44.1 kHz, 160k atoms (96k Mid, 64k Side), 6000 epochs.

| Metric | Value |
| --- | ---: |
| Stereo SNR | 27.78 dB |
| Stereo SI-SDR | 27.77 dB |
| Stereo PSNR | 44.24 dB |
| Mean LSD | 4.57 dB |
| Mean Mel-L1 | 0.1985 |
| Mean MR-STFT | 1.8532 |
| Mid SNR | 28.50 dB |
| Side SNR | 20.01 dB |

Gabor atoms are markedly better suited than plain Gaussians for signals with
oscillation, which was the point of switching primitives.

Configs: `configs/audio/gabor/`, `configs/audio/audio_only/`

## Full outputs

Rendered videos, checkpoints and complete per-experiment outputs are published
separately: TODO_DRIVE_URL

The lightweight records — `metricas.json`, `metricas_por_frame.csv`,
`config_usada.json`, `info_clip.json` — are included in the repository, so the
numbers above can be checked without retraining.
