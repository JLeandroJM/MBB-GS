"""
Baseline de interpolacion LINEAL de pixeles para el termometro retenidos-x2.

Piso de comparacion del slow-mo: si el modelo GS2D no le gana a PROMEDIAR los dos
frames vecinos, no esta aportando nada sobre interpolacion trivial.

Protocolo (igual que holdout multiplos k=2):
    - frames PARES (0,2,4,...) = "vistos".
    - frames IMPARES (1,3,5,...) = "retenidos": se PREDICEN como
          pred[j] = 0.5 * (frame[j-1] + frame[j+1])
      y se comparan contra el frame real (que si tenemos).

Reporta PSNR/SSIM promedio sobre los frames retenidos. Corre en CPU (Mac).

Uso:
    python scripts/interp_baseline_lineal.py --clip data/clips/gota_s6s16_480x270
    python scripts/interp_baseline_lineal.py --clip data/clips/gota_s6s16_480x270 \
        --max_frames 300 --csv outputs/baseline_gota.csv
"""
import argparse
import csv
import os
import sys
from pathlib import Path

import numpy as np
from PIL import Image

try:
    import torch
    from pytorch_msssim import ssim as _ssim_externo
    _HAY_SSIM = True
except Exception:
    _HAY_SSIM = False


def _cargar_frames(carpeta, max_frames=None):
    archivos = sorted(f for f in os.listdir(carpeta)
                      if f.startswith("frame_") and f.endswith(".png"))
    if max_frames is not None:
        archivos = archivos[:max_frames]
    if not archivos:
        sys.exit(f"no hay frames PNG en {carpeta}")
    arrs = [np.asarray(Image.open(os.path.join(carpeta, a)).convert("RGB"),
                       dtype=np.float32) / 255.0 for a in archivos]
    return np.stack(arrs, axis=0)   # (T, H, W, 3) en [0,1]


def _psnr(a, b):
    mse = float(np.mean((a - b) ** 2))
    if mse <= 0:
        return float("inf")
    return float(-10.0 * np.log10(mse))


def _ssim(a, b):
    if not _HAY_SSIM:
        return None
    # a,b: (H,W,3) en [0,1] -> (1,3,H,W)
    ta = torch.from_numpy(a).permute(2, 0, 1).unsqueeze(0)
    tb = torch.from_numpy(b).permute(2, 0, 1).unsqueeze(0)
    return float(_ssim_externo(ta, tb, data_range=1.0, size_average=True, win_size=11))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--clip", required=True, help="carpeta con frame_%04d.png")
    ap.add_argument("--max_frames", type=int, default=None)
    ap.add_argument("--csv", default=None, help="ruta opcional para csv por frame")
    args = ap.parse_args()

    frames = _cargar_frames(args.clip, args.max_frames)
    T = frames.shape[0]
    print(f"clip={Path(args.clip).name}  n_frames={T}  res={frames.shape[1]}x{frames.shape[2]}",
          flush=True)
    if not _HAY_SSIM:
        print("  (pytorch_msssim no disponible: solo PSNR)", flush=True)

    filas = []
    psnrs, ssims = [], []
    for j in range(1, T - 1, 2):                     # impares interiores
        pred = 0.5 * (frames[j - 1] + frames[j + 1])
        p = _psnr(frames[j], pred)
        s = _ssim(frames[j], pred)
        psnrs.append(p)
        if s is not None:
            ssims.append(s)
        filas.append((j, p, s))

    psnr_mean = float(np.mean(psnrs)) if psnrs else float("nan")
    ssim_mean = float(np.mean(ssims)) if ssims else None

    print(f"\n=== BASELINE LINEAL (retenidos x2, {len(psnrs)} frames impares) ===")
    print(f"  PSNR_interp = {psnr_mean:.4f} dB")
    if ssim_mean is not None:
        print(f"  SSIM_interp = {ssim_mean:.4f}")
    print("  -> este es el PISO: el modelo debe SUPERAR estos numeros.", flush=True)

    if args.csv:
        Path(args.csv).parent.mkdir(parents=True, exist_ok=True)
        with open(args.csv, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["frame_idx", "psnr", "ssim"])
            for j, p, s in filas:
                w.writerow([j, f"{p:.6f}", "" if s is None else f"{s:.6f}"])
        print(f"  csv -> {args.csv}", flush=True)


if __name__ == "__main__":
    main()
