"""
SLOW-MOTION / frame interpolation evaluando el polinomio en t NO entero.

Como los parametros de cada gaussiana son polinomios CONTINUOS en el tiempo,
podemos pedir frames "entre" los frames originales: para un factor f, generamos
instantes j = 0, 1/f, 2/f, ..., n-1, cada uno con su t normalizado
    t(j) = 2 * j / (n_frames - 1) - 1
y rasterizamos. El resultado es un video a f x fps que nunca existio.

Necesita GPU (rasterizador CUDA) -> correr en Khipu.

Uso (en Khipu):
    python scripts/render_slowmo.py \
        --checkpoint outputs/dvd_20s_holdout_mult2_l1dssim/checkpoints/checkpoint_final.pt \
        --salida outputs/dvd_20s_holdout_mult2_l1dssim/slowmo --factor 2 --crear_video --fps 60
"""
import argparse
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image

RAIZ = Path(__file__).resolve().parents[1]
SRC = RAIZ / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from gs2d_video.core.bases import construir_matriz_chebyshev_en_t
from gs2d_video.render.renderer import render_frame
from _carga_checkpoint import cargar_modelo_desde_checkpoint


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--salida", required=True)
    ap.add_argument("--factor", type=int, default=2, help="factor de slow-motion (2 = doble de frames)")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--crear_video", action="store_true")
    ap.add_argument("--fps", type=int, default=60)
    ap.add_argument("--limpiar_cache_cada", type=int, default=25)
    args = ap.parse_args()

    device = torch.device(args.device)
    salida = Path(args.salida).resolve()
    salida.mkdir(parents=True, exist_ok=True)

    modelo, config, _matrices_enteras, info = cargar_modelo_desde_checkpoint(args.checkpoint, device=device)
    n_frames, H, W = info["n_frames"], info["H"], info["W"]
    grados = info["grados"]

    # instantes intermedios: j = 0, 1/f, 2/f, ..., n-1
    f = int(args.factor)
    n_out = (n_frames - 1) * f + 1
    js = np.arange(n_out, dtype=np.float64) / f                    # 0, 0.5, 1, ...
    ts = 2.0 * js / (n_frames - 1) - 1.0                           # normalizado a [-1,1]
    ts_t = torch.from_numpy(ts)

    grados_distintos = sorted(set(grados.values()))
    matrices_slow = {
        g: construir_matriz_chebyshev_en_t(ts_t, g, device=device, dtype=torch.float32)
        for g in grados_distintos
    }

    print(f"=== render_slowmo ===", flush=True)
    print(f"  checkpoint : {args.checkpoint}", flush=True)
    print(f"  n_frames   : {n_frames}  ->  n_out : {n_out}  (factor {f})", flush=True)
    print(f"  resolucion : {H}x{W}   device: {device}", flush=True)

    for i in range(n_out):
        params_i = modelo.evaluar_en_frame(i, matrices_slow)
        r = render_frame(params_i, H, W, config).clamp(0, 1)
        r = torch.nan_to_num(r, nan=0.0, posinf=1.0, neginf=0.0)
        img = (r.detach().cpu().numpy() * 255).astype(np.uint8)
        Image.fromarray(img).save(salida / f"frame_{i:05d}.png")
        del params_i, r
        if device.type == "cuda" and args.limpiar_cache_cada > 0 and (i + 1) % args.limpiar_cache_cada == 0:
            torch.cuda.empty_cache()
        if i == 0 or (i + 1) % 50 == 0 or i == n_out - 1:
            print(f"  slowmo {i + 1}/{n_out}", flush=True)

    print(f"=== frames en {salida} ===", flush=True)

    if args.crear_video:
        ruta_video = salida.parent / f"slowmo_x{f}.mp4"
        cmd = [sys.executable, str(RAIZ / "scripts" / "frames_a_video.py"),
               "--frames", str(salida), "--salida", str(ruta_video), "--fps", str(int(args.fps))]
        print(" ".join(cmd), flush=True)
        subprocess.run(cmd, cwd=str(RAIZ), check=True)
        print(f"video -> {ruta_video}", flush=True)


if __name__ == "__main__":
    main()
