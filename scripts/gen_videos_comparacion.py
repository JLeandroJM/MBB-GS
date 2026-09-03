"""
Genera los videos de COMPARACION para el slow-mo, en CPU (Mac, env lpips).
Sirven para mirar al lado del slowmo_x3.mp4 del modelo cuando vuelva de Khipu.

Por cada clip genera 3 mp4 en comparacion_slowmo/<clip>/:

  1) lineal_x3.mp4    -> interpolacion LINEAL de pixeles x3 (el baseline a vencer).
     Misma grilla de instantes que render_slowmo (factor 3): entre el frame j y
     j+1 mete 2 frames promediados (alpha=1/3, 2/3).

  2) sininterp_x3.mp4 -> MISMA duracion que el slowmo x3, pero SIN interpolar:
     cada frame real se repite 3 veces (slow-mo "ingenuo", se ve a saltos).
     Es el "antes": muestra por que hace falta interpolar.

  3) rapido_60fps.mp4 -> los frames REALES a 60 fps: dura la MITAD, se ve rapido.
     Solo para ver de un vistazo que contiene cada clip.

Las tres comparten la grilla/fps con el modelo:
  - lineal_x3 y sininterp_x3 van a fps_slow (=30 por defecto, = el de render_slowmo).
  - asi el del modelo, el lineal y el sin-interpolar duran lo mismo y se comparan.

Uso:
    python scripts/gen_videos_comparacion.py            # los 4 clips por defecto
    python scripts/gen_videos_comparacion.py --clips gota_s6s16_480x270
"""
import argparse
import itertools
from pathlib import Path

import numpy as np
from PIL import Image
import imageio.v2 as imageio

RAIZ = Path(__file__).resolve().parents[1]
CLIPS_DIR = RAIZ / "data" / "clips"
DEST = RAIZ / "comparacion_slowmo"

CLIPS_DEFAULT = [
    "gota_s6s16_480x270",
    "rampa_alta_s4s9_480x270",
    "rampa_baja_s9s15_480x270",
    "estructura_s6s14_480x270",
]


def _cargar(clip_dir, max_frames=None):
    """Carga como uint8 (memoria segura a 1080p)."""
    archivos = sorted(Path(clip_dir).glob("frame_*.png"))
    if not archivos:
        raise FileNotFoundError(f"sin frames en {clip_dir}")
    if max_frames is not None:
        archivos = archivos[:max_frames]
    arrs = [np.asarray(Image.open(a).convert("RGB"), dtype=np.uint8) for a in archivos]
    return np.stack(arrs, 0)            # (N,H,W,3) uint8


def _escribir_stream(gen, ruta, fps):
    """Escribe frame por frame desde un generador (no apila la salida en RAM)."""
    ruta.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with imageio.get_writer(str(ruta), fps=fps, codec="libx264", quality=8) as w:
        for fr in gen:
            w.append_data(fr)
            n += 1
    return n


def _gen_lineal_x3(F, f):
    """Interpolacion lineal en la grilla de render_slowmo (n_out=(N-1)*f+1)."""
    N = F.shape[0]
    n_out = (N - 1) * f + 1
    for i in range(n_out):
        p = i / f
        j = int(np.floor(p))
        a = p - j
        if j + 1 <= N - 1:
            fr = (1.0 - a) * F[j].astype(np.float32) + a * F[j + 1].astype(np.float32)
            yield np.clip(fr, 0, 255).astype(np.uint8)
        else:
            yield F[j]


def _gen_sininterp_x3(F, f):
    """Misma duracion que el x3 pero SIN interpolar: cada frame real repetido f veces."""
    N = F.shape[0]
    n_out = (N - 1) * f + 1
    for i in range(n_out):
        yield F[int(np.floor(i / f))]     # se mantiene el frame real -> escalones


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--clips", nargs="+", default=CLIPS_DEFAULT)
    ap.add_argument("--factor", type=int, default=3)
    ap.add_argument("--max_frames", type=int, default=None,
                    help="usar solo las primeras N frames (para igualar a un modelo con max_frames)")
    ap.add_argument("--n_salida", type=int, default=None,
                    help="cap de frames de salida (para igualar a un slowmo PARCIAL del modelo)")
    ap.add_argument("--fps_slow", type=int, default=30, help="fps del slowmo (= render_slowmo)")
    ap.add_argument("--fps_rapido", type=int, default=60)
    args = ap.parse_args()

    f = args.factor
    cap = (lambda g: itertools.islice(g, args.n_salida)) if args.n_salida else (lambda g: g)
    for clip in args.clips:
        clip_dir = CLIPS_DIR / clip
        F = _cargar(clip_dir, args.max_frames)
        N = F.shape[0]
        salida = DEST / clip
        print(f"\n=== {clip}  N={N}  res={F.shape[1]}x{F.shape[2]}  factor x{f} ===")

        nl = _escribir_stream(cap(_gen_lineal_x3(F, f)), salida / f"lineal_x{f}.mp4", args.fps_slow)
        print(f"  lineal_x{f}.mp4     {nl} frames @ {args.fps_slow}fps  = {nl/args.fps_slow:.1f}s")

        ns = _escribir_stream(cap(_gen_sininterp_x3(F, f)), salida / f"sininterp_x{f}.mp4", args.fps_slow)
        print(f"  sininterp_x{f}.mp4  {ns} frames @ {args.fps_slow}fps  = {ns/args.fps_slow:.1f}s  (a saltos)")

        nr = _escribir_stream(iter(F), salida / "rapido_60fps.mp4", args.fps_rapido)
        print(f"  rapido_60fps.mp4  {nr} frames @ {args.fps_rapido}fps  = {nr/args.fps_rapido:.1f}s  (rapido)")

    print(f"\n=== videos en {DEST} ===")


if __name__ == "__main__":
    main()
