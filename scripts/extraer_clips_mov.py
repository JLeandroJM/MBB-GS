"""
Extrae TODOS los frames de una ventana temporal de un .MOV (iPhone slow-mo) a
PNGs en data/clips/<nombre_clip>/ con el patron frame_%04d.png (0-based), que es
el que espera el loader (io/video.py).

Importante (slow-mo iPhone):
    El .MOV reporta 30 fps de REPRODUCCION pero contiene todos los frames
    capturados en camara lenta (240 fps de captura). NO resampleamos fps:
    extraemos TODOS los frames de la ventana tal cual, porque cada uno es un
    frame real distinto y esa densidad es justo lo que el modelo interpola.

Uso:
    python scripts/extraer_clips_mov.py \
        --video data/videos/GOTA.MOV \
        --inicio_seg 6 --fin_seg 16 \
        --H 270 --W 480 \
        --nombre_clip gota_s6s16_480x270
"""
import argparse
import os
import subprocess
import sys
from pathlib import Path

RAIZ = Path(__file__).resolve().parents[1]
RUTA_CLIPS = RAIZ / "data" / "clips"


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--video", required=True, help="ruta al .MOV")
    p.add_argument("--inicio_seg", type=float, required=True)
    p.add_argument("--fin_seg", type=float, required=True)
    p.add_argument("--H", type=int, required=True, help="alto de salida")
    p.add_argument("--W", type=int, required=True, help="ancho de salida")
    p.add_argument("--nombre_clip", required=True,
                   help="carpeta destino dentro de data/clips/")
    p.add_argument("--forzar", action="store_true",
                   help="reextrae aunque la carpeta ya tenga PNGs")
    return p.parse_args()


def main():
    args = parse_args()
    video = Path(args.video).resolve()
    if not video.is_file():
        sys.exit(f"no existe el video: {video}")

    dur = float(args.fin_seg) - float(args.inicio_seg)
    if dur <= 0:
        sys.exit("fin_seg debe ser > inicio_seg")

    destino = RUTA_CLIPS / args.nombre_clip
    destino.mkdir(parents=True, exist_ok=True)

    ya = [f for f in os.listdir(destino) if f.startswith("frame_") and f.endswith(".png")]
    if ya and not args.forzar:
        sys.exit(f"{destino} ya tiene {len(ya)} PNGs (usa --forzar para reextraer)")
    for f in ya:
        os.remove(destino / f)

    # ffmpeg numera desde 1; luego renombramos a base 0 para el loader.
    patron_tmp = str(destino / "tmp_%05d.png")
    cmd = [
        "ffmpeg", "-v", "error",
        "-ss", str(args.inicio_seg),
        "-i", str(video),
        "-t", str(dur),
        "-vf", f"scale={args.W}:{args.H}:flags=area",
        "-start_number", "1",
        patron_tmp,
    ]
    print("  " + " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True)

    tmps = sorted(f for f in os.listdir(destino)
                  if f.startswith("tmp_") and f.endswith(".png"))
    for i, f in enumerate(tmps):
        os.rename(destino / f, destino / f"frame_{i:04d}.png")

    print(f"=== {args.nombre_clip}: {len(tmps)} frames -> {destino} "
          f"({args.W}x{args.H})", flush=True)


if __name__ == "__main__":
    main()
