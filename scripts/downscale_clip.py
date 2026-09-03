"""
Baja la resolucion de un clip de frames YA extraidos (frame_%04d.png), sin
tocar el .MOV. Util para generar rapido una version chica y entrenar mas rapido.

Uso:
    python scripts/downscale_clip.py \
        --src data/clips/video_final_gota_1080p \
        --dst data/clips/video_final_gota_480x270 --H 270 --W 480
"""
import argparse
import os
from pathlib import Path
from PIL import Image


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True)
    ap.add_argument("--dst", required=True)
    ap.add_argument("--H", type=int, required=True)
    ap.add_argument("--W", type=int, required=True)
    args = ap.parse_args()

    src, dst = Path(args.src), Path(args.dst)
    dst.mkdir(parents=True, exist_ok=True)
    archivos = sorted(f for f in os.listdir(src)
                      if f.startswith("frame_") and f.endswith(".png"))
    if not archivos:
        raise SystemExit(f"sin frames en {src}")

    for i, nombre in enumerate(archivos):
        img = Image.open(src / nombre).convert("RGB").resize((args.W, args.H), Image.LANCZOS)
        img.save(dst / f"frame_{i:04d}.png")
    print(f"=== {len(archivos)} frames {args.W}x{args.H} -> {dst} ===")


if __name__ == "__main__":
    main()
