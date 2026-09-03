"""
Une dos videos LADO A LADO (hstack) con etiquetas, para presentaciones.
Ambos videos deben tener la misma cantidad de frames y fps para ir sincronizados.

Uso:
    python scripts/video_lado_a_lado.py \
        --izq outputs_khipu/slowmo_gota_mejorado_s2s4/slowmo_x3.mp4 \
        --der comparacion_slowmo/gota_mejorado_s2s4_1080p/sininterp_x3.mp4 \
        --label_izq "Modelo (slow-mo x3)" \
        --label_der "Original sin interpolar" \
        --salida comparacion_slowmo/gota_mejorado_s2s4_1080p/comparativa_lado_a_lado.mp4
"""
import argparse
import subprocess
from pathlib import Path

FUENTE = "/System/Library/Fonts/Supplemental/Arial.ttf"


def _drawtext(texto, fuente):
    t = texto.replace(":", r"\:").replace("'", "")
    return (f"drawtext=fontfile={fuente}:text='{t}':x=20:y=20:"
            f"fontsize=30:fontcolor=white:box=1:boxcolor=black@0.55:boxborderw=10")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--izq", required=True)
    ap.add_argument("--der", required=True)
    ap.add_argument("--label_izq", default="")
    ap.add_argument("--label_der", default="")
    ap.add_argument("--salida", required=True)
    ap.add_argument("--altura", type=int, default=540, help="alto de cada panel (px)")
    ap.add_argument("--fuente", default=FUENTE)
    ap.add_argument("--sin_etiquetas", action="store_true")
    args = ap.parse_args()

    for p in (args.izq, args.der):
        if not Path(p).is_file():
            raise SystemExit(f"no existe: {p}")
    Path(args.salida).parent.mkdir(parents=True, exist_ok=True)

    H = args.altura
    usar_txt = (not args.sin_etiquetas) and Path(args.fuente).is_file()

    izq = f"[0:v]scale=-2:{H}"
    der = f"[1:v]scale=-2:{H}"
    if usar_txt and args.label_izq:
        izq += "," + _drawtext(args.label_izq, args.fuente)
    if usar_txt and args.label_der:
        der += "," + _drawtext(args.label_der, args.fuente)
    filtro = f"{izq}[l];{der}[r];[l][r]hstack=inputs=2[v]"

    cmd = [
        "ffmpeg", "-v", "error", "-y",
        "-i", args.izq, "-i", args.der,
        "-filter_complex", filtro,
        "-map", "[v]",
        "-c:v", "libx264", "-crf", "18", "-preset", "veryfast", "-pix_fmt", "yuv420p",
        args.salida,
    ]
    print("  " + " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True)
    print(f"=== lado a lado -> {args.salida} ===", flush=True)


if __name__ == "__main__":
    main()
