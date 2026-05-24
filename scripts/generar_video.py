"""
Genera un MP4 desde los PNGs rasterizados de una corrida.

Uso (Windows):
    python scripts\\generar_video.py outputs\\runs\\<exp>\\<clip>
    python scripts\\generar_video.py outputs\\runs\\<exp>\\<clip>\\frames_rasterizados
    python scripts\\generar_video.py <ruta> --fps 25 --crf 18 --out reconstruido.mp4

El primer argumento puede ser:
  - la carpeta de la corrida (busca frames_rasterizados/ adentro)
  - una carpeta directa con archivos frame_NNNN.png

Requiere ffmpeg en el PATH (mismo requisito que io/video.py).
"""
import argparse
import os
import shutil
import subprocess
import sys


PREFIJO = "frame_"
EXTENSION = ".png"


def _listar_frames(carpeta):
    return sorted(
        f for f in os.listdir(carpeta)
        if f.startswith(PREFIJO) and f.endswith(EXTENSION)
    )


def resolver_carpeta_frames(ruta):
    """Devuelve la carpeta que efectivamente contiene los frame_NNNN.png."""
    if not os.path.isdir(ruta):
        raise FileNotFoundError(f"no es una carpeta: {ruta}")

    if _listar_frames(ruta):
        return ruta

    candidato = os.path.join(ruta, "frames_rasterizados")
    if os.path.isdir(candidato) and _listar_frames(candidato):
        return candidato

    raise FileNotFoundError(
        f"no encontre archivos {PREFIJO}NNNN{EXTENSION} en:\n"
        f"  {ruta}\n"
        f"  {candidato}"
    )


def detectar_padding(nombre_archivo):
    """frame_0000.png -> 4. frame_00000.png -> 5."""
    return len(nombre_archivo) - len(PREFIJO) - len(EXTENSION)


def construir_mp4(carpeta_frames, ruta_salida, fps, crf, codec):
    if shutil.which("ffmpeg") is None:
        print(
            "ERROR: ffmpeg no se encuentra en el PATH.\n"
            "  Instalalo y reintenta. En Windows: https://www.gyan.dev/ffmpeg/builds/",
            file=sys.stderr,
        )
        sys.exit(2)

    pngs = _listar_frames(carpeta_frames)
    if not pngs:
        raise RuntimeError(f"sin frames en {carpeta_frames}")

    padding = detectar_padding(pngs[0])
    patron = os.path.join(carpeta_frames, f"{PREFIJO}%0{padding}d{EXTENSION}")

    print(f"  fuente   : {carpeta_frames}", flush=True)
    print(f"  frames   : {len(pngs)} ({pngs[0]} ... {pngs[-1]})", flush=True)
    print(f"  patron   : {patron}", flush=True)
    print(f"  destino  : {ruta_salida}", flush=True)
    print(f"  fps={fps}  codec={codec}  crf={crf}", flush=True)

    os.makedirs(os.path.dirname(os.path.abspath(ruta_salida)) or ".", exist_ok=True)

    cmd = [
        "ffmpeg", "-y",
        "-framerate", str(fps),
        "-i", patron,
        "-c:v", codec,
        "-crf", str(crf),
        "-pix_fmt", "yuv420p",
        "-movflags", "+faststart",
        ruta_salida,
    ]
    print(f"  cmd: {' '.join(cmd)}", flush=True)

    resultado = subprocess.run(cmd, capture_output=True, text=True)
    if resultado.returncode != 0:
        raise RuntimeError(
            f"ffmpeg fallo (returncode={resultado.returncode}):\n"
            f"--- stderr ---\n{resultado.stderr}"
        )

    bytes_video = os.path.getsize(ruta_salida)
    duracion_seg = len(pngs) / fps
    print(
        f"listo: {ruta_salida}  "
        f"({bytes_video / 1024:.1f} KiB, {duracion_seg:.2f}s @ {fps}fps)",
        flush=True,
    )


def ruta_salida_default(carpeta_frames):
    """
    Si la entrada fue ../<run>/frames_rasterizados/ -> salida en ../<run>/reconstruido.mp4.
    Si fue cualquier otra carpeta -> salida dentro de ella.
    """
    base = os.path.basename(carpeta_frames.rstrip(os.sep))
    if base == "frames_rasterizados":
        padre = os.path.dirname(carpeta_frames.rstrip(os.sep))
        return os.path.join(padre, "reconstruido.mp4")
    return os.path.join(carpeta_frames, "reconstruido.mp4")


def main():
    parser = argparse.ArgumentParser(
        description="Genera un MP4 desde los PNGs rasterizados de una corrida."
    )
    parser.add_argument(
        "ruta",
        help="carpeta de la corrida (outputs/runs/<exp>/<clip>) "
             "o carpeta directa con frame_NNNN.png",
    )
    parser.add_argument("--fps", type=int, default=25,
                        help="frames por segundo (default 25)")
    parser.add_argument("--crf", type=int, default=18,
                        help="calidad x264 0-51, menor=mejor (default 18 = alta calidad)")
    parser.add_argument("--codec", default="libx264",
                        help="codec ffmpeg (default libx264; alternativas: libx265, libaom-av1)")
    parser.add_argument("--out", default=None,
                        help="ruta del MP4 de salida (default: <ruta>/reconstruido.mp4)")
    args = parser.parse_args()

    carpeta_frames = resolver_carpeta_frames(args.ruta)
    destino = args.out or ruta_salida_default(carpeta_frames)

    construir_mp4(carpeta_frames, destino, args.fps, args.crf, args.codec)


if __name__ == "__main__":
    main()
