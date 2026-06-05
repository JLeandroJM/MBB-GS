import argparse
import subprocess
from pathlib import Path

import imageio_ffmpeg


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--inicio", type=float, default=0.0)
    parser.add_argument("--segundos", type=float, default=5.0)
    parser.add_argument("--sr", type=int, default=44100)
    parser.add_argument("--mono", action="store_true")
    args = parser.parse_args()

    entrada = Path(args.input)
    salida = Path(args.output)
    salida.parent.mkdir(parents=True, exist_ok=True)

    ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()

    cmd = [
        ffmpeg,
        "-y",
        "-i", str(entrada),
        "-ss", str(args.inicio),
        "-t", str(args.segundos),
        "-vn",
        "-acodec", "pcm_s16le",
        "-ar", str(args.sr),
    ]

    if args.mono:
        cmd += ["-ac", "1"]
    else:
        cmd += ["-ac", "2"]

    cmd += [str(salida)]

    print("Ejecutando:")
    print(" ".join(cmd))
    subprocess.run(cmd, check=True)

    print("")
    print(f"Audio limpio guardado en: {salida}")


if __name__ == "__main__":
    main()
