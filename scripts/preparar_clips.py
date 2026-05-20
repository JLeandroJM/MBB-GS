"""
Acepta un mp4, lo recorta temporalmente, lo redimensiona a 256x256 con crop
centrado, y lo extrae como secuencia de PNGs en clips/<nombre>/

Uso:
    python tools/preparar_clips.py video.mp4 nombre_clip [--start S] [--end E]

Requiere ffmpeg en el PATH (ya viene en macOS via homebrew).
"""
import argparse
import json
import os
import subprocess
import sys

from PIL import Image


RES = 256
FPS = 30



def asegurar_carpeta(ruta):
    os.makedirs(ruta, exist_ok=True)



def extraer_frames_temporal(mp4_path, carpeta_tmp, start=None, end=None, fps=FPS):
    """Llama a ffmpeg para extraer frames a fps fijo, con recorte temporal opcional."""
    asegurar_carpeta(carpeta_tmp)
    cmd = ["ffmpeg", "-y"]
    if start is not None:
        cmd += ["-ss", str(start)]
    cmd += ["-i", mp4_path]
    if end is not None:
        dur = float(end) - float(start or 0.0)
        cmd += ["-t", str(dur)]
    cmd += ["-vf", f"fps={fps}", "-q:v", "2",
            os.path.join(carpeta_tmp, "raw_%05d.png")]

    print("ejecutando:", " ".join(cmd))
    subprocess.run(cmd, check=True)



def redimensionar_y_cropear(carpeta_tmp, carpeta_final):
    """Redimensiona cada frame a RES x RES con crop centrado si el aspect no coincide."""
    asegurar_carpeta(carpeta_final)
    archivos = sorted(f for f in os.listdir(carpeta_tmp) if f.endswith(".png"))

    for i, archivo in enumerate(archivos):
        img = Image.open(os.path.join(carpeta_tmp, archivo)).convert("RGB")
        w, h = img.size
        # crop centrado al menor lado
        lado = min(w, h)
        izq = (w - lado) // 2
        arr = (h - lado) // 2
        img = img.crop((izq, arr, izq + lado, arr + lado))
        img = img.resize((RES, RES), Image.LANCZOS)
        img.save(os.path.join(carpeta_final, f"frame_{i:04d}.png"))

    return len(archivos)



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("mp4", help="ruta al video mp4")
    parser.add_argument("nombre", help="nombre del clip")
    parser.add_argument("--start", type=float, default=None,
                        help="segundo de inicio (opcional)")
    parser.add_argument("--end", type=float, default=None,
                        help="segundo de fin (opcional)")
    args = parser.parse_args()

    aqui = os.path.dirname(os.path.abspath(__file__))
    raiz = os.path.abspath(os.path.join(aqui, ".."))
    clips_dir = os.path.join(raiz, "clips")
    carpeta_tmp = os.path.join(raiz, "_tmp_extraccion")
    carpeta_final = os.path.join(clips_dir, args.nombre)

    # limpiar tmp y extraer frames crudos a fps fijo
    if os.path.isdir(carpeta_tmp):
        import shutil; shutil.rmtree(carpeta_tmp)
    extraer_frames_temporal(args.mp4, carpeta_tmp, args.start, args.end, FPS)

    # redimensionar a 256x256
    n_frames = redimensionar_y_cropear(carpeta_tmp, carpeta_final)

    # metadatos
    info = {
        "nombre": args.nombre,
        "num_frames": n_frames,
        "fps": FPS,
        "resolucion": [RES, RES],
        "origen": os.path.abspath(args.mp4),
    }
    with open(os.path.join(carpeta_final, "clip_info.json"), "w") as f:
        json.dump(info, f, indent=2)

    # limpiar tmp
    import shutil; shutil.rmtree(carpeta_tmp)

    print(f"\nlisto. {n_frames} frames en: {carpeta_final}")



if __name__ == "__main__":
    main()
