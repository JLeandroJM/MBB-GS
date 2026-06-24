import argparse
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image


RAIZ = Path(__file__).resolve().parents[1]
SRC = RAIZ / "src"

if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from gs2d_video.core.bases import construir_matriz_chebyshev
from gs2d_video.core.modelo import GaussianasPolinomial2D
from gs2d_video.render.renderer import render_frame


def elegir_device(nombre):
    if nombre == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA no está disponible")
        return torch.device("cuda")
    return torch.device(nombre)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--salida", required=True)
    parser.add_argument("--fps_origen", type=float, default=30.0)
    parser.add_argument("--fps_salida", type=float, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--forzar", action="store_true")
    args = parser.parse_args()

    device = elegir_device(args.device)
    ruta_checkpoint = Path(args.checkpoint)
    carpeta_salida = Path(args.salida)

    if not ruta_checkpoint.is_file():
        raise FileNotFoundError(f"No existe: {ruta_checkpoint}")

    carpeta_salida.mkdir(parents=True, exist_ok=True)

    existentes = sorted(carpeta_salida.glob("frame_*.png"))
    if existentes and not args.forzar:
        raise RuntimeError(
            f"Ya existen {len(existentes)} frames en {carpeta_salida}. "
            "Usa --forzar para reemplazarlos."
        )

    if args.forzar:
        for archivo in existentes:
            archivo.unlink()

    checkpoint = torch.load(ruta_checkpoint, map_location=device)
    state = checkpoint["state_dict_coefs"]
    config = checkpoint["config"]

    grados = state["grados"]
    n_gaussianas = int(state["N"])
    n_frames_origen = int(state["n_frames"])
    H = int(state["H"])
    W = int(state["W"])

    n_frames_salida = int(
        round((n_frames_origen - 1) * args.fps_salida / args.fps_origen)
    ) + 1
    n_frames_salida = max(2, n_frames_salida)

    print("=== regeneración temporal interpolada ===")
    print(f"checkpoint       : {ruta_checkpoint}")
    print(f"gaussianas       : {n_gaussianas}")
    print(f"resolución       : {H}x{W}")
    print(f"frames originales: {n_frames_origen}")
    print(f"fps original     : {args.fps_origen}")
    print(f"fps salida       : {args.fps_salida}")
    print(f"frames salida    : {n_frames_salida}")
    print(f"destino          : {carpeta_salida}")

    modelo = GaussianasPolinomial2D(
        n_gaussianas=n_gaussianas,
        n_frames=n_frames_origen,
        grados=grados,
        H=H,
        W=W,
        device=device,
        escala_inicial_px=float(config.get("escala_inicial_px", 2.5)),
        frame_0_imagen=None,
        semilla=int(config.get("seed", 42)),
    )

    with torch.no_grad():
        for nombre in ["mu", "opacity", "color", "scale", "theta", "depth"]:
            getattr(modelo, f"{nombre}_a0").copy_(
                state[f"{nombre}_a0"].to(device)
            )
            getattr(modelo, f"{nombre}_high").copy_(
                state[f"{nombre}_high"].to(device)
            )

    modelo.eval()

    grados_unicos = sorted(set(grados.values()))
    matrices_base = {
        grado: construir_matriz_chebyshev(
            n_frames=n_frames_salida,
            grado_max=grado,
            device=device,
            dtype=torch.float32,
        )
        for grado in grados_unicos
    }

    with torch.no_grad():
        for j in range(n_frames_salida):
            params = modelo.evaluar_en_frame(j, matrices_base)
            render = render_frame(params, H, W, config).clamp(0, 1)

            imagen = (
                render.mul(255)
                .to(torch.uint8)
                .cpu()
                .numpy()
            )

            Image.fromarray(imagen).save(
                carpeta_salida / f"frame_{j:04d}.png"
            )

            if j == 0 or (j + 1) % 25 == 0 or j == n_frames_salida - 1:
                print(f"frame {j + 1}/{n_frames_salida}", flush=True)

            del params, render

    print(f"Listo: {n_frames_salida} frames generados.")


if __name__ == "__main__":
    main()
