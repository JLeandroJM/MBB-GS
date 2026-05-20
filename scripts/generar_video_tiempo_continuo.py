import argparse
import math
import os
import sys

import imageio.v2 as imageio
import numpy as np
import torch
from PIL import Image

AQUI = os.path.dirname(os.path.abspath(__file__))
RAIZ = os.path.abspath(os.path.join(AQUI, ".."))

EXP_DIR = os.path.join(
    RAIZ,
    "experimentos",
    "exp09_batchfull_comparacion_bases_CUDA"
)

if EXP_DIR not in sys.path:
    sys.path.insert(0, EXP_DIR)

from modelo import GaussianasPolinomial2D
from rasterizador import rasterizar_un_frame

try:
    from rasterizador_cuda_tiled_autograd import rasterizar_un_frame_cuda_tiled
except Exception:
    rasterizar_un_frame_cuda_tiled = None

try:
    from rasterizador_cuda_autograd import rasterizar_un_frame_cuda_conic
except Exception:
    rasterizar_un_frame_cuda_conic = None


def construir_matriz_desde_tiempos(base, tiempos_frame, grado_max, n_frames_train, device, dtype):
    tiempos_frame = torch.as_tensor(tiempos_frame, dtype=torch.float64)

    if n_frames_train < 2:
        t01 = torch.zeros_like(tiempos_frame)
    else:
        t01 = tiempos_frame / float(n_frames_train - 1)

    B = torch.empty((len(tiempos_frame), grado_max + 1), dtype=torch.float64)

    if base == "monomial":
        B[:, 0] = 1.0
        if grado_max >= 1:
            B[:, 1] = t01
        for k in range(2, grado_max + 1):
            B[:, k] = B[:, k - 1] * t01

    elif base == "chebyshev":
        x = 2.0 * t01 - 1.0
        B[:, 0] = 1.0
        if grado_max >= 1:
            B[:, 1] = x
        for k in range(2, grado_max + 1):
            B[:, k] = 2.0 * x * B[:, k - 1] - B[:, k - 2]

    else:
        raise ValueError(f"base no soportada: {base}")

    return B.to(device=device, dtype=dtype)


def cargar_modelo_desde_checkpoint(ruta_checkpoint, device):
    ckpt = torch.load(ruta_checkpoint, map_location=device)

    sd = ckpt["state_dict_coefs"]
    config = ckpt.get("config", {})

    grados = sd["grados"]
    base = sd["base"]
    N = int(sd["N"])
    H = int(sd["H"])
    W = int(sd["W"])
    n_frames_train = int(sd["n_frames"])

    modelo = GaussianasPolinomial2D(
        n_gaussianas=N,
        n_frames=n_frames_train,
        grados=grados,
        base=base,
        H=H,
        W=W,
        device=device,
        escala_inicial_px=float(config.get("escala_inicial_px", 5.0)),
        frame_0_imagen=None,
        semilla=int(config.get("seed", 42)),
    )

    with torch.no_grad():
        for nombre in ["mu", "opacity", "color", "scale", "theta", "depth"]:
            getattr(modelo, f"{nombre}_a0").copy_(sd[f"{nombre}_a0"].to(device))
            getattr(modelo, f"{nombre}_high").copy_(sd[f"{nombre}_high"].to(device))

    modelo.eval()
    return modelo, config, n_frames_train, H, W, base, grados


def rasterizar_segun_config(params, H, W, config):
    if bool(config.get("usar_cuda_tiled", False)):
        if rasterizar_un_frame_cuda_tiled is not None:
            return rasterizar_un_frame_cuda_tiled(
                params,
                H,
                W,
                tile_size=int(config.get("cuda_tile_size", 16)),
                k_sigma=float(config.get("cuda_k_sigma", 3.5)),
            )

    if bool(config.get("usar_cuda_conic", False)):
        if rasterizar_un_frame_cuda_conic is not None:
            return rasterizar_un_frame_cuda_conic(params, H, W)

    return rasterizar_un_frame(params, H, W)


def evaluar_en_tiempo_continuo(modelo, idx_t, matrices_base):
    salida = {}
    for nombre, (a0, hi, grado, dim_p) in modelo.parametros_temporales().items():
        B = matrices_base[grado][idx_t]  # (grado+1,)
        coefs = torch.cat([a0, hi], dim=-1)  # (N, dim_p, grado+1)
        val = torch.matmul(coefs, B)         # (N, dim_p)

        if nombre == "mu":
            salida["mu"] = val
        elif nombre == "opacity":
            salida["opacity"] = torch.sigmoid(val.squeeze(-1))
        elif nombre == "color":
            salida["color"] = torch.sigmoid(val)
        elif nombre == "scale":
            salida["scale"] = torch.exp(val)
        elif nombre == "theta":
            salida["theta"] = val.squeeze(-1)
        elif nombre == "depth":
            salida["depth"] = val.squeeze(-1)

    return salida


@torch.no_grad()
def generar_video_tiempo_continuo(
    ruta_checkpoint,
    carpeta_salida,
    t_inicio,
    t_fin,
    dt,
    fps,
    guardar_frames,
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}")

    modelo, config, n_frames_train, H, W, base, grados = cargar_modelo_desde_checkpoint(
        ruta_checkpoint,
        device,
    )

    tiempos = []
    t = float(t_inicio)
    while t <= float(t_fin) + 1e-12:
        tiempos.append(t)
        t += float(dt)

    os.makedirs(carpeta_salida, exist_ok=True)

    print("=== info checkpoint ===")
    print(f"base: {base}")
    print(f"N: {modelo.numero_gausianas()}")
    print(f"frames entrenamiento: {n_frames_train}")
    print(f"resolucion: {H}x{W}")
    print(f"t_inicio: {t_inicio}")
    print(f"t_fin: {t_fin}")
    print(f"dt: {dt}")
    print(f"n_samples: {len(tiempos)}")

    grados_distintos = sorted(set(grados.values()))
    matrices_base = {
        g: construir_matriz_desde_tiempos(
            base=base,
            tiempos_frame=tiempos,
            grado_max=g,
            n_frames_train=n_frames_train,
            device=device,
            dtype=torch.float32,
        )
        for g in grados_distintos
    }

    carpeta_frames = os.path.join(carpeta_salida, "frames")
    if guardar_frames:
        os.makedirs(carpeta_frames, exist_ok=True)

    frames_uint8 = []

    for i, t_real in enumerate(tiempos):
        params = evaluar_en_tiempo_continuo(modelo, i, matrices_base)
        render = rasterizar_segun_config(params, H, W, config)

        render = torch.nan_to_num(render, nan=0.0, posinf=1.0, neginf=0.0)
        render = render.clamp(0, 1)

        img = (render.detach().cpu().numpy() * 255).astype(np.uint8)
        frames_uint8.append(img)

        if guardar_frames:
            nombre = f"frame_{i:04d}_t_{t_real:08.3f}.png"
            Image.fromarray(img).save(os.path.join(carpeta_frames, nombre))

        if (i + 1) % 10 == 0 or i == 0 or i == len(tiempos) - 1:
            print(f"render {i + 1}/{len(tiempos)}   t={t_real:.3f}")

    ruta_gif = os.path.join(carpeta_salida, "video_tiempo_continuo.gif")
    imageio.mimsave(ruta_gif, frames_uint8, fps=fps)
    print(f"gif: {ruta_gif}")

    try:
        ruta_mp4 = os.path.join(carpeta_salida, "video_tiempo_continuo.mp4")
        imageio.mimsave(ruta_mp4, frames_uint8, fps=fps)
        print(f"mp4: {ruta_mp4}")
    except Exception as e:
        print(f"[warn] no se pudo guardar mp4: {e}")
        print("Instala: pip install imageio[ffmpeg]")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--salida", required=True)
    parser.add_argument("--t_inicio", type=float, required=True)
    parser.add_argument("--t_fin", type=float, required=True)
    parser.add_argument("--dt", type=float, default=0.1)
    parser.add_argument("--fps", type=int, default=25)
    parser.add_argument("--no_guardar_frames", action="store_true")
    args = parser.parse_args()

    generar_video_tiempo_continuo(
        ruta_checkpoint=args.checkpoint,
        carpeta_salida=args.salida,
        t_inicio=args.t_inicio,
        t_fin=args.t_fin,
        dt=args.dt,
        fps=args.fps,
        guardar_frames=not args.no_guardar_frames,
    )


if __name__ == "__main__":
    main()