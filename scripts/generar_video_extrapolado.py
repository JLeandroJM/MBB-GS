import argparse
import os
import sys
import numpy as np
import torch
from PIL import Image
import imageio.v2 as imageio


# ============================================================
# Paths del proyecto
# ============================================================

AQUI = os.path.dirname(os.path.abspath(__file__))

# tools/.. = raiz tesis_2dgs_video
RAIZ = os.path.abspath(os.path.join(AQUI, ".."))

# carpeta donde estan modelo.py, rasterizador.py, etc.
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

# ============================================================
# Matriz temporal extrapolada
# ============================================================

def construir_matriz_extrapolada(base, n_frames_salida, grado_max, n_frames_train, device, dtype):
    """
    Evalua la base usando el tiempo original del entrenamiento.

    Si el entrenamiento tuvo T frames:
        t01 = j / (T - 1)

    Entonces:
        j = T - 1  => t01 = 1
        j = 2T - 2 => t01 = 2

    Eso permite extrapolar mas alla del tiempo entrenado.
    """
    if n_frames_train < 2:
        t01 = torch.zeros(n_frames_salida, dtype=torch.float64)
    else:
        j = torch.arange(n_frames_salida, dtype=torch.float64)
        t01 = j / float(n_frames_train - 1)

    B = torch.empty(n_frames_salida, grado_max + 1, dtype=torch.float64)

    if base == "monomial":
        B[:, 0] = 1.0
        if grado_max >= 1:
            B[:, 1] = t01
        for k in range(2, grado_max + 1):
            B[:, k] = B[:, k - 1] * t01

    elif base == "chebyshev":
        # En entrenamiento: x esta en [-1, 1]
        # En extrapolacion: x puede pasar de 1
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


def rasterizar_segun_config(params, H, W, config, forzar_pytorch=False):
    if not forzar_pytorch and bool(config.get("usar_cuda_tiled", False)):
        if rasterizar_un_frame_cuda_tiled is None:
            print("[warn] CUDA tiled no disponible. Usando PyTorch.")
            return rasterizar_un_frame(params, H, W)

        return rasterizar_un_frame_cuda_tiled(
            params,
            H,
            W,
            tile_size=int(config.get("cuda_tile_size", 16)),
            k_sigma=float(config.get("cuda_k_sigma", 3.5)),
        )

    if not forzar_pytorch and bool(config.get("usar_cuda_conic", False)):
        if rasterizar_un_frame_cuda_conic is None:
            print("[warn] CUDA conic no disponible. Usando PyTorch.")
            return rasterizar_un_frame(params, H, W)

        return rasterizar_un_frame_cuda_conic(params, H, W)

    return rasterizar_un_frame(params, H, W)


@torch.no_grad()
def generar_video_extrapolado(
    ruta_checkpoint,
    carpeta_salida,
    n_frames_salida=None,
    factor_tiempo=2.0,
    fps=25,
    guardar_frames=True,
    forzar_pytorch=False,
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}")

    modelo, config, n_frames_train, H, W, base, grados = cargar_modelo_desde_checkpoint(
        ruta_checkpoint,
        device,
    )

    if n_frames_salida is None:
        n_frames_salida = int(round(n_frames_train * factor_tiempo))

    os.makedirs(carpeta_salida, exist_ok=True)

    print("=== info checkpoint ===")
    print(f"base: {base}")
    print(f"N: {modelo.numero_gausianas()}")
    print(f"frames entrenamiento: {n_frames_train}")
    print(f"frames salida: {n_frames_salida}")
    print(f"resolucion: {H}x{W}")
    print(f"factor_tiempo: {n_frames_salida / max(1, n_frames_train):.2f}x")

    grados_distintos = sorted(set(grados.values()))
    matrices_base = {
        g: construir_matriz_extrapolada(
            base=base,
            n_frames_salida=n_frames_salida,
            grado_max=g,
            n_frames_train=n_frames_train,
            device=device,
            dtype=torch.float32,
        )
        for g in grados_distintos
    }

    carpeta_frames = os.path.join(carpeta_salida, "frames_extrapolados")
    if guardar_frames:
        os.makedirs(carpeta_frames, exist_ok=True)

    frames_uint8 = []

    for j in range(n_frames_salida):
        params_j = modelo.evaluar_en_frame(j, matrices_base)

        render = rasterizar_segun_config(
            params_j,
            H,
            W,
            config,
            forzar_pytorch=forzar_pytorch,
        ).clamp(0, 1)

        render = torch.nan_to_num(render, nan=0.0, posinf=1.0, neginf=0.0)
        render = render.clamp(0, 1)

        img = (render.detach().cpu().numpy() * 255).astype(np.uint8)        
        frames_uint8.append(img)

        if guardar_frames:
            Image.fromarray(img).save(os.path.join(carpeta_frames, f"frame_{j:04d}.png"))

        if (j + 1) % 10 == 0 or j == 0 or j == n_frames_salida - 1:
            print(f"render {j + 1}/{n_frames_salida}")

    ruta_mp4 = os.path.join(carpeta_salida, "video_extrapolado.mp4")
    ruta_gif = os.path.join(carpeta_salida, "video_extrapolado.gif")

    try:
        imageio.mimsave(ruta_mp4, frames_uint8, fps=fps)
        print(f"mp4: {ruta_mp4}")
    except Exception as e:
        print(f"[warn] no se pudo guardar mp4: {e}")
        print("Instala con: pip install imageio[ffmpeg]")

    imageio.mimsave(ruta_gif, frames_uint8, fps=fps)
    print(f"gif: {ruta_gif}")

    print("=== listo ===")
    print(f"mp4: {ruta_mp4}")
    print(f"gif: {ruta_gif}")
    if guardar_frames:
        print(f"frames: {carpeta_frames}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True, help="ruta a checkpoint_final.pt")
    parser.add_argument("--salida", default="salidas_extrapolacion", help="carpeta de salida")
    parser.add_argument("--factor_tiempo", type=float, default=2.0, help="2.0 genera el doble de duracion")
    parser.add_argument("--n_frames_salida", type=int, default=None, help="cantidad exacta de frames a generar")
    parser.add_argument("--fps", type=int, default=25)
    parser.add_argument("--no_guardar_frames", action="store_true")
    parser.add_argument("--forzar_pytorch", action="store_true")
    args = parser.parse_args()

    generar_video_extrapolado(
        ruta_checkpoint=args.checkpoint,
        carpeta_salida=args.salida,
        n_frames_salida=args.n_frames_salida,
        factor_tiempo=args.factor_tiempo,
        fps=args.fps,
        guardar_frames=not args.no_guardar_frames,
        forzar_pytorch=args.forzar_pytorch,
    )


if __name__ == "__main__":
    main()