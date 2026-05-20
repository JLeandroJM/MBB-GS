import os
import sys
import argparse

import torch
import numpy as np
from PIL import Image


# ==========================
# Rutas base
# ==========================

AQUI = os.path.dirname(os.path.abspath(__file__))
RAIZ_TESIS_VIDEO_DEFAULT = os.path.abspath(os.path.join(AQUI, ".."))


def resolver_rutas(raiz_tesis_video_arg=None, ruta_cuda_arg=None):
    if raiz_tesis_video_arg is not None:
        raiz_tesis_video = os.path.abspath(raiz_tesis_video_arg)
    else:
        raiz_tesis_video = RAIZ_TESIS_VIDEO_DEFAULT

    raiz_rasterizador = os.path.abspath(os.path.join(raiz_tesis_video, ".."))

    ruta_exp09 = os.path.join(
        raiz_tesis_video,
        "experimentos",
        "exp09_batchfull_comparacion_bases_CUDA",
    )

    ruta_cuda = (
        os.path.abspath(ruta_cuda_arg)
        if ruta_cuda_arg is not None
        else os.environ.get(
            "RUTA_RASTER_CUDA",
            os.path.join(raiz_rasterizador, "raster_cuda_new"),
        )
    )

    return raiz_tesis_video, raiz_rasterizador, ruta_exp09, ruta_cuda


def preparar_imports(raiz_tesis_video, ruta_exp09, ruta_cuda):
    if not os.path.isdir(raiz_tesis_video):
        raise FileNotFoundError(f"No existe raiz_tesis_video: {raiz_tesis_video}")

    if not os.path.isdir(ruta_exp09):
        raise FileNotFoundError(f"No existe carpeta exp09: {ruta_exp09}")

    if not os.path.isdir(ruta_cuda):
        raise FileNotFoundError(f"No existe carpeta CUDA: {ruta_cuda}")

    # Primero el experimento para usar sus wrappers correctos.
    if ruta_exp09 not in sys.path:
        sys.path.insert(0, ruta_exp09)

    # Luego CUDA para que los wrappers encuentren raster_cuda.pyd.
    if ruta_cuda not in sys.path:
        sys.path.append(ruta_cuda)

    os.environ["RUTA_RASTER_CUDA"] = ruta_cuda
    os.environ.setdefault("USAR_PREPROCESS_TILED_CUDA", "1")


# Los imports del experimento se hacen despues de preparar sys.path.
construir_matriz = None
GaussianasPolinomial2D = None
rasterizar_un_frame = None
rasterizar_un_frame_cuda_conic = None
rasterizar_un_frame_cuda_tiled = None


def importar_modulos_exp09():
    global construir_matriz
    global GaussianasPolinomial2D
    global rasterizar_un_frame
    global rasterizar_un_frame_cuda_conic
    global rasterizar_un_frame_cuda_tiled

    from bases import construir_matriz as _construir_matriz
    from modelo import GaussianasPolinomial2D as _GaussianasPolinomial2D
    from rasterizador import rasterizar_un_frame as _rasterizar_un_frame

    construir_matriz = _construir_matriz
    GaussianasPolinomial2D = _GaussianasPolinomial2D
    rasterizar_un_frame = _rasterizar_un_frame

    try:
        from rasterizador_cuda_autograd import rasterizar_un_frame_cuda_conic as _conic
        rasterizar_un_frame_cuda_conic = _conic
    except Exception as e:
        print(f"[warning] CUDA conic no disponible: {e}", flush=True)
        rasterizar_un_frame_cuda_conic = None

    try:
        from rasterizador_cuda_tiled_autograd import rasterizar_un_frame_cuda_tiled as _tiled
        rasterizar_un_frame_cuda_tiled = _tiled
    except Exception as e:
        print(f"[warning] CUDA tiled no disponible: {e}", flush=True)
        rasterizar_un_frame_cuda_tiled = None


def elegir_device(device_str):
    if device_str == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("cuda no disponible")
        return torch.device("cuda")
    if device_str == "cpu":
        return torch.device("cpu")
    return torch.device(device_str)


def rasterizar_segun_config(params_j, H, W, config):
    if bool(config.get("usar_cuda_tiled", False)):
        if rasterizar_un_frame_cuda_tiled is None:
            raise RuntimeError("cuda tiled no disponible")
        return rasterizar_un_frame_cuda_tiled(
            params_j,
            H,
            W,
            tile_size=int(config.get("cuda_tile_size", 16)),
            k_sigma=float(config.get("cuda_k_sigma", 4.0)),
        )

    if bool(config.get("usar_cuda_conic", False)):
        if rasterizar_un_frame_cuda_conic is None:
            raise RuntimeError("cuda conic no disponible")
        return rasterizar_un_frame_cuda_conic(params_j, H, W)

    return rasterizar_un_frame(params_j, H, W)


def contar_frames(carpeta_clip):
    if not os.path.isdir(carpeta_clip):
        raise FileNotFoundError(f"No existe carpeta_clip: {carpeta_clip}")

    archivos = sorted(
        f for f in os.listdir(carpeta_clip)
        if f.startswith("frame_") and f.endswith(".png")
    )
    if not archivos:
        raise RuntimeError(f"No se encontraron frame_*.png en: {carpeta_clip}")
    return len(archivos)


def leer_resolucion_frame(carpeta_clip, frame_idx):
    ruta = os.path.join(carpeta_clip, f"frame_{frame_idx:04d}.png")
    if not os.path.isfile(ruta):
        raise FileNotFoundError(f"No existe frame objetivo: {ruta}")

    img = Image.open(ruta).convert("RGB")
    arr = np.array(img, dtype=np.uint8)
    H, W = arr.shape[0], arr.shape[1]
    return H, W, ruta


def inferir_n_gaussianas(state_dict_coefs, config):
    # Para modelo pruneado, el N real puede ser menor que el del config.
    for key in ["mu_a0", "color_a0", "opacity_a0", "scale_a0", "theta_a0", "depth_a0"]:
        v = state_dict_coefs.get(key, None)
        if torch.is_tensor(v) and v.ndim >= 1:
            return int(v.shape[0])
    return int(config["n_gaussianas_inicial"])


def cargar_state_dict_coefs_manual(modelo, state_dict_coefs, device):
    # Copia tensores del checkpoint al modelo reconstruido.
    for nombre, valor in state_dict_coefs.items():
        if not torch.is_tensor(valor):
            continue
        if not hasattr(modelo, nombre):
            print(f"[warning] el modelo no tiene atributo: {nombre}", flush=True)
            continue

        tensor_modelo = getattr(modelo, nombre)
        if tensor_modelo.shape != valor.shape:
            raise RuntimeError(
                f"shape incompatible para {nombre}: "
                f"modelo={tuple(tensor_modelo.shape)} checkpoint={tuple(valor.shape)}"
            )
        tensor_modelo.data.copy_(valor.to(device))


def construir_modelo_desde_checkpoint(ckpt, device, n_frames, H, W):
    config = ckpt["config"]
    state_dict_coefs = ckpt["state_dict_coefs"]

    grados = config["grados"]
    base = config["base"]
    n_gaussianas = inferir_n_gaussianas(state_dict_coefs, config)

    modelo = GaussianasPolinomial2D(
        n_gaussianas=n_gaussianas,
        n_frames=n_frames,
        grados=grados,
        base=base,
        H=H,
        W=W,
        device=device,
        escala_inicial_px=float(config.get("escala_inicial_px", 5.0)),
        frame_0_imagen=None,
        semilla=int(config.get("seed", 42)),
    )

    cargar_state_dict_coefs_manual(modelo, state_dict_coefs, device)
    return modelo


def calcular_metricas_simples(render_np, target_path):
    target = Image.open(target_path).convert("RGB")
    target_np = np.array(target, dtype=np.float32) / 255.0
    pred_np = render_np.astype(np.float32) / 255.0

    mse = float(np.mean((pred_np - target_np) ** 2))
    mae = float(np.mean(np.abs(pred_np - target_np)))
    psnr = float("inf") if mse <= 0 else float(-10.0 * np.log10(mse))
    return mse, mae, psnr


def guardar_comparacion(render_np, target_path, salida_comparacion, factor_diff=5.0):
    target = Image.open(target_path).convert("RGB")
    target_np = np.array(target, dtype=np.float32) / 255.0
    pred_np = render_np.astype(np.float32) / 255.0
    diff = np.clip(np.abs(target_np - pred_np) * factor_diff, 0, 1)
    concat = np.concatenate([target_np, pred_np, diff], axis=1)
    Image.fromarray((concat * 255).astype(np.uint8)).save(salida_comparacion)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True, help="Ruta a checkpoint_final.pt o modelo_pruneado.pt")
    parser.add_argument("--frame", type=int, default=0, help="Frame a regenerar")
    parser.add_argument("--salida", default=None, help="Ruta PNG de salida")
    parser.add_argument("--comparacion", default=None, help="Ruta PNG para original | render | diff")
    parser.add_argument("--clip", default=None, help="Override del clip del checkpoint")
    parser.add_argument("--device", default=None, help="Override de device: cuda o cpu")
    parser.add_argument("--raiz_tesis_video", default=None, help="Ruta a tesis_2dgs_video")
    parser.add_argument("--ruta_cuda", default=None, help="Ruta a raster_cuda_new")
    args = parser.parse_args()

    raiz_tesis_video, _, ruta_exp09, ruta_cuda = resolver_rutas(
        raiz_tesis_video_arg=args.raiz_tesis_video,
        ruta_cuda_arg=args.ruta_cuda,
    )
    preparar_imports(raiz_tesis_video, ruta_exp09, ruta_cuda)
    importar_modulos_exp09()

    ruta_ckpt = os.path.abspath(args.checkpoint)
    if not os.path.isfile(ruta_ckpt):
        raise FileNotFoundError(f"No existe checkpoint: {ruta_ckpt}")

    ckpt = torch.load(ruta_ckpt, map_location="cpu")
    if "config" not in ckpt or "state_dict_coefs" not in ckpt:
        raise RuntimeError("El checkpoint no tiene config o state_dict_coefs")

    config = dict(ckpt["config"])
    if args.clip is not None:
        config["clip"] = args.clip

    device_str = args.device if args.device is not None else config.get("device", "cuda")
    device = elegir_device(device_str)

    clip = config["clip"]
    carpeta_clip = os.path.join(raiz_tesis_video, "clips", clip)
    n_frames = contar_frames(carpeta_clip)

    if args.frame < 0 or args.frame >= n_frames:
        raise ValueError(f"frame fuera de rango: {args.frame}, n_frames={n_frames}")

    H, W, target_path = leer_resolucion_frame(carpeta_clip, args.frame)

    grados_distintos = sorted(set(config["grados"].values()))
    matrices_base = {
        g: construir_matriz(config["base"], n_frames, g, device=device, dtype=torch.float32)
        for g in grados_distintos
    }

    modelo = construir_modelo_desde_checkpoint(ckpt, device, n_frames, H, W)

    with torch.no_grad():
        params_j = modelo.evaluar_en_frame(args.frame, matrices_base)
        render = rasterizar_segun_config(params_j, H, W, config).clamp(0, 1)

    render_np = (render.detach().cpu().numpy() * 255).astype(np.uint8)

    if args.salida is None:
        salida = os.path.join(os.path.dirname(ruta_ckpt), f"reconstruida_frame_{args.frame:04d}.png")
    else:
        salida = os.path.abspath(args.salida)

    os.makedirs(os.path.dirname(salida), exist_ok=True)
    Image.fromarray(render_np).save(salida)

    mse, mae, psnr = calcular_metricas_simples(render_np, target_path)

    if args.comparacion is None:
        salida_comparacion = os.path.join(os.path.dirname(ruta_ckpt), f"comparacion_frame_{args.frame:04d}.png")
    else:
        salida_comparacion = os.path.abspath(args.comparacion)

    guardar_comparacion(render_np, target_path, salida_comparacion)

    print("checkpoint usado:", ruta_ckpt)
    print("clip:", clip)
    print("carpeta_clip:", carpeta_clip)
    print("frame:", args.frame)
    print("resolucion:", f"{H}x{W}")
    print("ruta CUDA:", ruta_cuda)
    print("imagen guardada en:", salida)
    print("comparacion guardada en:", salida_comparacion)
    print(f"MSE={mse:.8f}  MAE={mae:.8f}  PSNR={psnr:.2f} dB")


if __name__ == "__main__":
    main()
