"""
Entry point de entrenamiento.

Version corregida para entrenamientos grandes:
- Opcion frames_en_cpu=true para mantener los frames en RAM como uint8.
- Opcion frames_en_gpu_uint8=true para mantener los frames en VRAM como uint8.
- Solo se mueve a GPU el frame_0 cuando se usa para inicializar color.
- Guarda checkpoint_final.pt inmediatamente despues del entrenamiento.
- Evita renderizar el clip completo dentro de train.py cuando frames_en_cpu=true
  y evitar_render_completo_en_train=true.
- Para configs grandes, renderiza luego desde checkpoint con:
  scripts/regenerar_clip_desde_checkpoint_streaming.py
"""

import argparse
import csv
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image

RAIZ = Path(__file__).resolve().parents[1]
SRC = RAIZ / "src"

if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from gs2d_video.core.bases import construir_matriz
from gs2d_video.core.modelo import GaussianasPolinomial2D
from gs2d_video.core.optimizador import construir_optimizador
from gs2d_video.core.pruning_post import prunear_post

from gs2d_video.training.trainer import entrenar_batch_full
from gs2d_video.io.video import extraer_frames_de_video

from gs2d_video.metrics.calidad import reporte_completo
from gs2d_video.metrics.compresion import reporte_compresion

from gs2d_video.render.renderer import render_frame, render_clip

from gs2d_video.viz.visualizaciones import (
    generar_trayectorias_png,
    generar_heatmap_opacity,
    generar_evolucion_parametros,
    generar_coeficientes_magnitudes,
)


# ============================================================
# Helpers generales
# ============================================================

def _fmt_metric(x, nd=4):
    if x is None:
        return "n/a"
    if isinstance(x, float) and np.isinf(x):
        return "inf"
    return f"{x:.{nd}f}"


def frame_a_float_device(frame, device):
    """
    Convierte un frame CPU/GPU a float32 en el device indicado.

    - CPU uint8 -> CUDA/CPU float32 en [0, 1]
    - CPU float -> CUDA/CPU float32
    - CUDA float32 -> igual
    """
    if frame.device.type == device.type and frame.dtype == torch.float32:
        return frame

    if frame.dtype == torch.uint8:
        return frame.to(device=device, non_blocking=True).float().div_(255.0)

    return frame.to(device=device, dtype=torch.float32, non_blocking=True)


def frames_a_float_device(frames, device):
    """
    Convierte un tensor completo de frames a float32 en device.
    Usar solo para resoluciones pequenas porque puede ocupar mucha VRAM.
    """
    if frames.device.type == device.type and frames.dtype == torch.float32:
        return frames

    if frames.dtype == torch.uint8:
        return frames.to(device=device, non_blocking=True).float().div_(255.0)

    return frames.to(device=device, dtype=torch.float32, non_blocking=True)


def frames_a_uint8_numpy(frames):
    """
    Devuelve frames como numpy uint8 con shape [T,H,W,3].
    Compatible con:
    - frames CPU uint8
    - frames CPU/GPU float en [0,1]
    """
    if frames.dtype == torch.uint8:
        return frames.detach().cpu().numpy()

    return (
        frames.detach()
        .clamp(0, 1)
        .mul(255)
        .to(torch.uint8)
        .cpu()
        .numpy()
    )


def rasterizar_segun_config(params_j, H, W, config):
    return render_frame(params_j, H, W, config)


@torch.no_grad()
def renderizar_clip(modelo, matrices_base, H, W, n_frames, config):
    """
    Ojo: render_clip apila todos los renders en GPU.
    Para configs grandes, usar regenerar_clip_desde_checkpoint_streaming.py.
    """
    return render_clip(modelo, matrices_base, H, W, n_frames, config)


def guardar_gif_desde_render(render_batch, frames, ruta_gif, paso=2, factor_diff=5.0, duracion=0.066):
    """Genera GIF original | reconstruido | diff usando renders ya calculados."""
    import imageio.v2 as imageio

    cuadros = []
    frames_np = frames_a_uint8_numpy(frames).astype(np.float32) / 255.0
    render_np = render_batch.detach().clamp(0, 1).cpu().numpy()

    for j in range(0, render_np.shape[0], paso):
        target = frames_np[j]
        render = render_np[j]
        diff = np.clip(np.abs(target - render) * factor_diff, 0, 1)
        concat = np.concatenate([target, render, diff], axis=1)
        cuadros.append((concat * 255).astype(np.uint8))

    imageio.mimsave(ruta_gif, cuadros, duration=duracion)


# ============================================================
# Carga de frames
# ============================================================

def cargar_clip(carpeta_clip, device, max_frames=None, frames_en_cpu=False, frames_en_gpu_uint8=False):
    archivos = sorted(
        f for f in os.listdir(carpeta_clip)
        if f.startswith("frame_") and f.endswith(".png")
    )

    if max_frames is not None:
        archivos = archivos[:max_frames]

    if not archivos:
        raise RuntimeError(f"no se encontraron frames PNG en {carpeta_clip}")

    img0 = Image.open(os.path.join(carpeta_clip, archivos[0])).convert("RGB")
    arr0 = np.asarray(img0, dtype=np.uint8)
    H, W, C = arr0.shape

    data = np.empty((len(archivos), H, W, C), dtype=np.uint8)
    data[0] = arr0

    for i, nombre in enumerate(archivos[1:], start=1):
        img = Image.open(os.path.join(carpeta_clip, nombre)).convert("RGB")
        data[i] = np.asarray(img, dtype=np.uint8)

    frames = torch.from_numpy(data)

    if frames_en_cpu and frames_en_gpu_uint8:
        raise ValueError("No uses frames_en_cpu=true y frames_en_gpu_uint8=true al mismo tiempo")

    if frames_en_cpu:
        # Se queda en RAM como uint8.
        # Esto reduce fuerte la VRAM para videos grandes.
        try:
            frames = frames.pin_memory()
        except RuntimeError:
            pass
        return frames

    if frames_en_gpu_uint8:
        # Modo rapido para GPUs con VRAM grande: todo el clip queda en VRAM
        # como uint8. En cada iteracion se convierte solo el frame usado a
        # float32 [0, 1], evitando transferencias CPU->GPU por frame.
        try:
            frames = frames.pin_memory()
        except RuntimeError:
            pass
        return frames.to(device=device, dtype=torch.uint8, non_blocking=True)

    # Modo antiguo: todo el video en GPU como float32.
    return frames.to(device=device, dtype=torch.float32, non_blocking=True).div_(255.0)


def elegir_device(device_str):
    if device_str == "cuda":
        if not torch.cuda.is_available():
            print("ERROR: device='cuda' pero torch.cuda.is_available()==False", file=sys.stderr, flush=True)
            print(f"  torch version: {torch.__version__}", file=sys.stderr)
            print(f"  cuda compiled: {torch.version.cuda}", file=sys.stderr)
            sys.exit(2)
        return torch.device("cuda")

    if device_str == "mps":
        if not torch.backends.mps.is_available():
            print("ERROR: device='mps' pero MPS no disponible", file=sys.stderr, flush=True)
            sys.exit(2)
        return torch.device("mps")

    if device_str == "cpu":
        return torch.device("cpu")

    raise ValueError(f"device desconocido: {device_str!r}")


def guardar_curva(valores, titulo, ylabel, ruta, xlabel="iteracion"):
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8, 4), dpi=100)
    ax.plot(valores)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(titulo)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(ruta)
    plt.close(fig)


def _reporte_vacio(n_frames):
    return {
        "psnr_por_frame": [None] * n_frames,
        "psnr_promedio": None,
        "ssim_por_frame": [None] * n_frames,
        "ssim_promedio": None,
        "lpips_por_frame": [None] * n_frames,
        "lpips_promedio": None,
    }


def _guardar_json_metricas(
    salida,
    nombre_exp,
    base,
    clip,
    n_frames,
    H,
    W,
    n_orig,
    n_final,
    rep_pre,
    rep_post,
):
    with open(os.path.join(salida, "metricas_calidad.json"), "w", encoding="utf-8") as f:
        json.dump({
            "exp": nombre_exp,
            "base": base,
            "clip": clip,
            "n_frames": n_frames,
            "resolucion": [H, W],
            "pre_pruning": {
                "N": n_orig,
                "psnr_promedio": rep_pre["psnr_promedio"],
                "ssim_promedio": rep_pre["ssim_promedio"],
                "lpips_promedio": rep_pre["lpips_promedio"],
                "psnr_por_frame": rep_pre["psnr_por_frame"],
                "ssim_por_frame": rep_pre["ssim_por_frame"],
            },
            "post_pruning": {
                "N": n_final,
                "psnr_promedio": rep_post["psnr_promedio"],
                "ssim_promedio": rep_post["ssim_promedio"],
                "lpips_promedio": rep_post["lpips_promedio"],
                "psnr_por_frame": rep_post["psnr_por_frame"],
                "ssim_por_frame": rep_post["ssim_por_frame"],
            },
        }, f, indent=2, default=str)


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="ruta al config json")
    args = parser.parse_args()

    aqui = os.path.dirname(os.path.abspath(__file__))
    raiz = os.path.abspath(os.path.join(aqui, ".."))

    with open(args.config, encoding="utf-8") as f:
        config = json.load(f)

    # Flags de debug/performance.
    calcular_metricas = bool(config.get("calcular_metricas", True))
    usar_ssim = bool(config.get("usar_ssim", True))
    usar_lpips = bool(config.get("usar_lpips", False))
    ejecutar_pruning = bool(config.get("ejecutar_pruning_post", True))
    calcular_compresion = bool(config.get("calcular_compresion", True))
    guardar_visualizaciones = bool(config.get("guardar_visualizaciones", True))
    guardar_gif = bool(config.get("guardar_gif", True))
    guardar_frames = bool(config.get("guardar_frames_rasterizados", True))

    frames_en_cpu = bool(config.get("frames_en_cpu", False))
    frames_en_gpu_uint8 = bool(config.get("frames_en_gpu_uint8", False))

    if frames_en_cpu and frames_en_gpu_uint8:
        raise ValueError("No uses frames_en_cpu=true y frames_en_gpu_uint8=true al mismo tiempo")

    # Por defecto, si frames_en_cpu=true evitamos renderizar todo el clip dentro de train.py.
    # Si frames_en_gpu_uint8=true, se permite render/metricas porque asumimos VRAM grande.
    # Para forzarlo manualmente: "evitar_render_completo_en_train": true/false
    evitar_render_completo = bool(config.get("evitar_render_completo_en_train", frames_en_cpu))

    seed = int(config.get("seed", 42))
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)

    device = elegir_device(config["device"])
    print(f"dispositivo: {device}", flush=True)
    print(f"base: {config['base']}", flush=True)

    # === preparar clip ======================================================
    clip = config["clip"]
    carpeta_clip = os.path.join(raiz, "data", "clips", clip)

    if config.get("video_mp4"):
        ruta_video = config["video_mp4"]
        if not os.path.isabs(ruta_video):
            ruta_video = os.path.join(raiz, ruta_video)

        res = config.get("resolucion_extraccion", [256, 256])
        H_ext, W_ext = int(res[0]), int(res[1])

        print(f"=== extrayendo frames del video '{ruta_video}' ===", flush=True)
        n_extraidos = extraer_frames_de_video(
            ruta_mp4=ruta_video,
            carpeta_salida=carpeta_clip,
            n_frames=config.get("n_frames_extraer"),
            fps=config.get("fps_extraccion"),
            H=H_ext,
            W=W_ext,
            forzar=bool(config.get("forzar_extraccion", False)),
        )
        print(f"  frames disponibles en {carpeta_clip}: {n_extraidos}", flush=True)

    if not os.path.isdir(carpeta_clip):
        raise FileNotFoundError(
            f"clip no encontrado: {carpeta_clip}.\n"
            f"  - O bien colocaste un mp4 en video/ y agregaste video_mp4 al config,\n"
            f"  - o bien existe ya una secuencia PNG en data/clips/{clip}/."
        )

    frames = cargar_clip(
        carpeta_clip,
        device,
        max_frames=config.get("max_frames"),
        frames_en_cpu=frames_en_cpu,
        frames_en_gpu_uint8=frames_en_gpu_uint8,
    )

    n_frames, H, W, _ = frames.shape
    print(f"clip={clip}  n_frames={n_frames}  resolucion={H}x{W}", flush=True)

    mb = frames.numel() * frames.element_size() / (1024 ** 2)
    if frames_en_cpu:
        print(f"[train] frames_en_cpu=true: frames en RAM como {frames.dtype}, aprox {mb:.1f} MiB", flush=True)
    elif frames_en_gpu_uint8:
        print(f"[train] frames_en_gpu_uint8=true: frames en {frames.device} como {frames.dtype}, aprox {mb:.1f} MiB", flush=True)
    else:
        print(f"[train] frames en {frames.device} como {frames.dtype}, aprox {mb:.1f} MiB", flush=True)

    if evitar_render_completo:
        if calcular_metricas or guardar_frames or guardar_gif or calcular_compresion:
            print(
                "[train] modo seguro memoria activo: se desactivan metricas/render/frames/GIF/compresion "
                "dentro de train.py para evitar OOM. Usa regenerar_clip_desde_checkpoint_streaming.py "
                "despues del entrenamiento.",
                flush=True,
            )

        calcular_metricas = False
        usar_ssim = False
        usar_lpips = False
        calcular_compresion = False
        guardar_gif = False
        guardar_frames = False

        # Pruning puede cambiar el modelo y a veces dispara renders despues.
        # Para configs grandes lo mas seguro es no hacer pruning dentro del train.
        if ejecutar_pruning and bool(config.get("desactivar_pruning_en_modo_seguro", True)):
            print("[train] modo seguro memoria: pruning post-training desactivado.", flush=True)
            ejecutar_pruning = False

    # === salida =============================================================
    nombre_exp = config["nombre_experimento"]
    salida = os.path.join(raiz, "outputs", "runs", nombre_exp, clip)
    os.makedirs(salida, exist_ok=True)

    # === modelo y matrices ==================================================
    base = config["base"]
    grados = config["grados"]
    grados_distintos = sorted(set(grados.values()))
    matrices_base = {
        g: construir_matriz(base, n_frames, g, device=device, dtype=torch.float32)
        for g in grados_distintos
    }
    print(f"matrices ({base}) construidas para grados: {grados_distintos}", flush=True)

    usar_frame0_color = bool(config.get("inicializar_color_desde_frame0", True))

    if usar_frame0_color:
        frame_init_color = frame_a_float_device(frames[0], device)
    else:
        frame_init_color = None

    modelo = GaussianasPolinomial2D(
        n_gaussianas=config["n_gaussianas_inicial"],
        n_frames=n_frames,
        grados=grados,
        base=base,
        H=H,
        W=W,
        device=device,
        escala_inicial_px=config["escala_inicial_px"],
        frame_0_imagen=frame_init_color,
        semilla=seed,
    )
    print(f"modelo: N={modelo.numero_gausianas()}  grados={grados}", flush=True)

    # Ya no se necesita mantener frame_init_color como variable grande.
    del frame_init_color

    optimizer = construir_optimizador(modelo, config["lrs"])

    # === cargar checkpoint inicial opcional ================================
    checkpoint_inicial = config.get("checkpoint_inicial", None)

    if checkpoint_inicial:
        ruta_ckpt = checkpoint_inicial
        if not os.path.isabs(ruta_ckpt):
            ruta_ckpt = os.path.join(raiz, ruta_ckpt)

        print("\n=== cargando checkpoint inicial ===", flush=True)
        print(f"  ruta: {ruta_ckpt}", flush=True)

        ckpt = torch.load(ruta_ckpt, map_location=device)
        sd = ckpt["state_dict_coefs"]

        with torch.no_grad():
            for nombre in ["mu", "opacity", "color", "scale", "theta", "depth"]:
                getattr(modelo, f"{nombre}_a0").copy_(sd[f"{nombre}_a0"].to(device))
                getattr(modelo, f"{nombre}_high").copy_(sd[f"{nombre}_high"].to(device))

        if bool(config.get("cargar_optimizer_state", False)) and "optimizer_state" in ckpt:
            optimizer.load_state_dict(ckpt["optimizer_state"])
            print("  optimizer_state cargado desde checkpoint", flush=True)
        else:
            print("  optimizer nuevo: se usan los learning rates del config actual", flush=True)

    # === entrenamiento ======================================================
    print(f"\n=== entrenamiento ({config['n_epochs']} epochs) ===", flush=True)
    historial = entrenar_batch_full(
        modelo,
        frames,
        matrices_base,
        optimizer,
        config,
        carpeta_salida=salida,
    )

    # === DEBUG: rangos de parametros despues del entrenamiento =============
    print("\n=== DEBUG rangos de parametros activados ===", flush=True)
    with torch.no_grad():
        params_0 = modelo.evaluar_en_frame(0, matrices_base)
        params_mid = modelo.evaluar_en_frame(n_frames // 2, matrices_base)
        params_last = modelo.evaluar_en_frame(n_frames - 1, matrices_base)

        for nombre, p in [
            ("frame 0", params_0),
            ("frame mid", params_mid),
            ("frame last", params_last),
        ]:
            print(f"\n  === {nombre} ===")
            print(
                f"    mu        : min={p['mu'].min().item():.1f}  "
                f"max={p['mu'].max().item():.1f}  "
                f"std={p['mu'].std().item():.2f}"
            )
            print(
                f"    scale (px): min={p['scale'].min().item():.2f}  "
                f"mean={p['scale'].mean().item():.2f}  "
                f"max={p['scale'].max().item():.2f}"
            )
            print(
                f"    opacity   : min={p['opacity'].min().item():.3f}  "
                f"mean={p['opacity'].mean().item():.3f}  "
                f"max={p['opacity'].max().item():.3f}"
            )
            print(
                f"    theta     : min={p['theta'].min().item():.2f}  "
                f"max={p['theta'].max().item():.2f}"
            )
            print(
                f"    color     : min={p['color'].min().item():.3f}  "
                f"max={p['color'].max().item():.3f}"
            )
        print("", flush=True)

    t_train = historial["tiempo_total"]
    print(f"entrenamiento listo en {t_train:.1f}s  ({t_train / 60:.1f}min)", flush=True)

    # === curvas =============================================================
    guardar_curva(
        historial["losses_render"],
        f"loss_render - {nombre_exp}",
        "loss render",
        os.path.join(salida, "loss_curve.png"),
        xlabel="epoch",
    )
    guardar_curva(
        historial["losses_smooth"],
        f"loss_smoothness - {nombre_exp}",
        "loss smooth",
        os.path.join(salida, "loss_smooth_curve.png"),
        xlabel="epoch",
    )

    # Curvas adicionales para experimentos max-aware (mean RAW por epoch y MAX RAW por epoch).
    raw_mean = historial.get("losses_render_raw_mean")
    raw_max = historial.get("losses_render_raw_max")
    if raw_mean is not None and raw_max is not None:
        guardar_curva(
            raw_mean,
            f"loss_r_mean (raw) - {nombre_exp}",
            "loss render mean raw",
            os.path.join(salida, "loss_raw_mean_curve.png"),
            xlabel="epoch",
        )
        guardar_curva(
            raw_max,
            f"loss_r_max (peor frame del epoch) - {nombre_exp}",
            "loss render max raw",
            os.path.join(salida, "loss_raw_max_curve.png"),
            xlabel="epoch",
        )

    with open(os.path.join(salida, "log_entrenamiento.csv"), "w", newline="", encoding="utf-8") as f:
        w_csv = csv.writer(f)
        w_csv.writerow([
            "epoch",
            "loss_render",
            "loss_render_raw_mean",
            "loss_render_raw_max",
            "loss_smooth",
            "tiempo_seg",
        ])
        n_epochs_log = len(historial["losses_render"])
        for i in range(n_epochs_log):
            lr_ = historial["losses_render"][i]
            rm_ = raw_mean[i] if raw_mean is not None else lr_
            rx_ = raw_max[i] if raw_max is not None else lr_
            ls_ = historial["losses_smooth"][i]
            ts_ = historial["tiempos_por_epoch"][i]
            w_csv.writerow([i + 1, f"{lr_:.6f}", f"{rm_:.6f}", f"{rx_:.6f}", f"{ls_:.6f}", f"{ts_:.3f}"])

    # Guardar checkpoint inmediatamente despues del entrenamiento.
    torch.save({
        "state_dict_coefs": modelo.state_dict_coefs(),
        "config": config,
    }, os.path.join(salida, "checkpoint_final.pt"))

    print(f"\ncheckpoint final guardado en: {os.path.join(salida, 'checkpoint_final.pt')}", flush=True)

    # === metricas pre-pruning ==============================================
    render_pre = None

    if calcular_metricas:
        print("\n=== metricas pre-pruning ===", flush=True)

        render_pre = renderizar_clip(modelo, matrices_base, H, W, n_frames, config)

        frames_metricas = frames_a_float_device(frames, device)

        rep_pre = reporte_completo(
            render_pre,
            frames_metricas,
            device=device,
            usar_ssim=usar_ssim,
            usar_lpips=usar_lpips,
        )

        print(
            f"  PSNR={_fmt_metric(rep_pre['psnr_promedio'], 2)}  "
            f"SSIM={_fmt_metric(rep_pre['ssim_promedio'], 4)}  "
            f"LPIPS={_fmt_metric(rep_pre['lpips_promedio'], 4)}",
            flush=True,
        )
    else:
        print("\n=== metricas pre-pruning desactivadas ===", flush=True)
        rep_pre = _reporte_vacio(n_frames)

    # === pruning post-training =============================================
    n_orig = modelo.numero_gausianas()

    if ejecutar_pruning:
        print("\n=== pruning post-training ===", flush=True)
        n_orig, n_final, _ = prunear_post(
            modelo,
            base,
            umbral=float(config.get("umbral_pruning_post", 0.05)),
            n_samples=int(config.get("pruning_n_samples", 200)),
        )
        print(f"  N: {n_orig} -> {n_final}", flush=True)
    else:
        n_final = n_orig
        print("\n=== pruning post-training desactivado ===", flush=True)
        print(f"  N: {n_orig} -> {n_final}", flush=True)

    # === render post-pruning ===============================================
    render_post = None
    reuse_umbral = config.get("reutilizar_render_pre_si_pruning_menor_pct", None)

    puede_reusar = False

    if render_pre is not None and reuse_umbral is not None and n_orig > 0:
        pct_eliminado = (n_orig - n_final) / n_orig
        puede_reusar = pct_eliminado <= float(reuse_umbral)

    if puede_reusar:
        print("\n=== render post-pruning reutilizado desde pre-pruning ===", flush=True)
        render_post = render_pre
    else:
        necesita_render_post = calcular_metricas or guardar_frames or guardar_gif or calcular_compresion

        if necesita_render_post:
            print("\n=== render post-pruning ===", flush=True)
            render_post = renderizar_clip(modelo, matrices_base, H, W, n_frames, config)

    if render_post is not None:
        with torch.no_grad():
            d_0_mid = torch.mean(torch.abs(render_post[0] - render_post[n_frames // 2])).item()
            d_mid_last = torch.mean(torch.abs(render_post[n_frames // 2] - render_post[-1])).item()
            d_0_last = torch.mean(torch.abs(render_post[0] - render_post[-1])).item()

        print("\n=== debug movimiento render ===", flush=True)
        print(f"  diff frame0 vs mid  = {d_0_mid:.6f}", flush=True)
        print(f"  diff mid vs last    = {d_mid_last:.6f}", flush=True)
        print(f"  diff frame0 vs last = {d_0_last:.6f}", flush=True)

    if calcular_metricas and render_post is not None:
        frames_metricas = frames_a_float_device(frames, device)

        rep_post = reporte_completo(
            render_post,
            frames_metricas,
            device=device,
            usar_ssim=usar_ssim,
            usar_lpips=usar_lpips,
        )

        print(
            f"  PSNR_post={_fmt_metric(rep_post['psnr_promedio'], 2)}  "
            f"SSIM_post={_fmt_metric(rep_post['ssim_promedio'], 4)}  "
            f"LPIPS_post={_fmt_metric(rep_post['lpips_promedio'], 4)}",
            flush=True,
        )
    else:
        rep_post = _reporte_vacio(n_frames)

    torch.save({
        "state_dict_coefs": modelo.state_dict_coefs(),
        "config": config,
        "metricas_pre": rep_pre,
        "metricas_post": rep_post,
    }, os.path.join(salida, "modelo_pruneado.pt"))

    # === guardar frames rasterizados =======================================
    render_np = None

    if guardar_frames and render_post is not None:
        carpeta_frames_recon = os.path.join(salida, "frames_rasterizados")
        os.makedirs(carpeta_frames_recon, exist_ok=True)

        render_np = render_post.detach().clamp(0, 1).cpu().numpy()

        for j in range(n_frames):
            Image.fromarray((render_np[j] * 255).astype(np.uint8)).save(
                os.path.join(carpeta_frames_recon, f"frame_{j:04d}.png")
            )

    # === metricas de calidad JSON ==========================================
    _guardar_json_metricas(
        salida=salida,
        nombre_exp=nombre_exp,
        base=base,
        clip=clip,
        n_frames=n_frames,
        H=H,
        W=W,
        n_orig=n_orig,
        n_final=n_final,
        rep_pre=rep_pre,
        rep_post=rep_post,
    )

    # === metricas de compresion ============================================
    if calcular_compresion:
        print("\n=== metricas de compresion ===", flush=True)

        if render_post is None:
            print("  omitidas: no hay render_post disponible", flush=True)
        else:
            print("  convirtiendo frames a uint8 de forma segura...", flush=True)

            frames_np = frames_a_uint8_numpy(frames)

            render_np_u8 = (
                render_post
                .detach()
                .clamp(0, 1)
                .mul(255)
                .to(torch.uint8)
                .cpu()
                .numpy()
            )

            rep_comp = reporte_compresion(
                modelo,
                render_np_u8,
                frames_np,
                ruta_video_original=carpeta_clip,
                calidades_avif=tuple(config.get("calidades_avif", [80, 95])),
                carpeta_avif_originales=os.path.join(salida, "frames_originales_avif"),
                carpeta_avif_rasterizados=os.path.join(salida, "frames_rasterizados_avif"),
            )

            print(
                f"  bytes_modelo = {rep_comp['tamano_modelo_bytes']}  "
                f"({rep_comp['tamano_modelo_kb']:.1f} KiB)",
                flush=True,
            )
            print(f"  bytes_video_original = {rep_comp['tamano_video_original_bytes']}", flush=True)

            if rep_comp.get("avif_disponible", False):
                for q in config.get("calidades_avif", [80, 95]):
                    sz_o = rep_comp["avif_originales_por_calidad"][q]["total_bytes"]
                    sz_r = rep_comp["avif_rasterizados_por_calidad"][q]["total_bytes"]
                    print(
                        f"  AVIF q={q}: originales={sz_o} bytes  "
                        f"rasterizados={sz_r} bytes",
                        flush=True,
                    )
            else:
                print("  AVIF deshabilitado.", flush=True)

            with open(os.path.join(salida, "metricas_compresion.json"), "w", encoding="utf-8") as f:
                json.dump(rep_comp, f, indent=2, default=str)
    else:
        print("\n=== metricas de compresion desactivadas ===", flush=True)

    # === visualizaciones ===================================================
    if guardar_visualizaciones:
        print("\n=== visualizaciones ===", flush=True)

        frame0_viz = frame_a_float_device(frames[0], device)

        generar_trayectorias_png(
            modelo,
            frame0_viz,
            matrices_base,
            os.path.join(salida, "trayectorias.png"),
        )
        generar_heatmap_opacity(
            modelo,
            matrices_base,
            os.path.join(salida, "heatmap_opacity_temporal.png"),
        )
        generar_evolucion_parametros(
            modelo,
            matrices_base,
            os.path.join(salida, "evolucion_parametros.png"),
        )
        generar_coeficientes_magnitudes(
            modelo,
            os.path.join(salida, "coeficientes_magnitudes.png"),
        )
    else:
        print("\n=== visualizaciones desactivadas ===", flush=True)

    if guardar_gif:
        if render_post is None:
            print("\n=== GIF omitido: no hay render_post disponible ===", flush=True)
        else:
            print("\n=== generando GIF desde render_post ===", flush=True)
            guardar_gif_desde_render(
                render_post,
                frames,
                os.path.join(salida, "reconstruccion_vs_original.gif"),
                paso=int(config.get("gif_paso", 2)),
                factor_diff=float(config.get("gif_factor_diff", 5.0)),
                duracion=float(config.get("gif_duracion", 0.066)),
            )
    else:
        print("\n=== GIF desactivado ===", flush=True)

    print(f"\nlisto. resultados en: {salida}", flush=True)


if __name__ == "__main__":
    main()
