"""
Loop de entrenamiento epoch-based con gradient accumulation.

Cambios de rendimiento:
- Flags para desactivar PSNR durante training.
- Flags para no guardar checkpoints intermedios ni verificacion visual.
- Verificacion visual usa el mismo rasterizador del config si se activa.
- Opcion frames_por_epoch para debug/entrenamiento rapido con subset temporal.
"""
import os
import time

import torch

from gs2d_video.core.perdidas import loss_render_frame, loss_smoothness
from gs2d_video.render.renderer import render_frame, loss_frame_cuda



def _psnr(a, b):
    mse = torch.mean((a - b) ** 2)
    if mse <= 0:
        return float("inf")
    return float(-10.0 * torch.log10(mse))


def _rasterizar_segun_config(params_j, H, W, config):
    return render_frame(params_j, H, W, config)

@torch.no_grad()
def _evaluar_psnr_promedio(modelo, frames, matrices_base, config):
    """PSNR promedio. Usa el rasterizador indicado por config."""
    n_frames, H, W, _ = frames.shape
    suma = 0.0
    for j in range(n_frames):
        params_j = modelo.evaluar_en_frame(j, matrices_base)
        r = _rasterizar_segun_config(params_j, H, W, config).clamp(0, 1)
        suma += _psnr(r, frames[j])
    return suma / n_frames


def _indices_epoch(n_frames, frames_por_epoch, device):
    """
    Devuelve los indices de frames a usar en un epoch.
    Si frames_por_epoch es None, usa todos los frames.
    """
    if frames_por_epoch is None:
        return list(range(n_frames))

    k = int(frames_por_epoch)
    if k <= 0 or k >= n_frames:
        return list(range(n_frames))

    # randperm en CPU para que .tolist() sea barato y no sincronice tanto.
    return torch.randperm(n_frames, device="cpu")[:k].tolist()


def entrenar_batch_full(modelo, frames, matrices_base, optimizer, config, carpeta_salida=None):
    """
    Entrena el modelo con gradient accumulation.

    Flags utiles en config:
        calcular_psnr_durante_training: bool
        guardar_checkpoints_intermedios: bool
        guardar_verificacion_visual: bool
        frames_por_epoch: int | null
        usar_loss_cuda: bool
        loss_cuda_tipo: "l1" | "mse"
        usar_scheduler: bool
        scheduler_factor: float
        scheduler_patience: int
        scheduler_min_lr: float
    """
    n_frames, H, W, _ = frames.shape

    # ============================================================
    # Config general
    # ============================================================
    n_epochs = int(config["n_epochs"])
    beta = float(config["beta_smoothness"])
    pesos_smooth = config.get("pesos_smoothness", None)
    chk_each = int(config.get("checkpoint_cada_n_epochs", 50))

    sub_batch_fr = int(config.get("sub_batch_frames", 1))
    early_plateau = config.get("early_stop_plateau", None)
    frames_por_epoch = config.get("frames_por_epoch", None)

    # ============================================================
    # Loss
    # ============================================================
    lambda_dssim = float(config.get("lambda_dssim", 0.2))

    usar_loss_cuda = bool(config.get("usar_loss_cuda", False))
    loss_cuda_tipo = config.get("loss_cuda_tipo", "l1")

    if usar_loss_cuda and lambda_dssim != 0.0:
        raise RuntimeError(
            "usar_loss_cuda=true solo soporta lambda_dssim=0.0 por ahora. "
            "Para DSSIM usa el camino normal de PyTorch."
        )

    if usar_loss_cuda:
        print(f"[trainer] usando loss CUDA fusionada tipo={loss_cuda_tipo}", flush=True)

    # ============================================================
    # Scheduler
    # ============================================================
    usar_scheduler_lento = bool(config.get("usar_scheduler_lento", False))
    scheduler_lento_window = int(config.get("scheduler_lento_window", 40))
    scheduler_lento_min_delta = float(config.get("scheduler_lento_min_delta", 3e-4))
    scheduler_lento_factor = float(config.get("scheduler_lento_factor", 0.5))
    scheduler_lento_min_lr = float(config.get("scheduler_lento_min_lr", 1e-6))
    scheduler_lento_cooldown = int(config.get("scheduler_lento_cooldown", scheduler_lento_window))

  
    ultimo_epoch_reduce_lr = -10**9

    if usar_scheduler_lento:
        print(
            f"[trainer] scheduler lento activado "
            f"(window={scheduler_lento_window}, "
            f"min_delta={scheduler_lento_min_delta}, "
            f"factor={scheduler_lento_factor}, "
            f"min_lr={scheduler_lento_min_lr})",
            flush=True,
        )

        

    # ============================================================
    # Renderer CUDA
    # ============================================================
    usar_cuda_conic = bool(config.get("usar_cuda_conic", False))
    usar_cuda_tiled = bool(config.get("usar_cuda_tiled", False))
    tile_size = int(config.get("cuda_tile_size", 16))
    k_sigma = float(config.get("cuda_k_sigma", 3.5))

    if usar_cuda_conic and usar_cuda_tiled:
        raise RuntimeError("No puedes usar usar_cuda_conic=true y usar_cuda_tiled=true al mismo tiempo")

    if not usar_cuda_tiled:
        raise RuntimeError("Este repo limpio solo soporta usar_cuda_tiled=true")

    print(
        f"[trainer] usando rasterizador CUDA tiled diferenciable "
        f"(tile={tile_size}, k_sigma={k_sigma})",
        flush=True,
    )

    if frames_por_epoch is not None:
        print(
            f"[trainer] modo frames_por_epoch={frames_por_epoch} "
            f"(subset temporal por epoch)",
            flush=True,
        )

    # ============================================================
    # Flags de salida/debug
    # ============================================================
    calcular_psnr = bool(config.get("calcular_psnr_durante_training", False))
    guardar_ckpts = bool(config.get("guardar_checkpoints_intermedios", False))
    guardar_verif = bool(config.get("guardar_verificacion_visual", False))

    # ============================================================
    # Historial
    # ============================================================
    losses_render = []
    losses_smooth = []
    losses_total = []
    if usar_scheduler_lento and len(losses_render) > scheduler_lento_window:
        loss_antes = losses_render[-scheduler_lento_window - 1]
        loss_ahora = losses_render[-1]
        mejora_ventana = loss_antes - loss_ahora

        paso_cooldown = (epoch + 1) - ultimo_epoch_reduce_lr >= scheduler_lento_cooldown

        if mejora_ventana < scheduler_lento_min_delta and paso_cooldown:
            for grupo in optimizer.param_groups:
                lr_actual = grupo["lr"]
                nuevo_lr = max(lr_actual * scheduler_lento_factor, scheduler_lento_min_lr)
                grupo["lr"] = nuevo_lr

            ultimo_epoch_reduce_lr = epoch + 1

            lrs_actuales = [g["lr"] for g in optimizer.param_groups]
            print(
                f"[trainer] scheduler lento redujo LR en epoch {epoch + 1}: "
                f"mejora_ventana={mejora_ventana:.6f} < {scheduler_lento_min_delta:.6f} | "
                f"lr_min={min(lrs_actuales):.2e} lr_max={max(lrs_actuales):.2e}",
                flush=True,
            )
    psnrs_chk = []
    tiempos = []

    plateau_count = 0
    mejor_loss = float("inf")
    t_inicio = time.time()

    # ============================================================
    # Helper para imprimir LR
    # ============================================================
    def _lr_info():
        lrs_actuales = [g["lr"] for g in optimizer.param_groups]
        return f"  lr_min={min(lrs_actuales):.2e} lr_max={max(lrs_actuales):.2e}"

    # ============================================================
    # Loop principal
    # ============================================================
    for epoch in range(n_epochs):
        t_epoch = time.time()
        optimizer.zero_grad()

        idx_epoch = _indices_epoch(n_frames, frames_por_epoch, frames.device)
        n_usados = len(idx_epoch)

        loss_render_epoch_total = 0.0

        # --------------------------------------------------------
        # Render + loss con gradient accumulation
        # --------------------------------------------------------
        for inicio in range(0, n_usados, sub_batch_fr):
            sub_indices = idx_epoch[inicio:inicio + sub_batch_fr]
            loss_sub = None

            for j in sub_indices:
                params_j = modelo.evaluar_en_frame(j, matrices_base)

                if usar_loss_cuda:
                    l_j = loss_frame_cuda(
                        params_j,
                        frames[j],
                        H,
                        W,
                        config,
                    )
                else:
                    render_j = _rasterizar_segun_config(
                        params_j,
                        H,
                        W,
                        config,
                    )
                    l_j = loss_render_frame(
                        render_j,
                        frames[j],
                        lambda_dssim=lambda_dssim,
                    )

                # Promedio sobre los frames realmente usados en este epoch.
                l_j = l_j / n_usados
                loss_sub = l_j if loss_sub is None else (loss_sub + l_j)

            if loss_sub is not None:
                loss_sub.backward()
                loss_render_epoch_total += float(loss_sub.detach().item()) * n_usados

        loss_render_avg = loss_render_epoch_total / max(1, n_usados)

        # --------------------------------------------------------
        # Smoothness
        # --------------------------------------------------------
        loss_smooth = loss_smoothness(modelo, pesos_por_param=pesos_smooth)
        loss_smooth_escalado = beta * loss_smooth

        # Para beta_smoothness=0, smoothness puede no tener gradiente util.
        if loss_smooth_escalado.requires_grad and beta != 0.0:
            loss_smooth_escalado.backward()

        # --------------------------------------------------------
        # Optimizer step
        # --------------------------------------------------------
        optimizer.step()

        # --------------------------------------------------------
        # Reporte numerico
        # --------------------------------------------------------
        loss_render_reportado = float(loss_render_avg)
        loss_smooth_reportado = float(loss_smooth.detach().item())
        loss_total_reportado = loss_render_reportado + beta * loss_smooth_reportado

        # Scheduler despues del optimizer.step.

        losses_render.append(loss_render_reportado)
        losses_smooth.append(loss_smooth_reportado)
        losses_total.append(loss_total_reportado)

        if usar_scheduler_lento and len(losses_render) > scheduler_lento_window:
            loss_antes = losses_render[-scheduler_lento_window - 1]
            loss_ahora = losses_render[-1]
            mejora_ventana = loss_antes - loss_ahora

            paso_cooldown = (epoch + 1) - ultimo_epoch_reduce_lr >= scheduler_lento_cooldown

            if mejora_ventana < scheduler_lento_min_delta and paso_cooldown:
                for grupo in optimizer.param_groups:
                    lr_actual = grupo["lr"]
                    nuevo_lr = max(lr_actual * scheduler_lento_factor, scheduler_lento_min_lr)
                    grupo["lr"] = nuevo_lr

                ultimo_epoch_reduce_lr = epoch + 1

                lrs_actuales = [g["lr"] for g in optimizer.param_groups]
                print(
                    f"[trainer] scheduler lento redujo LR en epoch {epoch + 1}: "
                    f"mejora_ventana={mejora_ventana:.6f} < {scheduler_lento_min_delta:.6f} | "
                    f"lr_min={min(lrs_actuales):.2e} lr_max={max(lrs_actuales):.2e}",
                    flush=True,
                )
                
        tiempos.append(time.time() - t_epoch)

        # ========================================================
        # Logs/checkpoints
        # ========================================================
        es_checkpoint = (
            (epoch + 1) % chk_each == 0
            or epoch == 0
            or epoch == n_epochs - 1
        )

        if es_checkpoint:
            if calcular_psnr:
                psnr_avg = _evaluar_psnr_promedio(
                    modelo,
                    frames,
                    matrices_base,
                    config,
                )
            else:
                psnr_avg = -1.0

            psnrs_chk.append((epoch, psnr_avg))

            t_corrido = time.time() - t_inicio
            t_promedio = t_corrido / (epoch + 1)
            eta = t_promedio * (n_epochs - epoch - 1)

            print(
                f"  epoch {epoch + 1:4d}/{n_epochs}  "
                f"loss_r={loss_render_reportado:.5f}  "
                f"loss_s={loss_smooth_reportado:.3e}  "
                f"PSNR_avg={psnr_avg:.2f}  "
                f"t_epoch={tiempos[-1]:.1f}s  "
                f"eta={eta / 60:.1f}min"
                f"{_lr_info()}",
                flush=True,
            )

            if carpeta_salida is not None and guardar_ckpts:
                torch.save({
                    "state_dict_coefs": modelo.state_dict_coefs(),
                    "optimizer_state": optimizer.state_dict(),
                    "epoch_completado": epoch + 1,
                    "config": config,
                }, os.path.join(carpeta_salida, f"checkpoint_epoch{epoch + 1:04d}.pt"))

            if carpeta_salida is not None and guardar_verif:
                _guardar_verificacion_visual(
                    modelo,
                    frames,
                    matrices_base,
                    carpeta_salida,
                    epoch + 1,
                    config,
                )

        else:
            log_cada = max(1, chk_each // 5)

            if (epoch + 1) % log_cada == 0:
                t_corrido = time.time() - t_inicio
                t_promedio = t_corrido / (epoch + 1)
                eta = t_promedio * (n_epochs - epoch - 1)

                print(
                    f"  epoch {epoch + 1:4d}/{n_epochs}  "
                    f"loss_r={loss_render_reportado:.5f}  "
                    f"loss_s={loss_smooth_reportado:.3e}  "
                    f"t_epoch={tiempos[-1]:.1f}s  "
                    f"eta={eta / 60:.1f}min"
                    f"{_lr_info()}",
                    flush=True,
                )

        # ========================================================
        # Early stop manual por plateau
        # ========================================================
        if early_plateau is not None:
            if loss_total_reportado < mejor_loss - 1e-6:
                mejor_loss = loss_total_reportado
                plateau_count = 0
            else:
                plateau_count += 1

            if plateau_count >= early_plateau:
                print(
                    f"  early stop en epoch {epoch + 1} "
                    f"(plateau de {early_plateau} epochs)",
                    flush=True,
                )
                break

    return {
        "losses_render": losses_render,
        "losses_smooth": losses_smooth,
        "losses_total": losses_total,
        "psnrs_chk": psnrs_chk,
        "tiempos_por_epoch": tiempos,
        "tiempo_total": time.time() - t_inicio,
    }

@torch.no_grad()
def _guardar_verificacion_visual(modelo, frames, matrices_base, carpeta, epoch, config):
    """Renderiza primer y ultimo frame usando el rasterizador del config."""
    from PIL import Image
    import numpy as np

    n_frames, H, W, _ = frames.shape
    sub = os.path.join(carpeta, "verificacion")
    os.makedirs(sub, exist_ok=True)

    for nombre, j in [("primer", 0), ("ultimo", n_frames - 1)]:
        params_j = modelo.evaluar_en_frame(j, matrices_base)
        r = _rasterizar_segun_config(params_j, H, W, config).clamp(0, 1)

        if torch.is_tensor(r):
            r = r.detach().cpu().numpy()

        Image.fromarray((r * 255).astype(np.uint8)).save(
            os.path.join(sub, f"epoch{epoch:04d}_{nombre}_frame.png")
        )
