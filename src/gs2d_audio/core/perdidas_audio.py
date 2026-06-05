import torch
import torch.nn.functional as F

try:
    from pytorch_msssim import ssim as _ssim_externo
    _USAR_MSSSIM = True
except Exception:
    _USAR_MSSSIM = False


_EPS = 1e-8


def _get_float(config, key, default=0.0):
    return float(config.get(key, default))


def _weighted_mean(diff_abs, weight=None):
    if weight is None:
        return diff_abs.mean()

    weight = weight.to(device=diff_abs.device, dtype=diff_abs.dtype)
    return (diff_abs * weight).sum() / weight.sum().clamp_min(_EPS)


def _aggregate_error(diff_abs, weight, config):
    """
    Equivalente al exponente_pixel del video, pero sobre celdas tiempo-frecuencia.
    diff_abs: [B, F]
    """
    if bool(config.get("usar_max_pixel", False)):
        if weight is None:
            return diff_abs.max()
        return (diff_abs * weight).max()

    p = float(config.get("exponente_pixel", 1.0))

    if p == 1.0:
        return _weighted_mean(diff_abs, weight)

    base = diff_abs.clamp_min(_EPS) if p != int(p) else diff_abs
    pow_diff = base ** p

    if weight is None:
        result = pow_diff.mean()
    else:
        weight = weight.to(device=pow_diff.device, dtype=pow_diff.dtype)
        result = (pow_diff * weight).sum() / weight.sum().clamp_min(_EPS)

    if bool(config.get("usar_pnorm_root", False)):
        result = result.clamp_min(_EPS) ** (1.0 / p)

    return result


def _motion_weight(target_batch, target_full, indices, config):
    """
    Peso por cambio temporal del espectrograma:
    motion[t, f] = abs(S[t, f] - S[t-1, f])
    """
    lambda_motion = _get_float(config, "lambda_motion", 0.0)
    if lambda_motion <= 0.0:
        return None

    if target_full is None or indices is None:
        if target_batch.shape[0] < 2:
            return None
        motion = torch.zeros_like(target_batch)
        motion[1:] = torch.abs(target_batch[1:] - target_batch[:-1])
    else:
        idx = indices.to(device=target_batch.device, dtype=torch.long)
        prev_idx = torch.clamp(idx - 1, min=0)
        prev = target_full[prev_idx].to(device=target_batch.device, dtype=target_batch.dtype)
        motion = torch.abs(target_batch - prev)
        motion = torch.where((idx == 0).view(-1, 1), torch.zeros_like(motion), motion)

    mean = motion.mean().detach()
    if mean <= _EPS:
        motion_norm = torch.zeros_like(motion)
    else:
        motion_norm = motion / mean.clamp_min(_EPS)

    clip_val = _get_float(config, "motion_clip", 5.0)
    motion_norm = motion_norm.clamp(0.0, clip_val)

    return (1.0 + lambda_motion * motion_norm).detach()


def _hard_weight(diff_abs, config):
    lambda_hard = _get_float(config, "lambda_hard", 0.0)
    if lambda_hard <= 0.0:
        return None

    err_mean = diff_abs.mean().detach()

    if err_mean <= _EPS:
        err_norm = torch.zeros_like(diff_abs)
    else:
        err_norm = diff_abs / err_mean.clamp_min(_EPS)

    clip_val = _get_float(config, "hard_clip", 5.0)
    err_norm = err_norm.clamp(0.0, clip_val)

    return (1.0 + lambda_hard * err_norm).detach()


def _loss_temporal(pred, target):
    if pred.shape[0] < 2:
        return pred.new_tensor(0.0)

    d_pred = pred[1:] - pred[:-1]
    d_gt = target[1:] - target[:-1]
    return torch.mean(torch.abs(d_pred - d_gt))


def _loss_freq_edge(pred, target):
    if pred.shape[1] < 2:
        return pred.new_tensor(0.0)

    d_pred = pred[:, 1:] - pred[:, :-1]
    d_gt = target[:, 1:] - target[:, :-1]
    return torch.mean(torch.abs(d_pred - d_gt))


def _loss_dssim_spectrogram(pred, target):
    """
    DSSIM sobre el chunk como imagen 2D [tiempo, frecuencia].
    pred/target: [B, F]
    """
    if pred.shape[0] < 11 or pred.shape[1] < 11:
        return pred.new_tensor(0.0)

    x = pred.clamp(0, 1).unsqueeze(0).unsqueeze(0)
    y = target.clamp(0, 1).unsqueeze(0).unsqueeze(0)

    if _USAR_MSSSIM:
        ssim_val = _ssim_externo(x, y, data_range=1.0, size_average=True, win_size=11)
        return (1.0 - ssim_val) / 2.0

    # Fallback simple si pytorch_msssim no esta disponible.
    mu_x = F.avg_pool2d(x, kernel_size=11, stride=1, padding=5)
    mu_y = F.avg_pool2d(y, kernel_size=11, stride=1, padding=5)

    sx = F.avg_pool2d(x * x, kernel_size=11, stride=1, padding=5) - mu_x * mu_x
    sy = F.avg_pool2d(y * y, kernel_size=11, stride=1, padding=5) - mu_y * mu_y
    sxy = F.avg_pool2d(x * y, kernel_size=11, stride=1, padding=5) - mu_x * mu_y

    c1 = 0.01 ** 2
    c2 = 0.03 ** 2

    ssim_map = ((2 * mu_x * mu_y + c1) * (2 * sxy + c2)) / (
        (mu_x * mu_x + mu_y * mu_y + c1) * (sx + sy + c2)
    )

    return (1.0 - ssim_map.mean()) / 2.0


def loss_audio_batch(pred, target, config, target_full=None, indices=None):
    """
    Loss para espectrograma log-magnitude.
    pred/target: [B, F]

    Tipos:
    - baseline: L1 + lambda_mse*MSE
    - motion: L1 ponderado por variacion temporal del target
    - hard: L1 ponderado por error alto
    - temporal: baseline + lambda_temporal*derivada temporal
    - freq_edge: baseline + lambda_freq_edge*derivada en frecuencia
    - dssim: mezcla baseline con DSSIM espectral
    - combo: activa motion/hard/temporal/freq_edge/dssim si sus lambdas > 0
    """
    tipo = str(config.get("tipo_loss", "baseline")).lower().strip()

    diff_abs = torch.abs(pred - target)

    weight = None

    if tipo in ("motion", "combo"):
        w_motion = _motion_weight(target, target_full, indices, config)
        if w_motion is not None:
            weight = w_motion if weight is None else weight * w_motion

    if tipo in ("hard", "combo"):
        w_hard = _hard_weight(diff_abs, config)
        if w_hard is not None:
            weight = w_hard if weight is None else weight * w_hard

    loss_l1 = _aggregate_error(diff_abs, weight, config)
    loss_mse = torch.mean((pred - target) ** 2)

    lambda_mse = _get_float(config, "lambda_mse", 0.25)
    loss = loss_l1 + lambda_mse * loss_mse

    if tipo in ("temporal", "combo"):
        lambda_temporal = _get_float(config, "lambda_temporal", 0.0)
        if lambda_temporal > 0.0:
            loss = loss + lambda_temporal * _loss_temporal(pred, target)

    if tipo in ("freq_edge", "combo"):
        lambda_freq_edge = _get_float(config, "lambda_freq_edge", 0.0)
        if lambda_freq_edge > 0.0:
            loss = loss + lambda_freq_edge * _loss_freq_edge(pred, target)

    if tipo in ("dssim", "combo"):
        lambda_dssim = _get_float(config, "lambda_dssim", 0.0)
        if lambda_dssim > 0.0:
            dssim = _loss_dssim_spectrogram(pred, target)
            loss = (1.0 - lambda_dssim) * loss + lambda_dssim * dssim

    return loss, {
        "loss_l1": loss_l1.detach(),
        "loss_mse": loss_mse.detach(),
    }


@torch.no_grad()
def metricas_logmag(pred_tf, target_tf):
    """
    pred_tf/target_tf: [T, F] en CPU o GPU, rango [0,1].
    Devuelve PSNR global y PSNR por tiempo.
    """
    pred = pred_tf.float()
    target = target_tf.float()

    mse_por_t = torch.mean((pred - target) ** 2, dim=1).clamp_min(1e-12)
    psnr_por_t = -10.0 * torch.log10(mse_por_t)

    mse_global = torch.mean((pred - target) ** 2).clamp_min(1e-12)
    psnr_global = -10.0 * torch.log10(mse_global)

    vals = psnr_por_t.detach().cpu()

    return {
        "mse_logmag": float(mse_global.detach().cpu()),
        "psnr_logmag_promedio": float(psnr_global.detach().cpu()),
        "psnr_logmag_min": float(vals.min()),
        "psnr_logmag_max": float(vals.max()),
        "psnr_logmag_p5": float(torch.quantile(vals, 0.05)),
        "psnr_logmag_std": float(vals.std(unbiased=False)),
        "psnr_logmag_por_t": [float(x) for x in vals],
    }
