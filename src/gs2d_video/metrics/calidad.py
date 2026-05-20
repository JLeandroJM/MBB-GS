"""
Metricas de calidad por frame y agregadas.

PSNR  : log10 del MSE inverso. data_range = 1.0.
SSIM  : pytorch_msssim si esta, sino implementacion propia (ventana 11).
LPIPS : paquete lpips con backbone alex. Se puede desactivar desde config.

Modo debug recomendado:
    reporte_completo(..., usar_ssim=False, usar_lpips=False)

Modo final recomendado:
    reporte_completo(..., usar_ssim=True, usar_lpips=True)
"""
import numpy as np
import torch
import torch.nn.functional as F


# pytorch-msssim opcional
try:
    from pytorch_msssim import ssim as _ssim_externo
    _USAR_MSSSIM = True
except Exception:
    _USAR_MSSSIM = False


# LPIPS opcional, instanciado bajo demanda
_LPIPS_FN = None
_LPIPS_INTENTADO = False


def _obtener_lpips(device):
    global _LPIPS_FN, _LPIPS_INTENTADO
    if not _LPIPS_INTENTADO:
        _LPIPS_INTENTADO = True
        try:
            import lpips
            _LPIPS_FN = lpips.LPIPS(net="alex", verbose=False).to(device).eval()
        except Exception as e:
            print(f"[metricas_calidad] lpips no disponible ({e}), se reportara None", flush=True)
            _LPIPS_FN = None
    return _LPIPS_FN


def calcular_psnr(render, gt):
    """render, gt: (H, W, 3) en [0, 1]."""
    mse = torch.mean((render - gt) ** 2)
    if mse <= 0:
        return float("inf")
    return float(-10.0 * torch.log10(mse))


def _kernel_gauss_2d(tamano=11, sigma=1.5, device="cpu", dtype=torch.float32):
    coords = torch.arange(tamano, dtype=dtype, device=device) - (tamano - 1) / 2.0
    g1 = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
    g1 = g1 / g1.sum()
    return g1[:, None] * g1[None, :]


def calcular_ssim(render, gt):
    """render, gt: (H, W, 3) en [0, 1]. Devuelve float."""
    x_b = render.permute(2, 0, 1).unsqueeze(0).clamp(0, 1)
    y_b = gt.permute(2, 0, 1).unsqueeze(0).clamp(0, 1)

    if _USAR_MSSSIM:
        return float(_ssim_externo(x_b, y_b, data_range=1.0, size_average=True, win_size=11))

    C = x_b.shape[1]
    device = x_b.device
    kernel = _kernel_gauss_2d(11, 1.5, device, x_b.dtype).expand(C, 1, 11, 11)

    mu_x = F.conv2d(x_b, kernel, padding=5, groups=C)
    mu_y = F.conv2d(y_b, kernel, padding=5, groups=C)
    mu_x2 = mu_x * mu_x
    mu_y2 = mu_y * mu_y
    mu_xy = mu_x * mu_y
    sx2 = F.conv2d(x_b * x_b, kernel, padding=5, groups=C) - mu_x2
    sy2 = F.conv2d(y_b * y_b, kernel, padding=5, groups=C) - mu_y2
    sxy = F.conv2d(x_b * y_b, kernel, padding=5, groups=C) - mu_xy

    C1, C2 = 0.01 ** 2, 0.03 ** 2
    num = (2 * mu_xy + C1) * (2 * sxy + C2)
    den = (mu_x2 + mu_y2 + C1) * (sx2 + sy2 + C2)
    return float((num / den).mean())


def calcular_lpips(render, gt, device=None):
    """render, gt: (H, W, 3) en [0, 1]. Devuelve float o None."""
    if device is None:
        device = render.device

    fn = _obtener_lpips(device)
    if fn is None:
        return None

    x_b = (render.permute(2, 0, 1).unsqueeze(0).clamp(0, 1) * 2.0 - 1.0)
    y_b = (gt.permute(2, 0, 1).unsqueeze(0).clamp(0, 1) * 2.0 - 1.0)

    param_device = next(fn.parameters()).device
    x_b = x_b.to(param_device)
    y_b = y_b.to(param_device)

    with torch.no_grad():
        return float(fn(x_b, y_b).item())


def _promedio_o_none(valores):
    validos = [v for v in valores if v is not None]
    return float(np.mean(validos)) if validos else None


@torch.no_grad()
def reporte_completo(render_batch, frames_gt, device=None, usar_ssim=True, usar_lpips=False):
    """
    Calcula metricas por frame y promedios.

    Args:
        render_batch : (n_frames, H, W, 3) en [0, 1]
        frames_gt    : (n_frames, H, W, 3) en [0, 1]
        usar_ssim    : si False, no calcula SSIM. Util para debug rapido.
        usar_lpips   : si False, no carga AlexNet/LPIPS. Util para debug rapido.
    """
    n_frames = render_batch.shape[0]
    psnrs, ssims, lpipss = [], [], []

    for j in range(n_frames):
        psnrs.append(calcular_psnr(render_batch[j], frames_gt[j]))

        if usar_ssim:
            ssims.append(calcular_ssim(render_batch[j], frames_gt[j]))
        else:
            ssims.append(None)

        if usar_lpips:
            lpipss.append(calcular_lpips(render_batch[j], frames_gt[j], device=device))
        else:
            lpipss.append(None)

    return {
        "psnr_por_frame": psnrs,
        "psnr_promedio": float(np.mean(psnrs)),
        "ssim_por_frame": ssims,
        "ssim_promedio": _promedio_o_none(ssims),
        "lpips_por_frame": lpipss,
        "lpips_promedio": _promedio_o_none(lpipss),
    }
