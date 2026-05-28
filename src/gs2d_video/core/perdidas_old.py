"""
Funciones de perdida.

loss_render_frame:
    (1 - lambda)*L1 + lambda*DSSIM.
    Si lambda_dssim <= 0, retorna L1 directamente para evitar el costo de SSIM.

loss_smoothness:
    penaliza coeficientes de orden alto con sum (a_k)^2 * k^2.
"""
import torch
import torch.nn.functional as F


try:
    from pytorch_msssim import ssim as _ssim_externo
    _USAR_MSSSIM = True
except Exception:
    _USAR_MSSSIM = False


def _kernel_gauss_2d(tamano=11, sigma=1.5, device="cpu", dtype=torch.float32):
    coords = torch.arange(tamano, dtype=dtype, device=device) - (tamano - 1) / 2.0
    g1 = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
    g1 = g1 / g1.sum()
    return g1[:, None] * g1[None, :]


def _dssim_propio(x_b, y_b):
    """DSSIM diferenciable usando un kernel 11x11. x_b, y_b: (B, C, H, W)."""
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
    ssim_map = ((2 * mu_xy + C1) * (2 * sxy + C2)) / ((mu_x2 + mu_y2 + C1) * (sx2 + sy2 + C2))
    return (1.0 - ssim_map.mean()) / 2.0


## ---------------------------
## Tomar la norma L2 en vez de la norma L1. ----------
## ---------------------------

def l_distance(img1, img2, p):
    diff = img1.float() - img2.float()
    return torch.norm(diff, p)

def loss_render_frame(render_hw3, target_hw3, lambda_dssim=0.2):
    """Loss para un frame. Si lambda_dssim <= 0, usa L1 puro y evita SSIM."""
    #l1 = torch.mean(torch.square(render_hw3 - target_hw3))
    l1 = torch.mean(torch.abs(render_hw3 - target_hw3))
    #l1 = l_distance(render_hw3 , target_hw3,1) ## l1
    #l1 = l_distance(render_hw3, target_hw3,2) ## l2


    if lambda_dssim <= 0.0:
        return l1

    x_b = render_hw3.permute(2, 0, 1).unsqueeze(0).clamp(0, 1)
    y_b = target_hw3.permute(2, 0, 1).unsqueeze(0).clamp(0, 1)

    if _USAR_MSSSIM:
        ssim_val = _ssim_externo(x_b, y_b, data_range=1.0, size_average=True, win_size=11)
        dssim = (1.0 - ssim_val) / 2.0
    else:
        dssim = _dssim_propio(x_b, y_b)

    return (1.0 - lambda_dssim) * l1 + lambda_dssim * dssim


def loss_render_batch(render_batch, frames_gt, lambda_dssim=0.2):
    """Loss promediado sobre un batch de frames."""
    l1 = torch.mean(torch.abs(render_batch - frames_gt))

    if lambda_dssim <= 0.0:
        return l1

    x_b = render_batch.permute(0, 3, 1, 2).clamp(0, 1)
    y_b = frames_gt.permute(0, 3, 1, 2).clamp(0, 1)

    if _USAR_MSSSIM:
        ssim_val = _ssim_externo(x_b, y_b, data_range=1.0, size_average=True, win_size=11)
        dssim = (1.0 - ssim_val) / 2.0
    else:
        dssim = _dssim_propio(x_b, y_b)

    return (1.0 - lambda_dssim) * l1 + lambda_dssim * dssim


def loss_smoothness(modelo, pesos_por_param=None):
    """Penaliza coeficientes de orden alto usando factores cacheados."""
    total = None
    pesos_por_param = pesos_por_param or {}

    for nombre, (a0, a_high, grado, _) in modelo.parametros_temporales().items():
        peso = pesos_por_param.get(nombre, 1.0)

        if peso == 0.0 or grado == 0:
            continue

        factor = modelo.smooth_factors[nombre]

        if factor.device != a_high.device or factor.dtype != a_high.dtype:
            factor = factor.to(device=a_high.device, dtype=a_high.dtype)

        while factor.dim() < a_high.dim():
            factor = factor.unsqueeze(0)

        contrib = (a_high ** 2 * factor).sum() * peso
        total = contrib if total is None else total + contrib

    if total is None:
        return torch.tensor(0.0, device=modelo.mu_a0.device)

    return total
