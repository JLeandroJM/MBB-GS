import os
import sys
from pathlib import Path

import torch


# ============================================================
# Paths del proyecto
# Este archivo esta en:
# Gaussian2D_Optimized/cuda/raster_cuda/tests/test_cuda_fused_loss.py
# ============================================================

ROOT = Path(__file__).resolve().parents[3]
SRC = ROOT / "src"
CUDA_DIR = ROOT / "cuda" / "raster_cuda"

os.environ["RUTA_RASTER_CUDA"] = str(CUDA_DIR)

if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

if str(CUDA_DIR) not in sys.path:
    sys.path.insert(0, str(CUDA_DIR))


from gs2d_video.render.cuda_tiled import (
    rasterizar_un_frame_cuda_tiled,
    loss_un_frame_cuda_tiled,
)


def crear_params_random(N, H, W, device):
    torch.manual_seed(123)

    mu = torch.empty(N, 2, device=device).uniform_(0, 1)
    mu[:, 0] *= H
    mu[:, 1] *= W

    scale = torch.empty(N, 2, device=device).uniform_(2.0, 8.0)
    theta = torch.empty(N, device=device).uniform_(-0.5, 0.5)
    opacity = torch.empty(N, device=device).uniform_(0.05, 0.8)
    color = torch.empty(N, 3, device=device).uniform_(0.0, 1.0)
    depth = torch.empty(N, device=device).uniform_(0.0, 1.0)

    mu.requires_grad_(True)
    scale.requires_grad_(True)
    theta.requires_grad_(True)
    opacity.requires_grad_(True)
    color.requires_grad_(True)

    return {
        "mu": mu,
        "scale": scale,
        "theta": theta,
        "opacity": opacity,
        "color": color,
        "depth": depth,
    }


def clonar_params(params):
    out = {}

    for k, v in params.items():
        c = v.detach().clone()

        if k != "depth":
            c.requires_grad_(True)

        out[k] = c

    return out


def obtener_grads(params):
    return {
        "mu": params["mu"].grad.detach().clone(),
        "scale": params["scale"].grad.detach().clone(),
        "theta": params["theta"].grad.detach().clone(),
        "opacity": params["opacity"].grad.detach().clone(),
        "color": params["color"].grad.detach().clone(),
    }


def comparar_tensor(nombre, a, b, atol=2e-3, rtol=2e-3):
    max_abs = (a - b).abs().max().item()
    mean_abs = (a - b).abs().mean().item()
    ok = torch.allclose(a, b, atol=atol, rtol=rtol)

    print(
        f"{nombre:14s} max_abs_diff={max_abs:.6e}  "
        f"mean_abs_diff={mean_abs:.6e}  ok={ok}",
        flush=True,
    )

    if not ok:
        raise AssertionError(f"{nombre} no coincide. max_abs_diff={max_abs}")


def test_cuda_fused_l1_loss_vs_pytorch_l1():
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA no disponible")

    device = torch.device("cuda")

    H = 96
    W = 128
    N = 300
    tile_size = 16
    k_sigma = 4.0

    torch.manual_seed(999)
    target = torch.rand(H, W, 3, device=device, dtype=torch.float32)

    params_base = crear_params_random(N, H, W, device)

    # ========================================================
    # Ruta vieja:
    # render CUDA tiled + loss L1 en PyTorch
    # ========================================================
    print("\n=== ruta vieja: CUDA render + PyTorch L1 ===", flush=True)

    params_old = clonar_params(params_base)

    render_old = rasterizar_un_frame_cuda_tiled(
        params_old,
        H,
        W,
        tile_size=tile_size,
        k_sigma=k_sigma,
    )

    loss_old = torch.mean(torch.abs(render_old - target))
    loss_old.backward()

    grads_old = obtener_grads(params_old)

    # ========================================================
    # Ruta nueva:
    # render + loss L1 fusionada en CUDA
    # ========================================================
    print("\n=== ruta nueva: CUDA render + CUDA fused L1 ===", flush=True)

    params_new = clonar_params(params_base)

    loss_new = loss_un_frame_cuda_tiled(
        params_new,
        target,
        H,
        W,
        tile_size=tile_size,
        k_sigma=k_sigma,
        loss_type="l1",
    )

    loss_new.backward()

    grads_new = obtener_grads(params_new)

    torch.cuda.synchronize()

    # ========================================================
    # Comparacion
    # ========================================================
    print("\n=== comparacion loss ===", flush=True)
    print(f"loss_old = {loss_old.item():.8f}", flush=True)
    print(f"loss_new = {loss_new.item():.8f}", flush=True)
    print(f"diff     = {abs(loss_old.item() - loss_new.item()):.8e}", flush=True)

    assert abs(loss_old.item() - loss_new.item()) < 1e-4

    print("\n=== comparacion gradientes ===", flush=True)
    comparar_tensor("grad_mu", grads_old["mu"], grads_new["mu"])
    comparar_tensor("grad_scale", grads_old["scale"], grads_new["scale"])
    comparar_tensor("grad_theta", grads_old["theta"], grads_new["theta"])
    comparar_tensor("grad_opacity", grads_old["opacity"], grads_new["opacity"])
    comparar_tensor("grad_color", grads_old["color"], grads_new["color"])

    print("\nOK: CUDA fused L1 loss coincide con render CUDA + PyTorch L1", flush=True)


if __name__ == "__main__":
    print("=== iniciando test CUDA fused loss ===", flush=True)
    test_cuda_fused_l1_loss_vs_pytorch_l1()
    print("=== test finalizado OK ===", flush=True)