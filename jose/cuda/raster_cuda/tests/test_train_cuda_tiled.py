import torch
import torch.nn.functional as F

from rasterizador_cuda_tiled_autograd import rasterizar_un_frame_cuda_tiled


def crear_target(H, W, device):
    filas = torch.arange(H, device=device, dtype=torch.float32) + 0.5
    columnas = torch.arange(W, device=device, dtype=torch.float32) + 0.5
    rr, cc = torch.meshgrid(filas, columnas, indexing="ij")

    cx = H * 0.5
    cy = W * 0.5
    sigma = min(H, W) * 0.16

    dist2 = (rr - cx) ** 2 + (cc - cy) ** 2
    blob = torch.exp(-0.5 * dist2 / (sigma * sigma))

    target = torch.zeros(H, W, 3, device=device)
    target[:, :, 0] = blob
    target[:, :, 1] = 0.35 * blob
    target[:, :, 2] = 0.10 * blob

    return target.clamp(0, 1)


def main():
    torch.manual_seed(0)

    device = "cuda"
    H = 32
    W = 32
    N = 12
    iters = 200

    tile_size = 16
    k_sigma = 6.0

    target = crear_target(H, W, device)

    mu = torch.empty(N, 2, device=device)
    mu[:, 0] = torch.rand(N, device=device) * H
    mu[:, 1] = torch.rand(N, device=device) * W
    mu = torch.nn.Parameter(mu)

    scale_raw = torch.nn.Parameter(torch.zeros(N, 2, device=device) + 2.0)
    theta = torch.nn.Parameter(torch.zeros(N, device=device))
    opacity_raw = torch.nn.Parameter(torch.zeros(N, device=device) - 1.0)
    color_raw = torch.nn.Parameter(torch.randn(N, 3, device=device) * 0.2)

    depth = torch.linspace(0, 1, N, device=device)

    opt = torch.optim.Adam(
        [mu, scale_raw, theta, opacity_raw, color_raw],
        lr=0.03
    )

    print("=== Entrenamiento CUDA tiled diferenciable ===")
    print(f"H={H}, W={W}, N={N}, tile={tile_size}, k_sigma={k_sigma}")

    for it in range(iters):
        opt.zero_grad()

        scale = torch.exp(scale_raw.clamp(min=-1.0, max=4.0))
        opacity = torch.sigmoid(opacity_raw)
        color = torch.sigmoid(color_raw)

        params = {
            "mu": mu,
            "scale": scale,
            "theta": theta,
            "opacity": opacity,
            "color": color,
            "depth": depth,
        }

        render = rasterizar_un_frame_cuda_tiled(
            params,
            H,
            W,
            tile_size=tile_size,
            k_sigma=k_sigma
        )

        loss = F.l1_loss(render.clamp(0, 1), target)
        loss.backward()
        opt.step()

        if it == 0 or (it + 1) % 20 == 0:
            print(f"iter {it + 1:04d} | loss {loss.item():.6f}")

    print("listo")


if __name__ == "__main__":
    main()