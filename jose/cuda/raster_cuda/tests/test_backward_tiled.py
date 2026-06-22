import math
import torch
import raster_cuda


def construir_conic(scale, theta):
    sx = scale[:, 0]
    sy = scale[:, 1]

    c = torch.cos(theta)
    s = torch.sin(theta)

    inv_sx2 = 1.0 / (sx * sx + 1e-8)
    inv_sy2 = 1.0 / (sy * sy + 1e-8)

    m00 = c * c * inv_sx2 + s * s * inv_sy2
    m01 = c * s * (inv_sx2 - inv_sy2)
    m11 = s * s * inv_sx2 + c * c * inv_sy2

    return torch.stack([m00, m01, m11], dim=1).contiguous()


def grad_conic_a_scale_theta(scale, theta, grad_conic):
    sx = scale[:, 0]
    sy = scale[:, 1]

    c = torch.cos(theta)
    s = torch.sin(theta)

    den_x = sx * sx + 1e-8
    den_y = sy * sy + 1e-8

    A = 1.0 / den_x
    B = 1.0 / den_y

    gm00 = grad_conic[:, 0]
    gm01 = grad_conic[:, 1]
    gm11 = grad_conic[:, 2]

    grad_A = gm00 * c * c + gm01 * c * s + gm11 * s * s
    grad_B = gm00 * s * s - gm01 * c * s + gm11 * c * c

    dA_dsx = -2.0 * sx / (den_x * den_x)
    dB_dsy = -2.0 * sy / (den_y * den_y)

    grad_sx = grad_A * dA_dsx
    grad_sy = grad_B * dB_dsy

    grad_scale = torch.stack([grad_sx, grad_sy], dim=1)

    dm00_dtheta = 2.0 * c * s * (B - A)
    dm01_dtheta = (c * c - s * s) * (A - B)
    dm11_dtheta = 2.0 * c * s * (A - B)

    grad_theta = gm00 * dm00_dtheta + gm01 * dm01_dtheta + gm11 * dm11_dtheta

    return grad_scale, grad_theta


def calcular_tiles_tocados_gpu(mu, scale, theta, depth, H, W, tile_size=16, k_sigma=3.5):
    device = mu.device
    N = mu.shape[0]

    tiles_x = math.ceil(W / tile_size)
    tiles_y = math.ceil(H / tile_size)
    total_tiles = tiles_x * tiles_y

    fila = mu[:, 0]
    col = mu[:, 1]

    sx = scale[:, 0]
    sy = scale[:, 1]

    c = torch.cos(theta)
    s = torch.sin(theta)

    extent_fila = k_sigma * torch.sqrt((c * sx) ** 2 + (s * sy) ** 2)
    extent_col = k_sigma * torch.sqrt((s * sx) ** 2 + (c * sy) ** 2)

    min_f = torch.floor(fila - extent_fila).clamp(0, H - 1).to(torch.int64)
    max_f = torch.ceil(fila + extent_fila).clamp(0, H - 1).to(torch.int64)
    min_c = torch.floor(col - extent_col).clamp(0, W - 1).to(torch.int64)
    max_c = torch.ceil(col + extent_col).clamp(0, W - 1).to(torch.int64)

    tile_y_min = torch.div(min_f, tile_size, rounding_mode="floor")
    tile_y_max = torch.div(max_f, tile_size, rounding_mode="floor")
    tile_x_min = torch.div(min_c, tile_size, rounding_mode="floor")
    tile_x_max = torch.div(max_c, tile_size, rounding_mode="floor")

    num_y = tile_y_max - tile_y_min + 1
    num_x = tile_x_max - tile_x_min + 1
    counts = num_y * num_x

    total_instancias = int(counts.sum().item())

    gaussian_ids = torch.repeat_interleave(
        torch.arange(N, device=device, dtype=torch.int64),
        counts
    )

    starts = torch.cumsum(counts, dim=0) - counts
    local_offsets = torch.arange(total_instancias, device=device, dtype=torch.int64) - starts[gaussian_ids]

    nx = num_x[gaussian_ids]
    local_y = torch.div(local_offsets, nx, rounding_mode="floor")
    local_x = local_offsets - local_y * nx

    tx = tile_x_min[gaussian_ids] + local_x
    ty = tile_y_min[gaussian_ids] + local_y

    tile_ids = ty * tiles_x + tx

    depth_vals = depth[gaussian_ids]
    d_min = depth_vals.min()
    d_max = depth_vals.max()
    depth_norm = (depth_vals - d_min) / (d_max - d_min + 1e-8)
    DEPTH_SCALE = 1_000_000

    depth_q = torch.clamp(
        (depth_norm * (DEPTH_SCALE - 1)).to(torch.int64),
        0,
        DEPTH_SCALE - 1
    )

    keys = tile_ids * DEPTH_SCALE + depth_q
    order = torch.argsort(keys)

    tile_ids = tile_ids[order].contiguous()
    gaussian_ids = gaussian_ids[order].contiguous()

    ranges = torch.full((total_tiles, 2), -1, device=device, dtype=torch.int64)

    if tile_ids.numel() > 0:
        is_start = torch.empty_like(tile_ids, dtype=torch.bool)
        is_start[0] = True
        is_start[1:] = tile_ids[1:] != tile_ids[:-1]

        starts_idx = torch.nonzero(is_start, as_tuple=False).flatten()
        unique_tiles = tile_ids[starts_idx]

        ends_idx = torch.empty_like(starts_idx)
        if starts_idx.numel() > 1:
            ends_idx[:-1] = starts_idx[1:]
        ends_idx[-1] = tile_ids.numel()

        ranges[unique_tiles, 0] = starts_idx
        ranges[unique_tiles, 1] = ends_idx

    return gaussian_ids.contiguous(), ranges.contiguous()


class RasterTiledCUDA(torch.autograd.Function):
    @staticmethod
    def forward(ctx, mu, scale, theta, opacity, color, depth, H, W, tile_size, k_sigma):
        mu_c = mu.contiguous()
        scale_c = scale.contiguous()
        theta_c = theta.contiguous()
        opacity_c = opacity.contiguous()
        color_c = color.contiguous()
        depth_c = depth.contiguous()

        conic = construir_conic(scale_c, theta_c)

        gaussian_ids, ranges = calcular_tiles_tocados_gpu(
            mu_c,
            scale_c,
            theta_c,
            depth_c,
            H,
            W,
            tile_size=tile_size,
            k_sigma=k_sigma
        )

        out = raster_cuda.forward_tiled(
            mu_c,
            conic,
            opacity_c,
            color_c,
            gaussian_ids,
            ranges,
            H,
            W,
            tile_size
        )

        ctx.save_for_backward(
            mu_c,
            scale_c,
            theta_c,
            conic,
            opacity_c,
            color_c,
            gaussian_ids,
            ranges
        )

        ctx.H = H
        ctx.W = W
        ctx.tile_size = tile_size

        return out

    @staticmethod
    def backward(ctx, grad_out):
        mu, scale, theta, conic, opacity, color, gaussian_ids, ranges = ctx.saved_tensors

        grad_mu, grad_conic, grad_opacity, grad_color = raster_cuda.backward_tiled(
            mu,
            conic,
            opacity,
            color,
            gaussian_ids,
            ranges,
            grad_out.contiguous(),
            ctx.H,
            ctx.W,
            ctx.tile_size
        )

        grad_scale, grad_theta = grad_conic_a_scale_theta(scale, theta, grad_conic)

        return (
            grad_mu,
            grad_scale,
            grad_theta,
            grad_opacity,
            grad_color,
            None,
            None,
            None,
            None,
            None
        )


def comparar(nombre, a, b):
    diff = (a - b).abs()
    print(nombre)
    print("  max_diff :", diff.max().item())
    print("  mean_diff:", diff.mean().item())


def main():
    torch.manual_seed(123)

    device = "cuda"
    H = 32
    W = 32
    N = 20
    tile_size = 16
    k_sigma = 20


    mu = torch.empty(N, 2, device=device)
    mu[:, 0] = torch.rand(N, device=device) * H
    mu[:, 1] = torch.rand(N, device=device) * W
    mu.requires_grad_(True)

    scale = torch.empty(N, 2, device=device)
    scale[:, 0] = 2.0 + torch.rand(N, device=device) * 3.0
    scale[:, 1] = 2.0 + torch.rand(N, device=device) * 3.0
    scale.requires_grad_(True)

    theta = ((torch.rand(N, device=device) - 0.5) * 6.28318530718).requires_grad_(True)
    opacity = (torch.rand(N, device=device) * 0.20).requires_grad_(True)
    color = torch.rand(N, 3, device=device).requires_grad_(True)

    depth = torch.rand(N, device=device)
    idx = torch.argsort(depth)

    mu2 = mu.detach().clone()[idx].requires_grad_(True)
    scale2 = scale.detach().clone()[idx].requires_grad_(True)
    theta2 = theta.detach().clone()[idx].requires_grad_(True)
    opacity2 = opacity.detach().clone()[idx].requires_grad_(True)
    color2 = color.detach().clone()[idx].requires_grad_(True)
    depth2 = depth.detach().clone()[idx]

    out_tiled = RasterTiledCUDA.apply(
        mu2,
        scale2,
        theta2,
        opacity2,
        color2,
        depth2,
        H,
        W,
        tile_size,
        k_sigma
    )

    conic2 = construir_conic(scale2, theta2)
    out_conic = raster_cuda.forward_conic(
        mu2.contiguous(),
        conic2.contiguous(),
        opacity2.contiguous(),
        color2.contiguous(),
        H,
        W
    )

    comparar("forward tiled vs conic", out_tiled, out_conic)

    grad_fake = torch.randn_like(out_tiled)

    loss_tiled = (out_tiled * grad_fake).sum()
    loss_tiled.backward()

    # Referencia: usar backward_conic ya validado
    grad_mu_c, grad_conic_c, grad_opacity_c, grad_color_c = raster_cuda.backward_conic(
        mu2.detach().contiguous(),
        conic2.detach().contiguous(),
        opacity2.detach().contiguous(),
        color2.detach().contiguous(),
        grad_fake.contiguous(),
        H,
        W
    )

    grad_scale_c, grad_theta_c = grad_conic_a_scale_theta(
        scale2.detach(),
        theta2.detach(),
        grad_conic_c
    )

    print("\n=== Gradientes tiled vs conic referencia ===")
    comparar("grad_mu", mu2.grad, grad_mu_c)
    comparar("grad_scale", scale2.grad, grad_scale_c)
    comparar("grad_theta", theta2.grad, grad_theta_c)
    comparar("grad_opacity", opacity2.grad, grad_opacity_c)
    comparar("grad_color", color2.grad, grad_color_c)


if __name__ == "__main__":
    main()