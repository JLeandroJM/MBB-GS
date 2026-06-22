import math
import time
import torch
import raster_cuda


def calcular_tiles_tocados_torch(mu, scale, theta, depth, H, W, tile_size=16, k_sigma=3.5):
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
        counts,
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

    depth_scale = 1_000_000
    depth_q = torch.clamp((depth_norm * (depth_scale - 1)).to(torch.int64), 0, depth_scale - 1)
    keys = tile_ids * depth_scale + depth_q

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


def bench(fn, iters=100):
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        out = fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) / iters, out


def main():
    torch.manual_seed(7)
    device = "cuda"
    N = 3000
    H = 288
    W = 512
    tile_size = 16
    k_sigma = 4.0
    iters = 100

    mu = torch.empty(N, 2, device=device)
    mu[:, 0] = torch.rand(N, device=device) * H
    mu[:, 1] = torch.rand(N, device=device) * W
    scale = 2.0 + torch.rand(N, 2, device=device) * 5.0
    theta = (torch.rand(N, device=device) - 0.5) * 6.28318530718
    depth = torch.rand(N, device=device)

    # Warmup
    calcular_tiles_tocados_torch(mu, scale, theta, depth, H, W, tile_size, k_sigma)
    raster_cuda.preprocess_tiled(mu.contiguous(), scale.contiguous(), theta.contiguous(), depth.contiguous(), H, W, tile_size, k_sigma)

    t_torch, (ids_t, ranges_t) = bench(
        lambda: calcular_tiles_tocados_torch(mu, scale, theta, depth, H, W, tile_size, k_sigma),
        iters,
    )
    t_cuda, (ids_c, ranges_c) = bench(
        lambda: raster_cuda.preprocess_tiled(mu.contiguous(), scale.contiguous(), theta.contiguous(), depth.contiguous(), H, W, tile_size, k_sigma),
        iters,
    )

    same_count = ids_t.numel() == ids_c.numel()
    same_ranges = torch.equal(ranges_t, ranges_c)
    same_ids = torch.equal(ids_t, ids_c) if same_count else False

    print("=== preprocess tiled ===")
    print(f"N={N}, H={H}, W={W}, tile={tile_size}, k_sigma={k_sigma}")
    print(f"instancias torch: {ids_t.numel()}")
    print(f"instancias cuda : {ids_c.numel()}")
    print(f"torch preprocess: {t_torch * 1000:.4f} ms")
    print(f"cuda preprocess : {t_cuda * 1000:.4f} ms")
    print(f"speedup         : {t_torch / max(t_cuda, 1e-12):.2f}x")
    print(f"same_count      : {same_count}")
    print(f"same_ranges     : {same_ranges}")
    print(f"same_ids        : {same_ids}")

    if not same_ranges or not same_count:
        raise RuntimeError("preprocess CUDA no coincide con referencia")


if __name__ == "__main__":
    main()
