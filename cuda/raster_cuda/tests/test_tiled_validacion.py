import os
import sys
import time
import math
import argparse

import torch
import matplotlib.pyplot as plt
import numpy as np

import raster_cuda


RAIZ = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
EXP09 = os.path.join(
    RAIZ,
    "tesis_2dgs_video",
    "experimentos",
    "exp09_batchfull_comparacion_bases"
)
sys.path.insert(0, EXP09)

from rasterizador import rasterizar_un_frame


def medir_tiempo_cuda(fn, n_warmup=10, n_iters=100):
    for _ in range(n_warmup):
        _ = fn()

    torch.cuda.synchronize()
    t0 = time.perf_counter()

    for _ in range(n_iters):
        _ = fn()

    torch.cuda.synchronize()
    t1 = time.perf_counter()

    return (t1 - t0) / n_iters


def crear_gaussianas_aleatorias(N, H, W, device, seed=0):
    torch.manual_seed(seed)

    mu = torch.empty(N, 2, device=device, dtype=torch.float32)
    mu[:, 0] = torch.rand(N, device=device) * H
    mu[:, 1] = torch.rand(N, device=device) * W

    scale = torch.empty(N, 2, device=device, dtype=torch.float32)
    scale[:, 0] = 2.0 + torch.rand(N, device=device) * 18.0
    scale[:, 1] = 2.0 + torch.rand(N, device=device) * 18.0

    theta = (torch.rand(N, device=device, dtype=torch.float32) - 0.5) * 6.28318530718
    opacity = torch.rand(N, device=device, dtype=torch.float32) * 0.8
    color = torch.rand(N, 3, device=device, dtype=torch.float32)
    depth = torch.rand(N, device=device, dtype=torch.float32)

    idx = torch.argsort(depth)

    return {
        "mu": mu[idx].contiguous(),
        "scale": scale[idx].contiguous(),
        "theta": theta[idx].contiguous(),
        "opacity": opacity[idx].contiguous(),
        "color": color[idx].contiguous(),
        "depth": depth[idx].contiguous(),
    }


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


def calcular_tiles_tocados(mu, scale, theta, depth, H, W, tile_size=16, k_sigma=3.5):
    device = mu.device
    N = mu.shape[0]

    tiles_x = math.ceil(W / tile_size)
    tiles_y = math.ceil(H / tile_size)
    total_tiles = tiles_x * tiles_y

    pares_tile = []
    pares_depth = []
    pares_gauss = []

    mu_cpu = mu.detach().cpu()
    scale_cpu = scale.detach().cpu()
    theta_cpu = theta.detach().cpu()
    depth_cpu = depth.detach().cpu()

    for i in range(N):
        fila = float(mu_cpu[i, 0])
        col = float(mu_cpu[i, 1])

        sx = float(scale_cpu[i, 0])
        sy = float(scale_cpu[i, 1])
        th = float(theta_cpu[i])

        c = math.cos(th)
        s = math.sin(th)

        # AABB de elipse rotada.
        # k_sigma controla cuanto de la cola de la gaussiana incluimos.
        extent_fila = k_sigma * math.sqrt((c * sx) ** 2 + (s * sy) ** 2)
        extent_col = k_sigma * math.sqrt((s * sx) ** 2 + (c * sy) ** 2)

        min_f = max(0, int(math.floor(fila - extent_fila)))
        max_f = min(H - 1, int(math.ceil(fila + extent_fila)))
        min_c = max(0, int(math.floor(col - extent_col)))
        max_c = min(W - 1, int(math.ceil(col + extent_col)))

        tile_y_min = min_f // tile_size
        tile_y_max = max_f // tile_size
        tile_x_min = min_c // tile_size
        tile_x_max = max_c // tile_size

        for ty in range(tile_y_min, tile_y_max + 1):
            for tx in range(tile_x_min, tile_x_max + 1):
                tile_id = ty * tiles_x + tx
                pares_tile.append(tile_id)
                pares_depth.append(float(depth_cpu[i]))
                pares_gauss.append(i)

    tile_ids = torch.tensor(pares_tile, device=device, dtype=torch.int64)
    depths = torch.tensor(pares_depth, device=device, dtype=torch.float32)
    gaussian_ids = torch.tensor(pares_gauss, device=device, dtype=torch.int64)

    # Orden por tile y luego depth.
    # Version simple con torch.argsort. Luego se podria reemplazar por radix sort GPU.
    depth_norm = (depths * 1_000_000).to(torch.int64)
    keys = tile_ids * 1_000_000 + depth_norm
    order = torch.argsort(keys)

    tile_ids = tile_ids[order].contiguous()
    gaussian_ids = gaussian_ids[order].contiguous()

    ranges = torch.full((total_tiles, 2), -1, device=device, dtype=torch.int64)

    tile_cpu = tile_ids.detach().cpu()
    start = 0
    while start < tile_cpu.numel():
        tid = int(tile_cpu[start])
        end = start + 1

        while end < tile_cpu.numel() and int(tile_cpu[end]) == tid:
            end += 1

        ranges[tid, 0] = start
        ranges[tid, 1] = end
        start = end

    return gaussian_ids, ranges, tiles_x, tiles_y


def guardar_comparacion(out_torch, out_tiled, ruta):
    torch_img = out_torch.detach().clamp(0, 1).cpu().numpy()
    tiled_img = out_tiled.detach().clamp(0, 1).cpu().numpy()

    diff = np.abs(torch_img - tiled_img)
    diff_vis = np.clip(diff * 50.0, 0, 1)

    combinado = np.concatenate([torch_img, tiled_img, diff_vis], axis=1)

    plt.figure(figsize=(12, 4), dpi=150)
    plt.imshow(combinado)
    plt.axis("off")
    plt.title("PyTorch brute-force | CUDA tiled | diff x50")
    plt.tight_layout()
    plt.savefig(ruta)
    plt.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--N", type=int, default=1000)
    parser.add_argument("--H", type=int, default=256)
    parser.add_argument("--W", type=int, default=256)
    parser.add_argument("--tile", type=int, default=16)
    parser.add_argument("--iters", type=int, default=100)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--sigmas", type=str, default="3.0,3.5,4.0")
    parser.add_argument("--visual_sigma", type=float, default=3.5)
    args = parser.parse_args()

    device = "cuda"

    print("=== Config ===")
    print(f"N gaussianas : {args.N}")
    print(f"resolucion   : {args.H} x {args.W}")
    print(f"tile         : {args.tile}")
    print(f"iters        : {args.iters}")
    print(f"device       : {torch.cuda.get_device_name(0)}")

    params = crear_gaussianas_aleatorias(args.N, args.H, args.W, device, args.seed)
    conic = construir_conic(params["scale"], params["theta"])

    def render_torch():
        return rasterizar_un_frame(params, args.H, args.W)

    def render_conic():
        return raster_cuda.forward_conic(
            params["mu"],
            conic,
            params["opacity"],
            params["color"],
            args.H,
            args.W
        )

    with torch.no_grad():
        out_torch = render_torch().clamp(0, 1)
        out_conic = render_conic().clamp(0, 1)

    t_torch = medir_tiempo_cuda(render_torch, args.warmup, args.iters)
    t_conic = medir_tiempo_cuda(render_conic, args.warmup, args.iters)

    print("\n=== Base ===")
    print(f"PyTorch rasterizador actual : {t_torch * 1000:.4f} ms")
    print(f"CUDA conic brute-force      : {t_conic * 1000:.4f} ms")
    print("Torch vs CUDA conic")
    print("max_diff :", (out_torch - out_conic).abs().max().item())
    print("mean_diff:", (out_torch - out_conic).abs().mean().item())

    sigmas = [float(x.strip()) for x in args.sigmas.split(",") if x.strip()]

    print("\n=== Sweep k_sigma ===")
    print("k_sigma | instancias | gauss/tile avg | gauss/tile max | tiled ms | total ms | mean_diff | max_diff | speedup_total")
    print("-" * 120)

    mejor = None

    for k_sigma in sigmas:
        # Preprocess una vez para medicion de forward y error
        gaussian_ids, ranges, tiles_x, tiles_y = calcular_tiles_tocados(
            params["mu"],
            params["scale"],
            params["theta"],
            params["depth"],
            args.H,
            args.W,
            tile_size=args.tile,
            k_sigma=k_sigma
        )

        longitudes = ranges[:, 1] - ranges[:, 0]
        longitudes = longitudes[longitudes > 0]
        avg_tile = longitudes.float().mean().item()
        max_tile = longitudes.max().item()
        instancias = gaussian_ids.numel()

        def render_tiled():
            return raster_cuda.forward_tiled(
                params["mu"],
                conic,
                params["opacity"],
                params["color"],
                gaussian_ids,
                ranges,
                args.H,
                args.W,
                args.tile
            )

        def preprocess_y_render():
            gids, rng, _, _ = calcular_tiles_tocados(
                params["mu"],
                params["scale"],
                params["theta"],
                params["depth"],
                args.H,
                args.W,
                tile_size=args.tile,
                k_sigma=k_sigma
            )
            return raster_cuda.forward_tiled(
                params["mu"],
                conic,
                params["opacity"],
                params["color"],
                gids,
                rng,
                args.H,
                args.W,
                args.tile
            )

        with torch.no_grad():
            out_tiled = render_tiled().clamp(0, 1)

        max_diff = (out_torch - out_tiled).abs().max().item()
        mean_diff = (out_torch - out_tiled).abs().mean().item()

        t_tiled = medir_tiempo_cuda(render_tiled, args.warmup, args.iters)

        # Este tiempo incluye preprocess Python + sort + ranges + forward tiled.
        # Es el tiempo total real de esta version actual.
        t_total = medir_tiempo_cuda(preprocess_y_render, n_warmup=2, n_iters=max(5, args.iters // 10))

        speedup_total = t_torch / t_total

        print(
            f"{k_sigma:7.2f} | "
            f"{instancias:10d} | "
            f"{avg_tile:14.2f} | "
            f"{max_tile:14.0f} | "
            f"{t_tiled * 1000:8.4f} | "
            f"{t_total * 1000:8.4f} | "
            f"{mean_diff:9.7f} | "
            f"{max_diff:8.6f} | "
            f"{speedup_total:12.2f}x"
        )

        if abs(k_sigma - args.visual_sigma) < 1e-6:
            ruta = f"comparacion_tiled_k{k_sigma:.1f}.png".replace(".", "_", 1)
            guardar_comparacion(out_torch, out_tiled, ruta)
            print(f"  imagen guardada: {ruta}")

        if mejor is None or mean_diff < mejor["mean_diff"]:
            mejor = {
                "k_sigma": k_sigma,
                "mean_diff": mean_diff,
                "max_diff": max_diff,
                "t_tiled": t_tiled,
                "t_total": t_total,
            }

    print("\n=== Mejor por mean_diff ===")
    print(mejor)


if __name__ == "__main__":
    main()