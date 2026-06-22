"""
Profiler por partes para raster_cuda tiled.

Objetivo:
- Medir build_conic
- Medir preprocess_tiled
- Medir forward_tiled_train_loss
- Medir backward_tiled_fast
- Medir grad_conic_to_scale_theta
- Medir total end-to-end

Ubicacion recomendada:
    D:/TesisProyecto/Gaussian2D_Optimized/cuda/raster_cuda/tests/test_cuda_profile_parts.py

Uso desde la raiz del proyecto:
    python cuda/raster_cuda/tests/test_cuda_profile_parts.py --N 20000 --H 288 --W 512 --tile 16 --k_sigma 4.0 --iters 100

No cambia la loss. No entrena. Solo llama las funciones CUDA ya compiladas.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Callable, Dict, Tuple

import torch


TEST_FILE = Path(__file__).resolve()
# Si el archivo esta en cuda/raster_cuda/tests:
CUDA_DIR = TEST_FILE.parents[1]
PROJECT_ROOT = TEST_FILE.parents[3]

if str(CUDA_DIR) not in sys.path:
    sys.path.insert(0, str(CUDA_DIR))

import raster_cuda  # noqa: E402


def make_inputs(N: int, H: int, W: int, device: str, offscreen_frac: float = 0.0) -> Dict[str, torch.Tensor]:
    torch.manual_seed(123)
    dev = torch.device(device)

    mu = torch.empty((N, 2), device=dev, dtype=torch.float32)
    mu[:, 0] = torch.rand(N, device=dev) * (H - 1)
    mu[:, 1] = torch.rand(N, device=dev) * (W - 1)

    if offscreen_frac > 0:
        k = int(N * offscreen_frac)
        if k > 0:
            # Muy fuera de pantalla para probar early culling.
            mu[:k, 0] = -5000.0 - torch.rand(k, device=dev) * 5000.0
            mu[:k, 1] = -5000.0 - torch.rand(k, device=dev) * 5000.0

    scale = 0.8 + torch.rand((N, 2), device=dev, dtype=torch.float32) * 3.0
    theta = (torch.rand(N, device=dev, dtype=torch.float32) - 0.5) * 6.28318530718
    opacity = 0.03 + torch.rand(N, device=dev, dtype=torch.float32) * 0.55
    color = torch.rand((N, 3), device=dev, dtype=torch.float32)

    # Profundidad unica y monotona para mantener orden determinista.
    depth = torch.linspace(-1.0, 1.0, N, device=dev, dtype=torch.float32)
    target = torch.rand((H, W, 3), device=dev, dtype=torch.float32)

    return {
        "mu": mu.contiguous(),
        "scale": scale.contiguous(),
        "theta": theta.contiguous(),
        "opacity": opacity.contiguous(),
        "color": color.contiguous(),
        "depth": depth.contiguous(),
        "target": target.contiguous(),
    }


def cuda_time_ms(fn: Callable[[], object], iters: int, warmup: int = 10) -> Tuple[float, object]:
    """Tiempo GPU usando CUDA events. Retorna promedio ms y ultima salida."""
    out = None
    for _ in range(warmup):
        out = fn()
    torch.cuda.synchronize()

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)

    start.record()
    for _ in range(iters):
        out = fn()
    end.record()
    torch.cuda.synchronize()

    return start.elapsed_time(end) / max(1, iters), out


def wall_time_ms(fn: Callable[[], object], iters: int, warmup: int = 5) -> Tuple[float, object]:
    """Tiempo pared CPU+GPU sincronizado. Incluye overhead Python/launch/alloc mas claramente."""
    out = None
    for _ in range(warmup):
        out = fn()
    torch.cuda.synchronize()

    t0 = time.perf_counter()
    for _ in range(iters):
        out = fn()
    torch.cuda.synchronize()
    dt = time.perf_counter() - t0
    return (dt * 1000.0) / max(1, iters), out


def full_once(inp: Dict[str, torch.Tensor], H: int, W: int, tile: int, k_sigma: float, loss_type: str):
    conic = raster_cuda.build_conic(inp["scale"], inp["theta"])
    gaussian_ids, ranges = raster_cuda.preprocess_tiled(
        inp["mu"], inp["scale"], inp["theta"], inp["depth"], H, W, tile, float(k_sigma)
    )
    loss, grad_render, final_Ts, n_contrib = raster_cuda.forward_tiled_train_loss(
        inp["mu"], conic, inp["opacity"], inp["color"], gaussian_ids, ranges,
        inp["target"], H, W, tile, loss_type
    )
    grad_mu, grad_conic, grad_opacity, grad_color = raster_cuda.backward_tiled_fast(
        inp["mu"], conic, inp["opacity"], inp["color"], gaussian_ids, ranges,
        final_Ts, n_contrib, grad_render, H, W, tile
    )
    grad_scale, grad_theta = raster_cuda.grad_conic_to_scale_theta(
        inp["scale"], inp["theta"], grad_conic
    )
    return {
        "conic": conic,
        "gaussian_ids": gaussian_ids,
        "ranges": ranges,
        "loss": loss,
        "grad_render": grad_render,
        "final_Ts": final_Ts,
        "n_contrib": n_contrib,
        "grad_mu": grad_mu,
        "grad_conic": grad_conic,
        "grad_opacity": grad_opacity,
        "grad_color": grad_color,
        "grad_scale": grad_scale,
        "grad_theta": grad_theta,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--N", type=int, default=20000)
    parser.add_argument("--H", type=int, default=288)
    parser.add_argument("--W", type=int, default=512)
    parser.add_argument("--tile", type=int, default=16)
    parser.add_argument("--k_sigma", type=float, default=4.0)
    parser.add_argument("--iters", type=int, default=100)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--loss", choices=["l1", "mse"], default="l1")
    parser.add_argument("--offscreen_frac", type=float, default=0.0)
    parser.add_argument("--wall", action="store_true", help="tambien mide tiempo pared CPU+GPU")
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("ERROR: CUDA no disponible")

    print(f"[profile] raster_cuda cargado desde: {raster_cuda.__file__}")
    print("=== config ===")
    print(f"N             : {args.N}")
    print(f"resolucion    : {args.H}x{args.W}")
    print(f"tile          : {args.tile}")
    print(f"k_sigma       : {args.k_sigma}")
    print(f"loss          : {args.loss}")
    print(f"iters         : {args.iters}")
    print(f"offscreen_frac: {args.offscreen_frac}")

    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

    inp = make_inputs(args.N, args.H, args.W, "cuda", offscreen_frac=args.offscreen_frac)

    # Precompute dependencias para aislar cada parte.
    conic = raster_cuda.build_conic(inp["scale"], inp["theta"])
    gaussian_ids, ranges = raster_cuda.preprocess_tiled(
        inp["mu"], inp["scale"], inp["theta"], inp["depth"], args.H, args.W, args.tile, float(args.k_sigma)
    )
    loss, grad_render, final_Ts, n_contrib = raster_cuda.forward_tiled_train_loss(
        inp["mu"], conic, inp["opacity"], inp["color"], gaussian_ids, ranges,
        inp["target"], args.H, args.W, args.tile, args.loss
    )
    grad_mu, grad_conic, grad_opacity, grad_color = raster_cuda.backward_tiled_fast(
        inp["mu"], conic, inp["opacity"], inp["color"], gaussian_ids, ranges,
        final_Ts, n_contrib, grad_render, args.H, args.W, args.tile
    )
    torch.cuda.synchronize()

    num_instances = int(gaussian_ids.numel())
    num_tiles = int(ranges.shape[0])

    print("\n=== estructura tiled ===")
    print(f"num_tiles     : {num_tiles}")
    print(f"instancias    : {num_instances}")
    print(f"ids dtype     : {gaussian_ids.dtype}")
    print(f"ranges dtype  : {ranges.dtype}")
    print(f"loss ejemplo  : {float(loss.detach().cpu().item()):.8f}")

    pieces = []

    t_build, _ = cuda_time_ms(lambda: raster_cuda.build_conic(inp["scale"], inp["theta"]), args.iters, args.warmup)
    pieces.append(("build_conic", t_build))

    t_pre, _ = cuda_time_ms(
        lambda: raster_cuda.preprocess_tiled(
            inp["mu"], inp["scale"], inp["theta"], inp["depth"], args.H, args.W, args.tile, float(args.k_sigma)
        ),
        args.iters,
        args.warmup,
    )
    pieces.append(("preprocess_tiled", t_pre))

    t_fwd, _ = cuda_time_ms(
        lambda: raster_cuda.forward_tiled_train_loss(
            inp["mu"], conic, inp["opacity"], inp["color"], gaussian_ids, ranges,
            inp["target"], args.H, args.W, args.tile, args.loss
        ),
        args.iters,
        args.warmup,
    )
    pieces.append(("forward_train_loss", t_fwd))

    t_bwd, _ = cuda_time_ms(
        lambda: raster_cuda.backward_tiled_fast(
            inp["mu"], conic, inp["opacity"], inp["color"], gaussian_ids, ranges,
            final_Ts, n_contrib, grad_render, args.H, args.W, args.tile
        ),
        args.iters,
        args.warmup,
    )
    pieces.append(("backward_tiled_fast", t_bwd))

    t_grad_conic, _ = cuda_time_ms(
        lambda: raster_cuda.grad_conic_to_scale_theta(inp["scale"], inp["theta"], grad_conic),
        args.iters,
        args.warmup,
    )
    pieces.append(("grad_conic_scale_theta", t_grad_conic))

    t_total_gpu, out_total = cuda_time_ms(
        lambda: full_once(inp, args.H, args.W, args.tile, args.k_sigma, args.loss),
        args.iters,
        args.warmup,
    )

    sum_parts = sum(t for _, t in pieces)

    print("\n=== tiempos GPU por parte (CUDA events) ===")
    for name, t in pieces:
        pct = 100.0 * t / max(t_total_gpu, 1e-9)
        print(f"{name:24s}: {t:8.4f} ms   ({pct:6.2f}% del total)")
    print(f"{'suma_partes':24s}: {sum_parts:8.4f} ms")
    print(f"{'total_end_to_end':24s}: {t_total_gpu:8.4f} ms")

    if args.wall:
        t_total_wall, _ = wall_time_ms(
            lambda: full_once(inp, args.H, args.W, args.tile, args.k_sigma, args.loss),
            max(1, args.iters // 2),
            max(1, args.warmup // 2),
        )
        print("\n=== tiempo pared ===")
        print(f"total_end_to_end wall : {t_total_wall:8.4f} ms / iter")

    peak_alloc = torch.cuda.max_memory_allocated() / (1024 ** 2)
    peak_reserved = torch.cuda.max_memory_reserved() / (1024 ** 2)
    print("\n=== memoria CUDA del test ===")
    print(f"peak allocated: {peak_alloc:.2f} MiB")
    print(f"peak reserved : {peak_reserved:.2f} MiB")

    print("\n=== lectura rapida ===")
    sorted_pieces = sorted(pieces, key=lambda x: x[1], reverse=True)
    top_name, top_ms = sorted_pieces[0]
    print(f"Kernel/etapa mas pesada: {top_name} ({top_ms:.4f} ms)")
    if top_name == "backward_tiled_fast":
        print("Siguiente fase recomendada: optimizar backward/atomicAdd.")
    elif top_name == "preprocess_tiled":
        print("Siguiente fase recomendada: workspace/cache o reducir duplicados/sort.")
    elif top_name == "forward_train_loss":
        print("Siguiente fase recomendada: optimizar forward por pixel/tile.")
    else:
        print("Siguiente fase: revisar si esa etapa realmente aparece igual de pesada en entrenamiento real.")


if __name__ == "__main__":
    main()
