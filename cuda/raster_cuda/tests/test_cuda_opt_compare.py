"""
Test/benchmark para comparar raster_cuda antes y despues del parche fase 1+2.

Uso recomendado:

1) ANTES de aplicar el parche, desde la raiz del proyecto:
   python scripts/test_cuda_opt_compare.py --mode save --out outputs/cuda_baseline.pt

2) Aplicar parche, recompilar raster_cuda.

3) DESPUES de recompilar:
   python scripts/test_cuda_opt_compare.py --mode compare --out outputs/cuda_baseline.pt
   python scripts/test_cuda_opt_compare.py --mode bench

El test no modifica la loss. Compara preprocess + forward_tiled_train_loss + backward_tiled_fast.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Dict, Tuple

import torch


TEST_FILE = Path(__file__).resolve()

# test_cuda_opt_compare.py esta en:
# Gaussian2D_Optimized/cuda/raster_cuda/tests/test_cuda_opt_compare.py
CUDA_DIR = TEST_FILE.parents[1]       # .../cuda/raster_cuda
PROJECT_ROOT = TEST_FILE.parents[3]   # .../Gaussian2D_Optimized

if str(CUDA_DIR) not in sys.path:
    sys.path.insert(0, str(CUDA_DIR))

import raster_cuda  # noqa: E402

print(f"[test] raster_cuda cargado desde: {raster_cuda.__file__}")


def make_inputs(N: int, H: int, W: int, device: str, offscreen_frac: float = 0.0) -> Dict[str, torch.Tensor]:
    torch.manual_seed(123)
    dev = torch.device(device)

    # Caso base: gaussianas dentro de pantalla para que el resultado viejo/nuevo sea comparable.
    mu = torch.empty((N, 2), device=dev, dtype=torch.float32)
    mu[:, 0] = torch.rand(N, device=dev) * (H - 1)
    mu[:, 1] = torch.rand(N, device=dev) * (W - 1)

    # Opcional: manda algunas gaussianas fuera para medir culling. No lo uses para compare estricto.
    if offscreen_frac > 0:
        k = int(N * offscreen_frac)
        if k > 0:
            mu[:k, 0] = -50.0 - torch.rand(k, device=dev) * 200.0
            mu[:k, 1] = -50.0 - torch.rand(k, device=dev) * 200.0

    scale = 0.8 + torch.rand((N, 2), device=dev, dtype=torch.float32) * 3.0
    theta = (torch.rand(N, device=dev, dtype=torch.float32) - 0.5) * 6.28318530718
    opacity = 0.03 + torch.rand(N, device=dev, dtype=torch.float32) * 0.55
    color = torch.rand((N, 3), device=dev, dtype=torch.float32)

    # Depth monotono y unico. Asi el cambio de key int64->int32 conserva el orden.
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


def run_once(inp: Dict[str, torch.Tensor], H: int, W: int, tile: int, k_sigma: float) -> Dict[str, torch.Tensor]:
    conic = raster_cuda.build_conic(inp["scale"], inp["theta"])
    gaussian_ids, ranges = raster_cuda.preprocess_tiled(
        inp["mu"], inp["scale"], inp["theta"], inp["depth"], H, W, tile, float(k_sigma)
    )

    loss, grad_render, final_Ts, n_contrib = raster_cuda.forward_tiled_train_loss(
        inp["mu"], conic, inp["opacity"], inp["color"], gaussian_ids, ranges,
        inp["target"], H, W, tile, "l1"
    )

    grad_mu, grad_conic, grad_opacity, grad_color = raster_cuda.backward_tiled_fast(
        inp["mu"], conic, inp["opacity"], inp["color"], gaussian_ids, ranges,
        final_Ts, n_contrib, grad_render, H, W, tile
    )

    return {
        "loss": loss.detach().cpu(),
        "grad_render": grad_render.detach().cpu(),
        "final_Ts": final_Ts.detach().cpu(),
        "n_contrib": n_contrib.detach().cpu(),
        "grad_mu": grad_mu.detach().cpu(),
        "grad_conic": grad_conic.detach().cpu(),
        "grad_opacity": grad_opacity.detach().cpu(),
        "grad_color": grad_color.detach().cpu(),
        "ids_dtype": torch.tensor(0 if gaussian_ids.dtype == torch.int64 else 1),
        "ranges_dtype": torch.tensor(0 if ranges.dtype == torch.int64 else 1),
        "num_instances": torch.tensor(int(gaussian_ids.numel())),
    }


def max_abs(a: torch.Tensor, b: torch.Tensor) -> float:
    if a.dtype in (torch.int8, torch.int16, torch.int32, torch.int64, torch.uint8, torch.bool):
        return float((a.to(torch.int64) - b.to(torch.int64)).abs().max().item()) if a.numel() else 0.0
    return float((a - b).abs().max().item()) if a.numel() else 0.0


def compare_dict(ref: Dict[str, torch.Tensor], cur: Dict[str, torch.Tensor], atol: float) -> None:
    keys = [
        "loss", "grad_render", "final_Ts", "n_contrib",
        "grad_mu", "grad_conic", "grad_opacity", "grad_color", "num_instances"
    ]
    ok = True
    print("=== comparacion baseline vs actual ===")
    for k in keys:
        d = max_abs(ref[k], cur[k])
        print(f"{k:14s} max_abs_diff = {d:.6e}")
        if k == "n_contrib":
            ok = ok and d == 0.0
        elif k == "num_instances":
            ok = ok and d == 0.0
        else:
            ok = ok and d <= atol

    print(f"baseline ids dtype   : {'int64' if int(ref['ids_dtype']) == 0 else 'int32'}")
    print(f"actual ids dtype     : {'int64' if int(cur['ids_dtype']) == 0 else 'int32'}")
    print(f"baseline ranges dtype: {'int64' if int(ref['ranges_dtype']) == 0 else 'int32'}")
    print(f"actual ranges dtype  : {'int64' if int(cur['ranges_dtype']) == 0 else 'int32'}")

    if not ok:
        raise SystemExit(f"ERROR: diferencias mayores que atol={atol}")
    print("OK: el resultado numerico coincide dentro de la tolerancia.")


def bench(inp: Dict[str, torch.Tensor], H: int, W: int, tile: int, k_sigma: float, iters: int) -> None:
    # Warmup
    for _ in range(5):
        run_once(inp, H, W, tile, k_sigma)
    torch.cuda.synchronize()

    t0 = time.perf_counter()
    num_instances = None
    for _ in range(iters):
        out = run_once(inp, H, W, tile, k_sigma)
        num_instances = int(out["num_instances"].item())
    torch.cuda.synchronize()
    dt = (time.perf_counter() - t0) / iters

    print("=== benchmark end-to-end CUDA ===")
    print(f"N             : {inp['mu'].shape[0]}")
    print(f"resolucion    : {H}x{W}")
    print(f"tile          : {tile}")
    print(f"k_sigma       : {k_sigma}")
    print(f"instancias    : {num_instances}")
    print(f"tiempo medio  : {dt * 1000:.3f} ms / iter")
    print(f"ids dtype     : {'int32' if int(out['ids_dtype']) == 1 else 'int64'}")
    print(f"ranges dtype  : {'int32' if int(out['ranges_dtype']) == 1 else 'int64'}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["save", "compare", "bench", "culling"], required=True)
    parser.add_argument("--out", default="outputs/cuda_baseline.pt")
    parser.add_argument("--N", type=int, default=5000)
    parser.add_argument("--H", type=int, default=288)
    parser.add_argument("--W", type=int, default=512)
    parser.add_argument("--tile", type=int, default=16)
    parser.add_argument("--k_sigma", type=float, default=4.0)
    parser.add_argument("--iters", type=int, default=20)
    parser.add_argument("--atol", type=float, default=2e-5)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("ERROR: CUDA no disponible")

    out_path = PROJECT_ROOT / args.out
    out_path.parent.mkdir(parents=True, exist_ok=True)

    offscreen = 0.0 if args.mode in ("save", "compare") else 0.15
    inp = make_inputs(args.N, args.H, args.W, "cuda", offscreen_frac=offscreen)

    if args.mode == "save":
        res = run_once(inp, args.H, args.W, args.tile, args.k_sigma)
        torch.save({"args": vars(args), "result": res}, out_path)
        print(f"Baseline guardado en: {out_path}")
        print(f"ids dtype    : {'int32' if int(res['ids_dtype']) == 1 else 'int64'}")
        print(f"ranges dtype : {'int32' if int(res['ranges_dtype']) == 1 else 'int64'}")
        print(f"instancias   : {int(res['num_instances'].item())}")

    elif args.mode == "compare":
        ref = torch.load(out_path, map_location="cpu")["result"]
        cur = run_once(inp, args.H, args.W, args.tile, args.k_sigma)
        compare_dict(ref, cur, args.atol)

    elif args.mode == "bench":
        bench(inp, args.H, args.W, args.tile, args.k_sigma, args.iters)

    elif args.mode == "culling":
        res = run_once(inp, args.H, args.W, args.tile, args.k_sigma)
        print("=== culling/offscreen stats ===")
        print(f"offscreen_frac: {offscreen}")
        print(f"instancias    : {int(res['num_instances'].item())}")
        print(f"ids dtype     : {'int32' if int(res['ids_dtype']) == 1 else 'int64'}")
        print(f"ranges dtype  : {'int32' if int(res['ranges_dtype']) == 1 else 'int64'}")


if __name__ == "__main__":
    main()
