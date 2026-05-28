"""
Barre RASTER_BATCH_SIZE recompilando la extension CUDA y ejecutando los tests
que ya tienes: test_cuda_opt_compare.py y test_cuda_profile_parts.py.

Uso desde la raiz del proyecto:
    python cuda/raster_cuda/tests/test_sweep_raster_batch_size.py --baseline outputs/cuda_phase2_baseline.pt --N 20000 --H 288 --W 512 --tile 16 --k_sigma 4.0 --iters 100

El script asume que esta ubicado en:
    Gaussian2D_Optimized/cuda/raster_cuda/tests/test_sweep_raster_batch_size.py
"""
from __future__ import annotations

import argparse
import csv
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path


THIS = Path(__file__).resolve()
CUDA_DIR = THIS.parents[1]
PROJECT_ROOT = THIS.parents[3]
PYTHON = sys.executable


TIME_PATTERNS = {
    "build_conic_ms": re.compile(r"build_conic\s*:\s*([0-9.]+) ms"),
    "preprocess_tiled_ms": re.compile(r"preprocess_tiled\s*:\s*([0-9.]+) ms"),
    "forward_train_loss_ms": re.compile(r"forward_train_loss\s*:\s*([0-9.]+) ms"),
    "backward_tiled_fast_ms": re.compile(r"backward_tiled_fast\s*:\s*([0-9.]+) ms"),
    "grad_conic_scale_theta_ms": re.compile(r"grad_conic_scale_theta\s*:\s*([0-9.]+) ms"),
    "total_end_to_end_ms": re.compile(r"total_end_to_end\s*:\s*([0-9.]+) ms"),
    "wall_ms": re.compile(r"total_end_to_end wall\s*:\s*([0-9.]+) ms / iter"),
    "instancias": re.compile(r"instancias\s*:\s*([0-9]+)"),
}


def run(cmd: list[str], cwd: Path, env: dict[str, str] | None = None) -> str:
    print("\n$ " + " ".join(cmd))
    p = subprocess.run(
        cmd,
        cwd=str(cwd),
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    print(p.stdout)
    if p.returncode != 0:
        raise RuntimeError(f"Comando fallo con codigo {p.returncode}: {' '.join(cmd)}")
    return p.stdout


def clean_build() -> None:
    for pyd in CUDA_DIR.glob("raster_cuda*.pyd"):
        try:
            pyd.unlink()
        except PermissionError as exc:
            raise RuntimeError(
                f"No se pudo borrar {pyd}. Cierra cualquier Python que haya importado raster_cuda."
            ) from exc
    build = CUDA_DIR / "build"
    if build.exists():
        shutil.rmtree(build)


def compile_with_batch(batch_size: int) -> None:
    clean_build()
    env = os.environ.copy()
    env["RASTER_BATCH_SIZE"] = str(batch_size)
    run([PYTHON, "setup.py", "build_ext", "--inplace"], cwd=CUDA_DIR, env=env)


def parse_profile(text: str) -> dict[str, float | int | str]:
    out: dict[str, float | int | str] = {}
    for key, pat in TIME_PATTERNS.items():
        m = pat.search(text)
        if not m:
            continue
        val = m.group(1)
        out[key] = int(val) if key == "instancias" else float(val)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--baseline", default="outputs/cuda_phase2_baseline.pt")
    ap.add_argument("--batch_sizes", nargs="+", type=int, default=[64, 128, 256, 512])
    ap.add_argument("--N", type=int, default=20000)
    ap.add_argument("--H", type=int, default=288)
    ap.add_argument("--W", type=int, default=512)
    ap.add_argument("--tile", type=int, default=16)
    ap.add_argument("--k_sigma", type=float, default=4.0)
    ap.add_argument("--iters", type=int, default=100)
    ap.add_argument("--atol", type=float, default=1e-4)
    ap.add_argument("--out_csv", default="outputs/raster_batch_size_sweep.csv")
    args = ap.parse_args()

    results = []

    for bs in args.batch_sizes:
        print("\n" + "=" * 80)
        print(f"Probando RASTER_BATCH_SIZE={bs}")
        print("=" * 80)
        compile_with_batch(bs)

        compare_cmd = [
            PYTHON,
            "cuda/raster_cuda/tests/test_cuda_opt_compare.py",
            "--mode", "compare",
            "--out", args.baseline,
            "--N", str(min(args.N, 5000)),
            "--H", str(args.H),
            "--W", str(args.W),
            "--tile", str(args.tile),
            "--k_sigma", str(args.k_sigma),
            "--atol", str(args.atol),
        ]
        compare_txt = run(compare_cmd, cwd=PROJECT_ROOT)
        ok = "OK: el resultado numerico coincide" in compare_txt
        if not ok:
            print(f"ADVERTENCIA: compare no dio OK para RASTER_BATCH_SIZE={bs}")

        profile_cmd = [
            PYTHON,
            "cuda/raster_cuda/tests/test_cuda_profile_parts.py",
            "--N", str(args.N),
            "--H", str(args.H),
            "--W", str(args.W),
            "--tile", str(args.tile),
            "--k_sigma", str(args.k_sigma),
            "--iters", str(args.iters),
            "--wall",
        ]
        profile_txt = run(profile_cmd, cwd=PROJECT_ROOT)
        row = parse_profile(profile_txt)
        row["RASTER_BATCH_SIZE"] = bs
        row["compare_ok"] = str(ok)
        results.append(row)

    out_csv = PROJECT_ROOT / args.out_csv
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    cols = [
        "RASTER_BATCH_SIZE",
        "compare_ok",
        "instancias",
        "build_conic_ms",
        "preprocess_tiled_ms",
        "forward_train_loss_ms",
        "backward_tiled_fast_ms",
        "grad_conic_scale_theta_ms",
        "total_end_to_end_ms",
        "wall_ms",
    ]
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for r in results:
            w.writerow({c: r.get(c, "") for c in cols})

    print("\n" + "=" * 80)
    print(f"Resultados guardados en: {out_csv}")
    print("Resumen:")
    for r in results:
        print(
            f"  BS={r.get('RASTER_BATCH_SIZE')} | "
            f"wall={r.get('wall_ms', 'n/a')} ms | "
            f"total={r.get('total_end_to_end_ms', 'n/a')} ms | "
            f"forward={r.get('forward_train_loss_ms', 'n/a')} ms | "
            f"backward={r.get('backward_tiled_fast_ms', 'n/a')} ms | "
            f"OK={r.get('compare_ok')}"
        )


if __name__ == "__main__":
    main()
