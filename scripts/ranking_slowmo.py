"""
Ranking de la ablacion de slow-mo: lee cada experimento y tabula la calidad de
INTERPOLACION (frames retenidos = holdout) vs la de los frames vistos (train),
y la compara contra el baseline de interpolacion lineal (el piso).

Por experimento lee:
    outputs/<exp>/metricas_por_frame.csv   (frame_idx, psnr, ssim, ...)
    outputs/<exp>/holdout_indices.json     (supervisados / holdout)

Lo que importa: PSNR/SSIM en los frames HOLDOUT (interpolados). El ganador es el
que maximiza eso por encima del baseline lineal.

Uso (en Khipu, tras la tanda):
    python scripts/ranking_slowmo.py --exps outputs/slowmo_gota_* \
        --baseline outputs/baseline_lineal/gota.csv
"""
import argparse
import csv
import glob
import json
from pathlib import Path

import numpy as np


def _leer_csv_metricas(ruta):
    filas = {}
    with open(ruta, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            filas[int(row["frame_idx"])] = row
    return filas


def _prom(filas, indices, campo):
    vals = []
    for j in indices:
        v = filas.get(j, {}).get(campo, "")
        if v not in ("", None):
            try:
                vals.append(float(v))
            except ValueError:
                pass
    return float(np.mean(vals)) if vals else float("nan")


def _resumen_exp(exp):
    exp = Path(exp)
    csv_path = exp / "metricas_por_frame.csv"
    hold_path = exp / "holdout_indices.json"
    if not csv_path.is_file() or not hold_path.is_file():
        return None
    filas = _leer_csv_metricas(csv_path)
    hold = json.loads(hold_path.read_text(encoding="utf-8"))
    sup = set(hold["supervisados"])
    held = set(hold["holdout"])
    return {
        "exp": exp.name,
        "psnr_train": _prom(filas, sup, "psnr"),
        "psnr_interp": _prom(filas, held, "psnr"),
        "ssim_train": _prom(filas, sup, "ssim"),
        "ssim_interp": _prom(filas, held, "ssim"),
        "n_interp": len(held),
    }


def _baseline(ruta):
    if not ruta or not Path(ruta).is_file():
        return None
    filas = _leer_csv_metricas(ruta)
    todos = list(filas.keys())
    return {
        "psnr_interp": _prom(filas, todos, "psnr"),
        "ssim_interp": _prom(filas, todos, "ssim"),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--exps", nargs="+", required=True,
                    help="carpetas de experimentos (acepta glob ya expandido por el shell)")
    ap.add_argument("--baseline", default=None, help="csv del baseline lineal")
    args = ap.parse_args()

    # expandir por si el shell no lo hizo
    carpetas = []
    for e in args.exps:
        carpetas.extend(glob.glob(e)) if any(c in e for c in "*?[") else carpetas.append(e)

    resus = [r for r in (_resumen_exp(c) for c in sorted(set(carpetas))) if r]
    if not resus:
        print("no se encontro ningun experimento con metricas+holdout")
        return

    base = _baseline(args.baseline)

    resus.sort(key=lambda r: (-(r["psnr_interp"] if r["psnr_interp"] == r["psnr_interp"] else -1e9)))

    print(f"\n{'experimento':38} {'PSNR_int':>9} {'SSIM_int':>9} {'PSNR_tr':>9} {'vs base':>9}")
    print("-" * 78)
    base_psnr = base["psnr_interp"] if base else float("nan")
    if base:
        print(f"{'[BASELINE lineal]':38} {base_psnr:9.3f} "
              f"{base['ssim_interp'] if base['ssim_interp']==base['ssim_interp'] else float('nan'):9.3f} "
              f"{'-':>9} {'-':>9}")
        print("-" * 78)
    for r in resus:
        delta = r["psnr_interp"] - base_psnr if base else float("nan")
        marca = "  <<" if (base and delta > 0) else ""
        print(f"{r['exp']:38} {r['psnr_interp']:9.3f} {r['ssim_interp']:9.3f} "
              f"{r['psnr_train']:9.3f} {delta:+9.3f}{marca}")

    print("\nPSNR_int / SSIM_int = calidad en frames INTERPOLADOS (lo que importa).")
    print("'vs base' positivo = el modelo SUPERA promediar vecinos.")


if __name__ == "__main__":
    main()
