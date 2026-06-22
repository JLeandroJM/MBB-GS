"""
Analisis de los experimentos de holdout temporal (corre en la Mac, sin GPU).

Lee, por experimento:
  - metricas_por_frame.csv   (frame_idx, psnr, ssim, lpips, psnr_temporal)
  - holdout_indices.json     (supervisados / holdout)

y separa las metricas en:
  - TRAIN    : frames supervisados (el modelo los vio)
  - HOLDOUT  : frames reconstruidos (el modelo NUNCA los vio)  <- lo que importa

La brecha TRAIN vs HOLDOUT mide cuanto generaliza la interpolacion polinomica.




Dos modos:
  1) un experimento:    --exp outputs_khipu/dvd_20s_holdout_mult2_l1dssim
  2) curva (varios):    --curva outputs_khipu/dvd_20s_holdout_curva_mult2_l1dssim \\
                                outputs_khipu/dvd_20s_holdout_curva_mult3_l1dssim ...
     -> grafica PSNR_holdout vs fraccion supervisada.

Uso:
    python scripts/analizar_holdout.py --exp <carpeta>
    python scripts/analizar_holdout.py --curva <carpeta1> <carpeta2> ... --salida curva.png
"""
import argparse
import csv
import json
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt


def _leer_metricas(exp):
    exp = Path(exp)
    csv_path = exp / "metricas_por_frame.csv"
    hold_path = exp / "holdout_indices.json"
    if not csv_path.is_file():
        raise FileNotFoundError(f"no existe {csv_path}")
    if not hold_path.is_file():
        raise FileNotFoundError(
            f"no existe {hold_path} (el experimento no se corrio con holdout)"
        )

    filas = {}
    with open(csv_path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            j = int(row["frame_idx"])
            filas[j] = row

    hold = json.loads(hold_path.read_text(encoding="utf-8"))
    sup_idx = set(hold["supervisados"])
    hold_idx = set(hold["holdout"])
    return filas, sup_idx, hold_idx, hold


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
    filas, sup_idx, hold_idx, hold = _leer_metricas(exp)
    n = hold["n_frames"]
    frac_sup = len(sup_idx) / n
    out = {
        "exp": Path(exp).name,
        "modo": hold["holdout_cfg"],
        "n_frames": n,
        "n_sup": len(sup_idx),
        "n_hold": len(hold_idx),
        "frac_sup": frac_sup,
    }
    for campo in ("psnr", "ssim", "lpips"):
        out[f"{campo}_train"] = _prom(filas, sup_idx, campo)
        out[f"{campo}_holdout"] = _prom(filas, hold_idx, campo)
    return out


def _imprimir(r):
    print(f"\n=== {r['exp']} ===")
    print(f"  holdout: {r['modo']}")
    print(f"  frames: {r['n_frames']}  supervisados: {r['n_sup']} ({r['frac_sup']*100:.0f}%)  holdout: {r['n_hold']}")
    print(f"  {'metrica':8} {'TRAIN':>10} {'HOLDOUT':>10} {'gap':>10}")
    for campo in ("psnr", "ssim", "lpips"):
        t = r[f"{campo}_train"]; h = r[f"{campo}_holdout"]
        print(f"  {campo:8} {t:10.4f} {h:10.4f} {t-h:10.4f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp", default=None, help="un experimento")
    ap.add_argument("--curva", nargs="+", default=None, help="varios experimentos para la curva")
    ap.add_argument("--salida", default=None, help="ruta del PNG de la curva")
    args = ap.parse_args()

    if args.exp:
        r = _resumen_exp(args.exp)
        _imprimir(r)
        # tabla json al lado
        out_json = Path(args.exp) / "resumen_holdout.json"
        out_json.write_text(json.dumps(r, indent=2), encoding="utf-8")
        print(f"\n  -> {out_json}")

    if args.curva:
        ress = []
        for e in args.curva:
            try:
                r = _resumen_exp(e)
                _imprimir(r)
                ress.append(r)
            except FileNotFoundError as ex:
                print(f"  (salto {e}: {ex})")
        if len(ress) >= 2:
            ress.sort(key=lambda r: r["frac_sup"])
            fr = [r["frac_sup"] * 100 for r in ress]
            ph = [r["psnr_holdout"] for r in ress]
            pt = [r["psnr_train"] for r in ress]
            fig, ax = plt.subplots(figsize=(8, 5), dpi=120)
            ax.plot(fr, pt, "o--", color="tab:gray", label="PSNR train (frames vistos)")
            ax.plot(fr, ph, "o-", color="tab:red", label="PSNR holdout (reconstruidos)")
            for r in ress:
                ax.annotate(f"k={r['modo'].get('k','?')}", (r["frac_sup"]*100, r["psnr_holdout"]),
                            textcoords="offset points", xytext=(5, -10), fontsize=8)
            ax.set_xlabel("% de frames supervisados")
            ax.set_ylabel("PSNR (dB)")
            ax.set_title("Curva de degradacion: reconstruccion de frames no vistos")
            ax.grid(True, alpha=0.3); ax.legend()
            salida = Path(args.salida) if args.salida else Path("curva_holdout.png")
            fig.tight_layout(); fig.savefig(salida); plt.close(fig)
            print(f"\n  curva -> {salida}")

    if not args.exp and not args.curva:
        ap.error("pasa --exp o --curva")


if __name__ == "__main__":
    main()
