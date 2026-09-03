"""
Video comparativo de reconstruccion con frames ocultos (holdout).

Aisla el efecto del RELLENO de los frames ocultos: en los frames disponibles
(supervisados) todos los paneles muestran el GT real (identicos); solo difieren
en los frames holdout, donde cada estrategia rellena el hueco distinto.

5 paneles (+ leyenda) en grid 2x3, todos con los mismos n_frames y fps -> misma
duracion:
  1. GT             : original (referencia)
  2. Congelado      : hueco = ultimo frame disponible (zero-order hold)
  3. Negro          : hueco = frame negro
  4. Interp lineal  : hueco = blend lineal entre los disponibles vecinos
  5. Tu metodo      : hueco = render del polinomio (en disponibles = GT)
  6. Leyenda        : PSNR holdout de cada estrategia (cuantifica la mejora)

Un borde ROJO marca los paneles durante los frames holdout (cuando rellenan).

Corre en la Mac, sin GPU. Usa el GT en data/clips/<clip> y los renders en
outputs/<exp>/frames_renderizados.

Uso:
    python scripts/video_comparativo_holdout.py \
        --exp outputs_khipu/musical_20s_f1_mult3_l1dssim --fps 24
"""
import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw
import imageio.v2 as imageio

RAIZ = Path(__file__).resolve().parents[1]


def _cargar(carpeta, j):
    return np.asarray(Image.open(carpeta / f"frame_{j:04d}.png").convert("RGB"), dtype=np.uint8)


def _psnr(a, b):
    mse = np.mean((a.astype(np.float64) - b.astype(np.float64)) ** 2)
    return 99.0 if mse <= 0 else float(10.0 * np.log10(255.0 ** 2 / mse))


def _vecinos_sup(n, sup_sorted, hold_set):
    """Para cada frame holdout: (sup anterior, sup siguiente) por indice."""
    prev = {}
    nxt = {}
    ult = sup_sorted[0]
    for j in range(n):
        if j in hold_set:
            prev[j] = ult
        else:
            ult = j
    sig = sup_sorted[-1]
    for j in range(n - 1, -1, -1):
        if j in hold_set:
            nxt[j] = sig
        else:
            sig = j
    return prev, nxt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp", required=True, help="carpeta del experimento")
    ap.add_argument("--salida", default=None)
    ap.add_argument("--fps", type=int, default=24)
    ap.add_argument("--escala", type=float, default=0.5, help="downscale de cada panel (solo grid)")
    ap.add_argument("--separados", action="store_true",
                    help="genera un mp4 LIMPIO (full-res, sin grid) por estrategia "
                         "(congelado e interp lineal) en vez del grid comparativo")
    args = ap.parse_args()

    exp = Path(args.exp).resolve()
    config = json.loads((exp / "config_usada.json").read_text(encoding="utf-8"))
    clip = config["clip"]
    gt_dir = RAIZ / "data" / "clips" / clip
    render_dir = exp / "frames_renderizados"

    hi = json.loads((exp / "holdout_indices.json").read_text(encoding="utf-8"))
    n = int(hi["n_frames"])
    sup_sorted = sorted(hi["supervisados"])
    hold_set = set(hi["holdout"])
    prev, nxt = _vecinos_sup(n, sup_sorted, hold_set)

    salida = Path(args.salida) if args.salida else (exp / "comparativo_holdout.mp4")

    print(f"=== video_comparativo_holdout ===", flush=True)
    print(f"  exp     : {exp.name}", flush=True)
    print(f"  clip GT : {gt_dir}", flush=True)
    print(f"  frames  : {n}  (holdout={len(hold_set)})", flush=True)

    # --- pasada 1: PSNR holdout de cada estrategia (solo frames holdout vs GT) ---
    acc = {"Congelado": [], "Negro": [], "Interp lineal": [], "Tu metodo": []}
    negro = None
    for j in sorted(hold_set):
        gt = _cargar(gt_dir, j)
        if negro is None:
            negro = np.zeros_like(gt)
        cong = _cargar(gt_dir, prev[j])
        a, b = prev[j], nxt[j]
        w = 0.0 if b == a else (j - a) / (b - a)
        lin = ((1 - w) * _cargar(gt_dir, a).astype(np.float64) + w * _cargar(gt_dir, b).astype(np.float64)).astype(np.uint8)
        met = _cargar(render_dir, j)
        acc["Congelado"].append(_psnr(cong, gt))
        acc["Negro"].append(_psnr(negro, gt))
        acc["Interp lineal"].append(_psnr(lin, gt))
        acc["Tu metodo"].append(_psnr(met, gt))
    psnr_med = {k: float(np.mean(v)) for k, v in acc.items()}
    print("  PSNR holdout por estrategia:", flush=True)
    for k in ["Negro", "Congelado", "Interp lineal", "Tu metodo"]:
        print(f"    {k:14} {psnr_med[k]:6.2f} dB", flush=True)

    # --- modo separados: un mp4 limpio (full-res) por estrategia ---
    if args.separados:
        def _frame_estrategia(j, cual):
            if j not in hold_set:
                return _cargar(gt_dir, j)               # disponible -> GT real
            if cual == "congelado":
                return _cargar(gt_dir, prev[j])
            if cual == "interp_lineal":
                a, b = prev[j], nxt[j]
                w = 0.0 if b == a else (j - a) / (b - a)
                return ((1 - w) * _cargar(gt_dir, a).astype(np.float64)
                        + w * _cargar(gt_dir, b).astype(np.float64)).astype(np.uint8)
            raise ValueError(cual)

        for cual in ("congelado", "interp_lineal"):
            ruta = exp / f"holdout_{cual}.mp4"
            with imageio.get_writer(str(ruta), fps=args.fps, codec="libx264", quality=8) as wr:
                for j in range(n):
                    wr.append_data(_frame_estrategia(j, cual))
                    if j == 0 or (j + 1) % 100 == 0 or j == n - 1:
                        print(f"  [{cual}] frame {j + 1}/{n}", flush=True)
            print(f"  video -> {ruta}", flush=True)
        print("=== listo (separados) ===", flush=True)
        return

    # --- dimensiones del panel ---
    gt0 = _cargar(gt_dir, 0)
    H, W, _ = gt0.shape
    hs, ws = int(H * args.escala), int(W * args.escala)
    barra = 22  # franja de titulo
    cel_h, cel_w = hs + barra, ws
    grid_h, grid_w = cel_h * 2, cel_w * 3

    def _panel(arr, titulo, es_hold, marcar=True):
        img = Image.fromarray(arr).resize((ws, hs))
        cel = Image.new("RGB", (cel_w, cel_h), (20, 20, 20))
        cel.paste(img, (0, barra))
        d = ImageDraw.Draw(cel)
        d.text((4, 5), titulo, fill=(255, 255, 255))
        if marcar and es_hold:
            d.rectangle([0, barra, cel_w - 1, cel_h - 1], outline=(255, 40, 40), width=3)
            d.text((cel_w - 70, 5), "HOLDOUT", fill=(255, 80, 80))
        return np.asarray(cel)

    def _leyenda(es_hold):
        cel = Image.new("RGB", (cel_w, cel_h), (20, 20, 20))
        d = ImageDraw.Draw(cel)
        d.text((6, barra + 6), "PSNR en frames ocultos:", fill=(255, 255, 255))
        y = barra + 28
        for k in ["Negro", "Congelado", "Interp lineal", "Tu metodo"]:
            col = (120, 255, 120) if k == "Tu metodo" else (200, 200, 200)
            d.text((10, y), f"{k:14} {psnr_med[k]:5.1f} dB", fill=col)
            y += 18
        d.text((6, y + 6), f"frame {'(rellenado)' if es_hold else '(real)'}",
               fill=(255, 80, 80) if es_hold else (120, 200, 255))
        return np.asarray(cel)

    # --- pasada 2: construir el video ---
    salida.parent.mkdir(parents=True, exist_ok=True)
    with imageio.get_writer(str(salida), fps=args.fps, codec="libx264", quality=8) as wr:
        for j in range(n):
            es_hold = j in hold_set
            gt = _cargar(gt_dir, j)
            if es_hold:
                cong = _cargar(gt_dir, prev[j])
                a, b = prev[j], nxt[j]
                w = 0.0 if b == a else (j - a) / (b - a)
                lin = ((1 - w) * _cargar(gt_dir, a).astype(np.float64) + w * _cargar(gt_dir, b).astype(np.float64)).astype(np.uint8)
                met = _cargar(render_dir, j)
                neg = np.zeros_like(gt)
            else:
                cong = lin = met = gt
                neg = gt

            fila0 = np.concatenate([
                _panel(gt, "1. GT (original)", es_hold, marcar=False),
                _panel(cong, "2. Congelado", es_hold),
                _panel(neg, "3. Negro", es_hold),
            ], axis=1)
            fila1 = np.concatenate([
                _panel(lin, "4. Interp lineal", es_hold),
                _panel(met, "5. Tu metodo", es_hold),
                _leyenda(es_hold),
            ], axis=1)
            wr.append_data(np.concatenate([fila0, fila1], axis=0))

            if j == 0 or (j + 1) % 100 == 0 or j == n - 1:
                print(f"  frame {j + 1}/{n}", flush=True)

    print(f"=== video -> {salida} ===", flush=True)


if __name__ == "__main__":
    main()
