"""
EXP 1 - Evolucion temporal del entrenamiento.

Lee los artefactos LIGEROS que el trainer guardo durante el job (sin GPU):
  - outputs/<exp>/verificacion/epoch{NNNN}_frame{JJJJ}.png   (frames rasterizados)
  - outputs/<exp>/evol_mu/epoch{NNNN}.npz                      (trayectorias mu marcadas)

Genera:
  1. Una "tira de convergencia" por frame: el mismo frame visto en epoch
     1 | 400 | 800 | ... | final, en grid (PNG) y opcionalmente GIF.
  2. Un plot de como EVOLUCIONA la posicion de las gaussianas marcadas:
     para cada gaussiana, su trayectoria mu_i(t) dibujada en cada epoch
     (color = epoch), mostrando como se va ajustando el recorrido.

Todo es lectura de PNG/NPZ + matplotlib: corre en la Mac, sin CUDA.

Uso:
    python scripts/viz_tira_evolucion.py --exp outputs_khipu/musical_20s_1600ep
    python scripts/viz_tira_evolucion.py --exp <carpeta> --gif
"""
import argparse
import re
from collections import defaultdict
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image


_RE_VERIF = re.compile(r"epoch(\d+)_frame(\d+)\.png$")
_RE_EVOL = re.compile(r"epoch(\d+)\.npz$")


def _leer_verificacion(carpeta_verif):
    """Devuelve dict {frame_idx: [(epoch, ruta_png), ...] ordenado por epoch}."""
    por_frame = defaultdict(list)
    for p in sorted(carpeta_verif.glob("epoch*_frame*.png")):
        m = _RE_VERIF.search(p.name)
        if not m:
            continue
        epoch = int(m.group(1))
        frame = int(m.group(2))
        por_frame[frame].append((epoch, p))
    for frame in por_frame:
        por_frame[frame].sort(key=lambda t: t[0])
    return por_frame


def _grid_convergencia(epoch_pngs, ruta_salida, titulo):
    n = len(epoch_pngs)
    fig, axes = plt.subplots(1, n, figsize=(3.2 * n, 3.6), dpi=120)
    if n == 1:
        axes = [axes]
    for ax, (epoch, ruta) in zip(axes, epoch_pngs):
        ax.imshow(np.asarray(Image.open(ruta).convert("RGB")))
        ax.set_title(f"epoch {epoch}", fontsize=11)
        ax.axis("off")
    fig.suptitle(titulo, fontsize=13)
    fig.tight_layout()
    fig.savefig(ruta_salida)
    plt.close(fig)
    print(f"  grid -> {ruta_salida}", flush=True)


def _gif_convergencia(epoch_pngs, ruta_salida, fps=2):
    frames = []
    for epoch, ruta in epoch_pngs:
        img = Image.open(ruta).convert("RGB")
        frames.append(img)
    if not frames:
        return
    dur_ms = int(1000 / max(1, fps))
    frames[0].save(
        ruta_salida, save_all=True, append_images=frames[1:],
        duration=dur_ms, loop=0,
    )
    print(f"  gif  -> {ruta_salida}", flush=True)


def _plot_evolucion_mu(carpeta_evol, ruta_salida, max_gauss=12):
    """
    Para cada gaussiana marcada, dibuja su trayectoria mu(t) en cada epoch.
    Color del mas claro (epoch temprano) al mas oscuro (epoch final).
    """
    npzs = sorted(carpeta_evol.glob("epoch*.npz"), key=lambda p: int(_RE_EVOL.search(p.name).group(1)))
    if not npzs:
        print("  (no hay evol_mu/*.npz, salto plot de evolucion de mu)", flush=True)
        return

    datos = []  # (epoch, mu (k,2,T), idx (k,))
    for p in npzs:
        d = np.load(p)
        datos.append((int(d["epoch"]), d["mu"], d["idx"]))

    k_total = datos[0][1].shape[0]
    k = min(max_gauss, k_total)
    n_epochs = len(datos)
    cmap = plt.get_cmap("viridis")

    ncol = 4
    nrow = int(np.ceil(k / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(3.0 * ncol, 3.0 * nrow), dpi=110)
    axes = np.atleast_1d(axes).flatten()

    for gi in range(k):
        ax = axes[gi]
        for ei, (epoch, mu, idx) in enumerate(datos):
            col = cmap(ei / max(1, n_epochs - 1))
            fila = mu[gi, 0, :]  # y
            columna = mu[gi, 1, :]  # x
            ax.plot(columna, fila, color=col, alpha=0.85, linewidth=1.2,
                    label=f"ep{epoch}" if gi == 0 else None)
        ax.set_title(f"gauss {int(datos[-1][2][gi])}", fontsize=9)
        ax.invert_yaxis()
        ax.set_aspect("equal", adjustable="datalim")
        ax.tick_params(labelsize=7)
    for gi in range(k, len(axes)):
        axes[gi].axis("off")

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=min(n_epochs, 8), fontsize=8)
    fig.suptitle("Evolucion de la trayectoria mu_i(t) por epoch (claro=temprano, oscuro=final)", fontsize=12)
    fig.tight_layout(rect=(0, 0.05, 1, 0.97))
    fig.savefig(ruta_salida)
    plt.close(fig)
    print(f"  evol_mu -> {ruta_salida}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp", required=True, help="carpeta del experimento (outputs/<exp>)")
    ap.add_argument("--salida", default=None, help="carpeta de salida (default: <exp>/viz_evolucion)")
    ap.add_argument("--gif", action="store_true", help="ademas genera GIF de convergencia por frame")
    ap.add_argument("--fps", type=int, default=2, help="fps del GIF de convergencia")
    ap.add_argument("--max_gauss", type=int, default=12, help="cuantas gaussianas en el plot de evolucion de mu")
    args = ap.parse_args()

    exp = Path(args.exp).resolve()
    salida = Path(args.salida).resolve() if args.salida else (exp / "viz_evolucion")
    salida.mkdir(parents=True, exist_ok=True)

    carpeta_verif = exp / "verificacion"
    carpeta_evol = exp / "evol_mu"

    print(f"=== viz_tira_evolucion ===\n  exp    : {exp}\n  salida : {salida}", flush=True)

    if carpeta_verif.is_dir():
        por_frame = _leer_verificacion(carpeta_verif)
        for frame, epoch_pngs in sorted(por_frame.items()):
            titulo = f"Convergencia frame {frame}  ({len(epoch_pngs)} checkpoints)"
            _grid_convergencia(epoch_pngs, salida / f"convergencia_frame{frame:04d}.png", titulo)
            if args.gif:
                _gif_convergencia(epoch_pngs, salida / f"convergencia_frame{frame:04d}.gif", fps=args.fps)
    else:
        print(f"  AVISO: no existe {carpeta_verif} (corre el job con guardar_verificacion_visual=true)", flush=True)

    if carpeta_evol.is_dir():
        _plot_evolucion_mu(carpeta_evol, salida / "evolucion_mu_por_epoch.png", max_gauss=args.max_gauss)
    else:
        print(f"  AVISO: no existe {carpeta_evol} (corre el job con viz_gaussianas_marcadas_n>0)", flush=True)

    print("=== listo ===", flush=True)


if __name__ == "__main__":
    main()
