"""
Diagnostico MOVIMIENTO vs CROSS-FADE (corre en la Mac, CPU, sin GPU).

Carga un checkpoint y mira las gaussianas que mas ENCIENDEN/APAGAN su opacidad
(candidatas a "la gota"). Para cada una compara:
  - cuanto se DESPLAZA su mu(t)  (px recorridos)  -> si es grande = MOVIMIENTO
  - cuanto varia su opacidad(t)                    -> si mu no se mueve = CROSS-FADE

Veredicto: si las gaussianas que aparecen/desaparecen NO se mueven, el modelo
esta haciendo slow-mo por opacidad (cross-fade). Si SI se mueven, interpola
movimiento real.

Uso:
    python scripts/diag_trayectoria_gota.py \
        --checkpoint outputs_khipu/gota_motion_b5e6/checkpoints/checkpoint_final.pt \
        --salida outputs_khipu/gota_motion_b5e6/diag_trayectoria.png
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import torch
import matplotlib.pyplot as plt

RAIZ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RAIZ / "scripts"))
from _carga_checkpoint import cargar_modelo_desde_checkpoint


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--salida", default=None)
    ap.add_argument("--topk", type=int, default=10, help="cuantas gaussianas 'gota' analizar")
    args = ap.parse_args()

    modelo, config, matrices, info = cargar_modelo_desde_checkpoint(args.checkpoint, device="cpu")
    out = modelo.evaluar_batch_completo(matrices)
    mu = out["mu"].cpu().numpy()          # (T, N, 2) -> (fila, col)
    op = out["opacity"].cpu().numpy()     # (T, N)
    T, N = op.shape

    op_range = op.max(0) - op.min(0)                              # cuanto enciende/apaga
    fila_disp = mu[:, :, 0].max(0) - mu[:, :, 0].min(0)           # recorrido vertical (px)
    col_disp = mu[:, :, 1].max(0) - mu[:, :, 1].min(0)
    mu_disp = np.sqrt(fila_disp ** 2 + col_disp ** 2)            # recorrido total (px)

    # candidatas a "gota": las que mas encienden/apagan
    idx = np.argsort(op_range)[::-1][:args.topk]

    print(f"clip={config.get('clip')}  T={T}  N={N}")
    print(f"beta_smoothness={config.get('beta_smoothness')}  lambda_motion={config.get('lambda_motion')}")
    print(f"\n{'gauss':>7} {'op_range':>9} {'mu_disp(px)':>12} {'fila(px)':>9}  interpretacion")
    print("-" * 60)
    for i in idx:
        interp = "MUEVE" if mu_disp[i] > 15 else ("cross-fade" if op_range[i] > 0.25 else "quieta")
        print(f"{i:7d} {op_range[i]:9.3f} {mu_disp[i]:12.1f} {fila_disp[i]:9.1f}  {interp}")

    disp_medio = float(np.mean(mu_disp[idx]))
    print("-" * 60)
    print(f"desplazamiento medio de las {args.topk} 'gota' = {disp_medio:.1f} px")
    if disp_medio > 15:
        print(">> VEREDICTO: interpola MOVIMIENTO (las gaussianas que aparecen se mueven).")
    else:
        print(">> VEREDICTO: CROSS-FADE (aparecen/desaparecen sin moverse).")

    # plot: fila(t) y opacidad(t) de las candidatas
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5), dpi=110)
    for i in idx:
        ax1.plot(range(T), mu[:, i, 0], alpha=0.8)
        ax2.plot(range(T), op[:, i], alpha=0.8)
    ax1.set_title("fila mu(t) de las gaussianas 'gota'\n(pendiente = se mueve vertical)")
    ax1.set_xlabel("frame"); ax1.set_ylabel("fila (px)"); ax1.invert_yaxis(); ax1.grid(alpha=0.3)
    ax2.set_title("opacidad(t) de las mismas\n(picos = aparece/desaparece)")
    ax2.set_xlabel("frame"); ax2.set_ylabel("opacidad"); ax2.grid(alpha=0.3)
    fig.suptitle(f"{Path(args.checkpoint).parts[-3]}  |  desp_medio={disp_medio:.1f}px", fontsize=11)
    fig.tight_layout()

    salida = args.salida or str(Path(args.checkpoint).parent.parent / "diag_trayectoria.png")
    fig.savefig(salida); plt.close(fig)
    print(f"\n  grafico -> {salida}")


if __name__ == "__main__":
    main()
