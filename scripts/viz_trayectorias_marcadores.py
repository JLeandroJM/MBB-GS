"""
EXP 2 - Marcadores y seguimiento de gaussianas.

Lee un checkpoint .pt final, selecciona un set de gaussianas y:
  (a) PNG estatico: las trayectorias mu_i(t) de las marcadas sobre un frame,
      con circulo + numero en cada una.
  (b) GIF/secuencia: el marcador (circulo + id) MOVIENDOSE frame a frame,
      sobre el frame de fondo correspondiente (render reconstruido o GT).

Todo es evaluacion de polinomios Chebyshev (matmul) + matplotlib: corre en la
Mac, SIN CUDA. No rasteriza gaussianas; usa como fondo PNGs ya existentes.

Modos de seleccion (--modo):
  top      : las N de mayor opacidad media (las mas visibles).
  aleatorio: N al azar (semilla fija) entre las visibles.
  region   : N dentro de un bounding box --bbox y0 x0 y1 x1 (en el frame 0).

Uso:
    python scripts/viz_trayectorias_marcadores.py \
        --checkpoint outputs_khipu/musical_20s_1600ep/checkpoints/checkpoint_final.pt \
        --modo top --n 12 --fondo_dir outputs_khipu/musical_20s_1600ep/frames_renderizados \
        --gif
"""
import argparse
from pathlib import Path

import numpy as np
import torch
import matplotlib.pyplot as plt
from matplotlib.patches import Circle

from _carga_checkpoint import cargar_modelo_desde_checkpoint, cargar_frame_fondo, RAIZ


@torch.no_grad()
def _evaluar(a0, hi, B):
    return torch.cat([a0, hi], dim=-1) @ B.T


@torch.no_grad()
def _trayectorias_y_opacidad(modelo, matrices_base):
    g_mu = modelo.grados["mu"]
    g_op = modelo.grados["opacity"]
    mu_t = _evaluar(modelo.mu_a0, modelo.mu_high, matrices_base[g_mu])            # (N,2,T)
    op_t = torch.sigmoid(_evaluar(modelo.opacity_a0, modelo.opacity_high, matrices_base[g_op]).squeeze(1))  # (N,T)
    return mu_t, op_t


def _seleccionar(mu_t, op_t, modo, n, bbox, semilla):
    op_max = op_t.max(dim=-1).values
    op_mean = op_t.mean(dim=-1)
    visibles = (op_max > 0.1).nonzero(as_tuple=False).squeeze(-1)

    if modo == "top":
        orden = torch.argsort(op_mean[visibles], descending=True)
        return visibles[orden[:n]]
    if modo == "aleatorio":
        g = torch.Generator().manual_seed(int(semilla))
        perm = torch.randperm(len(visibles), generator=g)
        return visibles[perm[:n]]
    if modo == "region":
        y0, x0, y1, x1 = bbox
        mu0 = mu_t[:, :, 0]  # posicion en frame 0: (N,2) -> (fila,col)
        dentro = ((mu0[:, 0] >= y0) & (mu0[:, 0] <= y1) &
                  (mu0[:, 1] >= x0) & (mu0[:, 1] <= x1))
        cand = (dentro & (op_max > 0.1)).nonzero(as_tuple=False).squeeze(-1)
        orden = torch.argsort(op_mean[cand], descending=True)
        return cand[orden[:n]]
    raise ValueError(f"modo desconocido: {modo}")


def _fondo(fondo_dir, config, indice, H, W):
    """Carga el frame de fondo: prioriza fondo_dir, luego el clip GT, luego negro."""
    if fondo_dir is not None:
        ruta = Path(fondo_dir) / f"frame_{int(indice):04d}.png"
        if ruta.is_file():
            from PIL import Image
            return np.asarray(Image.open(ruta).convert("RGB"), dtype=np.float32) / 255.0
    gt = cargar_frame_fondo(config, indice)
    if gt is not None:
        return gt.numpy()
    return np.zeros((H, W, 3), dtype=np.float32)


def _png_estatico(mu_t, idx, fondo, ruta, op_mean):
    mu_np = mu_t.cpu().numpy()
    fig, ax = plt.subplots(figsize=(9, 9 * fondo.shape[0] / fondo.shape[1]), dpi=120)
    ax.imshow(np.clip(fondo, 0, 1))
    cmap = plt.get_cmap("tab20")
    for k, i in enumerate(idx.cpu().numpy()):
        col = cmap(k % 20)
        ax.plot(mu_np[i, 1, :], mu_np[i, 0, :], color=col, linewidth=1.6, alpha=0.9)
        y0, x0 = mu_np[i, 0, 0], mu_np[i, 1, 0]
        ax.add_patch(Circle((x0, y0), radius=6, fill=False, edgecolor=col, linewidth=2))
        ax.text(x0 + 8, y0 - 8, str(int(i)), color=col, fontsize=9, weight="bold")
    ax.set_xlim(0, fondo.shape[1]); ax.set_ylim(fondo.shape[0], 0)
    ax.axis("off")
    ax.set_title(f"Trayectorias de {len(idx)} gaussianas marcadas")
    fig.tight_layout(); fig.savefig(ruta); plt.close(fig)
    print(f"  png estatico -> {ruta}", flush=True)


def _gif_animado(mu_t, idx, config, fondo_dir, H, W, ruta, fps, paso):
    from PIL import Image
    T = mu_t.shape[-1]
    mu_np = mu_t.cpu().numpy()
    cmap = plt.get_cmap("tab20")
    idx_np = idx.cpu().numpy()
    frames_out = []
    for j in range(0, T, max(1, paso)):
        fondo = _fondo(fondo_dir, config, j, H, W)
        fig, ax = plt.subplots(figsize=(7, 7 * H / W), dpi=100)
        ax.imshow(np.clip(fondo, 0, 1))
        for k, i in enumerate(idx_np):
            col = cmap(k % 20)
            # estela corta de los ultimos frames
            j0 = max(0, j - 15)
            ax.plot(mu_np[i, 1, j0:j + 1], mu_np[i, 0, j0:j + 1], color=col, linewidth=1.2, alpha=0.7)
            ax.add_patch(Circle((mu_np[i, 1, j], mu_np[i, 0, j]), radius=6, fill=False, edgecolor=col, linewidth=2))
            ax.text(mu_np[i, 1, j] + 8, mu_np[i, 0, j] - 8, str(int(i)), color=col, fontsize=8, weight="bold")
        ax.set_xlim(0, W); ax.set_ylim(H, 0); ax.axis("off")
        ax.set_title(f"frame {j}", fontsize=10)
        fig.tight_layout()
        fig.canvas.draw()
        buf = np.asarray(fig.canvas.buffer_rgba())[:, :, :3]
        frames_out.append(Image.fromarray(buf.copy()))
        plt.close(fig)
    if frames_out:
        frames_out[0].save(ruta, save_all=True, append_images=frames_out[1:],
                           duration=int(1000 / max(1, fps)), loop=0)
        print(f"  gif animado -> {ruta}  ({len(frames_out)} frames)", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--salida", default=None)
    ap.add_argument("--modo", choices=["top", "aleatorio", "region"], default="top")
    ap.add_argument("--n", type=int, default=12)
    ap.add_argument("--bbox", type=float, nargs=4, default=None, metavar=("y0", "x0", "y1", "x1"))
    ap.add_argument("--semilla", type=int, default=0)
    ap.add_argument("--frame_fondo", type=int, default=0, help="frame de fondo para el PNG estatico")
    ap.add_argument("--fondo_dir", default=None, help="carpeta con frame_NNNN.png (render reconstruido)")
    ap.add_argument("--gif", action="store_true")
    ap.add_argument("--fps", type=int, default=12)
    ap.add_argument("--paso", type=int, default=2, help="usar 1 de cada 'paso' frames en el GIF")
    args = ap.parse_args()

    ckpt = Path(args.checkpoint).resolve()
    salida = Path(args.salida).resolve() if args.salida else (ckpt.parent.parent / "viz_marcadores")
    salida.mkdir(parents=True, exist_ok=True)

    print(f"=== viz_trayectorias_marcadores ===\n  checkpoint: {ckpt}\n  salida    : {salida}", flush=True)
    modelo, config, matrices_base, info = cargar_modelo_desde_checkpoint(ckpt, device="cpu")
    H, W = info["H"], info["W"]
    mu_t, op_t = _trayectorias_y_opacidad(modelo, matrices_base)
    op_mean = op_t.mean(dim=-1)

    idx = _seleccionar(mu_t, op_t, args.modo, args.n, args.bbox, args.semilla)
    print(f"  modo={args.modo}  seleccionadas={idx.tolist()}", flush=True)

    fondo = _fondo(args.fondo_dir, config, args.frame_fondo, H, W)
    _png_estatico(mu_t, idx, fondo, salida / f"marcadores_{args.modo}_estatico.png", op_mean)

    if args.gif:
        _gif_animado(mu_t, idx, config, args.fondo_dir, H, W,
                     salida / f"marcadores_{args.modo}_animado.gif", args.fps, args.paso)

    print("=== listo ===", flush=True)


if __name__ == "__main__":
    main()
