"""
EXP 3 - Elipses de las gaussianas + campo de velocidades.

Lee un checkpoint .pt final y, para un frame dado, genera:

  (a) ELIPSES: cada gaussiana 2D es una elipse definida por su scale (sigma_x,
      sigma_y) y su rotacion theta. Dibuja el contorno de las gaussianas mas
      opacas sobre el frame -> muestra como el modelo reparte y orienta su
      "presupuesto" de gaussianas (alargadas en bordes, pequenas en detalle).

  (b) VELOCIDADES: la velocidad de la gaussiana i en el frame t es
      v_i(t) = mu_i(t+1) - mu_i(t). Se dibuja como flechas (quiver) sobre el
      frame -> mapa de flujo de hacia donde se mueven las gaussianas. Util para
      contrastar zonas de mucho vs poco movimiento.

Todo es evaluacion de polinomios + matplotlib: corre en la Mac, SIN CUDA.

Uso:
    python scripts/viz_elipses_velocidades.py \
        --checkpoint outputs_khipu/musical_20s_1600ep/checkpoints/checkpoint_final.pt \
        --frame 240 --fondo_dir outputs_khipu/musical_20s_1600ep/frames_renderizados
"""
import argparse
from pathlib import Path

import numpy as np
import torch
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
from matplotlib.collections import PatchCollection

from _carga_checkpoint import cargar_modelo_desde_checkpoint, cargar_frame_fondo


@torch.no_grad()
def _evaluar(a0, hi, B):
    return torch.cat([a0, hi], dim=-1) @ B.T


@torch.no_grad()
def _params_en_frame(modelo, matrices_base, j):
    """mu (N,2), scale (N,2) activado, theta (N,), opacity (N,) en el frame j."""
    out = modelo.evaluar_en_frame(j, matrices_base)
    return out["mu"], out["scale"], out["theta"], out["opacity"]


def _fondo(fondo_dir, config, indice, H, W):
    if fondo_dir is not None:
        ruta = Path(fondo_dir) / f"frame_{int(indice):04d}.png"
        if ruta.is_file():
            from PIL import Image
            return np.asarray(Image.open(ruta).convert("RGB"), dtype=np.float32) / 255.0
    gt = cargar_frame_fondo(config, indice)
    if gt is not None:
        return gt.numpy()
    return np.zeros((H, W, 3), dtype=np.float32)


def _plot_elipses(mu, scale, theta, opacity, fondo, ruta, n_max, k_sigma):
    mu = mu.cpu().numpy(); scale = scale.cpu().numpy()
    theta = theta.cpu().numpy(); op = opacity.cpu().numpy()

    orden = np.argsort(-op)[:n_max]
    H, W = fondo.shape[:2]
    fig, ax = plt.subplots(figsize=(10, 10 * H / W), dpi=120)
    ax.imshow(np.clip(fondo, 0, 1))

    elipses = []
    for i in orden:
        cy, cx = mu[i, 0], mu[i, 1]
        sy, sx = scale[i, 0], scale[i, 1]
        ang = np.degrees(theta[i])
        e = Ellipse((cx, cy), width=2 * k_sigma * sx, height=2 * k_sigma * sy, angle=ang)
        elipses.append(e)
    pc = PatchCollection(elipses, facecolor="none", edgecolor="cyan", linewidths=0.5, alpha=0.6)
    ax.add_collection(pc)
    ax.set_xlim(0, W); ax.set_ylim(H, 0); ax.axis("off")
    ax.set_title(f"Elipses de las {len(orden)} gaussianas mas opacas (k_sigma={k_sigma})")
    fig.tight_layout(); fig.savefig(ruta); plt.close(fig)
    print(f"  elipses -> {ruta}", flush=True)


def _plot_velocidades(modelo, matrices_base, j, fondo, ruta, n_max, escala_flecha):
    n_frames = modelo.n_frames
    j1 = min(j + 1, n_frames - 1)
    mu_j, _, _, op_j = _params_en_frame(modelo, matrices_base, j)
    mu_j1, _, _, _ = _params_en_frame(modelo, matrices_base, j1)
    v = (mu_j1 - mu_j).cpu().numpy()      # (N,2) -> (dy, dx)
    mu = mu_j.cpu().numpy(); op = op_j.cpu().numpy()

    orden = np.argsort(-op)[:n_max]
    H, W = fondo.shape[:2]
    fig, ax = plt.subplots(figsize=(10, 10 * H / W), dpi=120)
    ax.imshow(np.clip(fondo, 0, 1))
    mag = np.linalg.norm(v[orden], axis=1)
    q = ax.quiver(mu[orden, 1], mu[orden, 0], v[orden, 1], v[orden, 0], mag,
                  angles="xy", scale_units="xy", scale=1.0 / max(1e-6, escala_flecha),
                  cmap="plasma", width=0.002)
    plt.colorbar(q, ax=ax, fraction=0.025, label="|v| px/frame")
    ax.set_xlim(0, W); ax.set_ylim(H, 0); ax.axis("off")
    ax.set_title(f"Campo de velocidades mu(t+1)-mu(t) en frame {j}")
    fig.tight_layout(); fig.savefig(ruta); plt.close(fig)
    print(f"  velocidades -> {ruta}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--salida", default=None)
    ap.add_argument("--frame", type=int, default=0)
    ap.add_argument("--fondo_dir", default=None)
    ap.add_argument("--n_max", type=int, default=2000, help="cuantas gaussianas (las mas opacas) dibujar")
    ap.add_argument("--k_sigma", type=float, default=2.0, help="tamano de elipse en sigmas")
    ap.add_argument("--escala_flecha", type=float, default=5.0, help="amplifica las flechas de velocidad")
    args = ap.parse_args()

    ckpt = Path(args.checkpoint).resolve()
    salida = Path(args.salida).resolve() if args.salida else (ckpt.parent.parent / "viz_elipses_vel")
    salida.mkdir(parents=True, exist_ok=True)

    print(f"=== viz_elipses_velocidades ===\n  checkpoint: {ckpt}\n  frame     : {args.frame}\n  salida    : {salida}", flush=True)
    modelo, config, matrices_base, info = cargar_modelo_desde_checkpoint(ckpt, device="cpu")
    H, W = info["H"], info["W"]
    fondo = _fondo(args.fondo_dir, config, args.frame, H, W)

    mu, scale, theta, opacity = _params_en_frame(modelo, matrices_base, args.frame)
    _plot_elipses(mu, scale, theta, opacity, fondo,
                  salida / f"elipses_frame{args.frame:04d}.png", args.n_max, args.k_sigma)
    _plot_velocidades(modelo, matrices_base, args.frame, fondo,
                      salida / f"velocidades_frame{args.frame:04d}.png", args.n_max, args.escala_flecha)
    print("=== listo ===", flush=True)


if __name__ == "__main__":
    main()
