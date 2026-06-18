"""
EXP - Inyeccion de un efecto de FUEGO (fogata) en un modelo .pt ya entrenado.

Idea (sistema de particulas con gaussianas):
  El modelo NO es semantico: ninguna gaussiana "sabe" que representa. En vez de
  mover gaussianas del contenido (lo cual dejaria huecos), AÑADIMOS M gaussianas
  nuevas que actuan como PARTICULAS de fuego, con sus parametros (mu, opacity,
  color, scale) definidos PROCEDIMENTALMENTE en el tiempo y proyectados a la base
  de Chebyshev. El contenido original queda intacto; el fuego es una "capa"
  encima, garantizada al frente via `depth` bajo (front-to-back blending).

Modelo de una fogata:
  - Un EMISOR fijo en la base (y0, x0).
  - M particulas. Cada una NACE abajo, SUBE (mu_y decrece), ONDULA en x, BRILLA y
    se APAGA (opacity = joroba sin^2 sobre su ventana de vida), y cambia de COLOR
    de rojo (base/joven) a amarillo (punta/madura). Las vidas estan desfasadas ->
    siempre hay particulas vivas -> la llama parpadea y se renueva.
  - CRECIMIENTO "poco a poco": las particulas que nacen mas tarde son mas grandes
    y mas dispersas (factor g crece con el frame de nacimiento), y hay mas
    nacimientos hacia el final -> la fogata empieza chica y crece a lo largo del
    clip, sin propagarse por toda la imagen.

Por que fuego y no lluvia: el movimiento del fuego (subida monotona + ondulacion
suave + flicker) es C1-suave -> se proyecta limpio a Chebyshev. La lluvia
periodica (gota que cae y reaparece) tiene discontinuidades -> Gibbs.

Que corre donde:
  - --preview  : Mac, SIN CUDA. Evalua los polinomios (matmul) y dibuja la
                 nube de particulas sobre frames de fondo -> valida posicion,
                 tamaño y crecimiento ANTES de rasterizar.
  - export .pt : siempre (CPU). Guarda <ckpt>_fuego.pt con N+M gaussianas, listo
                 para rasterizar en Khipu con
                 regenerar_clip_desde_checkpoint_streaming.py.

Uso:
  # 1) previsualizar en la Mac (env con torch CPU)
  python scripts/inyectar_efecto_fuego.py \
      --checkpoint outputs_khipu/musical_20s_1600ep/checkpoints/checkpoint_final.pt \
      --preview

  # 2) exportar el .pt con fuego (para mandar a Khipu)
  python scripts/inyectar_efecto_fuego.py \
      --checkpoint outputs_khipu/musical_20s_1600ep/checkpoints/checkpoint_final.pt

  # 3) en Khipu: rasterizar
  python scripts/regenerar_clip_desde_checkpoint_streaming.py \
      --checkpoint outputs_khipu/musical_20s_1600ep/checkpoints/checkpoint_final_fuego.pt \
      --salida outputs_khipu/musical_20s_1600ep/frames_fuego --device cuda --crear_video
"""
import argparse
from pathlib import Path

import numpy as np
import torch

from _carga_checkpoint import RAIZ
import sys
SRC = RAIZ / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
from gs2d_video.core.bases import construir_matriz_chebyshev


# ==========================================================================
# PERILLAS  (defaults pensados para musical_20s_1600ep: 720x1080, 480 frames)
# Ajusta mirando el --preview; no necesitas tocar el resto del script.
# ==========================================================================
CFG = {
    # --- emisor (base de la fogata), en pixeles (fila=y, col=x) -----------
    "x0": 540.0,            # centro horizontal (W/2 = 540)
    "y0": 620.0,            # base, abajo-centro (H=720)

    # --- tamaño de la llama AL FINAL del clip (cuando g=1) -----------------
    "altura_max": 380.0,    # cuanto sube la punta desde la base (px) ~0.53*H
    "w_disp": 150.0,        # semi-ancho del penacho (px) -> ancho total ~300 ~0.30*W
    "sigma_x": 9.0,         # sigma horizontal de cada particula (px)
    "sigma_y": 17.0,        # sigma vertical  (alargada -> lengua de fuego)
    "taper": 0.55,          # cuanto se afina la particula al llegar a la punta (0..1)

    # --- densidad / particulas -------------------------------------------
    "M": 280,               # numero de particulas
    "op_pico": 0.80,        # opacidad pico de la joroba de vida
    "amp_ond": 34.0,        # amplitud de la ondulacion lateral (px)

    # --- crecimiento temporal "poco a poco" ------------------------------
    "g_min": 0.12,          # tamaño relativo al inicio del clip (12%)
    "growth_exp": 0.85,     # curva de crecimiento del tamaño (g(p)=g_min+(1-g_min)*p^exp)
    "birth_bias": 0.6,      # <1 sesga los nacimientos hacia el final (mas fuego al final)
    "vida_min": 90,         # duracion de vida de una particula (frames)
    "vida_max": 150,

    # --- color (RGB en [0,1]) --------------------------------------------
    "color_base": (0.95, 0.16, 0.02),   # rojo (joven / base)
    "color_punta": (1.00, 0.85, 0.28),  # amarillo (madura / punta)

    # --- profundidad: < contenido para quedar DELANTE (front-to-back) -----
    "fire_depth": -8.0,

    "semilla": 7,
}


# ==========================================================================
# utilidades
# ==========================================================================
def smoothstep(x):
    x = np.clip(x, 0.0, 1.0)
    return x * x * (3.0 - 2.0 * x)


def _logit(p, lo=1e-4, hi=1 - 1e-4):
    p = np.clip(p, lo, hi)
    return np.log(p / (1.0 - p))


def proyectar_chebyshev(B, f):
    """
    Minimos cuadrados: encuentra coefs c (grado+1, K) tal que B @ c ~ f.
    B: (T, grado+1) float64.  f: (T,) o (T, K).  Devuelve c (grado+1, K) float32.
    """
    f2 = f.reshape(f.shape[0], -1).astype(np.float64)
    c, *_ = np.linalg.lstsq(B, f2, rcond=None)
    return c.astype(np.float32)            # (grado+1, K)


# ==========================================================================
# generacion de las particulas de fuego (en el espacio RAW, listo para proyectar)
# ==========================================================================
def generar_targets(cfg, H, W, T):
    """
    Devuelve un dict {nombre: array (M, dim, T)} con los valores OBJETIVO ya en el
    espacio RAW (antes de la activacion del modelo):
        mu_raw      : (M, 2, T)   px (fila, col)        -> sin activacion
        opacity_raw : (M, 1, T)   logit de la opacidad  -> sigmoid en el modelo
        color_raw   : (M, 3, T)   logit del color       -> sigmoid en el modelo
        scale_raw   : (M, 2, T)   log(sigma_px)         -> exp en el modelo
        theta_raw   : (M, 1, T)   radianes              -> sin activacion
        depth_raw   : (M, 1, T)   constante             -> sin activacion
    """
    rng = np.random.default_rng(cfg["semilla"])
    M = cfg["M"]
    x0, y0 = cfg["x0"], cfg["y0"]
    js = np.arange(T, dtype=np.float64)                      # frames 0..T-1

    mu = np.zeros((M, 2, T), dtype=np.float64)
    op = np.zeros((M, 1, T), dtype=np.float64)
    col = np.zeros((M, 3, T), dtype=np.float64)
    sca = np.zeros((M, 2, T), dtype=np.float64)
    cbase = np.array(cfg["color_base"]); cpunta = np.array(cfg["color_punta"])

    for i in range(M):
        # --- nacimiento sesgado al final + factor de crecimiento ----------
        u = rng.random()
        bc = u ** cfg["birth_bias"]                          # in [0,1], sesgo a 1
        jc = bc * (T - 1)                                    # frame centro de vida
        g = cfg["g_min"] + (1 - cfg["g_min"]) * (jc / (T - 1)) ** cfg["growth_exp"]

        L = rng.integers(cfg["vida_min"], cfg["vida_max"] + 1)
        j_start = jc - L / 2.0

        altura = cfg["altura_max"] * g * rng.uniform(0.7, 1.0)
        base_off = rng.uniform(-cfg["w_disp"], cfg["w_disp"]) * g
        sigx = cfg["sigma_x"] * g * rng.uniform(0.8, 1.2)
        sigy = cfg["sigma_y"] * g * rng.uniform(0.8, 1.2)
        amp = cfg["amp_ond"] * rng.uniform(0.6, 1.2)
        freq = rng.uniform(0.8, 1.8)
        phase = rng.uniform(0, 2 * np.pi)

        tau = (js - j_start) / L                             # progreso de vida (puede salir de [0,1])
        a = smoothstep(tau)                                  # 0..1 altura, C1-suave
        vivo = (tau >= 0.0) & (tau <= 1.0)
        hump = np.where(vivo, np.sin(np.pi * np.clip(tau, 0, 1)) ** 2, 0.0)  # joroba C1

        mu[i, 0, :] = y0 - altura * a                        # sube (fila decrece)
        mu[i, 1, :] = x0 + base_off + amp * a * np.sin(2 * np.pi * freq * tau + phase)
        op[i, 0, :] = cfg["op_pico"] * hump

        s = (1.0 - cfg["taper"] * a)                         # se afina hacia la punta
        sca[i, 0, :] = np.maximum(sigy * s, 0.6)             # fila = vertical
        sca[i, 1, :] = np.maximum(sigx * s, 0.6)             # col  = horizontal

        mezcla = a[None, :]                                  # (1,T)
        col[i, :, :] = (cbase[:, None] * (1 - mezcla) + cpunta[:, None] * mezcla)

    # --- a espacio RAW (activaciones inversas) ----------------------------
    targets = {
        "mu":      mu.astype(np.float32),                    # directo
        "opacity": _logit(op).astype(np.float32),
        "color":   _logit(col).astype(np.float32),
        "scale":   np.log(np.clip(sca, 0.5, max(H, W))).astype(np.float32),
        "theta":   np.zeros((M, 1, T), dtype=np.float32),
        "depth":   np.full((M, 1, T), cfg["fire_depth"], dtype=np.float32),
    }
    return targets


def targets_a_coefs(targets, grados, T):
    """
    Proyecta cada target (M, dim, T) a coefs de Chebyshev del grado del parametro
    y los parte en a0 / high con las MISMAS shapes que el modelo:
        params 'planos' (opacity, theta, depth): a0 (M,1)      high (M, grado)
        params 'dim>1'  (mu, color, scale):      a0 (M,dim,1)  high (M, dim, grado)
    """
    Bs = {g: construir_matriz_chebyshev(T, g, dtype=torch.float64).numpy()
          for g in sorted(set(grados.values()))}
    planos = {"opacity", "theta", "depth"}
    out = {}
    for nombre, arr in targets.items():
        g = grados[nombre]
        B = Bs[g]                                            # (T, g+1)
        M, dim, _ = arr.shape
        a0 = np.zeros((M, dim, 1), dtype=np.float32)
        hi = np.zeros((M, dim, g), dtype=np.float32)
        for i in range(M):
            c = proyectar_chebyshev(B, arr[i].T)             # (g+1, dim)
            a0[i, :, 0] = c[0]
            hi[i] = c[1:].T                                  # (dim, g)
        if nombre in planos:                                 # aplanar a (M, .)
            out[f"{nombre}_a0"] = torch.from_numpy(a0[:, 0, :])      # (M,1)
            out[f"{nombre}_high"] = torch.from_numpy(hi[:, 0, :])    # (M,g)
        else:
            out[f"{nombre}_a0"] = torch.from_numpy(a0)               # (M,dim,1)
            out[f"{nombre}_high"] = torch.from_numpy(hi)             # (M,dim,g)
    return out


# ==========================================================================
# preview (Mac, sin CUDA): nube de particulas sobre frames de fondo
# ==========================================================================
def preview(cfg, sd, fuego, grados, H, W, T, fondo_dir, salida, frames):
    import matplotlib.pyplot as plt
    from PIL import Image

    Bs = {g: construir_matriz_chebyshev(T, g, dtype=torch.float32)
          for g in sorted(set(grados.values()))}

    def eval_param(nombre, j):
        a0 = fuego[f"{nombre}_a0"]; hi = fuego[f"{nombre}_high"]
        B = Bs[grados[nombre]]
        coefs = torch.cat([a0, hi], dim=-1)                  # (M,[dim,]g+1)
        return coefs @ B[j]                                  # (M,[dim])

    def fondo(j):
        if fondo_dir is not None:
            ruta = Path(fondo_dir) / f"frame_{int(j):04d}.png"
            if ruta.is_file():
                return np.asarray(Image.open(ruta).convert("RGB"), np.float32) / 255.0
        return np.zeros((H, W, 3), np.float32)

    n = len(frames)
    fig, axes = plt.subplots(1, n, figsize=(4.2 * n, 4.2 * H / W), dpi=120)
    axes = np.atleast_1d(axes)
    for ax, j in zip(axes, frames):
        ax.imshow(np.clip(fondo(j), 0, 1))
        mu = eval_param("mu", j).numpy()                     # (M,2)
        op = torch.sigmoid(eval_param("opacity", j)).numpy() # (M,)
        cl = torch.sigmoid(eval_param("color", j)).numpy()   # (M,3)
        sc = torch.exp(eval_param("scale", j).clamp(np.log(0.5), np.log(max(H, W)))).numpy()
        vis = op > 0.05
        ax.scatter(mu[vis, 1], mu[vis, 0],
                   s=(sc[vis].mean(1) ** 2) * 0.05,
                   c=np.clip(cl[vis], 0, 1), alpha=np.clip(op[vis], 0, 1))
        ax.set_xlim(0, W); ax.set_ylim(H, 0); ax.axis("off")
        ax.set_title(f"frame {j}  (vivas={int(vis.sum())})", fontsize=10)
    fig.suptitle("Preview fogata (nube de particulas, NO es el render real)", fontsize=12)
    fig.tight_layout()
    ruta = Path(salida) / "preview_fuego.png"
    fig.savefig(ruta); plt.close(fig)
    print(f"  preview -> {ruta}", flush=True)


# ==========================================================================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--salida", default=None, help="ruta del .pt de salida (default: <ckpt>_fuego.pt)")
    ap.add_argument("--preview", action="store_true", help="solo genera preview PNG en la Mac, no exporta")
    ap.add_argument("--fondo_dir", default=None, help="carpeta frame_NNNN.png para el preview (default: frames_renderizados del exp)")
    ap.add_argument("--frames_preview", type=int, nargs="*", default=None, help="frames a previsualizar")
    args = ap.parse_args()

    ckpt_path = Path(args.checkpoint).resolve()
    ckpt = torch.load(str(ckpt_path), map_location="cpu")
    sd = ckpt["state_dict_coefs"]
    grados = dict(sd["grados"])
    H, W, T, N = int(sd["H"]), int(sd["W"]), int(sd["n_frames"]), int(sd["N"])
    print(f"=== inyectar_efecto_fuego ===\n  ckpt: {ckpt_path}\n  N={N} H={H} W={W} T={T}\n  grados={grados}", flush=True)

    targets = generar_targets(CFG, H, W, T)
    fuego = targets_a_coefs(targets, grados, T)
    print(f"  generadas M={CFG['M']} particulas de fuego en (y0={CFG['y0']}, x0={CFG['x0']})", flush=True)

    if args.preview:
        exp_dir = ckpt_path.parent.parent
        fondo_dir = args.fondo_dir or (exp_dir / "frames_renderizados")
        salida = exp_dir / "viz_fuego"; salida.mkdir(parents=True, exist_ok=True)
        frames = args.frames_preview or [0, T // 3, 2 * T // 3, T - 1]
        preview(CFG, sd, fuego, grados, H, W, T, fondo_dir, salida, frames)
        print("=== preview listo (no se exporto .pt) ===", flush=True)
        return

    # --- concatenar fuego al contenido y guardar nuevo .pt ----------------
    for nombre in ["mu", "opacity", "color", "scale", "theta", "depth"]:
        sd[f"{nombre}_a0"] = torch.cat([sd[f"{nombre}_a0"], fuego[f"{nombre}_a0"]], dim=0)
        sd[f"{nombre}_high"] = torch.cat([sd[f"{nombre}_high"], fuego[f"{nombre}_high"]], dim=0)
    sd["N"] = N + CFG["M"]

    salida = Path(args.salida).resolve() if args.salida else ckpt_path.with_name(ckpt_path.stem + "_fuego.pt")
    torch.save({"state_dict_coefs": sd, "config": ckpt["config"],
                "epoch_completado": ckpt.get("epoch_completado", -1)}, str(salida))
    print(f"  N_final = {sd['N']}  (contenido {N} + fuego {CFG['M']})", flush=True)
    print(f"=== exportado -> {salida} ===", flush=True)
    print("  rasterizalo en Khipu con regenerar_clip_desde_checkpoint_streaming.py", flush=True)


if __name__ == "__main__":
    main()
