"""
Helper compartido: cargar un checkpoint .pt y reconstruir el modelo
GaussianasPolinomial2D en CPU (o el device que se pida).

Lo usan las visualizaciones que corren en la Mac SIN GPU (trayectorias,
marcadores, elipses, velocidades), porque solo necesitan evaluar los
polinomios de Chebyshev (matmul), no el rasterizador CUDA.

La logica es la misma que scripts/regenerar_clip_desde_checkpoint_streaming.py
pero extraida aqui para reusarla sin duplicar.
"""
import sys
from pathlib import Path

import torch

RAIZ = Path(__file__).resolve().parents[1]
SRC = RAIZ / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from gs2d_video.core.bases import construir_matriz_chebyshev
from gs2d_video.core.modelo import GaussianasPolinomial2D


def _inferir_n_gaussianas(sd, config):
    for key in ["mu_a0", "color_a0", "opacity_a0", "scale_a0", "theta_a0", "depth_a0"]:
        value = sd.get(key)
        if torch.is_tensor(value) and value.ndim >= 1:
            return int(value.shape[0])
    return int(config["n_gaussianas_inicial"])


def cargar_modelo_desde_checkpoint(ruta_checkpoint, device="cpu"):
    """
    Devuelve (modelo, config, matrices_base, info) donde:
      - modelo: GaussianasPolinomial2D con los coefs del checkpoint, en .eval()
      - config: dict de configuracion guardado en el checkpoint
      - matrices_base: dict {grado: B (n_frames, grado+1)} ya en el device
      - info: dict con N, H, W, n_frames, grados
    """
    if isinstance(device, str):
        device = torch.device(device)

    ckpt = torch.load(str(ruta_checkpoint), map_location="cpu")
    if "config" not in ckpt or "state_dict_coefs" not in ckpt:
        raise RuntimeError("El checkpoint no tiene 'config' o 'state_dict_coefs'")

    config = dict(ckpt["config"])
    sd = ckpt["state_dict_coefs"]
    grados = dict(sd.get("grados", config["grados"]))
    N = int(sd.get("N", _inferir_n_gaussianas(sd, config)))
    H = int(sd.get("H"))
    W = int(sd.get("W"))
    n_frames = int(sd.get("n_frames", config.get("max_frames")))

    modelo = GaussianasPolinomial2D(
        n_gaussianas=N,
        n_frames=n_frames,
        grados=grados,
        H=H,
        W=W,
        device=device,
        escala_inicial_px=float(config.get("escala_inicial_px", 5.0)),
        frame_0_imagen=None,
        semilla=int(config.get("seed", 42)),
    )

    with torch.no_grad():
        for nombre in ["mu", "opacity", "color", "scale", "theta", "depth"]:
            getattr(modelo, f"{nombre}_a0").copy_(sd[f"{nombre}_a0"].to(device))
            getattr(modelo, f"{nombre}_high").copy_(sd[f"{nombre}_high"].to(device))
    modelo.eval()

    grados_distintos = sorted(set(grados.values()))
    matrices_base = {
        g: construir_matriz_chebyshev(n_frames, g, device=device, dtype=torch.float32)
        for g in grados_distintos
    }

    info = {"N": N, "H": H, "W": W, "n_frames": n_frames, "grados": grados}
    return modelo, config, matrices_base, info


def cargar_frame_fondo(config, indice, raiz=RAIZ):
    """
    Carga un frame PNG del clip de entrenamiento como tensor (H, W, 3) en [0,1].
    Devuelve None si el clip o el frame no existen localmente.
    """
    import numpy as np
    from PIL import Image

    clip = config.get("clip")
    if clip is None:
        return None
    carpeta = Path(raiz) / "data" / "clips" / clip
    ruta = carpeta / f"frame_{int(indice):04d}.png"
    if not ruta.is_file():
        return None
    img = np.asarray(Image.open(ruta).convert("RGB"), dtype=np.float32) / 255.0
    return torch.from_numpy(img)
