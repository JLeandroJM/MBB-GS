import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

RAIZ = Path(__file__).resolve().parents[1]
SRC = RAIZ / "src"

if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from gs2d_video.core.bases import construir_matriz_chebyshev
from gs2d_audio.core.modelo_audio import (
    GaussianasEspectralesTemporalesCheb,
    construir_optimizador_audio,
    loss_smoothness_audio,
)


def crear_target_sintetico(n_frames, n_bins, device):
    t = torch.linspace(0.0, 1.0, n_frames, device=device).view(n_frames, 1)
    f = torch.arange(n_bins, device=device, dtype=torch.float32).view(1, n_bins)

    c1 = 30.0 + 120.0 * t
    c2 = 190.0 - 70.0 * t
    c3 = 90.0 + 25.0 * torch.sin(2.0 * torch.pi * t * 2.0)

    b1 = torch.exp(-0.5 * ((f - c1) / 5.0) ** 2)
    b2 = 0.7 * torch.exp(-0.5 * ((f - c2) / 9.0) ** 2)
    b3 = 0.5 * torch.exp(-0.5 * ((f - c3) / 4.0) ** 2)

    target = b1 + b2 + b3
    target = target / target.max().clamp_min(1e-8)
    return target.clamp(0.0, 1.0)


def guardar_img_matriz(x_tf, ruta):
    arr = x_tf.detach().cpu().clamp(0, 1).numpy()
    arr = (arr.T * 255.0).astype(np.uint8)
    Image.fromarray(arr).save(ruta)


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}")

    n_frames = 160
    n_bins = 256
    n_gaussianas = 128

    grados = {
        "mu_f": 30,
        "sigma_f": 8,
        "amp": 30,
    }

    salida = RAIZ / "outputs" / "audio_fase2_smoke"
    salida.mkdir(parents=True, exist_ok=True)

    target = crear_target_sintetico(n_frames, n_bins, device)

    grados_distintos = sorted(set(grados.values()))
    matrices_base = {
        g: construir_matriz_chebyshev(
            n_frames=n_frames,
            grado_max=g,
            device=device,
            dtype=torch.float32,
        )
        for g in grados_distintos
    }

    modelo = GaussianasEspectralesTemporalesCheb(
        n_gaussianas=n_gaussianas,
        n_frames=n_frames,
        n_bins=n_bins,
        grados=grados,
        device=device,
        semilla=42,
        sigma_inicial_bins=6.0,
    )

    opt = construir_optimizador_audio(modelo, lrs={})

    n_epochs = 800
    beta_smooth = 1e-10
    losses = []

    for epoch in range(1, n_epochs + 1):
        opt.zero_grad(set_to_none=True)

        pred = modelo.render_all(matrices_base)

        loss_l1 = torch.mean(torch.abs(pred - target))
        loss_mse = F.mse_loss(pred, target)
        loss_smooth = loss_smoothness_audio(
            modelo,
            pesos_por_param={
                "mu_f": 1.0,
                "sigma_f": 0.5,
                "amp": 0.5,
            },
        )

        loss = loss_l1 + 0.25 * loss_mse + beta_smooth * loss_smooth
        loss.backward()
        opt.step()

        losses.append(float(loss.detach().cpu()))

        if epoch == 1 or epoch % 50 == 0 or epoch == n_epochs:
            print(
                f"epoch {epoch:04d}/{n_epochs} "
                f"loss={loss.item():.6f} "
                f"l1={loss_l1.item():.6f} "
                f"mse={loss_mse.item():.6f}"
            )

    with torch.no_grad():
        pred = modelo.render_all(matrices_base).clamp(0, 1)
        diff = torch.abs(pred - target).clamp(0, 1)

    guardar_img_matriz(target, salida / "target.png")
    guardar_img_matriz(pred, salida / "recon.png")
    guardar_img_matriz((diff * 5.0).clamp(0, 1), salida / "diff_x5.png")

    torch.save(
        {
            "state_dict_coefs": modelo.state_dict_coefs(),
            "grados": grados,
            "n_frames": n_frames,
            "n_bins": n_bins,
            "n_gaussianas": n_gaussianas,
            "loss_final": losses[-1],
        },
        salida / "checkpoint_final.pt",
    )

    np.savetxt(salida / "losses.txt", np.array(losses), fmt="%.8f")

    print("")
    print(f"listo. resultados en: {salida}")
    print("archivos: target.png, recon.png, diff_x5.png, checkpoint_final.pt")


if __name__ == "__main__":
    main()
