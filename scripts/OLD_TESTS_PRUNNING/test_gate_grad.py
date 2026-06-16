import argparse
import sys
from pathlib import Path

import torch

RAIZ = Path(__file__).resolve().parents[1]
SRC = RAIZ / "src"

if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from gs2d_video.core.bases import construir_matriz_chebyshev
from gs2d_video.core.modelo import GaussianasPolinomial2D
from gs2d_video.render.renderer import render_frame


def cargar_checkpoint(path, device):
    try:
        ckpt = torch.load(
            path,
            map_location="cpu",
            weights_only=False,
        )
    except TypeError:
        ckpt = torch.load(path, map_location="cpu")

    sd = ckpt["state_dict_coefs"]
    config = ckpt.get("config", {})

    grados = dict(sd.get("grados", config["grados"]))
    N = int(sd["N"])
    H = int(sd["H"])
    W = int(sd["W"])
    T = int(sd.get("n_frames", config.get("max_frames")))

    modelo = GaussianasPolinomial2D(
        n_gaussianas=N,
        n_frames=T,
        grados=grados,
        H=H,
        W=W,
        device=device,
        escala_inicial_px=float(config.get("escala_inicial_px", 5.0)),
        frame_0_imagen=None,
        semilla=int(config.get("seed", 42)),
    )

    claves_modelo = set(modelo.state_dict().keys())

    pesos = {
        k: v.to(device)
        for k, v in sd.items()
        if k in claves_modelo and torch.is_tensor(v)
    }

    resultado = modelo.load_state_dict(pesos, strict=False)

    if resultado.missing_keys:
        raise RuntimeError(
            f"Faltan parámetros: {resultado.missing_keys}"
        )

    modelo.eval()

    for parametro in modelo.parameters():
        parametro.requires_grad_(False)

    return modelo, config, grados, N, H, W, T


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--frame", type=int, default=None)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    device = torch.device(args.device)

    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA no está disponible.")

    modelo, config, grados, N, H, W, T = cargar_checkpoint(
        args.checkpoint,
        device,
    )

    frame = T // 2 if args.frame is None else args.frame

    if not 0 <= frame < T:
        raise ValueError(f"Frame fuera de rango: {frame}, T={T}")

    matrices = {
        grado: construir_matriz_chebyshev(
            T,
            grado,
            device=device,
            dtype=torch.float32,
        )
        for grado in sorted(set(grados.values()))
    }

    if device.type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

    params = modelo.evaluar_en_frame(frame, matrices)

    gate = torch.ones(
        N,
        dtype=params["opacity"].dtype,
        device=device,
        requires_grad=True,
    )

    params_gate = dict(params)
    params_gate["opacity"] = params["opacity"] * gate

    render = render_frame(
        params_gate,
        H,
        W,
        config,
    )

    # Proyección aleatoria para medir sensibilidad de todos los píxeles.
    probe = torch.empty_like(render)
    probe.bernoulli_(0.5)
    probe.mul_(2.0).sub_(1.0)

    scalar = (render * probe).mean()
    scalar.backward()

    if gate.grad is None:
        raise RuntimeError(
            "gate.grad es None: el rasterizador no propagó gradientes."
        )

    grad = gate.grad.detach().abs().float().cpu()

    print("======================================================")
    print("TEST GRADIENTE POR GAUSSIANA")
    print("======================================================")
    print(f"checkpoint       : {args.checkpoint}")
    print(f"frame            : {frame}/{T - 1}")
    print(f"N                : {N}")
    print(f"render shape     : {tuple(render.shape)}")
    print(f"grad nonzero     : {(grad > 0).sum().item()}/{N}")
    print(f"grad mean        : {grad.mean().item():.10e}")
    print(f"grad p50         : {torch.quantile(grad, 0.50).item():.10e}")
    print(f"grad p90         : {torch.quantile(grad, 0.90).item():.10e}")
    print(f"grad p99         : {torch.quantile(grad, 0.99).item():.10e}")
    print(f"grad max         : {grad.max().item():.10e}")

    if device.type == "cuda":
        peak_mb = torch.cuda.max_memory_allocated() / 1024**2
        print(f"VRAM pico MB     : {peak_mb:.2f}")

    print("======================================================")


if __name__ == "__main__":
    main()