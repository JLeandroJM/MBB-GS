import argparse
import csv
import math
import sys
import time
from pathlib import Path

import torch

RAIZ = Path(__file__).resolve().parents[1]
SCRIPTS = RAIZ / "scripts"

if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from test_gate_grad import cargar_checkpoint

SRC = RAIZ / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from gs2d_video.core.bases import construir_matriz_chebyshev
from gs2d_video.render.renderer import render_frame


def escribir_csv(ruta, indices, datos):
    ruta = Path(ruta)
    ruta.parent.mkdir(parents=True, exist_ok=True)

    campos = [
        "id",
        "rank_importance",
        "importance_rms",
        "importance_mean_abs",
        "importance_max",
        "frames_nonzero",
        "frac_frames_nonzero",
        "opacity_mean_sampled",
        "opacity_max_sampled",
    ]

    with ruta.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=campos)
        writer.writeheader()

        for idx in indices:
            i = int(idx)

            writer.writerow({
                "id": i,
                "rank_importance": int(datos["rank"][i]),
                "importance_rms": f"{datos['rms'][i]:.12e}",
                "importance_mean_abs": f"{datos['mean_abs'][i]:.12e}",
                "importance_max": f"{datos['max_abs'][i]:.12e}",
                "frames_nonzero": int(datos["frames_nonzero"][i]),
                "frac_frames_nonzero": f"{datos['frac_nonzero'][i]:.8f}",
                "opacity_mean_sampled": f"{datos['opacity_mean'][i]:.8f}",
                "opacity_max_sampled": f"{datos['opacity_max'][i]:.8f}",
            })


def formato_percentiles(x):
    porcentajes = [
        0.00,
        0.01,
        0.05,
        0.10,
        0.15,
        0.20,
        0.25,
        0.50,
        0.75,
        0.90,
        0.95,
        0.99,
        1.00,
    ]

    valores = torch.quantile(
        x.to(torch.float64),
        torch.tensor(porcentajes, dtype=torch.float64),
    )

    partes = []

    for p, valor in zip(porcentajes, valores):
        nombre = int(round(p * 100))
        partes.append(f"p{nombre}={float(valor):.8e}")

    return "  ".join(partes)


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--out_dir", default=None)
    parser.add_argument("--device", default="cuda")

    parser.add_argument(
        "--sample_every",
        type=int,
        default=5,
        help="Evalúa un frame cada N frames.",
    )

    parser.add_argument(
        "--probes",
        type=int,
        default=2,
        help="Proyecciones aleatorias por frame.",
    )

    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--top_k", type=int, default=1000)
    parser.add_argument("--nonzero_eps", type=float, default=0.0)

    args = parser.parse_args()

    checkpoint = Path(args.checkpoint).resolve()

    if not checkpoint.exists():
        raise FileNotFoundError(checkpoint)

    if args.sample_every <= 0:
        raise ValueError("sample_every debe ser mayor que cero.")

    if args.probes <= 0:
        raise ValueError("probes debe ser mayor que cero.")

    device = torch.device(args.device)

    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA no está disponible.")

    modelo, config, grados, N, H, W, T = cargar_checkpoint(
        checkpoint,
        device,
    )

    if args.out_dir:
        out_dir = Path(args.out_dir).resolve()
    else:
        experimento = checkpoint.parent.parent
        out_dir = experimento / "viz_importance"

    out_dir.mkdir(parents=True, exist_ok=True)

    matrices = {
        grado: construir_matriz_chebyshev(
            T,
            grado,
            device=device,
            dtype=torch.float32,
        )
        for grado in sorted(set(grados.values()))
    }

    frames = list(range(0, T, args.sample_every))

    if frames[-1] != T - 1:
        frames.append(T - 1)

    n_frames_eval = len(frames)
    n_probes_total = n_frames_eval * args.probes

    # Acumulación en CPU float64 para no perder los gradientes pequeños.
    suma_abs = torch.zeros(N, dtype=torch.float64)
    suma_sq = torch.zeros(N, dtype=torch.float64)
    max_abs = torch.zeros(N, dtype=torch.float32)

    frames_nonzero = torch.zeros(N, dtype=torch.int32)

    suma_opacity = torch.zeros(N, dtype=torch.float64)
    max_opacity = torch.zeros(N, dtype=torch.float32)

    generador = torch.Generator(device=device)
    generador.manual_seed(args.seed)

    if device.type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

    print("======================================================")
    print("IMPORTANCIA VISUAL POR GRADIENTE")
    print("======================================================")
    print(f"checkpoint       : {checkpoint}")
    print(f"salida           : {out_dir}")
    print(f"N                : {N}")
    print(f"frames totales   : {T}")
    print(f"frames evaluados : {n_frames_eval}")
    print(f"sample_every     : {args.sample_every}")
    print(f"probes/frame     : {args.probes}")
    print(f"backwards total  : {n_probes_total}")
    print("")

    inicio = time.time()

    for pos, frame_idx in enumerate(frames):
        params = modelo.evaluar_en_frame(frame_idx, matrices)

        opacity_cpu = params["opacity"].detach().float().cpu()
        suma_opacity += opacity_cpu.to(torch.float64)
        max_opacity = torch.maximum(max_opacity, opacity_cpu)

        gate = torch.ones(
            N,
            device=device,
            dtype=params["opacity"].dtype,
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

        frame_hit = torch.zeros(N, dtype=torch.bool)

        for probe_idx in range(args.probes):
            probe = torch.randint(
                low=0,
                high=2,
                size=render.shape,
                device=device,
                dtype=torch.int8,
                generator=generador,
            )

            probe = probe.to(render.dtype).mul_(2.0).sub_(1.0)

            # Escala estable. No altera el ranking relativo.
            scalar = (
                (render * probe).sum()
                / math.sqrt(float(render.numel()))
            )

            grad = torch.autograd.grad(
                outputs=scalar,
                inputs=gate,
                retain_graph=(probe_idx + 1 < args.probes),
                create_graph=False,
            )[0]

            grad_cpu = grad.detach().abs().float().cpu()
            grad64 = grad_cpu.to(torch.float64)

            suma_abs += grad64
            suma_sq += grad64.square()
            max_abs = torch.maximum(max_abs, grad_cpu)

            frame_hit |= grad_cpu > args.nonzero_eps

            del probe, scalar, grad, grad_cpu, grad64

        frames_nonzero += frame_hit.to(torch.int32)

        del params, params_gate, render, gate, opacity_cpu, frame_hit

        if device.type == "cuda" and (
            (pos + 1) % 10 == 0 or pos == n_frames_eval - 1
        ):
            torch.cuda.empty_cache()

        if (
            pos == 0
            or (pos + 1) % 10 == 0
            or pos == n_frames_eval - 1
        ):
            transcurrido = time.time() - inicio
            promedio = transcurrido / (pos + 1)
            eta = promedio * (n_frames_eval - pos - 1)

            print(
                f"frame eval {pos + 1}/{n_frames_eval} "
                f"(original={frame_idx}) "
                f"eta={eta / 60:.1f} min",
                flush=True,
            )

    importance_mean_abs = suma_abs / n_probes_total
    importance_rms = torch.sqrt(suma_sq / n_probes_total)
    opacity_mean = suma_opacity / n_frames_eval

    frac_nonzero = (
        frames_nonzero.to(torch.float64)
        / n_frames_eval
    )

    orden = torch.argsort(
        importance_rms,
        descending=False,
        stable=True,
    )

    rank = torch.empty(N, dtype=torch.int64)
    rank[orden] = torch.arange(N, dtype=torch.int64)

    datos = {
        "rank": rank,
        "rms": importance_rms,
        "mean_abs": importance_mean_abs,
        "max_abs": max_abs,
        "frames_nonzero": frames_nonzero,
        "frac_nonzero": frac_nonzero,
        "opacity_mean": opacity_mean,
        "opacity_max": max_opacity,
    }

    csv_completo = out_dir / "gaussian_importance.csv"
    escribir_csv(csv_completo, range(N), datos)

    top_k = min(args.top_k, N)

    csv_prescindibles = out_dir / "top_prescindibles.csv"
    escribir_csv(
        csv_prescindibles,
        orden[:top_k].tolist(),
        datos,
    )

    csv_importantes = out_dir / "top_importantes.csv"
    escribir_csv(
        csv_importantes,
        torch.flip(orden[-top_k:], dims=[0]).tolist(),
        datos,
    )

    resumen = out_dir / "resumen_importance.txt"

    cero_todos_frames = int(
        (frames_nonzero == 0).sum().item()
    )

    lineas = [
        "======================================================",
        "RESUMEN IMPORTANCIA VISUAL POR GRADIENTE",
        "======================================================",
        f"checkpoint: {checkpoint}",
        f"N: {N}",
        f"frames_totales: {T}",
        f"frames_evaluados: {n_frames_eval}",
        f"sample_every: {args.sample_every}",
        f"probes_por_frame: {args.probes}",
        f"backwards_total: {n_probes_total}",
        "",
        f"gaussianas_gradiente_cero_todos_frames: {cero_todos_frames}",
        (
            "porcentaje_gradiente_cero_todos_frames: "
            f"{100.0 * cero_todos_frames / N:.4f}%"
        ),
        "",
        "PERCENTILES importance_rms",
        formato_percentiles(importance_rms),
        "",
        "PERCENTILES importance_mean_abs",
        formato_percentiles(importance_mean_abs),
        "",
        "PERCENTILES frac_frames_nonzero",
        formato_percentiles(frac_nonzero),
        "",
        "ARCHIVOS",
        f"completo: {csv_completo}",
        f"prescindibles: {csv_prescindibles}",
        f"importantes: {csv_importantes}",
    ]

    if device.type == "cuda":
        pico = torch.cuda.max_memory_allocated() / 1024**2
        lineas.append(f"VRAM_pico_MB: {pico:.2f}")

    resumen.write_text(
        "\n".join(lineas),
        encoding="utf-8",
    )

    print("")
    print("======================================================")
    print("LISTO")
    print("======================================================")
    print(f"CSV completo     : {csv_completo}")
    print(f"Top prescindibles: {csv_prescindibles}")
    print(f"Top importantes  : {csv_importantes}")
    print(f"Resumen          : {resumen}")
    print(
        "Gradiente cero en todos los frames evaluados: "
        f"{cero_todos_frames}/{N} "
        f"({100.0 * cero_todos_frames / N:.4f}%)"
    )


if __name__ == "__main__":
    main()
