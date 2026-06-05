import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path
from datetime import datetime


RAIZ = Path(__file__).resolve().parents[2]


def agregar_arg(cmd, flag, valor):
    if valor is None:
        return

    if isinstance(valor, bool):
        if valor:
            cmd.append(flag)
        return

    cmd.extend([flag, str(valor)])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="ruta al config json de audio")
    args = parser.parse_args()

    config_path = Path(args.config)
    if not config_path.is_absolute():
        config_path = RAIZ / config_path

    if not config_path.exists():
        raise FileNotFoundError(f"No existe config: {config_path}")

    with open(config_path, "r", encoding="utf-8-sig") as f:
        cfg = json.load(f)

    nombre = cfg.get("nombre_experimento") or cfg.get("nombre")
    if not nombre:
        nombre = "audio_exp_" + datetime.now().strftime("%Y%m%d_%H%M%S")
        cfg["nombre_experimento"] = nombre

    salida = RAIZ / "outputs"/ "audio"  / nombre
    salida.mkdir(parents=True, exist_ok=True)

    # Snapshot del config usado.
    with open(salida / "config_usada.json", "w", encoding="utf-8") as f:
        json.dump(cfg, f, indent=2)

    shutil.copy2(config_path, salida / "config_original.json")

    script = RAIZ / "scripts" / "AUDIO_SCRIPTS" / "audio_fase2_real_wav.py"

    cmd = [
        sys.executable,
        str(script),
    ]

    # Campos principales
    agregar_arg(cmd, "--audio", cfg.get("audio"))
    agregar_arg(cmd, "--nombre", nombre)
    agregar_arg(cmd, "--max-segundos", cfg.get("max_segundos"))
    agregar_arg(cmd, "--sr", cfg.get("sr"))
    agregar_arg(cmd, "--n-fft", cfg.get("n_fft"))
    agregar_arg(cmd, "--hop", cfg.get("hop"))
    agregar_arg(cmd, "--n-gaussianas", cfg.get("n_gaussianas"))
    agregar_arg(cmd, "--epochs", cfg.get("epochs"))
    agregar_arg(cmd, "--batch-temporal", cfg.get("batch_temporal"))
    agregar_arg(cmd, "--log-cada", cfg.get("log_cada"))
    agregar_arg(cmd, "--canal", cfg.get("canal"))
    # Loss
    agregar_arg(cmd, "--tipo-loss", cfg.get("tipo_loss"))
    agregar_arg(cmd, "--lambda-mse", cfg.get("lambda_mse"))
    agregar_arg(cmd, "--lambda-motion", cfg.get("lambda_motion"))
    agregar_arg(cmd, "--lambda-hard", cfg.get("lambda_hard"))
    agregar_arg(cmd, "--lambda-temporal", cfg.get("lambda_temporal"))
    agregar_arg(cmd, "--lambda-freq-edge", cfg.get("lambda_freq_edge"))
    agregar_arg(cmd, "--lambda-dssim", cfg.get("lambda_dssim"))
    agregar_arg(cmd, "--motion-clip", cfg.get("motion_clip"))
    agregar_arg(cmd, "--hard-clip", cfg.get("hard_clip"))
    agregar_arg(cmd, "--exponente-pixel", cfg.get("exponente_pixel"))
    agregar_arg(cmd, "--usar-pnorm-root", cfg.get("usar_pnorm_root"))

    # Grados Chebyshev
    agregar_arg(cmd, "--grado-mu-f", cfg.get("grado_mu_f"))
    agregar_arg(cmd, "--grado-sigma-f", cfg.get("grado_sigma_f"))
    agregar_arg(cmd, "--grado-amp", cfg.get("grado_amp"))
    agregar_arg(cmd, "--usar-loss-cuda-audio", cfg.get("usar_loss_cuda_audio"))
    agregar_arg(cmd, "--usar-render-cuda-audio", cfg.get("usar_render_cuda_audio"))
    agregar_arg(cmd, "--griffin-lim-iters", cfg.get("griffin_lim_iters"))

    with open(salida / "comando.txt", "w", encoding="utf-8") as f:
        f.write(" ".join(cmd) + "\n")

    print("============================================================")
    print(" TRAIN AUDIO CONFIG")
    print("============================================================")
    print(f"config : {config_path}")
    print(f"salida : {salida}")
    print(f"cmd    : {' '.join(cmd)}")
    print("============================================================")

    proceso = subprocess.Popen(
        cmd,
        cwd=str(RAIZ),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",
        bufsize=1,
    )

    for linea in proceso.stdout:
        print(linea, end="")

    codigo = proceso.wait()
    if codigo != 0:
        raise SystemExit(codigo)


if __name__ == "__main__":
    main()
