"""
Entrenamiento de Gabor splatting de audio (waveform directa).

Idea (propuesta del profesor): representar la senal de audio cruda x(t) como
una superposicion de atomos de Gabor (gaussianas moduladas por sinusoides)
optimizados por gradiente. Sin STFT, sin fase separada, sin polinomios
temporales. Reconstruccion directa a waveform.

Uso:
    python scripts/GABOR_SCRIPTS/train_gabor.py --config configs/gabor/gabor_rock_8s_smoke.json

Salida en outputs/gabor/<nombre_experimento>/:
    config_usada.json
    info_audio.json
    metricas.json
    log_entrenamiento.csv
    loss_curve.png
    recon.wav            <- audio reconstruido
    target.wav           <- audio original (recortado/resampleado)
    checkpoints/checkpoint_final.pt
"""
import argparse
import csv
import json
import math
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch

RAIZ = Path(__file__).resolve().parents[2]
SRC = RAIZ / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from gs2d_gabor.core.modelo_gabor import GaborAudio1D, construir_optimizador_gabor
from gs2d_gabor.core.perdidas_gabor import loss_gabor, metricas_audio
from gs2d_gabor.core.metricas_perceptuales import calcular_metricas_perceptuales


# ======================================================================
# IO de audio
# ======================================================================

def cargar_wav_mono(ruta, sr_objetivo=None, max_segundos=None, canal="mono"):
    from scipy.io import wavfile
    from scipy import signal

    sr, data = wavfile.read(str(ruta))

    if data.ndim == 2:
        if canal == "left":
            data = data[:, 0]
        elif canal == "right":
            data = data[:, 1]
        else:
            data = data.mean(axis=1)

    if data.dtype == np.int16:
        x = data.astype(np.float32) / 32768.0
    elif data.dtype == np.int32:
        x = data.astype(np.float32) / 2147483648.0
    elif data.dtype == np.uint8:
        x = (data.astype(np.float32) - 128.0) / 128.0
    else:
        x = data.astype(np.float32)

    x = np.nan_to_num(x)
    x = np.clip(x, -1.0, 1.0)

    if sr_objetivo is not None and int(sr_objetivo) != int(sr):
        gcd = np.gcd(int(sr), int(sr_objetivo))
        up = int(sr_objetivo) // gcd
        down = int(sr) // gcd
        x = signal.resample_poly(x, up, down).astype(np.float32)
        sr = int(sr_objetivo)

    if max_segundos is not None:
        n = int(float(max_segundos) * sr)
        x = x[:n]

    if len(x) < 1024:
        raise RuntimeError("audio demasiado corto")

    return int(sr), x


def guardar_wav(ruta, sr, x):
    from scipy.io import wavfile

    x = np.asarray(x, dtype=np.float32)
    x = np.nan_to_num(x)
    max_abs = np.max(np.abs(x))
    if max_abs > 1.0:
        x = x / max_abs
    x_i16 = (np.clip(x, -1.0, 1.0) * 32767.0).astype(np.int16)
    wavfile.write(str(ruta), int(sr), x_i16)


def guardar_curva(valores, titulo, ylabel, ruta):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8, 4), dpi=100)
    ax.plot(valores)
    ax.set_xlabel("epoch")
    ax.set_ylabel(ylabel)
    ax.set_title(titulo)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(ruta)
    plt.close(fig)


def tamano_modelo_bytes(modelo):
    """5 params float32 por atomo."""
    return modelo.numero_atomos() * 5 * 4


# ======================================================================
# main
# ======================================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--nombre-experimento", default=None)
    args = parser.parse_args()

    with open(args.config, encoding="utf-8") as f:
        config = json.load(f)

    if args.nombre_experimento is not None:
        config["nombre_experimento"] = args.nombre_experimento
    if not config.get("nombre_experimento"):
        from datetime import datetime
        config["nombre_experimento"] = "gabor_" + datetime.now().strftime("%Y%m%d_%H%M%S")

    seed = int(config.get("seed", 42))
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)

    device = torch.device("cuda" if (config.get("device", "cuda") == "cuda" and torch.cuda.is_available()) else "cpu")
    print(f"device: {device}", flush=True)

    # === audio ===
    ruta_audio = Path(config["audio"])
    if not ruta_audio.is_absolute():
        ruta_audio = RAIZ / ruta_audio
    if not ruta_audio.exists():
        raise FileNotFoundError(f"No existe el audio: {ruta_audio}")

    sr, x_np = cargar_wav_mono(
        ruta_audio,
        sr_objetivo=config.get("sr"),
        max_segundos=config.get("max_segundos"),
        canal=config.get("canal", "mono"),
    )
    x = torch.from_numpy(x_np).to(device=device, dtype=torch.float32)
    T = x.shape[0]
    print(f"audio: {ruta_audio.name}  sr={sr}  samples={T}  ({T/sr:.2f}s)", flush=True)

    # === salida ===
    nombre_exp = config["nombre_experimento"]
    salida = RAIZ / "outputs" / "gabor" / nombre_exp
    sobreescribir = bool(config.get("sobreescribir_salida", False))
    if salida.exists() and not sobreescribir:
        raise FileExistsError(
            f"La carpeta ya existe: {salida}. Usa sobreescribir_salida=true o cambia el nombre."
        )
    (salida / "checkpoints").mkdir(parents=True, exist_ok=True)

    with open(salida / "config_usada.json", "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2, ensure_ascii=False)
    with open(salida / "info_audio.json", "w", encoding="utf-8") as f:
        json.dump({
            "audio": str(ruta_audio),
            "sr": sr,
            "samples": T,
            "duracion_s": T / sr,
            "canal": config.get("canal", "mono"),
            "seed": seed,
        }, f, indent=2)

    guardar_wav(salida / "target.wav", sr, x_np)

    # === modelo ===
    modelo = GaborAudio1D(
        n_atomos=int(config["n_atomos"]),
        n_samples=T,
        sr=sr,
        device=device,
        sigma_inicial_samples=config.get("sigma_inicial_samples"),
        f_min_hz=float(config.get("f_min_hz", 40.0)),
        f_max_hz=config.get("f_max_hz"),
        k_sigma=float(config.get("k_sigma", 4.0)),
        semilla=seed,
        sigma_min_samples=float(config.get("sigma_min_samples", 2.0)),
        sigma_max_frac=float(config.get("sigma_max_frac", 0.1)),
    )
    print(f"modelo Gabor: N={modelo.numero_atomos()} atomos  k_sigma={modelo.k_sigma}", flush=True)

    if bool(config.get("forzar_pytorch", False)):
        modelo._forzar_pytorch = True
        print("[train] forzando render PyTorch (fallback, lento)", flush=True)

    optimizer = construir_optimizador_gabor(modelo, config.get("lrs"))

    # === scheduler simple por plateau (opcional) ===
    usar_sched = bool(config.get("usar_scheduler", True))
    sched_factor = float(config.get("scheduler_factor", 0.5))
    sched_paciencia = int(config.get("scheduler_paciencia", 150))
    sched_min_lr = float(config.get("scheduler_min_lr", 1e-5))
    mejor_loss = float("inf")
    epochs_sin_mejora = 0

    # === loop ===
    n_epochs_base = int(config["epochs"])
    finetune_epochs = int(config.get("finetune_epochs", 0))
    finetune_lr_scale = float(config.get("finetune_lr_scale", 1.0))
    finetune_reset_scheduler = bool(config.get("finetune_reset_scheduler", True))

    if finetune_epochs < 0:
        finetune_epochs = 0

    n_epochs = n_epochs_base + finetune_epochs
    log_cada = int(config.get("log_cada", 50))

    config_loss_actual = dict(config)

    historial_loss = []
    historial_snr = []
    tiempos = []
    t0 = time.time()

    for epoch in range(n_epochs):
        if epoch == n_epochs_base and finetune_epochs > 0:
            print("", flush=True)
            print("=== FINETUNING GABOR ===", flush=True)
            print(f"epochs extra: {finetune_epochs}", flush=True)
            print(f"lr scale    : {finetune_lr_scale}", flush=True)

            for grupo in optimizer.param_groups:
                old_lr = float(grupo["lr"])
                grupo["lr"] = max(old_lr * finetune_lr_scale, sched_min_lr)
                nombre = grupo.get("name", "grupo")
                print(f"  lr {nombre}: {old_lr:.6e} -> {grupo['lr']:.6e}", flush=True)

            overrides = config.get("finetune_loss_overrides", {})
            if overrides:
                print("  aplicando finetune_loss_overrides:", flush=True)
                for k, v in overrides.items():
                    print(f"    {k}: {config_loss_actual.get(k, None)} -> {v}", flush=True)
                    config_loss_actual[k] = v

            if finetune_reset_scheduler:
                mejor_loss = float("inf")
                epochs_sin_mejora = 0
                print("  scheduler reiniciado para fine-tuning", flush=True)

            print("========================", flush=True)
            print("", flush=True)

        te = time.time()
        optimizer.zero_grad(set_to_none=True)

        x_hat = modelo.render()
        loss, partes = loss_gabor(x_hat, x, config_loss_actual)

        # Regularizaciones opcionales anti-zumbido para Gabor.
        lambda_amp_l2 = float(config_loss_actual.get("lambda_amp_l2", 0.0))
        lambda_highfreq_amp = float(config_loss_actual.get("lambda_highfreq_amp", 0.0))
        lambda_sigma_small = float(config_loss_actual.get("lambda_sigma_small", 0.0))

        if lambda_amp_l2 > 0.0 or lambda_highfreq_amp > 0.0 or lambda_sigma_small > 0.0:
            _, sigma_act, amp_act, fnorm_act, _ = modelo.activar_parametros()

            if lambda_amp_l2 > 0.0:
                l_amp_l2 = torch.mean(amp_act * amp_act)
                loss = loss + lambda_amp_l2 * l_amp_l2
                partes["l_amp_l2"] = float(l_amp_l2.detach())

            if lambda_highfreq_amp > 0.0:
                highfreq_start_hz = float(config_loss_actual.get("highfreq_start_hz", 6000.0))
                freq_hz = fnorm_act * float(sr)
                denom = max(1.0, float(sr) * 0.5 - highfreq_start_hz)
                w = torch.clamp((freq_hz - highfreq_start_hz) / denom, min=0.0, max=1.0)
                l_high = torch.mean((amp_act * w) ** 2)
                loss = loss + lambda_highfreq_amp * l_high
                partes["l_highfreq_amp"] = float(l_high.detach())

            if lambda_sigma_small > 0.0:
                sigma_target = float(config_loss_actual.get("sigma_small_target_samples", 8.0))
                l_sig = torch.mean(torch.relu(sigma_target - sigma_act) ** 2)
                loss = loss + lambda_sigma_small * l_sig
                partes["l_sigma_small"] = float(l_sig.detach())

        loss.backward()
        optimizer.step()

        loss_val = float(loss.detach())
        historial_loss.append(loss_val)
        tiempos.append(time.time() - te)

        if usar_sched:
            if loss_val < mejor_loss - 1e-6:
                mejor_loss = loss_val
                epochs_sin_mejora = 0
            else:
                epochs_sin_mejora += 1
                if epochs_sin_mejora >= sched_paciencia:
                    for grupo in optimizer.param_groups:
                        grupo["lr"] = max(grupo["lr"] * sched_factor, sched_min_lr)
                    epochs_sin_mejora = 0
                    lrs = [g["lr"] for g in optimizer.param_groups]
                    print(f"[sched] epoch {epoch+1}: LR reducido. min={min(lrs):.2e} max={max(lrs):.2e}", flush=True)

        if (epoch == 0) or ((epoch + 1) % log_cada == 0) or (epoch == n_epochs - 1):
            with torch.no_grad():
                met = metricas_audio(x_hat.detach(), x)
            historial_snr.append((epoch, met["snr_db"]))
            eta = (time.time() - t0) / (epoch + 1) * (n_epochs - epoch - 1)
            extra = "  ".join(f"{k}={v:.4f}" for k, v in partes.items())
            print(
                f"  epoch {epoch+1:5d}/{n_epochs}  loss={loss_val:.5f}  "
                f"{extra}  SNR={met['snr_db']:.2f}dB  PSNR={met['psnr_db']:.2f}dB  "
                f"t={tiempos[-1]*1000:.0f}ms  eta={eta/60:.1f}min",
                flush=True,
            )

    # === reconstruccion final ===
    with torch.no_grad():
        x_hat = modelo.render()
        met_final = metricas_audio(x_hat, x)
        met_perceptuales = calcular_metricas_perceptuales(x_hat, x, sr=sr, config=config)
    x_hat_np = x_hat.detach().cpu().numpy()
    guardar_wav(salida / "recon.wav", sr, x_hat_np)

    # === metricas + compresion ===
    bytes_modelo = tamano_modelo_bytes(modelo)
    bytes_wav = T * 2  # int16 mono
    metricas = {
        "snr_db": met_final["snr_db"],
        "psnr_db": met_final["psnr_db"],
        "mse_wave": met_final["mse_wave"],

        "si_sdr_db": met_perceptuales.get("si_sdr_db"),
        "lsd_db_promedio": met_perceptuales.get("lsd_db_promedio"),
        "mel_l1_log": met_perceptuales.get("mel_l1_log"),
        "mrstft_final": met_perceptuales.get("mrstft_final"),
        "n_atomos": modelo.numero_atomos(),
        "samples": T,
        "sr": sr,
        "duracion_s": T / sr,
        "epochs_base": n_epochs_base,
        "finetune_epochs": finetune_epochs,
        "finetune_lr_scale": finetune_lr_scale,
        "epochs_total": n_epochs,
        "bytes_modelo": bytes_modelo,
        "bytes_wav_int16": bytes_wav,
        "ratio_compresion_vs_wav": bytes_wav / max(1, bytes_modelo),
        "loss_final": historial_loss[-1] if historial_loss else None,
        "tiempo_total_s": time.time() - t0,
    }
    metricas.update({
        k: v for k, v in met_perceptuales.items()
        if k not in metricas
    })

    with open(salida / "metricas.json", "w", encoding="utf-8") as f:
        json.dump(metricas, f, indent=2)

    print("\n=== resultado ===", flush=True)
    print(f"  SNR={met_final['snr_db']:.2f} dB  PSNR={met_final['psnr_db']:.2f} dB", flush=True)
    print(
        f"  SI-SDR={metricas.get('si_sdr_db', float('nan')):.2f} dB  "
        f"LSD={metricas.get('lsd_db_promedio', float('nan')):.4f}  "
        f"Mel-L1={metricas.get('mel_l1_log', float('nan')):.4f}  "
        f"MRSTFT={metricas.get('mrstft_final', float('nan')):.4f}",
        flush=True,
    )
    print(f"  bytes_modelo={bytes_modelo}  bytes_wav={bytes_wav}  "
          f"ratio={metricas['ratio_compresion_vs_wav']:.2f}x", flush=True)

    # === curva + log csv ===
    guardar_curva(historial_loss, f"loss - {nombre_exp}", "loss", salida / "loss_curve.png")

    with open(salida / "log_entrenamiento.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["epoch", "loss", "tiempo_s"])
        for i, (lo, ti) in enumerate(zip(historial_loss, tiempos)):
            w.writerow([i + 1, f"{lo:.6f}", f"{ti:.4f}"])

    # === checkpoint ===
    torch.save({
        "state_dict_coefs": modelo.state_dict_coefs(),
        "config": config,
        "metricas": metricas,
    }, salida / "checkpoints" / "checkpoint_final.pt")

    print(f"\nlisto. resultados en: {salida}", flush=True)


if __name__ == "__main__":
    main()
