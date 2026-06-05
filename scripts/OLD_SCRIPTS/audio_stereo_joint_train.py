import argparse
import csv
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from scipy.io import wavfile
from scipy import signal

RAIZ = Path(__file__).resolve().parents[1]
SRC = RAIZ / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from gs2d_audio_stereo.core.modelo_stereo import (
    GaussianasStereoTemporalesCheb,
    construir_bases,
    construir_optimizador_stereo,
    loss_smoothness_stereo,
)


def cargar_wav_stereo(ruta, sr_objetivo=None, max_segundos=None):
    sr, data = wavfile.read(str(ruta))

    if data.ndim != 2 or data.shape[1] != 2:
        raise RuntimeError("Este script requiere audio stereo shape [samples, 2].")

    if data.dtype == np.int16:
        x = data.astype(np.float32) / 32768.0
    elif data.dtype == np.int32:
        x = data.astype(np.float32) / 2147483648.0
    else:
        x = data.astype(np.float32)

    x = np.nan_to_num(x)
    x = np.clip(x, -1.0, 1.0)

    if sr_objetivo is not None and int(sr_objetivo) != int(sr):
        gcd = np.gcd(int(sr), int(sr_objetivo))
        up = int(sr_objetivo) // gcd
        down = int(sr) // gcd
        x_l = signal.resample_poly(x[:, 0], up, down).astype(np.float32)
        x_r = signal.resample_poly(x[:, 1], up, down).astype(np.float32)
        x = np.stack([x_l, x_r], axis=1)
        sr = int(sr_objetivo)

    if max_segundos is not None:
        n = int(float(max_segundos) * sr)
        x = x[:n]

    return sr, x


def guardar_wav_stereo(ruta, sr, left, right):
    n = min(len(left), len(right))
    y = np.stack([left[:n], right[:n]], axis=1).astype(np.float32)
    y = np.nan_to_num(y)

    max_abs = np.max(np.abs(y))
    if max_abs > 1.0:
        y = y / max_abs

    wavfile.write(str(ruta), sr, (np.clip(y, -1, 1) * 32767).astype(np.int16))


def psnr_from_mse(mse):
    if mse <= 0:
        return float("inf")
    return -10.0 * np.log10(mse)


def hard_loss(pred, target, lambda_mse=0.25, lambda_hard=1.0, hard_clip=3.0):
    diff = pred - target
    absdiff = torch.abs(diff)

    mean_abs = absdiff.mean().clamp_min(1e-8)
    w = 1.0 + lambda_hard * torch.clamp(absdiff / mean_abs, 0.0, hard_clip)

    l1 = torch.sum(w * absdiff) / w.sum().clamp_min(1e-8)
    mse = torch.mean(diff ** 2)
    loss = l1 + lambda_mse * mse

    return loss, l1.detach(), mse.detach()


def griffin_lim_desde_magnitud(mag, n_fft, hop, n_iter, length, device):
    window = torch.hann_window(n_fft, device=device)

    phase = torch.exp(2j * torch.pi * torch.rand_like(mag))
    X = mag * phase

    for _ in range(int(n_iter)):
        wav = torch.istft(
            X,
            n_fft=n_fft,
            hop_length=hop,
            win_length=n_fft,
            window=window,
            center=True,
            length=length,
        )

        X_est = torch.stft(
            wav,
            n_fft=n_fft,
            hop_length=hop,
            win_length=n_fft,
            window=window,
            center=True,
            return_complex=True,
        )

        phase = X_est / torch.abs(X_est).clamp_min(1e-8)
        X = mag * phase

    return torch.istft(
        X,
        n_fft=n_fft,
        hop_length=hop,
        win_length=n_fft,
        window=window,
        center=True,
        length=length,
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args_cli = parser.parse_args()

    ruta_cfg = Path(args_cli.config)
    if not ruta_cfg.is_absolute():
        ruta_cfg = RAIZ / ruta_cfg

    with open(ruta_cfg, "r", encoding="utf-8-sig") as f:
        cfg = json.load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    nombre = cfg["nombre_experimento"]
    salida = RAIZ / "outputs" / "audio_stereo_joint" / nombre
    salida.mkdir(parents=True, exist_ok=True)

    ruta_audio = Path(cfg["audio"])
    if not ruta_audio.is_absolute():
        ruta_audio = RAIZ / ruta_audio

    sr, x_np = cargar_wav_stereo(
        ruta_audio,
        sr_objetivo=int(cfg.get("sr", 44100)),
        max_segundos=float(cfg.get("max_segundos", 6)),
    )

    x_l = torch.tensor(x_np[:, 0], dtype=torch.float32, device=device)
    x_r = torch.tensor(x_np[:, 1], dtype=torch.float32, device=device)

    n_fft = int(cfg.get("n_fft", 2048))
    hop = int(cfg.get("hop", 512))
    window = torch.hann_window(n_fft, device=device)

    X_l = torch.stft(x_l, n_fft=n_fft, hop_length=hop, win_length=n_fft, window=window, center=True, return_complex=True)
    X_r = torch.stft(x_r, n_fft=n_fft, hop_length=hop, win_length=n_fft, window=window, center=True, return_complex=True)

    mag_l = torch.abs(X_l)
    mag_r = torch.abs(X_r)
    phase_l = torch.angle(X_l)
    phase_r = torch.angle(X_r)

    log_l = torch.log1p(mag_l)
    log_r = torch.log1p(mag_r)

    max_l = log_l.max().clamp_min(1e-8)
    max_r = log_r.max().clamp_min(1e-8)

    target_l = (log_l / max_l).T.contiguous()
    target_r = (log_r / max_r).T.contiguous()

    n_frames, n_bins = target_l.shape

    grados = {
        "mu_f": int(cfg.get("grado_mu_f", 80)),
        "sigma_f": int(cfg.get("grado_sigma_f", 20)),
        "amp_l": int(cfg.get("grado_amp_l", 100)),
        "amp_r": int(cfg.get("grado_amp_r", 100)),
    }

    modelo = GaussianasStereoTemporalesCheb(
        n_gaussianas=int(cfg.get("n_gaussianas", 1000)),
        n_frames=n_frames,
        n_bins=n_bins,
        grados=grados,
        device=device,
        semilla=42,
    )

    bases = construir_bases(n_frames, grados, device)
    opt = construir_optimizador_stereo(modelo, cfg.get("lrs", None))

    epochs = int(cfg.get("epochs", 400))
    batch_temporal = int(cfg.get("batch_temporal", 16))
    log_cada = int(cfg.get("log_cada", 50))

    lambda_mse = float(cfg.get("lambda_mse", 0.25))
    lambda_hard = float(cfg.get("lambda_hard", 1.0))
    hard_clip = float(cfg.get("hard_clip", 3.0))
    gl_iters = int(cfg.get("griffin_lim_iters", 150))

    print(f"device: {device}")
    print(f"salida: {salida}")
    print(f"audio: {ruta_audio}")
    print(f"sr: {sr}")
    print(f"duracion: {len(x_np)/sr:.2f}s")
    print(f"target stereo: T={n_frames}, F={n_bins}")
    print(f"grados: {grados}")
    print(f"N: {modelo.n_gaussianas}")
    print("")

    log_csv = salida / "log_entrenamiento.csv"
    historial = []
    t0 = time.time()

    with open(log_csv, "w", newline="", encoding="utf-8") as fcsv:
        wcsv = csv.writer(fcsv)
        wcsv.writerow(["epoch", "loss", "l1_l", "l1_r", "mse_l", "mse_r", "psnr_l", "psnr_r", "t_epoch", "eta_min"])

        for epoch in range(1, epochs + 1):
            te = time.time()
            opt.zero_grad(set_to_none=True)

            loss_epoch = 0.0
            l1_l_epoch = 0.0
            l1_r_epoch = 0.0
            mse_l_epoch = 0.0
            mse_r_epoch = 0.0

            for ini in range(0, n_frames, batch_temporal):
                fin = min(ini + batch_temporal, n_frames)
                idx = torch.arange(ini, fin, dtype=torch.long, device=device)
                peso = float(fin - ini) / float(n_frames)

                pred_l, pred_r = modelo.render_indices(idx, bases)

                loss_l, l1_l, mse_l = hard_loss(pred_l, target_l[idx], lambda_mse, lambda_hard, hard_clip)
                loss_r, l1_r, mse_r = hard_loss(pred_r, target_r[idx], lambda_mse, lambda_hard, hard_clip)

                loss = 0.5 * (loss_l + loss_r)
                (loss * peso).backward()

                loss_epoch += float(loss.detach().cpu()) * peso
                l1_l_epoch += float(l1_l.cpu()) * peso
                l1_r_epoch += float(l1_r.cpu()) * peso
                mse_l_epoch += float(mse_l.cpu()) * peso
                mse_r_epoch += float(mse_r.cpu()) * peso

            beta_smooth = float(cfg.get("beta_smoothness", 1e-10))
            pesos_smoothness = cfg.get("pesos_smoothness", {
                "mu_f": 0.0,
                "sigma_f": 0.5,
                "amp_l": 0.0,
                "amp_r": 0.0,
            })

            loss_smooth = loss_smoothness_stereo(modelo, pesos_smoothness)

            if beta_smooth != 0.0:
                (beta_smooth * loss_smooth).backward()


            opt.step()

            t_epoch = time.time() - te
            t_avg = (time.time() - t0) / max(1, epoch)
            eta = t_avg * (epochs - epoch) / 60.0

            psnr_l = psnr_from_mse(mse_l_epoch)
            psnr_r = psnr_from_mse(mse_r_epoch)

            row = [epoch, loss_epoch, l1_l_epoch, l1_r_epoch, mse_l_epoch, mse_r_epoch, psnr_l, psnr_r, t_epoch, eta]
            wcsv.writerow(row)
            fcsv.flush()

            historial.append({
                "epoch": epoch,
                "loss": loss_epoch,
                "psnr_l": psnr_l,
                "psnr_r": psnr_r,
                "mse_l": mse_l_epoch,
                "mse_r": mse_r_epoch,
            })

            if epoch == 1 or epoch % log_cada == 0 or epoch == epochs:
                print(
                    f"epoch {epoch:04d}/{epochs} "
                    f"loss={loss_epoch:.6f} "
                    f"psnr_L={psnr_l:.2f} "
                    f"psnr_R={psnr_r:.2f} "
                    f"t_epoch={t_epoch:.2f}s eta={eta:.1f}min",
                    flush=True,
                )

    with open(salida / "historial_entrenamiento.json", "w", encoding="utf-8") as f:
        json.dump(historial, f, indent=2)

    print("")
    print("=== reconstruccion final ===")

    with torch.no_grad():
        partes_l = []
        partes_r = []

        for ini in range(0, n_frames, batch_temporal):
            fin = min(ini + batch_temporal, n_frames)
            idx = torch.arange(ini, fin, dtype=torch.long, device=device)
            pl, pr = modelo.render_indices(idx, bases)
            partes_l.append(pl.detach().cpu())
            partes_r.append(pr.detach().cpu())

        pred_l_cpu = torch.cat(partes_l, dim=0).clamp(0, 1)
        pred_r_cpu = torch.cat(partes_r, dim=0).clamp(0, 1)

        pred_l = pred_l_cpu.to(device).T * max_l
        pred_r = pred_r_cpu.to(device).T * max_r

        pred_mag_l = torch.expm1(pred_l).clamp_min(0)
        pred_mag_r = torch.expm1(pred_r).clamp_min(0)

        X_rec_l = pred_mag_l * torch.exp(1j * phase_l)
        X_rec_r = pred_mag_r * torch.exp(1j * phase_r)

        y_l = torch.istft(X_rec_l, n_fft=n_fft, hop_length=hop, win_length=n_fft, window=window, center=True, length=x_l.shape[0])
        y_r = torch.istft(X_rec_r, n_fft=n_fft, hop_length=hop, win_length=n_fft, window=window, center=True, length=x_r.shape[0])

        gl_l = griffin_lim_desde_magnitud(pred_mag_l, n_fft, hop, gl_iters, x_l.shape[0], device)
        gl_r = griffin_lim_desde_magnitud(pred_mag_r, n_fft, hop, gl_iters, x_r.shape[0], device)

    guardar_wav_stereo(
        salida / "original_stereo.wav",
        sr,
        x_l.detach().cpu().numpy(),
        x_r.detach().cpu().numpy(),
    )

    guardar_wav_stereo(
        salida / "recon_stereo_fase_original.wav",
        sr,
        y_l.detach().cpu().numpy(),
        y_r.detach().cpu().numpy(),
    )

    guardar_wav_stereo(
        salida / "recon_stereo_griffinlim.wav",
        sr,
        gl_l.detach().cpu().numpy(),
        gl_r.detach().cpu().numpy(),
    )

    mse_l = float(torch.mean((y_l - x_l) ** 2).detach().cpu())
    mse_r = float(torch.mean((y_r - x_r) ** 2).detach().cpu())

    snr_l = 10 * np.log10(float(torch.mean(x_l ** 2).detach().cpu()) / max(mse_l, 1e-12))
    snr_r = 10 * np.log10(float(torch.mean(x_r ** 2).detach().cpu()) / max(mse_r, 1e-12))

    metricas = {
        "mse_wave_l": mse_l,
        "mse_wave_r": mse_r,
        "snr_l_db": float(snr_l),
        "snr_r_db": float(snr_r),
        "psnr_logmag_l": historial[-1]["psnr_l"],
        "psnr_logmag_r": historial[-1]["psnr_r"],
        "n_gaussianas": modelo.n_gaussianas,
        "grados": grados,
        "epochs": epochs,
        "batch_temporal": batch_temporal,
        "griffin_lim_iters": gl_iters,
    }

    with open(salida / "metricas_stereo.json", "w", encoding="utf-8") as f:
        json.dump(metricas, f, indent=2)

    torch.save(
        {
            "state_dict": modelo.state_dict(),
            "metricas": metricas,
            "config": cfg,
            "grados": grados,
            "n_frames": n_frames,
            "n_bins": n_bins,
            "max_l": float(max_l.detach().cpu()),
            "max_r": float(max_r.detach().cpu()),
        },
        salida / "checkpoint_final.pt",
    )

    print(f"snr_L={snr_l:.2f} dB")
    print(f"snr_R={snr_r:.2f} dB")
    print(f"listo. salida: {salida}")
    print("escucha: recon_stereo_fase_original.wav y recon_stereo_griffinlim.wav")


if __name__ == "__main__":
    main()
