import argparse
import csv
import json
import time
import atexit
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from scipy.io import wavfile
from scipy import signal

RAIZ = Path(__file__).resolve().parents[2]
SRC = RAIZ / "src"

if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from gs2d_video.core.bases import construir_matriz_chebyshev
from gs2d_audio.core.modelo_audio import (
    GaussianasEspectralesTemporalesCheb,
    construir_optimizador_audio,
    loss_smoothness_audio,
)
from gs2d_audio.core.perdidas_audio import loss_audio_batch, metricas_logmag

try:
    from gs2d_audio.render.cuda_spectral import (
        loss_audio_espectral_cuda,
        construir_motion_weight_audio,
        render_audio_espectral_cuda,
        loss_audio_hard_cuda,
    )
    _CUDA_AUDIO_DISPONIBLE = True
    _CUDA_AUDIO_ERROR = None
except Exception as e:
    _CUDA_AUDIO_DISPONIBLE = False
    _CUDA_AUDIO_ERROR = e

class TeeLogger:
    """
    Duplica stdout/stderr:
    - sigue mostrando mensajes en consola
    - guarda los mismos mensajes en un archivo .log
    """

    def __init__(self, *streams):
        self.streams = streams

    def write(self, data):
        for stream in self.streams:
            stream.write(data)
            stream.flush()

    def flush(self):
        for stream in self.streams:
            stream.flush()


def activar_log_dual(ruta_log):
    ruta_log.parent.mkdir(parents=True, exist_ok=True)

    log_file = open(ruta_log, "w", encoding="utf-8", errors="replace", buffering=1)

    stdout_original = sys.stdout
    stderr_original = sys.stderr

    sys.stdout = TeeLogger(stdout_original, log_file)
    sys.stderr = TeeLogger(stderr_original, log_file)

    def cerrar_log():
        try:
            sys.stdout = stdout_original
            sys.stderr = stderr_original
            log_file.close()
        except Exception:
            pass

    atexit.register(cerrar_log)

    return ruta_log


def cargar_wav_mono(ruta_audio, sr_objetivo=None, max_segundos=None, canal="mono"):
    sr, data = wavfile.read(str(ruta_audio))

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
        raise RuntimeError("audio demasiado corto para STFT")

    return sr, x


def guardar_wav(ruta, sr, x):
    x = np.asarray(x, dtype=np.float32)
    x = np.nan_to_num(x)
    max_abs = np.max(np.abs(x))

    if max_abs > 1.0:
        x = x / max_abs

    x_i16 = np.clip(x, -1.0, 1.0)
    x_i16 = (x_i16 * 32767.0).astype(np.int16)
    wavfile.write(str(ruta), int(sr), x_i16)


def guardar_img_matriz(x_tf, ruta):
    arr = x_tf.detach().cpu().clamp(0, 1).numpy()
    arr = (arr.T * 255.0).astype(np.uint8)
    Image.fromarray(arr).save(str(ruta))


def psnr_from_mse(mse):
    if mse <= 0:
        return float("inf")
    return -10.0 * np.log10(mse)


def griffin_lim_desde_magnitud(mag, n_fft, hop, n_iter, length, device):
    window = torch.hann_window(n_fft, device=device)

    phase = torch.exp(2j * torch.pi * torch.rand_like(mag))
    X = mag * phase

    for _ in range(n_iter):
        wav = torch.istft(
            X,
            n_fft=n_fft,
            hop_length=hop,
            window=window,
            length=length,
        )

        X_est = torch.stft(
            wav,
            n_fft=n_fft,
            hop_length=hop,
            window=window,
            return_complex=True,
        )

        phase = X_est / torch.abs(X_est).clamp_min(1e-8)
        X = mag * phase

    return torch.istft(
        X,
        n_fft=n_fft,
        hop_length=hop,
        window=window,
        length=length,
    )

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--audio", default="data/audio/prueba.wav")
    parser.add_argument("--nombre", default="audio_fase2_real")
    parser.add_argument("--max-segundos", type=float, default=5.0)
    parser.add_argument("--sr", type=int, default=22050)
    parser.add_argument("--n-fft", type=int, default=1024)
    parser.add_argument("--hop", type=int, default=256)
    parser.add_argument("--n-gaussianas", type=int, default=256)
    parser.add_argument("--epochs", type=int, default=1200)
    parser.add_argument("--batch-temporal", type=int, default=64)
    parser.add_argument("--log-cada", type=int, default=50)
    parser.add_argument("--tipo-loss", default="baseline", choices=["baseline", "motion", "hard", "temporal", "freq_edge", "dssim", "combo"])
    parser.add_argument("--lambda-mse", type=float, default=0.25)
    parser.add_argument("--lambda-motion", type=float, default=0.0)
    parser.add_argument("--lambda-hard", type=float, default=0.0)
    parser.add_argument("--lambda-temporal", type=float, default=0.0)
    parser.add_argument("--lambda-freq-edge", type=float, default=0.0)
    parser.add_argument("--lambda-dssim", type=float, default=0.0)
    parser.add_argument("--motion-clip", type=float, default=5.0)
    parser.add_argument("--hard-clip", type=float, default=5.0)
    parser.add_argument("--exponente-pixel", type=float, default=1.0)
    parser.add_argument("--usar-pnorm-root", action="store_true")
    parser.add_argument("--grado-mu-f", type=int, default=40)
    parser.add_argument("--grado-sigma-f", type=int, default=10)
    parser.add_argument("--grado-amp", type=int, default=40)
    parser.add_argument("--usar-loss-cuda-audio", action="store_true")
    parser.add_argument("--usar-render-cuda-audio", action="store_true")
    parser.add_argument("--griffin-lim-iters", type=int, default=150)
    parser.add_argument("--canal", default="mono", choices=["mono", "left", "right"])
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}")

    ruta_audio = Path(args.audio)
    if not ruta_audio.is_absolute():
        ruta_audio = RAIZ / ruta_audio

    if not ruta_audio.exists():
        raise FileNotFoundError(
            f"No existe el audio: {ruta_audio}\n"
            "Coloca un WAV en data/audio/prueba.wav o usa --audio ruta/al/audio.wav"
        )

    salida = RAIZ / "outputs"/ "audio" / args.nombre
    salida.mkdir(parents=True, exist_ok=True)

    ruta_log = activar_log_dual(salida / "entrenamiento.log")
    print(f"log: {ruta_log}")

    sr, x_np = cargar_wav_mono(
        ruta_audio,
        sr_objetivo=args.sr,
        max_segundos=args.max_segundos,
        canal=args.canal,
    )

    print(f"audio: {ruta_audio}")
    print(f"sr: {sr}")
    print(f"samples: {len(x_np)}")
    print(f"duracion: {len(x_np) / sr:.2f}s")

    x = torch.tensor(x_np, dtype=torch.float32, device=device)

    window = torch.hann_window(args.n_fft, device=device)

    X = torch.stft(
        x,
        n_fft=args.n_fft,
        hop_length=args.hop,
        win_length=args.n_fft,
        window=window,
        center=True,
        return_complex=True,
    )

    # X: [F, T]
    mag = torch.abs(X)
    phase = torch.angle(X)

    log_mag = torch.log1p(mag)
    log_mag_max = log_mag.max().clamp_min(1e-8)

    target = (log_mag / log_mag_max).T.contiguous()
    # target: [T, F]

    n_frames, n_bins = target.shape
    print(f"target spectrogram: T={n_frames}, F={n_bins}")

    grados = {
        "mu_f": int(args.grado_mu_f),
        "sigma_f": int(args.grado_sigma_f),
        "amp": int(args.grado_amp),
    }

    config_loss = {
        "tipo_loss": args.tipo_loss,
        "lambda_mse": float(args.lambda_mse),
        "lambda_motion": float(args.lambda_motion),
        "lambda_hard": float(args.lambda_hard),
        "lambda_temporal": float(args.lambda_temporal),
        "lambda_freq_edge": float(args.lambda_freq_edge),
        "lambda_dssim": float(args.lambda_dssim),
        "motion_clip": float(args.motion_clip),
        "hard_clip": float(args.hard_clip),
        "exponente_pixel": float(args.exponente_pixel),
        "usar_pnorm_root": bool(args.usar_pnorm_root),
    }

    print(f"grados audio: {grados}")
    print(f"loss audio: {config_loss}")
    print(f"usar_loss_cuda_audio: {bool(args.usar_loss_cuda_audio)}")
    print(f"usar_render_cuda_audio: {bool(args.usar_render_cuda_audio)}")

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
        n_gaussianas=args.n_gaussianas,
        n_frames=n_frames,
        n_bins=n_bins,
        grados=grados,
        device=device,
        semilla=42,
        sigma_inicial_bins=6.0,
    )

    opt = construir_optimizador_audio(
        modelo,
        lrs={
            "mu_f_a0": 1e-2,
            "mu_f_high": 1e-3,
            "sigma_f_a0": 5e-3,
            "sigma_f_high": 5e-4,
            "amp_a0": 2e-2,
            "amp_high": 2e-3,
        },
    )

    beta_smooth = 1e-10
    losses = []

    guardar_img_matriz(target, salida / "target_logmag.png")

    print("")
    print("=== entrenamiento audio ===")

    ruta_log_csv = salida / "log_entrenamiento.csv"
    historial = []
    t_inicio_train = time.time()

    with open(ruta_log_csv, "w", newline="", encoding="utf-8") as fcsv:
        writer = csv.writer(fcsv)
        writer.writerow([
            "epoch",
            "loss",
            "l1",
            "mse",
            "psnr_logmag",
            "loss_smooth",
            "t_epoch_seg",
            "eta_min",
            "tipo_loss",
            "n_gaussianas",
            "batch_temporal",
        ])

        for epoch in range(1, args.epochs + 1):
            t_epoch = time.time()
            opt.zero_grad(set_to_none=True)

            # Entrenamiento FULL por chunks temporales.
            # Usa TODO el espectrograma en cada epoch, pero renderiza por bloques.
            chunk = int(args.batch_temporal)
            if chunk <= 0:
                chunk = n_frames

            loss_l1_epoch = 0.0
            loss_mse_epoch = 0.0
            loss_data_epoch = 0.0

            for ini_chunk in range(0, n_frames, chunk):
                fin_chunk = min(ini_chunk + chunk, n_frames)
                idx = torch.arange(ini_chunk, fin_chunk, device=device, dtype=torch.long)

                target_batch = target[idx]

                if bool(args.usar_loss_cuda_audio):
                    if not _CUDA_AUDIO_DISPONIBLE:
                        raise RuntimeError(
                            f"usar_loss_cuda_audio=true pero no se pudo importar CUDA audio: {_CUDA_AUDIO_ERROR}"
                        )

                    if args.tipo_loss not in ("baseline", "motion", "hard"):
                        raise RuntimeError(
                            "CUDA audio inicial soporta baseline, motion y hard. "
                            "Usa PyTorch para dssim, freq_edge, temporal o combo."
                        )

                    params_audio = modelo.evaluar_indices(idx, matrices_base)

                    if args.tipo_loss == "hard":
                        loss_chunk, l1_cuda, mse_cuda = loss_audio_hard_cuda(
                            params_audio["mu_f"],
                            params_audio["sigma_f"],
                            params_audio["amp"],
                            target_batch,
                            lambda_mse=float(args.lambda_mse),
                            lambda_hard=float(args.lambda_hard),
                            hard_clip=float(args.hard_clip),
                        )

                        partes_loss = {
                            "loss_l1": l1_cuda.detach(),
                            "loss_mse": mse_cuda.detach(),
                        }

                        del params_audio

                    else:
                        if args.tipo_loss == "motion":
                            weight = construir_motion_weight_audio(
                                target,
                                idx,
                                lambda_motion=float(args.lambda_motion),
                                motion_clip=float(args.motion_clip),
                            )
                        else:
                            weight = torch.ones_like(target_batch)

                        loss_chunk, l1_cuda, mse_cuda = loss_audio_espectral_cuda(
                            params_audio["mu_f"],
                            params_audio["sigma_f"],
                            params_audio["amp"],
                            target_batch,
                            weight=weight,
                            lambda_mse=float(args.lambda_mse),
                        )

                        partes_loss = {
                            "loss_l1": l1_cuda.detach(),
                            "loss_mse": mse_cuda.detach(),
                        }

                        del params_audio, weight

                else:
                    if bool(args.usar_render_cuda_audio):
                        if not _CUDA_AUDIO_DISPONIBLE:
                            raise RuntimeError(
                                f"usar_render_cuda_audio=true pero no se pudo importar CUDA audio: {_CUDA_AUDIO_ERROR}"
                            )

                        params_audio = modelo.evaluar_indices(idx, matrices_base)

                        pred = render_audio_espectral_cuda(
                            params_audio["mu_f"],
                            params_audio["sigma_f"],
                            params_audio["amp"],
                            n_bins,
                        )

                        del params_audio
                    else:
                        pred = modelo.render_indices(idx, matrices_base)

                    loss_chunk, partes_loss = loss_audio_batch(
                        pred,
                        target_batch,
                        config_loss,
                        target_full=target,
                        indices=idx,
                    )

                    del pred

                peso_chunk = float(fin_chunk - ini_chunk) / float(n_frames)

                # Pondera el chunk para que el epoch represente el promedio completo.
                loss_chunk_pond = loss_chunk * peso_chunk
                loss_chunk_pond.backward()

                l1_val = float(partes_loss["loss_l1"].detach().cpu())
                mse_val = float(partes_loss["loss_mse"].detach().cpu())
                loss_val = float(loss_chunk.detach().cpu())

                loss_l1_epoch += l1_val * peso_chunk
                loss_mse_epoch += mse_val * peso_chunk
                loss_data_epoch += loss_val * peso_chunk

                del idx, target_batch, loss_chunk, loss_chunk_pond

            loss_smooth = loss_smoothness_audio(
                modelo,
                pesos_por_param={
                    "mu_f": 1.0,
                    "sigma_f": 0.5,
                    "amp": 0.5,
                },
            )

            if beta_smooth != 0.0:
                (beta_smooth * loss_smooth).backward()

            opt.step()

            t_epoch_seg = time.time() - t_epoch
            t_corrido = time.time() - t_inicio_train
            t_prom = t_corrido / max(1, epoch)
            eta = t_prom * (args.epochs - epoch)

            loss_smooth_val = float(loss_smooth.detach().cpu())
            loss_total_epoch = loss_data_epoch + beta_smooth * loss_smooth_val

            psnr_epoch = psnr_from_mse(loss_mse_epoch)

            losses.append(float(loss_total_epoch))
            historial.append({
                "epoch": epoch,
                "loss": float(loss_total_epoch),
                "l1": float(loss_l1_epoch),
                "mse": float(loss_mse_epoch),
                "psnr_logmag": float(psnr_epoch),
                "loss_smooth": float(loss_smooth_val),
                "t_epoch_seg": float(t_epoch_seg),
                "eta_min": float(eta / 60.0),
                "tipo_loss": args.tipo_loss,
                "n_gaussianas": args.n_gaussianas,
                "batch_temporal": args.batch_temporal,
            })

            writer.writerow([
                epoch,
                f"{loss_total_epoch:.8f}",
                f"{loss_l1_epoch:.8f}",
                f"{loss_mse_epoch:.8f}",
                f"{psnr_epoch:.4f}",
                f"{loss_smooth_val:.8e}",
                f"{t_epoch_seg:.3f}",
                f"{eta / 60.0:.3f}",
                args.tipo_loss,
                args.n_gaussianas,
                args.batch_temporal,
            ])
            fcsv.flush()

            if epoch == 1 or epoch % int(args.log_cada) == 0 or epoch == args.epochs:
                print(
                    f"epoch {epoch:04d}/{args.epochs} "
                    f"loss={loss_total_epoch:.6f} "
                    f"l1={loss_l1_epoch:.6f} "
                    f"mse={loss_mse_epoch:.6f} "
                    f"psnr_logmag={psnr_epoch:.2f} "
                    f"t_epoch={t_epoch_seg:.1f}s "
                    f"eta={eta / 60.0:.1f}min "
                    f"loss_type={args.tipo_loss}",
                    flush=True,
                )

    with open(salida / "historial_entrenamiento.json", "w", encoding="utf-8") as f:
        json.dump(historial, f, indent=2)

    with torch.no_grad():
        partes = []
        bs_eval = max(1, int(args.batch_temporal))

        for ini in range(0, n_frames, bs_eval):
            fin = min(ini + bs_eval, n_frames)
            idx_eval = torch.arange(ini, fin, device=device, dtype=torch.long)

            # Para evaluacion final usamos el renderer normal del modelo.
            # Esto reconstruye la magnitud logaritmica normalizada predicha.
            parte = modelo.render_indices(idx_eval, matrices_base).detach().cpu()
            partes.append(parte)

        pred_cpu = torch.cat(partes, dim=0).clamp(0, 1)
        target_cpu = target.detach().cpu()
        diff_cpu = torch.abs(pred_cpu - target_cpu).clamp(0, 1)

        guardar_img_matriz(pred_cpu, salida / "recon_logmag.png")
        guardar_img_matriz((diff_cpu * 5.0).clamp(0, 1), salida / "diff_logmag_x5.png")

        metricas_lm = metricas_logmag(pred_cpu, target_cpu)

        with open(salida / "metricas_audio.json", "w", encoding="utf-8") as f:
            json.dump(metricas_lm, f, indent=2)

        with open(salida / "metricas_logmag_por_t.csv", "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["t_idx", "psnr_logmag"])
            for i, v in enumerate(metricas_lm["psnr_logmag_por_t"]):
                w.writerow([i, f"{v:.6f}"])

        # Volver de log-magnitude normalizada [T,F] a magnitud lineal [F,T].
        pred = pred_cpu.to(device)
        pred_log_mag = pred.T.contiguous() * log_mag_max
        pred_mag = torch.expm1(pred_log_mag).clamp_min(0.0)

        # 1) Reconstruccion usando fase original.
        X_rec = pred_mag * torch.exp(1j * phase)

        x_rec = torch.istft(
            X_rec,
            n_fft=args.n_fft,
            hop_length=args.hop,
            win_length=args.n_fft,
            window=window,
            center=True,
            length=x.shape[0],
        )

        # 2) Reconstruccion sin fase original usando Griffin-Lim.
        x_griffin = griffin_lim_desde_magnitud(
            mag=pred_mag,
            n_fft=args.n_fft,
            hop=args.hop,
            n_iter=int(args.griffin_lim_iters),
            length=x.shape[0],
            device=device,
        )

    x_rec_np = x_rec.detach().cpu().numpy()
    x_griffin_np = x_griffin.detach().cpu().numpy()
    x_orig_np = x.detach().cpu().numpy()

    guardar_wav(salida / "original_mono.wav", sr, x_orig_np)
    guardar_wav(salida / "recon_fase_original.wav", sr, x_rec_np)
    guardar_wav(salida / "recon_griffinlim.wav", sr, x_griffin_np)

    mse_wave = float(np.mean((x_orig_np - x_rec_np) ** 2))
    snr = 10.0 * np.log10(
        np.mean(x_orig_np ** 2) / max(mse_wave, 1e-12)
    )

    mse_wave_griffin = float(np.mean((x_orig_np - x_griffin_np) ** 2))
    snr_griffin = 10.0 * np.log10(
        np.mean(x_orig_np ** 2) / max(mse_wave_griffin, 1e-12)
    )

    # Metricas mas justas para Griffin-Lim:
    # La senal puede sonar parecida pero no estar alineada sample a sample.
    with torch.no_grad():
        X_griffin = torch.stft(
            x_griffin,
            n_fft=args.n_fft,
            hop_length=args.hop,
            win_length=args.n_fft,
            window=window,
            center=True,
            return_complex=True,
        )

        mag_griffin = torch.abs(X_griffin)
        log_mag_griffin = torch.log1p(mag_griffin)
        target_log_mag = log_mag

        mse_logmag_griffin = torch.mean(
            (log_mag_griffin - target_log_mag) ** 2
        ).item()

        l1_logmag_griffin = torch.mean(
            torch.abs(log_mag_griffin - target_log_mag)
        ).item()

        psnr_logmag_griffin = psnr_from_mse(mse_logmag_griffin)

        mag_orig = torch.abs(X)
        mse_mag_griffin = torch.mean(
            (mag_griffin - mag_orig) ** 2
        ).item()

        l1_mag_griffin = torch.mean(
            torch.abs(mag_griffin - mag_orig)
        ).item()

    np.savetxt(salida / "losses.txt", np.array(losses), fmt="%.8f")

    torch.save(
        {
            "state_dict_coefs": modelo.state_dict_coefs(),
            "grados": grados,
            "n_frames": n_frames,
            "n_bins": n_bins,
            "n_gaussianas": args.n_gaussianas,
            "sr": sr,
            "n_fft": args.n_fft,
            "hop": args.hop,
            "log_mag_max": float(log_mag_max.detach().cpu()),
            "loss_final": losses[-1],
            "mse_wave": mse_wave,
            "snr_db": float(snr),
            "mse_wave_griffinlim": float(mse_wave_griffin),
            "snr_griffinlim_db": float(snr_griffin),
            "griffin_lim_iters": int(args.griffin_lim_iters),
            "mse_logmag_griffinlim": float(mse_logmag_griffin),
            "l1_logmag_griffinlim": float(l1_logmag_griffin),
            "psnr_logmag_griffinlim": float(psnr_logmag_griffin),
            "mse_mag_griffinlim": float(mse_mag_griffin),
            "l1_mag_griffinlim": float(l1_mag_griffin),
            "mse_wave_griffinlim": mse_wave_griffin,
            "snr_griffinlim_db": float(snr_griffin),
        },
        salida / "checkpoint_final.pt",
    )

    with open(salida / "metricas_audio.txt", "w", encoding="utf-8") as f:
        f.write(f"audio={ruta_audio}\n")
        f.write(f"sr={sr}\n")
        f.write(f"duracion={len(x_np) / sr:.4f}\n")
        f.write(f"n_fft={args.n_fft}\n")
        f.write(f"hop={args.hop}\n")
        f.write(f"n_frames={n_frames}\n")
        f.write(f"n_bins={n_bins}\n")
        f.write(f"n_gaussianas={args.n_gaussianas}\n")
        f.write(f"loss_final={losses[-1]:.8f}\n")
        f.write(f"mse_wave={mse_wave:.10f}\n")
        f.write(f"snr_db={snr:.4f}\n")
        f.write(f"griffin_lim_iters={int(args.griffin_lim_iters)}\n")
        f.write(f"mse_wave_griffinlim={mse_wave_griffin:.10f}\n")
        f.write(f"snr_griffinlim_db={snr_griffin:.4f}\n")
        f.write(f"mse_logmag_griffinlim={mse_logmag_griffin:.10f}\n")
        f.write(f"l1_logmag_griffinlim={l1_logmag_griffin:.10f}\n")
        f.write(f"psnr_logmag_griffinlim={psnr_logmag_griffin:.4f}\n")
        f.write(f"mse_mag_griffinlim={mse_mag_griffin:.10f}\n")
        f.write(f"l1_mag_griffinlim={l1_mag_griffin:.10f}\n")
        f.write(f"mse_wave_griffinlim={mse_wave_griffin:.10f}\n")
        f.write(f"snr_griffinlim_db={snr_griffin:.4f}\n")

        try:
            f.write(f"psnr_logmag_promedio={metricas_lm['psnr_logmag_promedio']:.4f}\n")
            f.write(f"psnr_logmag_min={metricas_lm['psnr_logmag_min']:.4f}\n")
            f.write(f"psnr_logmag_max={metricas_lm['psnr_logmag_max']:.4f}\n")
            f.write(f"psnr_logmag_p5={metricas_lm['psnr_logmag_p5']:.4f}\n")
            f.write(f"psnr_logmag_std={metricas_lm['psnr_logmag_std']:.4f}\n")
        except Exception:
            pass

    print("")
    print("=== metricas audio ===")
    print(f"mse_wave={mse_wave:.10f}")
    print(f"snr_db={snr:.2f}")
    print(f"griffin_lim_iters={int(args.griffin_lim_iters)}")
    print(f"mse_wave_griffinlim={mse_wave_griffin:.10f}")
    print(f"snr_griffinlim_db={snr_griffin:.2f}")
    print(f"psnr_logmag_griffinlim={psnr_logmag_griffin:.2f}")
    print(f"l1_logmag_griffinlim={l1_logmag_griffin:.6f}")
    print(f"mse_wave_griffinlim={mse_wave_griffin:.10f}")
    print(f"snr_griffinlim_db={snr_griffin:.2f}")

    try:
        print(
            f"psnr_logmag_prom={metricas_lm['psnr_logmag_promedio']:.2f} "
            f"min={metricas_lm['psnr_logmag_min']:.2f} "
            f"max={metricas_lm['psnr_logmag_max']:.2f} "
            f"p5={metricas_lm['psnr_logmag_p5']:.2f} "
            f"std={metricas_lm['psnr_logmag_std']:.2f}"
        )
    except Exception:
        pass
    print("")
    print(f"listo. resultados en: {salida}")
    print("escucha: original_mono.wav, recon_fase_original.wav y recon_griffinlim.wav")


if __name__ == "__main__":
    main()
