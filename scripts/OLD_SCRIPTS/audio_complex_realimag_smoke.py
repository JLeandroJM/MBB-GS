import argparse
import sys
from pathlib import Path

import imageio_ffmpeg
import numpy as np
import torch
import torch.nn.functional as F
from scipy.io import wavfile

RAIZ = Path(__file__).resolve().parents[1]
SRC = RAIZ / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from gs2d_complex.core.modelo_complex import (
    GaussianasComplejasTemporalesCheb,
    construir_bases_por_grado,
    smoothness_complex,
)


def cargar_wav(path, sr_objetivo, max_segundos):
    sr, data = wavfile.read(str(path))

    if data.ndim == 2:
        data = data.mean(axis=1)

    data = data.astype(np.float32)

    if data.max() > 1.5 or data.min() < -1.5:
        data = data / 32768.0

    if sr != sr_objetivo:
        raise ValueError(f"El wav tiene sr={sr}, pero pediste sr={sr_objetivo}. Convierte antes con ffmpeg.")

    n = int(sr_objetivo * max_segundos)
    data = data[:n]

    return sr, torch.from_numpy(data.astype(np.float32))


def guardar_wav(path, sr, audio):
    audio_np = audio.detach().cpu().numpy()
    audio_np = np.clip(audio_np, -1.0, 1.0)
    wavfile.write(str(path), sr, (audio_np * 32767.0).astype(np.int16))


def psnr_from_mse(mse):
    mse = max(float(mse), 1e-12)
    return -10.0 * np.log10(mse)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--audio", default="data/audio/RockThatBody_clean_mono_44k.wav")
    parser.add_argument("--nombre", default="complex_smoke_5s")
    parser.add_argument("--max-segundos", type=float, default=5.0)
    parser.add_argument("--sr", type=int, default=44100)
    parser.add_argument("--n-fft", type=int, default=2048)
    parser.add_argument("--hop", type=int, default=512)
    parser.add_argument("--n-gaussianas", type=int, default=512)
    parser.add_argument("--epochs", type=int, default=800)
    parser.add_argument("--batch-temporal", type=int, default=64)
    parser.add_argument("--log-cada", type=int, default=50)

    parser.add_argument("--grado-mu-f", type=int, default=40)
    parser.add_argument("--grado-sigma-f", type=int, default=10)
    parser.add_argument("--grado-amp-real", type=int, default=40)
    parser.add_argument("--grado-amp-imag", type=int, default=40)

    parser.add_argument("--lr", type=float, default=2e-3)
    parser.add_argument("--lambda-mse", type=float, default=0.25)
    parser.add_argument("--lambda-smooth", type=float, default=1e-8)

    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    salida = RAIZ / "outputs" / "audio_complex" / args.nombre
    salida.mkdir(parents=True, exist_ok=True)

    print(f"device: {device}")
    print(f"salida: {salida}")

    sr, wav = cargar_wav(RAIZ / args.audio, args.sr, args.max_segundos)
    wav = wav.to(device)

    window = torch.hann_window(args.n_fft, device=device)

    X = torch.stft(
        wav,
        n_fft=args.n_fft,
        hop_length=args.hop,
        window=window,
        return_complex=True,
    )

    # X viene [F, T]. Lo pasamos a [T, F].
    target_real = X.real.T.contiguous()
    target_imag = X.imag.T.contiguous()

    # Normalizacion compartida para real/imag.
    scale = torch.sqrt(target_real.pow(2) + target_imag.pow(2)).max().clamp_min(1e-8)

    target_real_n = target_real / scale
    target_imag_n = target_imag / scale

    n_tiempos, n_bins = target_real_n.shape

    print(f"audio: {RAIZ / args.audio}")
    print(f"sr: {sr}")
    print(f"duracion: {wav.numel() / sr:.2f}s")
    print(f"target complex: T={n_tiempos}, F={n_bins}")
    print(f"scale complejo: {float(scale.detach().cpu()):.6f}")

    grados = {
        "mu_f": args.grado_mu_f,
        "sigma_f": args.grado_sigma_f,
        "amp_real": args.grado_amp_real,
        "amp_imag": args.grado_amp_imag,
    }

    modelo = GaussianasComplejasTemporalesCheb(
        n_gaussianas=args.n_gaussianas,
        n_bins=n_bins,
        n_tiempos=n_tiempos,
        grados=grados,
        device=device,
    ).to(device)

    bases = construir_bases_por_grado(n_tiempos, grados, device)

    opt = torch.optim.Adam(modelo.parameters(), lr=args.lr)

    print(f"grados: {grados}")
    print("")
    print("=== entrenamiento complex real+imag ===")

    losses = []

    for epoch in range(1, args.epochs + 1):
        modelo.train()
        opt.zero_grad(set_to_none=True)

        loss_l1_epoch = 0.0
        loss_mse_epoch = 0.0
        loss_data_epoch = 0.0

        chunk = max(1, int(args.batch_temporal))

        for ini in range(0, n_tiempos, chunk):
            fin = min(ini + chunk, n_tiempos)
            idx = torch.arange(ini, fin, device=device, dtype=torch.long)

            pred_real, pred_imag = modelo.render_indices(idx, bases)

            gt_real = target_real_n[idx]
            gt_imag = target_imag_n[idx]
            mag_gt = torch.sqrt(gt_real ** 2 + gt_imag ** 2)
            weight = 1.0 + 5.0 * (mag_gt / mag_gt.mean().clamp_min(1e-8)).clamp(0.0, 5.0)

            diff_real = pred_real - gt_real
            diff_imag = pred_imag - gt_imag

            loss_l1 = 0.5 * (
                torch.sum(torch.abs(diff_real) * weight) / weight.sum().clamp_min(1e-8) +
                torch.sum(torch.abs(diff_imag) * weight) / weight.sum().clamp_min(1e-8)
            )

            loss_mse = 0.5 * (
                torch.sum((diff_real ** 2) * weight) / weight.sum().clamp_min(1e-8) +
                torch.sum((diff_imag ** 2) * weight) / weight.sum().clamp_min(1e-8)
            )

            peso = float(fin - ini) / float(n_tiempos)
            loss_chunk = (loss_l1 + args.lambda_mse * loss_mse) * peso
            loss_chunk.backward()

            loss_l1_epoch += float(loss_l1.detach().cpu()) * peso
            loss_mse_epoch += float(loss_mse.detach().cpu()) * peso
            loss_data_epoch += float((loss_l1 + args.lambda_mse * loss_mse).detach().cpu()) * peso

        loss_smooth = smoothness_complex(modelo)

        if args.lambda_smooth != 0:
            (args.lambda_smooth * loss_smooth).backward()

        opt.step()

        loss_total = loss_data_epoch + args.lambda_smooth * float(loss_smooth.detach().cpu())
        losses.append(loss_total)

        if epoch == 1 or epoch % args.log_cada == 0 or epoch == args.epochs:
            print(
                f"epoch {epoch:04d}/{args.epochs} "
                f"loss={loss_total:.6f} "
                f"l1={loss_l1_epoch:.6f} "
                f"mse={loss_mse_epoch:.6f} "
                f"psnr_complex={psnr_from_mse(loss_mse_epoch):.2f}",
                flush=True,
            )

    print("")
    print("=== reconstruccion final ===")

    modelo.eval()

    with torch.no_grad():
        partes_real = []
        partes_imag = []

        chunk = max(1, int(args.batch_temporal))

        for ini in range(0, n_tiempos, chunk):
            fin = min(ini + chunk, n_tiempos)
            idx = torch.arange(ini, fin, device=device, dtype=torch.long)

            pr, pi = modelo.render_indices(idx, bases)
            partes_real.append(pr.detach())
            partes_imag.append(pi.detach())

        pred_real_n = torch.cat(partes_real, dim=0)
        pred_imag_n = torch.cat(partes_imag, dim=0)

        pred_real = pred_real_n.T.contiguous() * scale
        pred_imag = pred_imag_n.T.contiguous() * scale

        X_hat = torch.complex(pred_real, pred_imag)

        wav_hat = torch.istft(
            X_hat,
            n_fft=args.n_fft,
            hop_length=args.hop,
            window=window,
            length=wav.numel(),
        )

        mse_wave = torch.mean((wav_hat - wav) ** 2).item()
        power = torch.mean(wav ** 2).item()
        snr = 10.0 * np.log10(max(power, 1e-12) / max(mse_wave, 1e-12))

    guardar_wav(salida / "original_mono.wav", sr, wav)
    guardar_wav(salida / "recon_complex.wav", sr, wav_hat)

    rms_orig = torch.sqrt(torch.mean(wav ** 2)).clamp_min(1e-8)
    rms_recon = torch.sqrt(torch.mean(wav_hat ** 2)).clamp_min(1e-8)
    gain = (rms_orig / rms_recon).clamp(max=20.0)

    wav_hat_gain = wav_hat * gain
    guardar_wav(salida / "recon_complex_gainmatched.wav", sr, wav_hat_gain)

    torch.save(
        {
            "modelo_state_dict": modelo.state_dict(),
            "grados": grados,
            "n_gaussianas": args.n_gaussianas,
            "n_bins": n_bins,
            "n_tiempos": n_tiempos,
            "sr": sr,
            "n_fft": args.n_fft,
            "hop": args.hop,
            "scale": float(scale.detach().cpu()),
            "args": vars(args),
        },
        salida / "checkpoint_final.pt",
    )

    with open(salida / "metricas_complex.txt", "w", encoding="utf-8") as f:
        f.write(f"loss_final={losses[-1]:.8f}\n")
        f.write(f"mse_wave={mse_wave:.10f}\n")
        f.write(f"snr_db={snr:.4f}\n")
        f.write(f"n_tiempos={n_tiempos}\n")
        f.write(f"n_bins={n_bins}\n")
        f.write(f"scale={float(scale.detach().cpu()):.8f}\n")

    print(f"mse_wave={mse_wave:.10f}")
    print(f"snr_db={snr:.2f}")
    print(f"listo. resultados en: {salida}")
    print("escucha: original_mono.wav y recon_complex.wav")


if __name__ == "__main__":
    main()
