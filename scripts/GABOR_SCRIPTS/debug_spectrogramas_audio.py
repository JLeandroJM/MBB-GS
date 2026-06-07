from pathlib import Path
import argparse
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile
from scipy.signal import stft


def cargar_wav(path, sr_objetivo=None, max_segundos=None):
    sr, data = wavfile.read(str(path))

    if data.dtype == np.int16:
        data = data.astype(np.float32) / 32768.0
    elif data.dtype == np.int32:
        data = data.astype(np.float32) / 2147483648.0
    else:
        data = data.astype(np.float32)

    if data.ndim == 2:
        data = data[:, 0]

    if max_segundos is not None:
        n = int(sr * max_segundos)
        data = data[:n]

    if sr_objetivo is not None and sr != sr_objetivo:
        raise ValueError(f"sr distinto: archivo={sr}, esperado={sr_objetivo}")

    return sr, data


def guardar_wav(path, sr, x):
    x = np.nan_to_num(x)
    mx = np.max(np.abs(x)) if x.size else 1.0
    if mx > 1.0:
        x = x / mx
    wavfile.write(str(path), sr, (np.clip(x, -1, 1) * 32767).astype(np.int16))


def stft_logmag(x, sr, n_fft=2048, hop=512):
    f, t, Z = stft(
        x,
        fs=sr,
        window="hann",
        nperseg=n_fft,
        noverlap=n_fft - hop,
        nfft=n_fft,
        boundary=None,
        padded=False,
    )
    mag = np.abs(Z)
    logmag = 20.0 * np.log10(np.maximum(mag, 1e-8))
    return f, t, logmag, mag


def lsd_por_frame(log_a, log_b):
    d = log_a - log_b
    return np.sqrt(np.mean(d * d, axis=0))


def energia_banda(mag, f, f0, f1):
    mask = (f >= f0) & (f < f1)
    if not np.any(mask):
        return 0.0
    return float(np.mean(mag[mask] ** 2))


def plot_spec(path, f, t, logmag, titulo, vmin=None, vmax=None):
    plt.figure(figsize=(12, 5))
    plt.imshow(
        logmag,
        origin="lower",
        aspect="auto",
        extent=[t[0] if len(t) else 0, t[-1] if len(t) else 0, f[0], f[-1]],
        vmin=vmin,
        vmax=vmax,
    )
    plt.colorbar(label="dB")
    plt.xlabel("Tiempo (s)")
    plt.ylabel("Frecuencia (Hz)")
    plt.title(titulo)
    plt.tight_layout()
    plt.savefig(path, dpi=140)
    plt.close()


def plot_line(path, x, ys, labels, titulo, xlabel, ylabel):
    plt.figure(figsize=(12, 5))
    for y, label in zip(ys, labels):
        plt.plot(x, y, label=label)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(titulo)
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=140)
    plt.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--original", required=True)
    ap.add_argument("--recon", required=True)
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--sr", type=int, default=None)
    ap.add_argument("--max-segundos", type=float, default=None)
    ap.add_argument("--n-fft", type=int, default=2048)
    ap.add_argument("--hop", type=int, default=512)
    args = ap.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    sr_o, x = cargar_wav(args.original, sr_objetivo=args.sr, max_segundos=args.max_segundos)
    sr_r, y = cargar_wav(args.recon, sr_objetivo=args.sr, max_segundos=args.max_segundos)

    if sr_o != sr_r:
        raise ValueError("Los sample rates no coinciden")

    n = min(len(x), len(y))
    x = x[:n]
    y = y[:n]

    e = y - x

    guardar_wav(outdir / "original.wav", sr_o, x)
    guardar_wav(outdir / "recon.wav", sr_o, y)
    guardar_wav(outdir / "residual.wav", sr_o, e)
    guardar_wav(outdir / "residual_x5.wav", sr_o, e * 5.0)

    f, t, log_x, mag_x = stft_logmag(x, sr_o, n_fft=args.n_fft, hop=args.hop)
    _, _, log_y, mag_y = stft_logmag(y, sr_o, n_fft=args.n_fft, hop=args.hop)
    _, _, log_e, mag_e = stft_logmag(e, sr_o, n_fft=args.n_fft, hop=args.hop)

    diff_log = log_y - log_x
    abs_diff_log = np.abs(diff_log)

    vmin = min(np.min(log_x), np.min(log_y))
    vmax = max(np.max(log_x), np.max(log_y))

    plot_spec(outdir / "spec_original.png", f, t, log_x, "Original log-spectrogram", vmin=vmin, vmax=vmax)
    plot_spec(outdir / "spec_recon.png", f, t, log_y, "Recon log-spectrogram", vmin=vmin, vmax=vmax)
    plot_spec(outdir / "spec_residual.png", f, t, log_e, "Residual log-spectrogram")
    plot_spec(outdir / "spec_diff.png", f, t, diff_log, "Recon - Original (logmag diff)")
    plot_spec(outdir / "spec_abs_diff.png", f, t, abs_diff_log, "Abs logmag diff")

    mean_log_x = np.mean(log_x, axis=1)
    mean_log_y = np.mean(log_y, axis=1)
    mean_log_e = np.mean(log_e, axis=1)
    mean_abs_diff = np.mean(abs_diff_log, axis=1)

    plot_line(
        outdir / "mean_spectrum.png",
        f,
        [mean_log_x, mean_log_y, mean_log_e],
        ["original", "recon", "residual"],
        "Mean log-spectrum",
        "Frecuencia (Hz)",
        "dB",
    )

    plot_line(
        outdir / "mean_abs_diff_per_freq.png",
        f,
        [mean_abs_diff],
        ["mean abs diff"],
        "Mean abs logmag diff per frequency",
        "Frecuencia (Hz)",
        "dB",
    )

    lsd_frames = lsd_por_frame(log_x, log_y)
    plt.figure(figsize=(12, 4))
    plt.plot(t[: len(lsd_frames)], lsd_frames)
    plt.xlabel("Tiempo (s)")
    plt.ylabel("LSD frame")
    plt.title("LSD por frame")
    plt.tight_layout()
    plt.savefig(outdir / "lsd_por_frame.png", dpi=140)
    plt.close()

    bandas = [
        ("0_1k", 0, 1000),
        ("1k_3k", 1000, 3000),
        ("3k_6k", 3000, 6000),
        ("6k_10k", 6000, 10000),
        ("10k_16k", 10000, 16000),
        ("16k_nyq", 16000, sr_o / 2.0),
    ]

    lineas = []
    lineas.append("DEBUG AUDIO ESPECTROGRAMAS")
    lineas.append("=" * 70)
    lineas.append(f"original={args.original}")
    lineas.append(f"recon={args.recon}")
    lineas.append(f"outdir={outdir}")
    lineas.append(f"sr={sr_o}")
    lineas.append(f"samples={n}")
    lineas.append(f"duracion={n / sr_o:.2f}s")
    lineas.append(f"n_fft={args.n_fft}")
    lineas.append(f"hop={args.hop}")
    lineas.append("")

    mse = float(np.mean((y - x) ** 2))
    snr = 10.0 * np.log10(np.sum(x ** 2) / max(np.sum((x - y) ** 2), 1e-12))
    lineas.append(f"mse_wave={mse:.10f}")
    lineas.append(f"snr_db={snr:.6f}")
    lineas.append(f"lsd_promedio={float(np.mean(lsd_frames)):.6f}")
    lineas.append(f"lsd_std={float(np.std(lsd_frames)):.6f}")
    lineas.append("")

    lineas.append("ENERGIA POR BANDAS")
    lineas.append("-" * 70)
    for nombre, f0, f1 in bandas:
        ex = energia_banda(mag_x, f, f0, f1)
        ey = energia_banda(mag_y, f, f0, f1)
        ee = energia_banda(mag_e, f, f0, f1)
        ratio = ee / max(ex, 1e-12)
        lineas.append(
            f"{nombre:10s} | orig={ex:.8e} | recon={ey:.8e} | residual={ee:.8e} | residual/orig={ratio:.6f}"
        )

    (outdir / "metricas_debug.txt").write_text("\n".join(lineas), encoding="utf-8")
    print("\n".join(lineas))
    print("")
    print("Archivos generados:")
    print(outdir / "spec_original.png")
    print(outdir / "spec_recon.png")
    print(outdir / "spec_residual.png")
    print(outdir / "spec_diff.png")
    print(outdir / "mean_spectrum.png")
    print(outdir / "metricas_debug.txt")


if __name__ == "__main__":
    main()