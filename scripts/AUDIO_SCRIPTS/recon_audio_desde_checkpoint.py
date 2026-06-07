import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from scipy.io import wavfile
from scipy import signal

RAIZ = Path(__file__).resolve().parents[2]
SRC = RAIZ / "src"

if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from gs2d_video.core.bases import construir_matriz_chebyshev
from gs2d_audio.core.modelo_audio import GaussianasEspectralesTemporalesCheb


def audio_to_float32(data):
    if data.dtype == np.int16:
        y = data.astype(np.float32) / 32768.0
    elif data.dtype == np.int32:
        y = data.astype(np.float32) / 2147483648.0
    elif data.dtype == np.uint8:
        y = (data.astype(np.float32) - 128.0) / 128.0
    else:
        y = data.astype(np.float32)

    y = np.nan_to_num(y)
    y = np.clip(y, -1.0, 1.0)
    return y


def cargar_wav_mono(ruta_audio, sr_objetivo=None, max_segundos=None, canal="mono"):
    sr, data = wavfile.read(str(ruta_audio))

    if data.ndim == 2:
        if canal == "left":
            data = data[:, 0]
        elif canal == "right":
            data = data[:, 1]
        else:
            data = data.mean(axis=1)

    x = audio_to_float32(data)

    if sr_objetivo is not None and int(sr_objetivo) != int(sr):
        gcd = np.gcd(int(sr), int(sr_objetivo))
        up = int(sr_objetivo) // gcd
        down = int(sr) // gcd
        x = signal.resample_poly(x, up, down).astype(np.float32)
        sr = int(sr_objetivo)

    if max_segundos is not None:
        n = int(float(max_segundos) * sr)
        x = x[:n]

    return sr, x


def guardar_wav(ruta, sr, x):
    x = np.asarray(x, dtype=np.float32)
    x = np.nan_to_num(x)

    max_abs = np.max(np.abs(x))
    if max_abs > 1.0:
        x = x / max_abs

    x_i16 = (np.clip(x, -1.0, 1.0) * 32767.0).astype(np.int16)
    wavfile.write(str(ruta), int(sr), x_i16)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    ckpt_path = Path(args.ckpt)
    cfg_path = Path(args.config)
    out_path = Path(args.out)

    if not ckpt_path.is_absolute():
        ckpt_path = RAIZ / ckpt_path
    if not cfg_path.is_absolute():
        cfg_path = RAIZ / cfg_path
    if not out_path.is_absolute():
        out_path = RAIZ / out_path

    with open(cfg_path, "r", encoding="utf-8-sig") as f:
        cfg = json.load(f)

    ckpt = torch.load(ckpt_path, map_location="cpu")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ruta_audio = Path(cfg["audio"])
    if not ruta_audio.is_absolute():
        ruta_audio = RAIZ / ruta_audio

    sr, x_np = cargar_wav_mono(
        ruta_audio,
        sr_objetivo=cfg.get("sr", ckpt["sr"]),
        max_segundos=cfg.get("max_segundos", None),
        canal=cfg.get("canal", "mono"),
    )

    x = torch.tensor(x_np, dtype=torch.float32, device=device)

    n_fft = int(ckpt.get("n_fft", cfg.get("n_fft")))
    hop = int(ckpt.get("hop", cfg.get("hop")))
    n_frames = int(ckpt["n_frames"])
    n_bins = int(ckpt["n_bins"])
    n_gaussianas = int(ckpt["n_gaussianas"])
    grados = ckpt["grados"]
    log_mag_max = float(ckpt["log_mag_max"])

    window = torch.hann_window(n_fft, device=device)

    X = torch.stft(
        x,
        n_fft=n_fft,
        hop_length=hop,
        win_length=n_fft,
        window=window,
        center=True,
        return_complex=True,
    )

    phase = torch.angle(X)

    modelo = GaussianasEspectralesTemporalesCheb(
        n_gaussianas=n_gaussianas,
        n_frames=n_frames,
        n_bins=n_bins,
        grados=grados,
        device=device,
        semilla=42,
        sigma_inicial_bins=float(cfg.get("sigma_inicial_bins", 6.0)),
    )

    state = ckpt["state_dict_coefs"]

    try:
        modelo.load_state_dict(state, strict=False)
    except Exception:
        modelo.load_state_dict_coefs(state)

    modelo.to(device)
    modelo.eval()

    grados_distintos = sorted(set(int(v) for v in grados.values()))
    matrices_base = {
        g: construir_matriz_chebyshev(
            n_frames=n_frames,
            grado_max=g,
            device=device,
            dtype=torch.float32,
        )
        for g in grados_distintos
    }

    partes = []
    bs = int(cfg.get("batch_temporal", 64))
    bs = max(1, bs)

    with torch.no_grad():
        for ini in range(0, n_frames, bs):
            fin = min(ini + bs, n_frames)
            idx = torch.arange(ini, fin, device=device, dtype=torch.long)
            pred = modelo.render_indices(idx, matrices_base)
            partes.append(pred.detach().cpu())

        pred_cpu = torch.cat(partes, dim=0).clamp(0, 1)
        pred = pred_cpu.to(device)

        pred_log_mag = pred.T.contiguous() * log_mag_max
        pred_mag = torch.expm1(pred_log_mag).clamp_min(0.0)

        X_rec = pred_mag * torch.exp(1j * phase)

        x_rec = torch.istft(
            X_rec,
            n_fft=n_fft,
            hop_length=hop,
            win_length=n_fft,
            window=window,
            center=True,
            length=x.shape[0],
        )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    guardar_wav(out_path, sr, x_rec.detach().cpu().numpy())

    print("OK")
    print("ckpt:", ckpt_path)
    print("out :", out_path)
    print("sr  :", sr)


if __name__ == "__main__":
    main()
