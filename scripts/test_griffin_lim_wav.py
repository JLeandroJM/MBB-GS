import argparse
from pathlib import Path

import numpy as np
import torch
from scipy.io import wavfile


RAIZ = Path(__file__).resolve().parents[1]


def cargar_wav(path):
    sr, data = wavfile.read(str(path))

    if data.ndim == 2:
        data = data.mean(axis=1)

    data = data.astype(np.float32)

    if np.max(np.abs(data)) > 1.5:
        data = data / 32768.0

    return sr, torch.from_numpy(data)


def guardar_wav(path, sr, audio):
    x = audio.detach().cpu().numpy()
    x = np.clip(x, -1.0, 1.0)
    wavfile.write(str(path), sr, (x * 32767.0).astype(np.int16))


def griffin_lim(mag, n_fft, hop, n_iter, length, device):
    window = torch.hann_window(n_fft, device=device)

    # fase aleatoria inicial
    phase = torch.exp(2j * torch.pi * torch.rand_like(mag))

    X = mag * phase

    for i in range(n_iter):
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

        if (i + 1) % 20 == 0:
            print(f"iter {i+1}/{n_iter}")

    wav = torch.istft(
        X,
        n_fft=n_fft,
        hop_length=hop,
        window=window,
        length=length,
    )

    return wav


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--n-fft", type=int, default=2048)
    parser.add_argument("--hop", type=int, default=512)
    parser.add_argument("--iters", type=int, default=80)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    input_path = Path(args.input)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    sr, wav = cargar_wav(input_path)
    wav = wav.to(device)

    print(f"device: {device}")
    print(f"input : {input_path}")
    print(f"output: {output_path}")
    print(f"sr: {sr}")
    print(f"samples: {wav.numel()}")

    window = torch.hann_window(args.n_fft, device=device)

    X = torch.stft(
        wav,
        n_fft=args.n_fft,
        hop_length=args.hop,
        window=window,
        return_complex=True,
    )

    mag = torch.abs(X)

    wav_gl = griffin_lim(
        mag=mag,
        n_fft=args.n_fft,
        hop=args.hop,
        n_iter=args.iters,
        length=wav.numel(),
        device=device,
    )

    guardar_wav(output_path, sr, wav_gl)

    print("listo")


if __name__ == "__main__":
    main()
