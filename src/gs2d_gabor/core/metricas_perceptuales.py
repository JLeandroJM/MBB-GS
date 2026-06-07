import math
import torch

from gs2d_gabor.core.perdidas_gabor import mrstft_loss, spectral_overshoot_loss


_EPS = 1e-8


@torch.no_grad()
def si_sdr_db(x_hat, x):
    x = x.float()
    x_hat = x_hat.float()

    x_zm = x - torch.mean(x)
    y_zm = x_hat - torch.mean(x_hat)

    scale = torch.sum(y_zm * x_zm) / torch.sum(x_zm * x_zm).clamp_min(_EPS)
    s_target = scale * x_zm
    e_noise = y_zm - s_target

    ratio = torch.sum(s_target * s_target).clamp_min(_EPS) / torch.sum(e_noise * e_noise).clamp_min(_EPS)
    return float((10.0 * torch.log10(ratio)).detach().cpu())


def _stft_mag(x, n_fft, hop, win):
    X = torch.stft(
        x,
        n_fft=int(n_fft),
        hop_length=int(hop),
        win_length=int(n_fft),
        window=win,
        center=True,
        return_complex=True,
        pad_mode="reflect",
    )
    return X.abs().clamp_min(_EPS)


@torch.no_grad()
def lsd_db_multi(x_hat, x, ffts=(512, 1024, 2048)):
    """
    Log Spectral Distance promedio.
    Menor es mejor.
    """
    device = x.device
    dtype = x.dtype

    valores = {}
    vals = []

    for n_fft in ffts:
        n_fft = int(n_fft)
        if x.shape[-1] < n_fft:
            continue

        hop = max(1, n_fft // 4)
        win = torch.hann_window(n_fft, device=device, dtype=dtype)

        S = _stft_mag(x, n_fft, hop, win)
        S_hat = _stft_mag(x_hat, n_fft, hop, win)

        db = 20.0 * torch.log10(S)
        db_hat = 20.0 * torch.log10(S_hat)

        # RMS por frame y promedio temporal
        lsd_frame = torch.sqrt(torch.mean((db - db_hat) ** 2, dim=0).clamp_min(_EPS))
        lsd = torch.mean(lsd_frame)

        key = f"lsd_db_fft{n_fft}"
        valores[key] = float(lsd.detach().cpu())
        vals.append(lsd)

    if vals:
        valores["lsd_db_promedio"] = float(torch.stack(vals).mean().detach().cpu())
    else:
        valores["lsd_db_promedio"] = float("nan")

    return valores


def _hz_to_mel(hz):
    return 2595.0 * math.log10(1.0 + hz / 700.0)


def _mel_to_hz(mel):
    return 700.0 * (10.0 ** (mel / 2595.0) - 1.0)


def _mel_filterbank(sr, n_fft, n_mels, f_min, f_max, device, dtype):
    n_freqs = n_fft // 2 + 1

    m_min = _hz_to_mel(float(f_min))
    m_max = _hz_to_mel(float(f_max))

    mels = torch.linspace(m_min, m_max, n_mels + 2, device=device, dtype=dtype)
    hz = torch.tensor([_mel_to_hz(float(m)) for m in mels], device=device, dtype=dtype)

    bins = torch.floor((n_fft + 1) * hz / float(sr)).long()
    bins = torch.clamp(bins, 0, n_freqs - 1)

    fb = torch.zeros(n_mels, n_freqs, device=device, dtype=dtype)

    for m in range(1, n_mels + 1):
        left = int(bins[m - 1])
        center = int(bins[m])
        right = int(bins[m + 1])

        if center <= left:
            center = min(left + 1, n_freqs - 1)
        if right <= center:
            right = min(center + 1, n_freqs - 1)

        if center > left:
            fb[m - 1, left:center] = (
                torch.arange(left, center, device=device, dtype=dtype) - left
            ) / max(1, center - left)

        if right > center:
            fb[m - 1, center:right] = (
                right - torch.arange(center, right, device=device, dtype=dtype)
            ) / max(1, right - center)

    return fb


@torch.no_grad()
def mel_l1_log(x_hat, x, sr, n_fft=2048, hop=None, n_mels=80, f_min=0.0, f_max=None):
    """
    Log-Mel L1.
    Menor es mejor.
    """
    if f_max is None:
        f_max = sr / 2.0

    if hop is None:
        hop = n_fft // 4

    device = x.device
    dtype = x.dtype

    if x.shape[-1] < n_fft:
        return float("nan")

    win = torch.hann_window(int(n_fft), device=device, dtype=dtype)

    S = _stft_mag(x, n_fft, hop, win)
    S_hat = _stft_mag(x_hat, n_fft, hop, win)

    fb = _mel_filterbank(
        sr=sr,
        n_fft=int(n_fft),
        n_mels=int(n_mels),
        f_min=f_min,
        f_max=f_max,
        device=device,
        dtype=dtype,
    )

    mel = torch.matmul(fb, S)
    mel_hat = torch.matmul(fb, S_hat)

    log_mel = torch.log(mel + _EPS)
    log_mel_hat = torch.log(mel_hat + _EPS)

    return float(torch.mean(torch.abs(log_mel - log_mel_hat)).detach().cpu())


@torch.no_grad()
def calcular_metricas_perceptuales(x_hat, x, sr, config):
    ffts = config.get("mrstft_ffts", [512, 1024, 2048])

    out = {}

    out["si_sdr_db"] = si_sdr_db(x_hat, x)

    out.update(lsd_db_multi(x_hat, x, ffts=ffts))

    out["mel_l1_log"] = mel_l1_log(
        x_hat,
        x,
        sr=int(sr),
        n_fft=int(config.get("metricas_mel_n_fft", 2048)),
        hop=config.get("metricas_mel_hop", None),
        n_mels=int(config.get("metricas_n_mels", 80)),
        f_min=float(config.get("metricas_mel_f_min", 0.0)),
        f_max=config.get("metricas_mel_f_max", None),
    )

    out["mrstft_final"] = float(
        mrstft_loss(x_hat, x, ffts=ffts, device=x_hat.device).detach().cpu()
    )

    out["spectral_overshoot_final"] = float(
        spectral_overshoot_loss(
            x_hat,
            x,
            ffts=config.get("overshoot_ffts", ffts),
            margin_db=float(config.get("overshoot_margin_db", 1.5)),
            highfreq_start_hz=config.get("overshoot_highfreq_start_hz", None),
            sr=sr,
            device=x_hat.device,
        ).detach().cpu()
    )

    return out
