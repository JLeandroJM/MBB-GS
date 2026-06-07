"""
Perdidas para Gabor splatting de audio en el dominio del tiempo.

Para audio, comparar la waveform sample-a-sample (L1/MSE) es perceptualmente
malo: dos senales que suenan identicas pueden diferir mucho muestra a muestra
por desfases minusculos. El estandar moderno (Parallel WaveGAN, HiFi-GAN, DDSP)
es la perdida multi-resolution STFT (MR-STFT): comparar la MAGNITUD del
espectrograma a varias resoluciones.

loss_total = lambda_wave * L1(x_hat, x)
           + lambda_mrstft * MR-STFT(x_hat, x)

MR-STFT a cada resolucion combina:
    - spectral convergence:  ||S - S_hat||_F / ||S||_F
    - log-magnitude L1:      mean(|log(S+eps) - log(S_hat+eps)|)
"""
import torch


_EPS = 1e-7


def _stft_mag(x, n_fft, hop, win):
    """x: [T] -> magnitud STFT [F, frames]."""
    X = torch.stft(
        x,
        n_fft=n_fft,
        hop_length=hop,
        win_length=n_fft,
        window=win,
        center=True,
        return_complex=True,
        pad_mode="reflect",
    )
    return X.abs()


def _stft_loss_una_resolucion(x_hat, x, n_fft, hop, win):
    S = _stft_mag(x, n_fft, hop, win)
    S_hat = _stft_mag(x_hat, n_fft, hop, win)

    # spectral convergence
    num = torch.linalg.norm(S - S_hat)
    den = torch.linalg.norm(S).clamp_min(_EPS)
    sc = num / den

    # log-magnitude L1
    mag = torch.mean(torch.abs(torch.log(S + _EPS) - torch.log(S_hat + _EPS)))

    return sc + mag


def mrstft_loss(x_hat, x, ffts=(512, 1024, 2048), device=None):
    """
    Multi-resolution STFT loss. ffts: lista de tamanos de ventana.
    hop = n_fft // 4 para cada resolucion.
    """
    if device is None:
        device = x_hat.device

    total = x_hat.new_tensor(0.0)
    for n_fft in ffts:
        n_fft = int(n_fft)
        # Si la senal es mas corta que la ventana, saltar esa resolucion.
        if x_hat.shape[-1] < n_fft:
            continue
        hop = max(1, n_fft // 4)
        win = torch.hann_window(n_fft, device=device, dtype=x_hat.dtype)
        total = total + _stft_loss_una_resolucion(x_hat, x, n_fft, hop, win)

    return total


def loss_gabor(x_hat, x, config):
    """
    Perdida combinada para Gabor audio.

    config:
        lambda_wave   : peso del L1 en waveform (default 0.0)
        lambda_mrstft : peso de la MR-STFT (default 1.0)
        mrstft_ffts   : lista de tamanos de ventana (default [512,1024,2048])
    """
    lambda_wave = float(config.get("lambda_wave", 0.0))
    lambda_mrstft = float(config.get("lambda_mrstft", 1.0))
    ffts = config.get("mrstft_ffts", [512, 1024, 2048])

    loss = x_hat.new_tensor(0.0)
    partes = {}

    if lambda_wave > 0.0:
        l_wave = torch.mean(torch.abs(x_hat - x))
        loss = loss + lambda_wave * l_wave
        partes["l_wave"] = float(l_wave.detach())

    if lambda_mrstft > 0.0:
        l_mr = mrstft_loss(x_hat, x, ffts=ffts, device=x_hat.device)
        loss = loss + lambda_mrstft * l_mr
        partes["l_mrstft"] = float(l_mr.detach())

    return loss, partes


@torch.no_grad()
def metricas_audio(x_hat, x):
    """
    SNR y PSNR en el dominio del tiempo, mas MSE.

    SNR_dB = 10 log10( ||x||^2 / ||x - x_hat||^2 )
    """
    x = x.float()
    x_hat = x_hat.float()

    err = x - x_hat
    pot_senal = torch.sum(x * x).clamp_min(1e-12)
    pot_error = torch.sum(err * err).clamp_min(1e-12)

    snr = 10.0 * torch.log10(pot_senal / pot_error)

    mse = torch.mean(err * err).clamp_min(1e-12)
    # PSNR asumiendo rango de senal [-1, 1] (amplitud pico 1, rango 2).
    psnr = 10.0 * torch.log10((2.0 ** 2) / mse)

    return {
        "snr_db": float(snr.detach().cpu()),
        "psnr_db": float(psnr.detach().cpu()),
        "mse_wave": float(mse.detach().cpu()),
    }
