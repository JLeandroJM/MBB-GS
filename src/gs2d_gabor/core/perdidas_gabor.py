"""
Perdidas para Gabor splatting de audio en waveform directa.

Soporta:
  - L1 waveform
  - Charbonnier waveform
  - MSE waveform
  - MR-STFT multi-resolucion

loss_total =
    lambda_wave * wave_loss
  + lambda_mse_wave * MSE_wave
  + lambda_mrstft * MR-STFT
"""
import torch


_EPS = 1e-7


def _stft_mag(x, n_fft, hop, win):
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

    # Spectral convergence
    num = torch.linalg.norm(S - S_hat)
    den = torch.linalg.norm(S).clamp_min(_EPS)
    sc = num / den

    # Log-magnitude L1
    mag = torch.mean(torch.abs(torch.log(S + _EPS) - torch.log(S_hat + _EPS)))

    return sc + mag


def mrstft_loss(x_hat, x, ffts=(512, 1024, 2048), device=None):
    if device is None:
        device = x_hat.device

    total = x_hat.new_tensor(0.0)
    usados = 0

    for n_fft in ffts:
        n_fft = int(n_fft)

        if x_hat.shape[-1] < n_fft:
            continue

        hop = max(1, n_fft // 4)
        win = torch.hann_window(n_fft, device=device, dtype=x_hat.dtype)

        total = total + _stft_loss_una_resolucion(x_hat, x, n_fft, hop, win)
        usados += 1

    if usados == 0:
        return x_hat.new_tensor(0.0)

    return total / usados


def wave_loss_fn(x_hat, x, config):
    tipo = str(config.get("tipo_wave_loss", "l1")).lower().strip()

    if tipo == "l1":
        return torch.mean(torch.abs(x_hat - x)), "l_wave_l1"

    if tipo == "charbonnier":
        eps = float(config.get("charbonnier_eps", 1e-3))
        return torch.mean(torch.sqrt((x_hat - x) ** 2 + eps ** 2)), "l_wave_charbonnier"

    if tipo == "mse":
        return torch.mean((x_hat - x) ** 2), "l_wave_mse"

    raise ValueError(f"tipo_wave_loss no soportado: {tipo}")



def spectral_overshoot_loss(
    x_hat,
    x,
    ffts=(512, 1024, 2048),
    margin_db=1.5,
    highfreq_start_hz=None,
    sr=None,
    device=None,
):
    """
    Penaliza energia espectral INVENTADA.

    No castiga que falte energia.
    Solo castiga cuando:
        log_mag_recon > log_mag_original + margen

    Esto apunta al hiss/zumbido:
    energia extra repartida en el espectro.
    """
    if device is None:
        device = x_hat.device

    total = x_hat.new_tensor(0.0)
    usados = 0

    margin_log = float(margin_db) / 20.0 * torch.log(x_hat.new_tensor(10.0))

    for n_fft in ffts:
        n_fft = int(n_fft)

        if x_hat.shape[-1] < n_fft:
            continue

        hop = max(1, n_fft // 4)
        win = torch.hann_window(n_fft, device=device, dtype=x_hat.dtype)

        S = _stft_mag(x, n_fft, hop, win)
        S_hat = _stft_mag(x_hat, n_fft, hop, win)

        log_s = torch.log(S + _EPS)
        log_hat = torch.log(S_hat + _EPS)

        overshoot = torch.relu(log_hat - log_s - margin_log)

        if highfreq_start_hz is not None and sr is not None:
            # Peso suave para enfocarse mas en media/alta frecuencia.
            freqs = torch.linspace(
                0.0,
                float(sr) / 2.0,
                S.shape[0],
                device=device,
                dtype=x_hat.dtype,
            ).view(-1, 1)

            denom = max(1.0, float(sr) / 2.0 - float(highfreq_start_hz))
            w = torch.clamp((freqs - float(highfreq_start_hz)) / denom, 0.0, 1.0)

            # Mantener una penalizacion base y reforzar altas.
            overshoot = overshoot * (0.25 + 0.75 * w)

        total = total + torch.mean(overshoot * overshoot)
        usados += 1

    if usados == 0:
        return x_hat.new_tensor(0.0)

    return total / usados

def loss_gabor(x_hat, x, config):
    lambda_wave = float(config.get("lambda_wave", 0.0))
    lambda_mse_wave = float(config.get("lambda_mse_wave", 0.0))
    lambda_mrstft = float(config.get("lambda_mrstft", 1.0))
    lambda_spectral_overshoot = float(config.get("lambda_spectral_overshoot", 0.0))

    ffts = config.get("mrstft_ffts", [512, 1024, 2048])
    overshoot_ffts = config.get("overshoot_ffts", ffts)
    overshoot_margin_db = float(config.get("overshoot_margin_db", 1.5))
    overshoot_highfreq_start_hz = config.get("overshoot_highfreq_start_hz", None)
    sr = config.get("sr", None)

    loss = x_hat.new_tensor(0.0)
    partes = {}

    if lambda_wave > 0.0:
        l_wave, nombre = wave_loss_fn(x_hat, x, config)
        loss = loss + lambda_wave * l_wave
        partes[nombre] = float(l_wave.detach())

    if lambda_mse_wave > 0.0:
        l_mse = torch.mean((x_hat - x) ** 2)
        loss = loss + lambda_mse_wave * l_mse
        partes["l_mse_wave"] = float(l_mse.detach())

    if lambda_mrstft > 0.0:
        l_mr = mrstft_loss(x_hat, x, ffts=ffts, device=x_hat.device)
        loss = loss + lambda_mrstft * l_mr
        partes["l_mrstft"] = float(l_mr.detach())

    if lambda_spectral_overshoot > 0.0:
        l_over = spectral_overshoot_loss(
            x_hat,
            x,
            ffts=overshoot_ffts,
            margin_db=overshoot_margin_db,
            highfreq_start_hz=overshoot_highfreq_start_hz,
            sr=sr,
            device=x_hat.device,
        )
        loss = loss + lambda_spectral_overshoot * l_over
        partes["l_spectral_overshoot"] = float(l_over.detach())

    return loss, partes


@torch.no_grad()
def metricas_audio(x_hat, x):
    x = x.float()
    x_hat = x_hat.float()

    err = x - x_hat
    pot_senal = torch.sum(x * x).clamp_min(1e-12)
    pot_error = torch.sum(err * err).clamp_min(1e-12)

    snr = 10.0 * torch.log10(pot_senal / pot_error)

    mse = torch.mean(err * err).clamp_min(1e-12)

    # PSNR asumiendo rango [-1, 1], rango total 2.
    psnr = 10.0 * torch.log10((2.0 ** 2) / mse)

    return {
        "snr_db": float(snr.detach().cpu()),
        "psnr_db": float(psnr.detach().cpu()),
        "mse_wave": float(mse.detach().cpu()),
    }
