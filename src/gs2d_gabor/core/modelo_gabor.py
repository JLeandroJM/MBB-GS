"""
Modelo de Gabor splatting para audio en el dominio del tiempo (waveform).

A diferencia del enfoque de espectrograma (gaussianas en frecuencia que
evolucionan en el tiempo via Chebyshev), aqui cada atomo es ESTATICO y vive
directamente sobre el eje del tiempo de la senal:

    x_hat(t) = sum_i A_i * exp(-(t - mu_i)^2 / (2 sigma_i^2)) * cos(2 pi f_i (t - mu_i) + phi_i)

No hay STFT, no hay fase separada, no hay polinomios temporales. La senal cruda
se aproxima como superposicion de atomos de Gabor (gaussiana modulada por una
sinusoide). Esto resuelve el problema de Nyquist de las gaussianas puras: la
modulacion cos permite representar oscilaciones de audio con pocos atomos.

Parametrizacion (raw -> activado):
    mu_t   : posicion temporal en samples. raw directo, se clampa en [0, T-1].
    sigma  : ancho gaussiano. sigma = exp(log_sigma), clampeado a un rango.
    amp    : amplitud con signo. raw directo.
    fnorm  : frecuencia normalizada en ciclos/sample. fnorm = sigmoid(freq_raw)*0.5
             (acota a Nyquist).
    phi    : fase. raw directo (periodica).
"""
import math

import torch
from torch import nn

from gs2d_gabor.render.cuda_gabor import (
    render_gabor_cuda,
    render_gabor_pytorch,
    cuda_disponible,
)


class GaborAudio1D(nn.Module):
    def __init__(
        self,
        n_atomos,
        n_samples,
        sr,
        device,
        sigma_inicial_samples=None,
        f_min_hz=40.0,
        f_max_hz=None,
        k_sigma=4.0,
        semilla=42,
        sigma_min_samples=2.0,
        sigma_max_frac=0.1,
    ):
        super().__init__()

        self.N = int(n_atomos)
        self.T = int(n_samples)
        self.sr = int(sr)
        self.device = device
        self.k_sigma = float(k_sigma)

        g = torch.Generator(device="cpu").manual_seed(int(semilla))

        # Limites de sigma en samples.
        # min: un par de samples; max: una fraccion de la senal completa.
        sigma_min_cfg = max(1.0, float(sigma_min_samples))
        sigma_max_cfg = max(4.0, self.T * float(sigma_max_frac))

        self.log_sigma_min = math.log(sigma_min_cfg)
        self.log_sigma_max = math.log(sigma_max_cfg)

        if sigma_inicial_samples is None:
            # Espaciado promedio entre atomos: T / N. La envolvente inicial
            # cubre aproximadamente ese hueco.
            sigma_inicial_samples = max(4.0, self.T / max(1, self.N))
        sigma0 = float(min(max(sigma_inicial_samples, math.exp(self.log_sigma_min)),
                           math.exp(self.log_sigma_max)))

        # mu_t: posiciones uniformes en el tiempo, con pequeno jitter.
        base = torch.linspace(0, self.T - 1, self.N)
        jitter = (torch.rand(self.N, generator=g) - 0.5) * (self.T / max(1, self.N))
        mu_init = (base + jitter).clamp(0, self.T - 1)
        self.mu_t = nn.Parameter(mu_init.to(device))

        # log_sigma: constante inicial.
        self.log_sigma = nn.Parameter(
            torch.full((self.N,), math.log(sigma0), device=device)
        )

        # amp: pequeno aleatorio con signo.
        amp_init = (torch.rand(self.N, generator=g) - 0.5) * 0.02
        self.amp = nn.Parameter(amp_init.to(device))

        # freq_raw: tal que fnorm cubra [f_min, f_max] en ciclos/sample.
        if f_max_hz is None:
            f_max_hz = 0.45 * self.sr   # un poco por debajo de Nyquist
        fnorm_min = max(1e-4, float(f_min_hz) / self.sr)
        fnorm_max = min(0.499, float(f_max_hz) / self.sr)
        # Distribuye log-espaciado (mejor para audio: mas resolucion en graves).
        fnorm_init = torch.logspace(
            math.log10(fnorm_min), math.log10(fnorm_max), self.N
        )[torch.randperm(self.N, generator=g)]
        # invertir sigmoid*0.5: fnorm = sigmoid(raw)*0.5  ->  raw = logit(fnorm/0.5)
        ratio = (fnorm_init / 0.5).clamp(1e-4, 1 - 1e-4)
        freq_raw_init = torch.log(ratio / (1.0 - ratio))
        self.freq_raw = nn.Parameter(freq_raw_init.to(device))

        # phi: fase uniforme.
        phi_init = torch.rand(self.N, generator=g) * (2.0 * math.pi)
        self.phi = nn.Parameter(phi_init.to(device))

        self._forzar_pytorch = False

    # ------------------------------------------------------------------
    def activar_parametros(self):
        """Convierte los params raw a los activados que espera el render."""
        mu = self.mu_t.clamp(0.0, float(self.T - 1))
        sigma = torch.exp(self.log_sigma.clamp(self.log_sigma_min, self.log_sigma_max))
        amp = self.amp
        fnorm = torch.sigmoid(self.freq_raw) * 0.5
        phi = self.phi
        return mu, sigma, amp, fnorm, phi

    def render(self):
        """Devuelve x_hat [T] reconstruido."""
        mu, sigma, amp, fnorm, phi = self.activar_parametros()

        usar_cuda = (not self._forzar_pytorch) and self.mu_t.is_cuda and cuda_disponible()

        if usar_cuda:
            return render_gabor_cuda(mu, sigma, amp, fnorm, phi, self.T, self.k_sigma)

        # Fallback. Chunk para acotar memoria en audio largo.
        chunk = None if self.T <= 20000 else 20000
        return render_gabor_pytorch(
            mu, sigma, amp, fnorm, phi, self.T, self.k_sigma, chunk_t=chunk
        )

    def numero_atomos(self):
        return self.N

    def frecuencias_hz(self):
        """fnorm activado -> Hz, para inspeccion."""
        with torch.no_grad():
            fnorm = torch.sigmoid(self.freq_raw) * 0.5
            return (fnorm * self.sr).detach().cpu()

    def state_dict_coefs(self):
        return {
            "mu_t": self.mu_t.detach().cpu(),
            "log_sigma": self.log_sigma.detach().cpu(),
            "amp": self.amp.detach().cpu(),
            "freq_raw": self.freq_raw.detach().cpu(),
            "phi": self.phi.detach().cpu(),
            "N": self.N,
            "T": self.T,
            "sr": self.sr,
            "k_sigma": self.k_sigma,
        }


# ======================================================================
# optimizador
# ======================================================================

def construir_optimizador_gabor(modelo, lrs=None):
    """
    Adam con param_groups por tipo de parametro.

    Convencion de claves en lrs:
        mu_t, log_sigma, amp, freq_raw, phi
    """
    lrs = lrs or {}

    defaults = {
        "mu_t":     1.0,      # samples: lr alto porque la escala es grande
        "log_sigma": 5e-3,
        "amp":      5e-3,
        "freq_raw": 5e-3,
        "phi":      5e-2,
    }

    def lr(clave):
        return float(lrs.get(clave, defaults[clave]))

    grupos = [
        {"params": [modelo.mu_t],     "lr": lr("mu_t"),     "name": "mu_t"},
        {"params": [modelo.log_sigma], "lr": lr("log_sigma"), "name": "log_sigma"},
        {"params": [modelo.amp],      "lr": lr("amp"),      "name": "amp"},
        {"params": [modelo.freq_raw], "lr": lr("freq_raw"), "name": "freq_raw"},
        {"params": [modelo.phi],      "lr": lr("phi"),      "name": "phi"},
    ]

    try:
        return torch.optim.Adam(grupos, fused=True)
    except Exception:
        pass
    try:
        return torch.optim.Adam(grupos, foreach=True)
    except Exception:
        pass
    return torch.optim.Adam(grupos)
