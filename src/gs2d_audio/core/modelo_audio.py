import math

import torch
from torch import nn
import torch.nn.functional as F


class GaussianasEspectralesTemporalesCheb(nn.Module):
    """
    Modelo de audio fase 2.

    Representa un espectrograma S[t, f] como suma de gaussianas 1D
    sobre frecuencia. Los parametros de cada gaussiana dependen del tiempo
    usando coeficientes Chebyshev.

    S(f, t) = sum_i A_i(t) * G(f; mu_i(t), sigma_i(t))
    """

    def __init__(
        self,
        n_gaussianas,
        n_frames,
        n_bins,
        grados,
        device,
        semilla=42,
        sigma_inicial_bins=6.0,
    ):
        super().__init__()

        self.N = int(n_gaussianas)
        self.n_frames = int(n_frames)
        self.n_bins = int(n_bins)
        self.grados = dict(grados)
        self.device = device

        g = torch.Generator(device="cpu").manual_seed(int(semilla))

        # mu_f raw: se activa con sigmoid y se escala a [0, n_bins-1]
        mu01 = torch.rand(self.N, 1, generator=g).clamp(1e-4, 1.0 - 1e-4)
        mu_raw = torch.log(mu01 / (1.0 - mu01))
        self.mu_f_a0 = nn.Parameter(mu_raw.to(device))
        self.mu_f_high = nn.Parameter(torch.zeros(self.N, self.grados["mu_f"], device=device))

        # sigma_f raw: se interpreta como log(sigma)
        sigma0 = math.log(float(sigma_inicial_bins))
        self.sigma_f_a0 = nn.Parameter(torch.full((self.N, 1), sigma0, device=device))
        self.sigma_f_high = nn.Parameter(torch.zeros(self.N, self.grados["sigma_f"], device=device))

        # amp raw: se activa con softplus para que sea positiva
        amp0 = torch.full((self.N, 1), -6.0)
        amp0 = amp0 + 0.05 * torch.randn(self.N, 1, generator=g)
        self.amp_a0 = nn.Parameter(amp0.to(device))
        self.amp_high = nn.Parameter(torch.zeros(self.N, self.grados["amp"], device=device))

        self.smooth_factors = {}
        for nombre, grado in self.grados.items():
            if grado > 0:
                k_idx = torch.arange(1, grado + 1, device=device, dtype=torch.float32)
                self.smooth_factors[nombre] = k_idx ** 2

    def parametros_temporales(self):
        return {
            "mu_f": (self.mu_f_a0, self.mu_f_high, self.grados["mu_f"], 1),
            "sigma_f": (self.sigma_f_a0, self.sigma_f_high, self.grados["sigma_f"], 1),
            "amp": (self.amp_a0, self.amp_high, self.grados["amp"], 1),
        }

    def _coefs_completos(self, a0, hi):
        return torch.cat([a0, hi], dim=-1)

    def _evaluar_raw_batch(self, a0, hi, grado, matrices_base):
        B = matrices_base[grado]
        coefs = self._coefs_completos(a0, hi)
        return B @ coefs.T

    def evaluar_batch_completo(self, matrices_base):
        raw = {}

        for nombre, (a0, hi, grado, _) in self.parametros_temporales().items():
            raw[nombre + "_raw"] = self._evaluar_raw_batch(a0, hi, grado, matrices_base)

        mu_f = torch.sigmoid(raw["mu_f_raw"]) * float(self.n_bins - 1)

        log_sigma_min = math.log(0.5)
        log_sigma_max = math.log(max(1.0, self.n_bins / 2.0))
        sigma_f = torch.exp(raw["sigma_f_raw"].clamp(log_sigma_min, log_sigma_max))

        amp = F.softplus(raw["amp_raw"])

        return {
            "mu_f": mu_f,
            "sigma_f": sigma_f,
            "amp": amp,
            "mu_f_raw": raw["mu_f_raw"],
            "sigma_f_raw": raw["sigma_f_raw"],
            "amp_raw": raw["amp_raw"],
        }


    def evaluar_indices(self, frame_indices, matrices_base):
        """
        Evalua parametros solo para un subconjunto de tiempos.

        frame_indices:
            tensor/list de indices temporales con shape [B]

        Devuelve parametros con shape [B, N].
        """
        if not torch.is_tensor(frame_indices):
            frame_indices = torch.tensor(frame_indices, device=self.device, dtype=torch.long)
        else:
            frame_indices = frame_indices.to(device=self.device, dtype=torch.long)

        raw = {}

        for nombre, (a0, hi, grado, _) in self.parametros_temporales().items():
            B = matrices_base[grado][frame_indices]  # [B, grado+1]
            coefs = self._coefs_completos(a0, hi)    # [N, grado+1]
            raw[nombre + "_raw"] = B @ coefs.T       # [B, N]

        mu_f = torch.sigmoid(raw["mu_f_raw"]) * float(self.n_bins - 1)

        log_sigma_min = math.log(0.5)
        log_sigma_max = math.log(max(1.0, self.n_bins / 2.0))
        sigma_f = torch.exp(raw["sigma_f_raw"].clamp(log_sigma_min, log_sigma_max))

        amp = F.softplus(raw["amp_raw"])

        return {
            "mu_f": mu_f,
            "sigma_f": sigma_f,
            "amp": amp,
            "mu_f_raw": raw["mu_f_raw"],
            "sigma_f_raw": raw["sigma_f_raw"],
            "amp_raw": raw["amp_raw"],
        }

    def render_indices(self, frame_indices, matrices_base):
        """
        Renderiza solo algunos tiempos del espectrograma.

        Devuelve tensor [B, F].
        """
        p = self.evaluar_indices(frame_indices, matrices_base)

        f = torch.arange(self.n_bins, device=self.device, dtype=torch.float32)
        f = f.view(1, 1, self.n_bins)

        mu = p["mu_f"].unsqueeze(-1)                 # [B, N, 1]
        sigma = p["sigma_f"].unsqueeze(-1).clamp_min(1e-6)
        amp = p["amp"].unsqueeze(-1)

        g = torch.exp(-0.5 * ((f - mu) / sigma) ** 2)
        density = torch.sum(amp * g, dim=1)          # [B, F]

        return 1.0 - torch.exp(-density)


    def render_all(self, matrices_base):
        """
        Devuelve espectrograma reconstruido con shape [T, F], en rango [0, 1].
        """
        p = self.evaluar_batch_completo(matrices_base)

        f = torch.arange(self.n_bins, device=self.device, dtype=torch.float32)
        f = f.view(1, 1, self.n_bins)

        mu = p["mu_f"].unsqueeze(-1)
        sigma = p["sigma_f"].unsqueeze(-1).clamp_min(1e-6)
        amp = p["amp"].unsqueeze(-1)

        g = torch.exp(-0.5 * ((f - mu) / sigma) ** 2)
        density = torch.sum(amp * g, dim=1)

        # Mapea densidad positiva a [0, 1], parecido a acumulacion de opacidad.
        return 1.0 - torch.exp(-density)

    def numero_gausianas(self):
        return self.N

    def state_dict_coefs(self):
        d = {}
        for nombre, (a0, hi, _, _) in self.parametros_temporales().items():
            d[f"{nombre}_a0"] = a0.detach().cpu()
            d[f"{nombre}_high"] = hi.detach().cpu()
        d["grados"] = self.grados
        d["N"] = self.N
        d["n_frames"] = self.n_frames
        d["n_bins"] = self.n_bins
        return d


def loss_smoothness_audio(modelo, pesos_por_param=None):
    total = None
    pesos_por_param = pesos_por_param or {}

    for nombre, (a0, a_high, grado, _) in modelo.parametros_temporales().items():
        peso = float(pesos_por_param.get(nombre, 1.0))

        if peso == 0.0 or grado == 0:
            continue

        factor = modelo.smooth_factors[nombre]

        while factor.dim() < a_high.dim():
            factor = factor.unsqueeze(0)

        contrib = (a_high ** 2 * factor).sum() * peso
        total = contrib if total is None else total + contrib

    if total is None:
        return torch.tensor(0.0, device=next(modelo.parameters()).device)

    return total


def construir_optimizador_audio(modelo, lrs):
    defaults = {
        "mu_f_a0": 1e-2,
        "mu_f_high": 1e-3,
        "sigma_f_a0": 5e-3,
        "sigma_f_high": 5e-4,
        "amp_a0": 2e-2,
        "amp_high": 2e-3,
    }

    def lr(clave):
        return float(lrs.get(clave, defaults[clave]))

    grupos = []

    for nombre in ["mu_f", "sigma_f", "amp"]:
        a0 = getattr(modelo, f"{nombre}_a0")
        hi = getattr(modelo, f"{nombre}_high")

        grupos.append({
            "params": [a0],
            "lr": lr(f"{nombre}_a0"),
            "name": f"{nombre}_a0",
        })
        grupos.append({
            "params": [hi],
            "lr": lr(f"{nombre}_high"),
            "name": f"{nombre}_high",
        })

    try:
        return torch.optim.Adam(grupos, fused=True)
    except Exception:
        pass

    try:
        return torch.optim.Adam(grupos, foreach=True)
    except Exception:
        pass

    return torch.optim.Adam(grupos)
