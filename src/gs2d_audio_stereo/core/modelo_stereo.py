import math
import torch
import torch.nn as nn

from gs2d_video.core.bases import construir_matriz_chebyshev


def construir_bases(n_frames, grados, device):
    grados_distintos = sorted(set(int(v) for v in grados.values()))
    return {
        g: construir_matriz_chebyshev(
            n_frames=n_frames,
            grado_max=g,
            device=device,
            dtype=torch.float32,
        )
        for g in grados_distintos
    }


class GaussianasStereoTemporalesCheb(nn.Module):
    def __init__(self, n_gaussianas, n_frames, n_bins, grados, device, semilla=42):
        super().__init__()

        torch.manual_seed(int(semilla))

        self.n_gaussianas = int(n_gaussianas)
        self.n_frames = int(n_frames)
        self.n_bins = int(n_bins)
        self.grados = dict(grados)
        self.device = device

        N = self.n_gaussianas

        self.mu_f_a0 = nn.Parameter(torch.randn(N, 1, device=device) * 0.5)
        self.mu_f_high = nn.Parameter(torch.randn(N, self.grados["mu_f"], device=device) * 0.02)

        self.sigma_f_a0 = nn.Parameter(torch.full((N, 1), math.log(6.0), device=device))
        self.sigma_f_high = nn.Parameter(torch.randn(N, self.grados["sigma_f"], device=device) * 0.01)

        amp_init = -6.0

        self.amp_l_a0 = nn.Parameter(torch.full((N, 1), amp_init, device=device))
        self.amp_l_high = nn.Parameter(torch.randn(N, self.grados["amp_l"], device=device) * 0.01)

        self.amp_r_a0 = nn.Parameter(torch.full((N, 1), amp_init, device=device))
        self.amp_r_high = nn.Parameter(torch.randn(N, self.grados["amp_r"], device=device) * 0.01)

    def _eval(self, a0, high, grado, idx, bases):
        C = torch.cat([a0, high], dim=1)
        B = bases[grado][idx]
        return B @ C.T

    def evaluar_indices(self, idx, bases):
        if not torch.is_tensor(idx):
            idx = torch.tensor(idx, dtype=torch.long, device=self.device)
        else:
            idx = idx.to(self.device, dtype=torch.long)

        raw_mu = self._eval(self.mu_f_a0, self.mu_f_high, self.grados["mu_f"], idx, bases)
        raw_sigma = self._eval(self.sigma_f_a0, self.sigma_f_high, self.grados["sigma_f"], idx, bases)
        raw_amp_l = self._eval(self.amp_l_a0, self.amp_l_high, self.grados["amp_l"], idx, bases)
        raw_amp_r = self._eval(self.amp_r_a0, self.amp_r_high, self.grados["amp_r"], idx, bases)

        mu_f = torch.sigmoid(raw_mu) * float(self.n_bins - 1)
        sigma_f = torch.exp(raw_sigma).clamp(0.5, self.n_bins / 2)

        amp_l = torch.nn.functional.softplus(raw_amp_l)
        amp_r = torch.nn.functional.softplus(raw_amp_r)

        return {
            "mu_f": mu_f,
            "sigma_f": sigma_f,
            "amp_l": amp_l,
            "amp_r": amp_r,
        }

    def render_indices(self, idx, bases):
        p = self.evaluar_indices(idx, bases)

        f = torch.arange(self.n_bins, device=self.device, dtype=torch.float32)
        f = f.view(1, 1, self.n_bins)

        mu = p["mu_f"].unsqueeze(-1)
        sigma = p["sigma_f"].unsqueeze(-1).clamp_min(1e-6)

        g = torch.exp(-0.5 * ((f - mu) / sigma) ** 2)

        density_l = torch.sum(p["amp_l"].unsqueeze(-1) * g, dim=1)
        density_r = torch.sum(p["amp_r"].unsqueeze(-1) * g, dim=1)

        pred_l = 1.0 - torch.exp(-density_l)
        pred_r = 1.0 - torch.exp(-density_r)

        return pred_l, pred_r


def loss_smoothness_stereo(modelo, pesos):
    loss = 0.0

    def penalizar(high, peso):
        if peso == 0.0:
            return 0.0
        if high.numel() == 0:
            return 0.0
        k = torch.arange(1, high.shape[1] + 1, device=high.device, dtype=high.dtype)
        k = k.view(1, -1)
        return float(peso) * torch.mean((high * (k ** 2)) ** 2)

    loss = loss + penalizar(modelo.mu_f_high, pesos.get("mu_f", 0.0))
    loss = loss + penalizar(modelo.sigma_f_high, pesos.get("sigma_f", 0.0))
    loss = loss + penalizar(modelo.amp_l_high, pesos.get("amp_l", 0.0))
    loss = loss + penalizar(modelo.amp_r_high, pesos.get("amp_r", 0.0))

    return loss

def construir_optimizador_stereo(modelo, lrs=None):
    if lrs is None:
        lrs = {}

    return torch.optim.Adam([
        {"params": [modelo.mu_f_a0], "lr": float(lrs.get("mu_f_a0", 0.005))},
        {"params": [modelo.mu_f_high], "lr": float(lrs.get("mu_f_high", 0.006))},

        {"params": [modelo.sigma_f_a0], "lr": float(lrs.get("sigma_f_a0", 0.004))},
        {"params": [modelo.sigma_f_high], "lr": float(lrs.get("sigma_f_high", 0.001))},

        {"params": [modelo.amp_l_a0, modelo.amp_r_a0], "lr": float(lrs.get("amp_a0", 0.03))},
        {"params": [modelo.amp_l_high, modelo.amp_r_high], "lr": float(lrs.get("amp_high", 0.01))},
    ])
