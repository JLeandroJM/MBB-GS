import math
import torch
import torch.nn as nn
import torch.nn.functional as F

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


class GaussianasStereoHybridCheb(nn.Module):
    def __init__(self, n_shared, n_private_l, n_private_r, n_frames, n_bins, grados, device, semilla=42):
        super().__init__()
        torch.manual_seed(int(semilla))

        self.n_shared = int(n_shared)
        self.n_private_l = int(n_private_l)
        self.n_private_r = int(n_private_r)
        self.n_frames = int(n_frames)
        self.n_bins = int(n_bins)
        self.grados = dict(grados)
        self.device = device

        self._crear_shared()
        self._crear_private_l()
        self._crear_private_r()

    def _crear_param(self, n, grado, init_a0=0.0, std_high=0.01):
        a0 = nn.Parameter(torch.full((n, 1), float(init_a0), device=self.device))
        high = nn.Parameter(torch.randn(n, grado, device=self.device) * float(std_high))
        return a0, high

    def _crear_shared(self):
        Ns = self.n_shared
        self.mu_s_a0, self.mu_s_high = self._crear_param(Ns, self.grados["mu_shared"], 0.0, 0.02)
        self.sigma_s_a0, self.sigma_s_high = self._crear_param(Ns, self.grados["sigma_shared"], math.log(3.0), 0.01)
        self.amp_s_l_a0, self.amp_s_l_high = self._crear_param(Ns, self.grados["amp_shared_l"], -7.0, 0.01)
        self.amp_s_r_a0, self.amp_s_r_high = self._crear_param(Ns, self.grados["amp_shared_r"], -7.0, 0.01)

    def _crear_private_l(self):
        N = self.n_private_l
        self.mu_l_a0, self.mu_l_high = self._crear_param(N, self.grados["mu_private_l"], 0.0, 0.02)
        self.sigma_l_a0, self.sigma_l_high = self._crear_param(N, self.grados["sigma_private_l"], math.log(6.0), 0.01)
        self.amp_l_a0, self.amp_l_high = self._crear_param(N, self.grados["amp_private_l"], -7.0, 0.01)

    def _crear_private_r(self):
        N = self.n_private_r
        self.mu_r_a0, self.mu_r_high = self._crear_param(N, self.grados["mu_private_r"], 0.0, 0.02)
        self.sigma_r_a0, self.sigma_r_high = self._crear_param(N, self.grados["sigma_private_r"], math.log(6.0), 0.01)
        self.amp_r_a0, self.amp_r_high = self._crear_param(N, self.grados["amp_private_r"], -7.0, 0.01)

    def _eval(self, a0, high, grado, idx, bases):
        C = torch.cat([a0, high], dim=1)
        B = bases[grado][idx]
        return B @ C.T

    def _params_group(self, prefix, n, idx, bases, grado_mu, grado_sigma, amp_a0, amp_high, grado_amp, mu_a0, mu_high, sigma_a0, sigma_high):
        raw_mu = self._eval(mu_a0, mu_high, grado_mu, idx, bases)
        raw_sigma = self._eval(sigma_a0, sigma_high, grado_sigma, idx, bases)
        raw_amp = self._eval(amp_a0, amp_high, grado_amp, idx, bases)

        mu = torch.sigmoid(raw_mu) * float(self.n_bins - 1)
        sigma = torch.exp(raw_sigma).clamp(0.5, self.n_bins / 2)
        amp = F.softplus(raw_amp)

        return mu, sigma, amp

    def _render_density(self, mu, sigma, amp):
        if mu.shape[1] == 0:
            B = mu.shape[0]
            return torch.zeros((B, self.n_bins), device=self.device)

        f = torch.arange(self.n_bins, device=self.device, dtype=torch.float32)
        f = f.view(1, 1, self.n_bins)

        g = torch.exp(-0.5 * ((f - mu.unsqueeze(-1)) / sigma.unsqueeze(-1).clamp_min(1e-6)) ** 2)
        return torch.sum(amp.unsqueeze(-1) * g, dim=1)

    def render_indices(self, idx, bases):
        if not torch.is_tensor(idx):
            idx = torch.tensor(idx, dtype=torch.long, device=self.device)
        else:
            idx = idx.to(self.device, dtype=torch.long)

        mu_s, sig_s, amp_s_l = self._params_group(
            "shared_l", self.n_shared, idx, bases,
            self.grados["mu_shared"], self.grados["sigma_shared"],
            self.amp_s_l_a0, self.amp_s_l_high, self.grados["amp_shared_l"],
            self.mu_s_a0, self.mu_s_high, self.sigma_s_a0, self.sigma_s_high,
        )
        _, _, amp_s_r = self._params_group(
            "shared_r", self.n_shared, idx, bases,
            self.grados["mu_shared"], self.grados["sigma_shared"],
            self.amp_s_r_a0, self.amp_s_r_high, self.grados["amp_shared_r"],
            self.mu_s_a0, self.mu_s_high, self.sigma_s_a0, self.sigma_s_high,
        )

        mu_l, sig_l, amp_l = self._params_group(
            "private_l", self.n_private_l, idx, bases,
            self.grados["mu_private_l"], self.grados["sigma_private_l"],
            self.amp_l_a0, self.amp_l_high, self.grados["amp_private_l"],
            self.mu_l_a0, self.mu_l_high, self.sigma_l_a0, self.sigma_l_high,
        )

        mu_r, sig_r, amp_r = self._params_group(
            "private_r", self.n_private_r, idx, bases,
            self.grados["mu_private_r"], self.grados["sigma_private_r"],
            self.amp_r_a0, self.amp_r_high, self.grados["amp_private_r"],
            self.mu_r_a0, self.mu_r_high, self.sigma_r_a0, self.sigma_r_high,
        )

        density_l = self._render_density(mu_s, sig_s, amp_s_l) + self._render_density(mu_l, sig_l, amp_l)
        density_r = self._render_density(mu_s, sig_s, amp_s_r) + self._render_density(mu_r, sig_r, amp_r)

        pred_l = 1.0 - torch.exp(-density_l)
        pred_r = 1.0 - torch.exp(-density_r)

        return pred_l, pred_r


def construir_optimizador_hybrid(modelo, lrs=None):
    if lrs is None:
        lrs = {}

    return torch.optim.Adam([
        {"params": [modelo.mu_s_a0, modelo.mu_l_a0, modelo.mu_r_a0], "lr": float(lrs.get("mu_a0", 0.005))},
        {"params": [modelo.mu_s_high, modelo.mu_l_high, modelo.mu_r_high], "lr": float(lrs.get("mu_high", 0.006))},

        {"params": [modelo.sigma_s_a0, modelo.sigma_l_a0, modelo.sigma_r_a0], "lr": float(lrs.get("sigma_a0", 0.004))},
        {"params": [modelo.sigma_s_high, modelo.sigma_l_high, modelo.sigma_r_high], "lr": float(lrs.get("sigma_high", 0.001))},

        {
            "params": [
                modelo.amp_s_l_a0, modelo.amp_s_r_a0,
                modelo.amp_l_a0, modelo.amp_r_a0,
            ],
            "lr": float(lrs.get("amp_a0", 0.03)),
        },
        {
            "params": [
                modelo.amp_s_l_high, modelo.amp_s_r_high,
                modelo.amp_l_high, modelo.amp_r_high,
            ],
            "lr": float(lrs.get("amp_high", 0.01)),
        },
    ])


def loss_smoothness_hybrid(modelo, pesos):
    loss = 0.0

    def penalizar(high, peso):
        if float(peso) == 0.0:
            return 0.0
        if high.numel() == 0:
            return 0.0
        k = torch.arange(1, high.shape[1] + 1, device=high.device, dtype=high.dtype).view(1, -1)
        return float(peso) * torch.mean((high * (k ** 2)) ** 2)

    loss = loss + penalizar(modelo.mu_s_high, pesos.get("mu_shared", 0.0))
    loss = loss + penalizar(modelo.mu_l_high, pesos.get("mu_private_l", 0.0))
    loss = loss + penalizar(modelo.mu_r_high, pesos.get("mu_private_r", 0.0))

    loss = loss + penalizar(modelo.sigma_s_high, pesos.get("sigma_shared", 0.5))
    loss = loss + penalizar(modelo.sigma_l_high, pesos.get("sigma_private_l", 0.5))
    loss = loss + penalizar(modelo.sigma_r_high, pesos.get("sigma_private_r", 0.5))

    loss = loss + penalizar(modelo.amp_s_l_high, pesos.get("amp", 0.0))
    loss = loss + penalizar(modelo.amp_s_r_high, pesos.get("amp", 0.0))
    loss = loss + penalizar(modelo.amp_l_high, pesos.get("amp", 0.0))
    loss = loss + penalizar(modelo.amp_r_high, pesos.get("amp", 0.0))

    return loss
