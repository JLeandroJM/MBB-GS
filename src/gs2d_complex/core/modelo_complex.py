import math
import torch
import torch.nn as nn
import torch.nn.functional as F


def construir_base_chebyshev(n_tiempos, grado, device):
    t = torch.linspace(-1.0, 1.0, n_tiempos, device=device)

    bases = [torch.ones_like(t)]

    if grado >= 1:
        bases.append(t)

    for k in range(2, grado + 1):
        bases.append(2.0 * t * bases[-1] - bases[-2])

    return torch.stack(bases, dim=1)


class GaussianasComplejasTemporalesCheb(nn.Module):
    def __init__(
        self,
        n_gaussianas,
        n_bins,
        n_tiempos,
        grados,
        device,
    ):
        super().__init__()

        self.n_gaussianas = int(n_gaussianas)
        self.n_bins = int(n_bins)
        self.n_tiempos = int(n_tiempos)
        self.grados = dict(grados)
        self.device = device

        self.mu_f_a0 = nn.Parameter(torch.empty(n_gaussianas, 1, device=device))
        self.mu_f_high = nn.Parameter(torch.empty(n_gaussianas, self.grados["mu_f"], device=device))

        self.sigma_f_a0 = nn.Parameter(torch.empty(n_gaussianas, 1, device=device))
        self.sigma_f_high = nn.Parameter(torch.empty(n_gaussianas, self.grados["sigma_f"], device=device))

        self.amp_real_a0 = nn.Parameter(torch.empty(n_gaussianas, 1, device=device))
        self.amp_real_high = nn.Parameter(torch.empty(n_gaussianas, self.grados["amp_real"], device=device))

        self.amp_imag_a0 = nn.Parameter(torch.empty(n_gaussianas, 1, device=device))
        self.amp_imag_high = nn.Parameter(torch.empty(n_gaussianas, self.grados["amp_imag"], device=device))

        self.reset_parameters()

    def reset_parameters(self):
        with torch.no_grad():
            self.mu_f_a0.normal_(0.0, 1.0)
            self.mu_f_high.normal_(0.0, 0.05)

            self.sigma_f_a0.fill_(math.log(8.0))
            self.sigma_f_high.normal_(0.0, 0.03)

            self.amp_real_a0.normal_(0.0, 0.02)
            self.amp_real_high.normal_(0.0, 0.02)

            self.amp_imag_a0.normal_(0.0, 0.02)
            self.amp_imag_high.normal_(0.0, 0.02)

    def _coefs(self, a0, high):
        return torch.cat([a0, high], dim=1)

    def parametros_temporales(self):
        return {
            "mu_f": (self.mu_f_a0, self.mu_f_high, self.grados["mu_f"]),
            "sigma_f": (self.sigma_f_a0, self.sigma_f_high, self.grados["sigma_f"]),
            "amp_real": (self.amp_real_a0, self.amp_real_high, self.grados["amp_real"]),
            "amp_imag": (self.amp_imag_a0, self.amp_imag_high, self.grados["amp_imag"]),
        }

    def evaluar_indices(self, indices, bases):
        if not torch.is_tensor(indices):
            indices = torch.tensor(indices, device=self.device, dtype=torch.long)
        else:
            indices = indices.to(device=self.device, dtype=torch.long)

        raw = {}

        for nombre, (a0, high, grado) in self.parametros_temporales().items():
            B = bases[grado][indices]
            C = self._coefs(a0, high)
            raw[nombre] = B @ C.T

        mu_f = torch.sigmoid(raw["mu_f"]) * float(self.n_bins - 1)

        log_sigma_min = math.log(0.5)
        log_sigma_max = math.log(max(1.0, self.n_bins / 2.0))
        sigma_f = torch.exp(raw["sigma_f"].clamp(log_sigma_min, log_sigma_max))

        # Real/imag son firmados. No softplus.
        amp_real = raw["amp_real"]
        amp_imag = raw["amp_imag"]

        return {
            "mu_f": mu_f,
            "sigma_f": sigma_f,
            "amp_real": amp_real,
            "amp_imag": amp_imag,
        }

    def render_indices(self, indices, bases):
        p = self.evaluar_indices(indices, bases)

        f = torch.arange(self.n_bins, device=self.device, dtype=torch.float32)
        f = f.view(1, 1, self.n_bins)

        mu = p["mu_f"].unsqueeze(-1)
        sigma = p["sigma_f"].unsqueeze(-1).clamp_min(1e-6)

        g = torch.exp(-0.5 * ((f - mu) / sigma) ** 2)

        real = torch.sum(p["amp_real"].unsqueeze(-1) * g, dim=1)
        imag = torch.sum(p["amp_imag"].unsqueeze(-1) * g, dim=1)

        return real, imag


def smoothness_complex(modelo):
    loss = 0.0

    for _, high, _ in modelo.parametros_temporales().values():
        if high.numel() == 0:
            continue

        k = torch.arange(1, high.shape[1] + 1, device=high.device, dtype=high.dtype)
        peso = k.pow(2).view(1, -1)
        loss = loss + torch.mean((high * peso) ** 2)

    return loss


def construir_bases_por_grado(n_tiempos, grados, device):
    unicos = sorted(set(int(g) for g in grados.values()))
    return {
        g: construir_base_chebyshev(n_tiempos, g, device)
        for g in unicos
    }
