import os
import sys

import torch


AQUI = os.path.dirname(os.path.abspath(__file__))

RAIZ_REPO = os.path.abspath(
    os.path.join(AQUI, "..", "..", "..")
)

RUTA_AUDIO_CUDA = os.environ.get(
    "RUTA_AUDIO_SPECTRAL_CUDA",
    os.path.join(RAIZ_REPO, "cuda", "audio_spectral_cuda")
)

print(f"[audio_spectral_cuda] buscando extension en: {RUTA_AUDIO_CUDA}", flush=True)

if not os.path.isdir(RUTA_AUDIO_CUDA):
    raise FileNotFoundError(f"No existe la carpeta audio_spectral_cuda: {RUTA_AUDIO_CUDA}")

if RUTA_AUDIO_CUDA not in sys.path:
    sys.path.append(RUTA_AUDIO_CUDA)

try:
    import audio_spectral_cuda
except Exception as e:
    print("[audio_spectral_cuda] sys.path usado:", flush=True)
    for p in sys.path[:8]:
        print("  ", p, flush=True)
    raise e


class AudioSpectralLossCUDA(torch.autograd.Function):
    @staticmethod
    def forward(ctx, mu_f, sigma_f, amp, target, weight, lambda_mse):
        mu_f_c = mu_f.contiguous()
        sigma_f_c = sigma_f.contiguous()
        amp_c = amp.contiguous()
        target_c = target.contiguous()
        weight_c = weight.contiguous()

        if mu_f_c.dtype != torch.float32:
            raise TypeError("mu_f debe ser float32")
        if sigma_f_c.dtype != torch.float32:
            raise TypeError("sigma_f debe ser float32")
        if amp_c.dtype != torch.float32:
            raise TypeError("amp debe ser float32")
        if target_c.dtype != torch.float32:
            raise TypeError("target debe ser float32")
        if weight_c.dtype != torch.float32:
            raise TypeError("weight debe ser float32")

        inv_l1_norm = 1.0 / float(weight_c.sum().clamp_min(1e-8).item())
        inv_mse_norm = 1.0 / float(target_c.numel())

        loss, l1, mse, grad_mu, grad_sigma, grad_amp = audio_spectral_cuda.loss_forward(
            mu_f_c,
            sigma_f_c,
            amp_c,
            target_c,
            weight_c,
            float(lambda_mse),
            float(inv_l1_norm),
            float(inv_mse_norm),
        )

        ctx.save_for_backward(grad_mu, grad_sigma, grad_amp)

        ctx.mark_non_differentiable(l1, mse)

        return loss, l1, mse

    @staticmethod
    def backward(ctx, grad_loss, grad_l1=None, grad_mse=None):
        grad_mu, grad_sigma, grad_amp = ctx.saved_tensors

        if grad_loss is None:
            grad_loss = grad_mu.new_tensor(1.0)

        escala = grad_loss.to(device=grad_mu.device, dtype=grad_mu.dtype)

        return (
            grad_mu * escala,
            grad_sigma * escala,
            grad_amp * escala,
            None,
            None,
            None,
        )


def loss_audio_espectral_cuda(mu_f, sigma_f, amp, target, weight=None, lambda_mse=0.25):
    """
    Fused CUDA loss para audio espectral.

    Inputs:
        mu_f    : [B, N]
        sigma_f : [B, N]
        amp     : [B, N]
        target  : [B, F]
        weight  : [B, F] o None

    Return:
        loss, l1, mse
    """
    if weight is None:
        weight = torch.ones_like(target)

    return AudioSpectralLossCUDA.apply(
        mu_f,
        sigma_f,
        amp,
        target,
        weight,
        float(lambda_mse),
    )


@torch.no_grad()
def construir_motion_weight_audio(target_full, indices, lambda_motion=1.5, motion_clip=5.0):
    """
    Crea pesos motion sobre espectrograma:
        motion[t,f] = abs(target[t,f] - target[t-1,f])
        weight = 1 + lambda_motion * normalizar(motion)
    """
    idx = indices.to(device=target_full.device, dtype=torch.long)
    target_batch = target_full[idx]

    prev_idx = torch.clamp(idx - 1, min=0)
    prev = target_full[prev_idx]

    motion = torch.abs(target_batch - prev)

    es_cero = (idx == 0).view(-1, 1)
    motion = torch.where(es_cero, torch.zeros_like(motion), motion)

    mean = motion.mean()
    if mean <= 1e-8:
        motion_norm = torch.zeros_like(motion)
    else:
        motion_norm = motion / mean.clamp_min(1e-8)

    motion_norm = motion_norm.clamp(0.0, float(motion_clip))

    return (1.0 + float(lambda_motion) * motion_norm).detach()


class AudioSpectralRenderCUDA(torch.autograd.Function):
    @staticmethod
    def forward(ctx, mu_f, sigma_f, amp, n_bins):
        mu_f_c = mu_f.contiguous()
        sigma_f_c = sigma_f.contiguous()
        amp_c = amp.contiguous()

        if mu_f_c.dtype != torch.float32:
            raise TypeError("mu_f debe ser float32")
        if sigma_f_c.dtype != torch.float32:
            raise TypeError("sigma_f debe ser float32")
        if amp_c.dtype != torch.float32:
            raise TypeError("amp debe ser float32")

        if not hasattr(audio_spectral_cuda, "render_forward"):
            raise RuntimeError(
                "La extension audio_spectral_cuda no tiene render_forward. "
                "Recompila cuda/audio_spectral_cuda con setup.py build_ext --inplace."
            )

        pred = audio_spectral_cuda.render_forward(
            mu_f_c,
            sigma_f_c,
            amp_c,
            int(n_bins),
        )

        ctx.save_for_backward(mu_f_c, sigma_f_c, amp_c, pred)
        return pred

    @staticmethod
    def backward(ctx, grad_pred):
        mu_f, sigma_f, amp, pred = ctx.saved_tensors

        if not hasattr(audio_spectral_cuda, "render_backward"):
            raise RuntimeError(
                "La extension audio_spectral_cuda no tiene render_backward. "
                "Recompila cuda/audio_spectral_cuda con setup.py build_ext --inplace."
            )

        grad_mu, grad_sigma, grad_amp = audio_spectral_cuda.render_backward(
            mu_f,
            sigma_f,
            amp,
            pred,
            grad_pred.contiguous(),
        )

        return grad_mu, grad_sigma, grad_amp, None


def render_audio_espectral_cuda(mu_f, sigma_f, amp, n_bins):
    """
    Renderer CUDA general para audio.

    Inputs:
        mu_f    : [B, N]
        sigma_f : [B, N]
        amp     : [B, N]
        n_bins  : F

    Output:
        pred    : [B, F]

    Este renderer permite usar CUDA para la parte matematica pesada
    y luego calcular cualquier loss en PyTorch.
    """
    return AudioSpectralRenderCUDA.apply(mu_f, sigma_f, amp, int(n_bins))


class AudioSpectralHardLossCUDA(torch.autograd.Function):
    @staticmethod
    def forward(ctx, mu_f, sigma_f, amp, target, lambda_mse, lambda_hard, hard_clip):
        mu_f_c = mu_f.contiguous()
        sigma_f_c = sigma_f.contiguous()
        amp_c = amp.contiguous()
        target_c = target.contiguous()

        if not hasattr(audio_spectral_cuda, "hard_loss_forward"):
            raise RuntimeError(
                "La extension audio_spectral_cuda no tiene hard_loss_forward. "
                "Recompila cuda/audio_spectral_cuda con setup.py build_ext --inplace."
            )

        loss, l1, mse, grad_mu, grad_sigma, grad_amp = audio_spectral_cuda.hard_loss_forward(
            mu_f_c,
            sigma_f_c,
            amp_c,
            target_c,
            float(lambda_mse),
            float(lambda_hard),
            float(hard_clip),
        )

        ctx.save_for_backward(grad_mu, grad_sigma, grad_amp)
        ctx.mark_non_differentiable(l1, mse)

        return loss, l1, mse

    @staticmethod
    def backward(ctx, grad_loss, grad_l1=None, grad_mse=None):
        grad_mu, grad_sigma, grad_amp = ctx.saved_tensors

        if grad_loss is None:
            grad_loss = grad_mu.new_tensor(1.0)

        escala = grad_loss.to(device=grad_mu.device, dtype=grad_mu.dtype)

        return (
            grad_mu * escala,
            grad_sigma * escala,
            grad_amp * escala,
            None,
            None,
            None,
            None,
        )


def loss_audio_hard_cuda(mu_f, sigma_f, amp, target, lambda_mse=0.25, lambda_hard=1.0, hard_clip=3.0):
    """
    Fused CUDA hard loss para audio espectral.

    Inputs:
        mu_f    : [B, N]
        sigma_f : [B, N]
        amp     : [B, N]
        target  : [B, F]

    Return:
        loss, l1, mse
    """
    return AudioSpectralHardLossCUDA.apply(
        mu_f,
        sigma_f,
        amp,
        target,
        float(lambda_mse),
        float(lambda_hard),
        float(hard_clip),
    )

