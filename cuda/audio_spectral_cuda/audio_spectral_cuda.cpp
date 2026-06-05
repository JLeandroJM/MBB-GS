#include <torch/extension.h>
#include <vector>




torch::Tensor audio_spectral_render_cuda(
    torch::Tensor mu,
    torch::Tensor sigma,
    torch::Tensor amp,
    int F
);

std::vector<torch::Tensor> audio_spectral_render_backward_cuda(
    torch::Tensor mu,
    torch::Tensor sigma,
    torch::Tensor amp,
    torch::Tensor pred,
    torch::Tensor grad_pred
);



std::vector<torch::Tensor> audio_spectral_hard_loss_cuda(
    torch::Tensor mu,
    torch::Tensor sigma,
    torch::Tensor amp,
    torch::Tensor target,
    double lambda_mse,
    double lambda_hard,
    double hard_clip
);

std::vector<torch::Tensor> audio_spectral_loss_cuda(
    torch::Tensor mu,
    torch::Tensor sigma,
    torch::Tensor amp,
    torch::Tensor target,
    torch::Tensor weight,
    double lambda_mse,
    double inv_l1_norm,
    double inv_mse_norm
);


#define CHECK_CUDA(x) TORCH_CHECK(x.is_cuda(), #x " debe estar en CUDA")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " debe ser contiguo")
#define CHECK_FLOAT(x) TORCH_CHECK(x.scalar_type() == torch::kFloat32, #x " debe ser float32")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x); CHECK_FLOAT(x)


std::vector<torch::Tensor> loss_forward(
    torch::Tensor mu,
    torch::Tensor sigma,
    torch::Tensor amp,
    torch::Tensor target,
    torch::Tensor weight,
    double lambda_mse,
    double inv_l1_norm,
    double inv_mse_norm
) {
    CHECK_INPUT(mu);
    CHECK_INPUT(sigma);
    CHECK_INPUT(amp);
    CHECK_INPUT(target);
    CHECK_INPUT(weight);

    TORCH_CHECK(mu.dim() == 2, "mu debe tener shape [B,N]");
    TORCH_CHECK(sigma.dim() == 2, "sigma debe tener shape [B,N]");
    TORCH_CHECK(amp.dim() == 2, "amp debe tener shape [B,N]");
    TORCH_CHECK(target.dim() == 2, "target debe tener shape [B,F]");
    TORCH_CHECK(weight.dim() == 2, "weight debe tener shape [B,F]");

    TORCH_CHECK(mu.sizes() == sigma.sizes(), "mu y sigma deben tener la misma shape");
    TORCH_CHECK(mu.sizes() == amp.sizes(), "mu y amp deben tener la misma shape");
    TORCH_CHECK(target.sizes() == weight.sizes(), "target y weight deben tener la misma shape");
    TORCH_CHECK(mu.size(0) == target.size(0), "B debe coincidir entre params y target");

    return audio_spectral_loss_cuda(
        mu,
        sigma,
        amp,
        target,
        weight,
        lambda_mse,
        inv_l1_norm,
        inv_mse_norm
    );
}



torch::Tensor render_forward(
    torch::Tensor mu,
    torch::Tensor sigma,
    torch::Tensor amp,
    int64_t F
) {
    CHECK_INPUT(mu);
    CHECK_INPUT(sigma);
    CHECK_INPUT(amp);

    TORCH_CHECK(mu.dim() == 2, "mu debe tener shape [B,N]");
    TORCH_CHECK(sigma.dim() == 2, "sigma debe tener shape [B,N]");
    TORCH_CHECK(amp.dim() == 2, "amp debe tener shape [B,N]");

    TORCH_CHECK(mu.sizes() == sigma.sizes(), "mu y sigma deben tener la misma shape");
    TORCH_CHECK(mu.sizes() == amp.sizes(), "mu y amp deben tener la misma shape");
    TORCH_CHECK(F > 0, "F debe ser > 0");

    return audio_spectral_render_cuda(
        mu,
        sigma,
        amp,
        (int)F
    );
}


std::vector<torch::Tensor> render_backward(
    torch::Tensor mu,
    torch::Tensor sigma,
    torch::Tensor amp,
    torch::Tensor pred,
    torch::Tensor grad_pred
) {
    CHECK_INPUT(mu);
    CHECK_INPUT(sigma);
    CHECK_INPUT(amp);
    CHECK_INPUT(pred);
    CHECK_INPUT(grad_pred);

    TORCH_CHECK(mu.dim() == 2, "mu debe tener shape [B,N]");
    TORCH_CHECK(pred.dim() == 2, "pred debe tener shape [B,F]");
    TORCH_CHECK(grad_pred.dim() == 2, "grad_pred debe tener shape [B,F]");

    TORCH_CHECK(mu.sizes() == sigma.sizes(), "mu y sigma deben tener la misma shape");
    TORCH_CHECK(mu.sizes() == amp.sizes(), "mu y amp deben tener la misma shape");
    TORCH_CHECK(pred.sizes() == grad_pred.sizes(), "pred y grad_pred deben tener la misma shape");
    TORCH_CHECK(mu.size(0) == pred.size(0), "B debe coincidir");

    return audio_spectral_render_backward_cuda(
        mu,
        sigma,
        amp,
        pred,
        grad_pred
    );
}



std::vector<torch::Tensor> hard_loss_forward(
    torch::Tensor mu,
    torch::Tensor sigma,
    torch::Tensor amp,
    torch::Tensor target,
    double lambda_mse,
    double lambda_hard,
    double hard_clip
) {
    CHECK_INPUT(mu);
    CHECK_INPUT(sigma);
    CHECK_INPUT(amp);
    CHECK_INPUT(target);

    TORCH_CHECK(mu.dim() == 2, "mu debe tener shape [B,N]");
    TORCH_CHECK(sigma.dim() == 2, "sigma debe tener shape [B,N]");
    TORCH_CHECK(amp.dim() == 2, "amp debe tener shape [B,N]");
    TORCH_CHECK(target.dim() == 2, "target debe tener shape [B,F]");

    TORCH_CHECK(mu.sizes() == sigma.sizes(), "mu y sigma deben tener la misma shape");
    TORCH_CHECK(mu.sizes() == amp.sizes(), "mu y amp deben tener la misma shape");
    TORCH_CHECK(mu.size(0) == target.size(0), "B debe coincidir entre params y target");

    return audio_spectral_hard_loss_cuda(
        mu,
        sigma,
        amp,
        target,
        lambda_mse,
        lambda_hard,
        hard_clip
    );
}


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("loss_forward", &loss_forward, "Audio spectral Gaussian fused loss forward CUDA");
    m.def("hard_loss_forward", &hard_loss_forward, "Audio spectral Gaussian hard fused loss CUDA");
    m.def("render_forward", &render_forward, "Audio spectral Gaussian render forward CUDA");
    m.def("render_backward", &render_backward, "Audio spectral Gaussian render backward CUDA");
}
