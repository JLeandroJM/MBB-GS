#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda.h>
#include <cuda_runtime.h>

#include <vector>


static inline int div_up_int(int a, int b) {
    return (a + b - 1) / b;
}


__global__ void density_loss_kernel(
    const float* __restrict__ mu,
    const float* __restrict__ sigma,
    const float* __restrict__ amp,
    const float* __restrict__ target,
    const float* __restrict__ weight,
    float* __restrict__ grad_density,
    float* __restrict__ loss_parts,
    int B,
    int N,
    int F,
    float lambda_mse,
    float inv_l1_norm,
    float inv_mse_norm
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = B * F;

    if (idx >= total) {
        return;
    }

    int b = idx / F;
    int f = idx - b * F;

    float density = 0.0f;

    int base_bn = b * N;

    for (int i = 0; i < N; ++i) {
        float m = mu[base_bn + i];
        float s = fmaxf(sigma[base_bn + i], 1.0e-6f);
        float a = amp[base_bn + i];

        float d = ((float)f - m) / s;
        float g = __expf(-0.5f * d * d);

        density += a * g;
    }

    float exp_neg_density = __expf(-density);
    float pred = 1.0f - exp_neg_density;

    float y = target[idx];
    float w = weight[idx];

    float diff = pred - y;
    float abs_diff = fabsf(diff);

    float l1_contrib = w * abs_diff * inv_l1_norm;
    float mse_contrib = diff * diff * inv_mse_norm;

    float loss_contrib = l1_contrib + lambda_mse * mse_contrib;

    atomicAdd(&loss_parts[0], loss_contrib);
    atomicAdd(&loss_parts[1], l1_contrib);
    atomicAdd(&loss_parts[2], mse_contrib);

    float sign_diff = 0.0f;
    if (diff > 0.0f) {
        sign_diff = 1.0f;
    } else if (diff < 0.0f) {
        sign_diff = -1.0f;
    }

    float grad_pred = w * sign_diff * inv_l1_norm
                    + lambda_mse * 2.0f * diff * inv_mse_norm;

    // pred = 1 - exp(-density)
    // d pred / d density = exp(-density)
    grad_density[idx] = grad_pred * exp_neg_density;
}


__global__ void grad_params_kernel(
    const float* __restrict__ mu,
    const float* __restrict__ sigma,
    const float* __restrict__ amp,
    const float* __restrict__ grad_density,
    float* __restrict__ grad_mu,
    float* __restrict__ grad_sigma,
    float* __restrict__ grad_amp,
    int B,
    int N,
    int F
) {
    int b = blockIdx.x;
    int i = blockIdx.y;
    int tid = threadIdx.x;

    __shared__ float s_mu[256];
    __shared__ float s_sigma[256];
    __shared__ float s_amp[256];

    float local_mu = 0.0f;
    float local_sigma = 0.0f;
    float local_amp = 0.0f;

    int idx_bn = b * N + i;

    float m = mu[idx_bn];
    float s = fmaxf(sigma[idx_bn], 1.0e-6f);
    float a = amp[idx_bn];

    float s2 = s * s;
    float s3 = s2 * s;

    for (int f = tid; f < F; f += blockDim.x) {
        int idx_bf = b * F + f;

        float gd = grad_density[idx_bf];

        float dx = (float)f - m;
        float d = dx / s;
        float g = __expf(-0.5f * d * d);

        local_amp += gd * g;

        // density += amp * g
        // d g / d mu = g * (f - mu) / sigma^2
        local_mu += gd * a * g * dx / s2;

        // d g / d sigma = g * (f - mu)^2 / sigma^3
        local_sigma += gd * a * g * dx * dx / s3;
    }

    s_mu[tid] = local_mu;
    s_sigma[tid] = local_sigma;
    s_amp[tid] = local_amp;

    __syncthreads();

    for (int offset = blockDim.x / 2; offset > 0; offset >>= 1) {
        if (tid < offset) {
            s_mu[tid] += s_mu[tid + offset];
            s_sigma[tid] += s_sigma[tid + offset];
            s_amp[tid] += s_amp[tid + offset];
        }
        __syncthreads();
    }

    if (tid == 0) {
        grad_mu[idx_bn] = s_mu[0];
        grad_sigma[idx_bn] = s_sigma[0];
        grad_amp[idx_bn] = s_amp[0];
    }
}




__global__ void render_kernel(
    const float* __restrict__ mu,
    const float* __restrict__ sigma,
    const float* __restrict__ amp,
    float* __restrict__ pred,
    int B,
    int N,
    int F
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = B * F;

    if (idx >= total) {
        return;
    }

    int b = idx / F;
    int f = idx - b * F;

    float density = 0.0f;
    int base_bn = b * N;

    for (int i = 0; i < N; ++i) {
        float m = mu[base_bn + i];
        float s = fmaxf(sigma[base_bn + i], 1.0e-6f);
        float a = amp[base_bn + i];

        float d = ((float)f - m) / s;
        float g = __expf(-0.5f * d * d);

        density += a * g;
    }

    pred[idx] = 1.0f - __expf(-density);
}


__global__ void render_backward_params_kernel(
    const float* __restrict__ mu,
    const float* __restrict__ sigma,
    const float* __restrict__ amp,
    const float* __restrict__ pred,
    const float* __restrict__ grad_pred,
    float* __restrict__ grad_mu,
    float* __restrict__ grad_sigma,
    float* __restrict__ grad_amp,
    int B,
    int N,
    int F
) {
    int b = blockIdx.x;
    int i = blockIdx.y;
    int tid = threadIdx.x;

    __shared__ float s_mu[256];
    __shared__ float s_sigma[256];
    __shared__ float s_amp[256];

    float local_mu = 0.0f;
    float local_sigma = 0.0f;
    float local_amp = 0.0f;

    int idx_bn = b * N + i;

    float m = mu[idx_bn];
    float s = fmaxf(sigma[idx_bn], 1.0e-6f);
    float a = amp[idx_bn];

    float s2 = s * s;
    float s3 = s2 * s;

    for (int f = tid; f < F; f += blockDim.x) {
        int idx_bf = b * F + f;

        // pred = 1 - exp(-density)
        // d pred / d density = exp(-density) = 1 - pred
        float grad_density = grad_pred[idx_bf] * fmaxf(1.0f - pred[idx_bf], 0.0f);

        float dx = (float)f - m;
        float d = dx / s;
        float g = __expf(-0.5f * d * d);

        local_amp += grad_density * g;

        // density += amp * g
        // d g / d mu = g * (f - mu) / sigma^2
        local_mu += grad_density * a * g * dx / s2;

        // d g / d sigma = g * (f - mu)^2 / sigma^3
        local_sigma += grad_density * a * g * dx * dx / s3;
    }

    s_mu[tid] = local_mu;
    s_sigma[tid] = local_sigma;
    s_amp[tid] = local_amp;

    __syncthreads();

    for (int offset = blockDim.x / 2; offset > 0; offset >>= 1) {
        if (tid < offset) {
            s_mu[tid] += s_mu[tid + offset];
            s_sigma[tid] += s_sigma[tid + offset];
            s_amp[tid] += s_amp[tid + offset];
        }
        __syncthreads();
    }

    if (tid == 0) {
        grad_mu[idx_bn] = s_mu[0];
        grad_sigma[idx_bn] = s_sigma[0];
        grad_amp[idx_bn] = s_amp[0];
    }
}


torch::Tensor audio_spectral_render_cuda(
    torch::Tensor mu,
    torch::Tensor sigma,
    torch::Tensor amp,
    int F
) {
    const int B = (int)mu.size(0);
    const int N = (int)mu.size(1);

    auto pred = torch::empty({B, F}, mu.options());

    const int threads = 256;
    const int total = B * F;
    const int blocks = div_up_int(total, threads);

    cudaStream_t stream = at::cuda::getCurrentCUDAStream();

    render_kernel<<<blocks, threads, 0, stream>>>(
        mu.data_ptr<float>(),
        sigma.data_ptr<float>(),
        amp.data_ptr<float>(),
        pred.data_ptr<float>(),
        B,
        N,
        F
    );

    return pred;
}


std::vector<torch::Tensor> audio_spectral_render_backward_cuda(
    torch::Tensor mu,
    torch::Tensor sigma,
    torch::Tensor amp,
    torch::Tensor pred,
    torch::Tensor grad_pred
) {
    const int B = (int)mu.size(0);
    const int N = (int)mu.size(1);
    const int F = (int)pred.size(1);

    auto grad_mu = torch::zeros_like(mu);
    auto grad_sigma = torch::zeros_like(sigma);
    auto grad_amp = torch::zeros_like(amp);

    const int threads = 256;
    dim3 blocks(B, N);

    cudaStream_t stream = at::cuda::getCurrentCUDAStream();

    render_backward_params_kernel<<<blocks, threads, 0, stream>>>(
        mu.data_ptr<float>(),
        sigma.data_ptr<float>(),
        amp.data_ptr<float>(),
        pred.data_ptr<float>(),
        grad_pred.data_ptr<float>(),
        grad_mu.data_ptr<float>(),
        grad_sigma.data_ptr<float>(),
        grad_amp.data_ptr<float>(),
        B,
        N,
        F
    );

    return {grad_mu, grad_sigma, grad_amp};
}




__global__ void hard_pass1_render_diff_kernel(
    const float* __restrict__ mu,
    const float* __restrict__ sigma,
    const float* __restrict__ amp,
    const float* __restrict__ target,
    float* __restrict__ pred,
    float* __restrict__ diff,
    float* __restrict__ absdiff,
    float* __restrict__ sum_absdiff,
    int B,
    int N,
    int F
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = B * F;

    if (idx >= total) {
        return;
    }

    int b = idx / F;
    int f = idx - b * F;

    float density = 0.0f;
    int base_bn = b * N;

    for (int i = 0; i < N; ++i) {
        float m = mu[base_bn + i];
        float s = fmaxf(sigma[base_bn + i], 1.0e-6f);
        float a = amp[base_bn + i];

        float d = ((float)f - m) / s;
        float g = __expf(-0.5f * d * d);

        density += a * g;
    }

    float p = 1.0f - __expf(-density);
    float df = p - target[idx];
    float ad = fabsf(df);

    pred[idx] = p;
    diff[idx] = df;
    absdiff[idx] = ad;

    atomicAdd(sum_absdiff, ad);
}


__global__ void hard_pass2_weight_sum_kernel(
    const float* __restrict__ absdiff,
    float* __restrict__ weight,
    float* __restrict__ sum_weight,
    const float* __restrict__ sum_absdiff,
    int total,
    float lambda_hard,
    float hard_clip
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx >= total) {
        return;
    }

    float mean_abs = sum_absdiff[0] / fmaxf((float)total, 1.0f);
    mean_abs = fmaxf(mean_abs, 1.0e-8f);

    float norm = absdiff[idx] / mean_abs;
    norm = fminf(fmaxf(norm, 0.0f), hard_clip);

    float w = 1.0f + lambda_hard * norm;

    weight[idx] = w;
    atomicAdd(sum_weight, w);
}


__global__ void hard_pass3_loss_grad_density_kernel(
    const float* __restrict__ pred,
    const float* __restrict__ diff,
    const float* __restrict__ absdiff,
    const float* __restrict__ weight,
    const float* __restrict__ sum_weight,
    float* __restrict__ grad_density,
    float* __restrict__ loss_parts,
    int total,
    float lambda_mse
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx >= total) {
        return;
    }

    float sw = fmaxf(sum_weight[0], 1.0e-8f);

    float df = diff[idx];
    float ad = absdiff[idx];
    float w = weight[idx];

    float l1_contrib = w * ad / sw;
    float mse_contrib = df * df / fmaxf((float)total, 1.0f);
    float loss_contrib = l1_contrib + lambda_mse * mse_contrib;

    atomicAdd(&loss_parts[0], loss_contrib);
    atomicAdd(&loss_parts[1], l1_contrib);
    atomicAdd(&loss_parts[2], mse_contrib);

    float sign_df = 0.0f;
    if (df > 0.0f) {
        sign_df = 1.0f;
    } else if (df < 0.0f) {
        sign_df = -1.0f;
    }

    // Hard weight tratado como detached.
    float grad_pred = w * sign_df / sw
                    + lambda_mse * 2.0f * df / fmaxf((float)total, 1.0f);

    // pred = 1 - exp(-density)
    // d pred / d density = exp(-density) = 1 - pred
    float grad_d = grad_pred * fmaxf(1.0f - pred[idx], 0.0f);

    grad_density[idx] = grad_d;
}


std::vector<torch::Tensor> audio_spectral_hard_loss_cuda(
    torch::Tensor mu,
    torch::Tensor sigma,
    torch::Tensor amp,
    torch::Tensor target,
    double lambda_mse,
    double lambda_hard,
    double hard_clip
) {
    const int B = (int)mu.size(0);
    const int N = (int)mu.size(1);
    const int F = (int)target.size(1);
    const int total = B * F;

    auto opts = mu.options();

    auto pred = torch::empty({B, F}, opts);
    auto diff = torch::empty({B, F}, opts);
    auto absdiff = torch::empty({B, F}, opts);
    auto weight = torch::empty({B, F}, opts);
    auto grad_density = torch::empty({B, F}, opts);

    auto sum_absdiff = torch::zeros({1}, opts);
    auto sum_weight = torch::zeros({1}, opts);
    auto loss_parts = torch::zeros({3}, opts);

    auto grad_mu = torch::zeros_like(mu);
    auto grad_sigma = torch::zeros_like(sigma);
    auto grad_amp = torch::zeros_like(amp);

    const int threads = 256;
    const int blocks = div_up_int(total, threads);

    cudaStream_t stream = at::cuda::getCurrentCUDAStream();

    hard_pass1_render_diff_kernel<<<blocks, threads, 0, stream>>>(
        mu.data_ptr<float>(),
        sigma.data_ptr<float>(),
        amp.data_ptr<float>(),
        target.data_ptr<float>(),
        pred.data_ptr<float>(),
        diff.data_ptr<float>(),
        absdiff.data_ptr<float>(),
        sum_absdiff.data_ptr<float>(),
        B,
        N,
        F
    );

    hard_pass2_weight_sum_kernel<<<blocks, threads, 0, stream>>>(
        absdiff.data_ptr<float>(),
        weight.data_ptr<float>(),
        sum_weight.data_ptr<float>(),
        sum_absdiff.data_ptr<float>(),
        total,
        (float)lambda_hard,
        (float)hard_clip
    );

    hard_pass3_loss_grad_density_kernel<<<blocks, threads, 0, stream>>>(
        pred.data_ptr<float>(),
        diff.data_ptr<float>(),
        absdiff.data_ptr<float>(),
        weight.data_ptr<float>(),
        sum_weight.data_ptr<float>(),
        grad_density.data_ptr<float>(),
        loss_parts.data_ptr<float>(),
        total,
        (float)lambda_mse
    );

    const int threads_grad = 256;
    dim3 blocks_grad(B, N);

    grad_params_kernel<<<blocks_grad, threads_grad, 0, stream>>>(
        mu.data_ptr<float>(),
        sigma.data_ptr<float>(),
        amp.data_ptr<float>(),
        grad_density.data_ptr<float>(),
        grad_mu.data_ptr<float>(),
        grad_sigma.data_ptr<float>(),
        grad_amp.data_ptr<float>(),
        B,
        N,
        F
    );

    auto loss = loss_parts.select(0, 0);
    auto l1 = loss_parts.select(0, 1);
    auto mse = loss_parts.select(0, 2);

    return {loss, l1, mse, grad_mu, grad_sigma, grad_amp};
}


std::vector<torch::Tensor> audio_spectral_loss_cuda(
    torch::Tensor mu,
    torch::Tensor sigma,
    torch::Tensor amp,
    torch::Tensor target,
    torch::Tensor weight,
    double lambda_mse,
    double inv_l1_norm,
    double inv_mse_norm
) {
    const int B = (int)mu.size(0);
    const int N = (int)mu.size(1);
    const int F = (int)target.size(1);

    auto opts = mu.options();

    auto grad_density = torch::empty({B, F}, opts);
    auto grad_mu = torch::zeros_like(mu);
    auto grad_sigma = torch::zeros_like(sigma);
    auto grad_amp = torch::zeros_like(amp);

    auto loss_parts = torch::zeros({3}, opts);

    const int threads = 256;

    int total_bf = B * F;
    int blocks_bf = div_up_int(total_bf, threads);

    cudaStream_t stream = at::cuda::getCurrentCUDAStream();

    density_loss_kernel<<<blocks_bf, threads, 0, stream>>>(
        mu.data_ptr<float>(),
        sigma.data_ptr<float>(),
        amp.data_ptr<float>(),
        target.data_ptr<float>(),
        weight.data_ptr<float>(),
        grad_density.data_ptr<float>(),
        loss_parts.data_ptr<float>(),
        B,
        N,
        F,
        (float)lambda_mse,
        (float)inv_l1_norm,
        (float)inv_mse_norm
    );

    dim3 blocks_grad(B, N);

    grad_params_kernel<<<blocks_grad, threads, 0, stream>>>(
        mu.data_ptr<float>(),
        sigma.data_ptr<float>(),
        amp.data_ptr<float>(),
        grad_density.data_ptr<float>(),
        grad_mu.data_ptr<float>(),
        grad_sigma.data_ptr<float>(),
        grad_amp.data_ptr<float>(),
        B,
        N,
        F
    );

    auto loss = loss_parts.select(0, 0);
    auto l1 = loss_parts.select(0, 1);
    auto mse = loss_parts.select(0, 2);

    return {loss, l1, mse, grad_mu, grad_sigma, grad_amp};
}
