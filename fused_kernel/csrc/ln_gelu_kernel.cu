#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cmath>
#include <vector>

__device__ __forceinline__ float gelu(float x) {
    constexpr float kSqrtHalf = 0.70710678118654752440f; // 1 / sqrt(2)
    return 0.5f * x * (1.0f + erff(x * kSqrtHalf));
}

__device__ __forceinline__ float gelu_grad(float x) {
    constexpr float kSqrtHalf = 0.70710678118654752440f;
    constexpr float kInvSqrt2Pi = 0.39894228040143267794f; // 1 / sqrt(2 * pi)
    float cdf = 0.5f * (1.0f + erff(x * kSqrtHalf));
    float pdf = kInvSqrt2Pi * expf(-0.5f * x * x);
    return cdf + x * pdf;
}

// Block-wide sum reduction using warp shuffles and shared memory
template <int BLOCK_SIZE>
__device__ __forceinline__ float blockReduceSum(float val) {
    static __shared__ float shared[32]; // Max 1024 threads = 32 warps
    int lane = threadIdx.x % 32;
    int wid = threadIdx.x / 32;

    #pragma unroll
    for (int mask = 16; mask > 0; mask /= 2) {
        val += __shfl_down_sync(0xffffffff, val, mask);
    }

    if (lane == 0) {
        shared[wid] = val;
    }
    __syncthreads();

    val = (threadIdx.x < (BLOCK_SIZE / 32)) ? shared[lane] : 0.0f;
    if (wid == 0) {
        #pragma unroll
        for (int mask = 16; mask > 0; mask /= 2) {
            val += __shfl_down_sync(0xffffffff, val, mask);
        }
    }
    return val; // Returned result is valid on thread 0
}

// ============================================================================
// Forward Kernel
// ============================================================================

template <int BLOCK_SIZE>
__global__ void ln_gelu_forward_kernel(
    const float* __restrict__ X,
    const float* __restrict__ gamma,
    const float* __restrict__ beta,
    float* __restrict__ Y,
    float* __restrict__ mean,
    float* __restrict__ rsigma,
    int M,
    int N,
    float epsilon
) {
    int row = blockIdx.x;
    if (row >= M) return;

    const float* row_X = X + row * N;
    float* row_Y = Y + row * N;

    // Step 1: Mean calculation
    float thread_sum = 0.0f;
    for (int i = threadIdx.x; i < N; i += BLOCK_SIZE) {
        thread_sum += row_X[i];
    }
    float sum = blockReduceSum<BLOCK_SIZE>(thread_sum);

    __shared__ float s_mean;
    if (threadIdx.x == 0) {
        s_mean = sum / N;
        mean[row] = s_mean;
    }
    __syncthreads();
    float m = s_mean;

    // Step 2: Variance and reciprocal standard deviation calculation
    float thread_var_sum = 0.0f;
    for (int i = threadIdx.x; i < N; i += BLOCK_SIZE) {
        float diff = row_X[i] - m;
        thread_var_sum += diff * diff;
    }
    float var_sum = blockReduceSum<BLOCK_SIZE>(thread_var_sum);

    __shared__ float s_rsigma;
    if (threadIdx.x == 0) {
        float var = var_sum / N;
        s_rsigma = rsqrtf(var + epsilon);
        rsigma[row] = s_rsigma;
    }
    __syncthreads();
    float rs = s_rsigma;

    // Step 3: LayerNorm output + GELU activation
    for (int i = threadIdx.x; i < N; i += BLOCK_SIZE) {
        float x_val = row_X[i];
        float x_hat = (x_val - m) * rs;
        float g = (gamma != nullptr) ? gamma[i] : 1.0f;
        float b = (beta != nullptr) ? beta[i] : 0.0f;
        float ln_out = g * x_hat + b;
        row_Y[i] = gelu(ln_out);
    }
}

// ============================================================================
// Backward Kernel
// ============================================================================

template <int BLOCK_SIZE>
__global__ void ln_gelu_backward_kernel(
    const float* __restrict__ dY,
    const float* __restrict__ X,
    const float* __restrict__ gamma,
    const float* __restrict__ beta,
    const float* __restrict__ mean,
    const float* __restrict__ rsigma,
    float* __restrict__ dX,
    float* __restrict__ dgamma,
    float* __restrict__ dbeta,
    int M,
    int N
) {
    int row = blockIdx.x;
    if (row >= M) return;

    const float* row_dY = dY + row * N;
    const float* row_X = X + row * N;
    float* row_dX = dX + row * N;

    float m = mean[row];
    float rs = rsigma[row];

    // Step 1: Compute intermediate gradients (dz, dhat_x) and local statistics
    float thread_sum_dhat_x = 0.0f;
    float thread_sum_dhat_x_xhat = 0.0f;

    for (int i = threadIdx.x; i < N; i += BLOCK_SIZE) {
        float x_val = row_X[i];
        float x_hat = (x_val - m) * rs;
        float g = (gamma != nullptr) ? gamma[i] : 1.0f;
        float b = (beta != nullptr) ? beta[i] : 0.0f;

        float z = g * x_hat + b;
        float dy_val = row_dY[i];
        float dz = dy_val * gelu_grad(z);

        float dhat_x = dz * g;
        thread_sum_dhat_x += dhat_x;
        thread_sum_dhat_x_xhat += dhat_x * x_hat;

        // Accumulate gamma and beta gradients across batch dimension
        if (dgamma != nullptr) {
            atomicAdd(&dgamma[i], dz * x_hat);
        }
        if (dbeta != nullptr) {
            atomicAdd(&dbeta[i], dz);
        }
    }

    float sum_dhat_x = blockReduceSum<BLOCK_SIZE>(thread_sum_dhat_x);
    float sum_dhat_x_xhat = blockReduceSum<BLOCK_SIZE>(thread_sum_dhat_x_xhat);

    __shared__ float s_sum1, s_sum2;
    if (threadIdx.x == 0) {
        s_sum1 = sum_dhat_x;
        s_sum2 = sum_dhat_x_xhat;
    }
    __syncthreads();

    float sum1 = s_sum1;
    float sum2 = s_sum2;
    float inv_N = 1.0f / static_cast<float>(N);

    // Step 2: Compute dX
    for (int i = threadIdx.x; i < N; i += BLOCK_SIZE) {
        float x_val = row_X[i];
        float x_hat = (x_val - m) * rs;
        float g = (gamma != nullptr) ? gamma[i] : 1.0f;
        float b = (beta != nullptr) ? beta[i] : 0.0f;

        float z = g * x_hat + b;
        float dy_val = row_dY[i];
        float dz = dy_val * gelu_grad(z);
        float dhat_x = dz * g;

        row_dX[i] = rs * (dhat_x - (sum1 + x_hat * sum2) * inv_N);
    }
}

// ============================================================================
// C++ PyTorch Launchers
// ============================================================================

std::vector<at::Tensor> ln_gelu_forward_cuda(
    at::Tensor x,
    at::Tensor gamma,
    at::Tensor beta,
    float epsilon
) {
    TORCH_CHECK(x.is_cuda(), "x must be a CUDA tensor");
    TORCH_CHECK(x.is_contiguous(), "x must be contiguous");

    const int N = x.size(-1);
    const int M = x.numel() / N;

    auto y = torch::empty_like(x);
    auto mean = torch::empty({M}, x.options());
    auto rsigma = torch::empty({M}, x.options());

    const float* gamma_ptr = gamma.defined() ? gamma.data_ptr<float>() : nullptr;
    const float* beta_ptr = beta.defined() ? beta.data_ptr<float>() : nullptr;

    const int threads = 256;
    const int blocks = M;

    ln_gelu_forward_kernel<threads><<<blocks, threads>>>(
        x.data_ptr<float>(),
        gamma_ptr,
        beta_ptr,
        y.data_ptr<float>(),
        mean.data_ptr<float>(),
        rsigma.data_ptr<float>(),
        M,
        N,
        epsilon
    );

    return {y, mean, rsigma};
}

std::vector<at::Tensor> ln_gelu_backward_cuda(
    at::Tensor dy,
    at::Tensor x,
    at::Tensor gamma,
    at::Tensor beta,
    at::Tensor mean,
    at::Tensor rsigma
) {
    TORCH_CHECK(dy.is_cuda() && x.is_cuda(), "Inputs must be CUDA tensors");
    TORCH_CHECK(dy.is_contiguous() && x.is_contiguous(), "Inputs must be contiguous");

    const int N = x.size(-1);
    const int M = x.numel() / N;

    auto dx = torch::empty_like(x);
    auto dgamma = gamma.defined() ? torch::zeros_like(gamma) : torch::Tensor();
    auto dbeta = beta.defined() ? torch::zeros_like(beta) : torch::Tensor();

    const float* gamma_ptr = gamma.defined() ? gamma.data_ptr<float>() : nullptr;
    const float* beta_ptr = beta.defined() ? beta.data_ptr<float>() : nullptr;
    float* dgamma_ptr = dgamma.defined() ? dgamma.data_ptr<float>() : nullptr;
    float* dbeta_ptr = dbeta.defined() ? dbeta.data_ptr<float>() : nullptr;

    const int threads = 256;
    const int blocks = M;

    ln_gelu_backward_kernel<threads><<<blocks, threads>>>(
        dy.data_ptr<float>(),
        x.data_ptr<float>(),
        gamma_ptr,
        beta_ptr,
        mean.data_ptr<float>(),
        rsigma.data_ptr<float>(),
        dx.data_ptr<float>(),
        dgamma_ptr,
        dbeta_ptr,
        M,
        N
    );

    return {dx, dgamma, dbeta};
}