#include <torch/extension.h>
#include <vector>

// ============================================================================
// CUDA 파일(ln_gelu_kernel.cu)에 정의된 C++ 런처 함수 선언 (Forward Declarations)
// ============================================================================

std::vector<at::Tensor> ln_gelu_forward_cuda(
    at::Tensor x,
    at::Tensor gamma,
    at::Tensor beta,
    float epsilon
);

std::vector<at::Tensor> ln_gelu_backward_cuda(
    at::Tensor dy,
    at::Tensor x,
    at::Tensor gamma,
    at::Tensor beta,
    at::Tensor mean,
    at::Tensor rsigma
);

// ============================================================================
// pybind11 바인딩 모듈 정의
// ============================================================================

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.doc() = "Fused LayerNorm + GELU CUDA Extension";
    
    // Python에서 module.forward(...) 형태로 호출하도록 바인딩
    m.def(
        "forward", 
        &ln_gelu_forward_cuda, 
        "Fused LayerNorm + GELU Forward Pass (CUDA)"
    );

    // Python에서 module.backward(...) 형태로 호출하도록 바인딩
    m.def(
        "backward", 
        &ln_gelu_backward_cuda, 
        "Fused LayerNorm + GELU Backward Pass (CUDA)"
    );
}