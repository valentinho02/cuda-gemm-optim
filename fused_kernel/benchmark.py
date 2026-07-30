import torch
import torch.nn as nn
from ops import FusedLNGELU  # ops.py에 정의된 nn.Module 또는 autograd.Function


class NativeLNGELU(nn.Module):
    """최적화 대상인 기존 PyTorch Native 구현"""
    def __init__(self, hidden_dim, eps=1e-5):
        super().__init__()
        self.ln = nn.LayerNorm(hidden_dim, eps=eps)
        self.gelu = nn.GELU()

    def forward(self, x):
        return self.gelu(self.ln(x))


def measure_execution_time(model, x, num_warmup=20, num_iters=100):
    """torch.cuda.Event를 사용하여 Forward/Backward 실행 시간을 밀리초(ms) 단위로 측정"""
    
    # Warmup (GPU Clocks 안정화 및 CUDA Context 캐싱)
    for _ in range(num_warmup):
        out = model(x)
        loss = out.sum()
        loss.backward()
        model.zero_grad()
        if x.grad is not None:
            x.grad.zero_()

    torch.cuda.synchronize()

    # --- 1. Forward Pass 측정 ---
    start_fwd = torch.cuda.Event(enable_timing=True)
    end_fwd = torch.cuda.Event(enable_timing=True)

    start_fwd.record()
    for _ in range(num_iters):
        out = model(x)
    end_fwd.record()
    torch.cuda.synchronize()
    fwd_time_ms = start_fwd.elapsed_time(end_fwd) / num_iters

    # --- 2. Backward Pass 측정 ---
    start_bwd = torch.cuda.Event(enable_timing=True)
    end_bwd = torch.cuda.Event(enable_timing=True)

    # Gradient 누적 방지를 위해 매 iteration마다 grad 초기화
    start_bwd.record()
    for _ in range(num_iters):
        loss = out.sum()
        loss.backward(retain_graph=True)
    end_bwd.record()
    torch.cuda.synchronize()
    bwd_time_ms = start_bwd.elapsed_time(end_bwd) / num_iters

    total_time_ms = fwd_time_ms + bwd_time_ms
    return fwd_time_ms, bwd_time_ms, total_time_ms


def run_benchmark():
    print("=" * 80)
    print("      Fused LayerNorm + GELU vs PyTorch Native Benchmark")
    print("=" * 80)
    
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA를 사용할 수 없습니다. GPU 환경에서 실행해주세요.")

    device_name = torch.cuda.get_device_name(0)
    print(f"GPU Device: {device_name}\n")

    # 실험할 텐서 크기 설정 (Batch, Seq_Len, Hidden_Dim)
    configs = [
        {"batch": 16, "seq_len": 512,  "hidden": 768},   # BERT-Base 크기
        {"batch": 8,  "seq_len": 1024, "hidden": 1024},  # Large Transformer
        {"batch": 4,  "seq_len": 2048, "hidden": 2048},  # LLaMA 스타일 소형
        {"batch": 2,  "seq_len": 4096, "hidden": 4096},  # 대형 텐서 (메모리 병목 극대화)
    ]

    header = f"{'Shape (B, S, H)':<20} | {'Mode':<10} | {'Fwd (ms)':<10} | {'Bwd (ms)':<10} | {'Total (ms)':<10} | {'Speedup':<8}"
    print(header)
    print("-" * len(header))

    for cfg in configs:
        b, s, h = cfg["batch"], cfg["seq_len"], cfg["hidden"]
        shape_str = f"({b}, {s}, {h})"

        # 입력 데이터 준비
        x_native = torch.randn(b, s, h, device="cuda", dtype=torch.float32, requires_grad=True)
        x_fused = x_native.clone().detach().requires_grad_(True)

        # 모델 정의
        native_model = NativeLNGELU(h).cuda()
        fused_model = FusedLNGELU(h).cuda()

        # 가중치 동일화 (정확한 비교를 위함)
        with torch.no_grad():
            fused_model.gamma.copy_(native_model.ln.weight)
            fused_model.beta.copy_(native_model.ln.bias)

        # 측정 실행
        nat_fwd, nat_bwd, nat_tot = measure_execution_time(native_model, x_native)
        fus_fwd, fus_bwd, fus_tot = measure_execution_time(fused_model, x_fused)

        # Speedup 계산
        speedup_fwd = nat_fwd / fus_fwd
        speedup_tot = nat_tot / fus_tot

        # 결과 출력
        print(f"{shape_str:<20} | Native     | {nat_fwd:10.4f} | {nat_bwd:10.4f} | {nat_tot:10.4f} | 1.00x")
        print(f"{'':<20} | Fused      | {fus_fwd:10.4f} | {fus_bwd:10.4f} | {fus_tot:10.4f} | {speedup_tot:7.2f}x")
        print("-" * len(header))


if __name__ == "__main__":
    run_benchmark()


"""
copied result:
================================================================================
      Fused LayerNorm + GELU vs PyTorch Native Benchmark
================================================================================
GPU Device: Tesla T4

Shape (B, S, H)      | Mode       | Fwd (ms)   | Bwd (ms)   | Total (ms) | Speedup 
-----------------------------------------------------------------------------------
(16, 512, 768)       | Native     |     0.4894 |     1.2150 |     1.7043 | 1.00x
                     | Fused      |     0.3128 |     0.8735 |     1.1863 |    1.44x
-----------------------------------------------------------------------------------
(8, 1024, 1024)      | Native     |     0.5819 |     1.6257 |     2.2076 | 1.00x
                     | Fused      |     0.2846 |     1.1360 |     1.4206 |    1.55x
-----------------------------------------------------------------------------------
(4, 2048, 2048)      | Native     |     1.3258 |     3.5245 |     4.8503 | 1.00x
                     | Fused      |     0.5553 |     2.3382 |     2.8935 |    1.68x
-----------------------------------------------------------------------------------
(2, 4096, 4096)      | Native     |     2.7805 |     7.0837 |     9.8642 | 1.00x
                     | Fused      |     1.3829 |     5.3870 |     6.7699 |    1.46x
-----------------------------------------------------------------------------------
"""