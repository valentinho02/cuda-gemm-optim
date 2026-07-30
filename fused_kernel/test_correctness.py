import torch
import torch.nn as nn
from ops import FusedLNGELU


class NativeLNGELU(nn.Module):
    def __init__(self, hidden_dim, eps=1e-5):
        super().__init__()
        self.ln = nn.LayerNorm(hidden_dim, eps=eps)
        self.gelu = nn.GELU()

    def forward(self, x):
        return self.gelu(self.ln(x))


def test_correctness():
    print("=" * 75)
    print("      Fused LayerNorm + GELU Correctness Test")
    print("=" * 75)

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA를 사용할 수 없습니다. GPU 환경에서 실행해주세요.")

    shapes = [
        (2, 4, 8),          # 소형 텐서
        (16, 128, 768),     # BERT-Base 크기
        (8, 512, 1024),     # Large Transformer 크기
        (4, 1024, 4096),    # LLaMA 스타일 대형 크기
        (3, 17, 257),       # 홀수/비정형 텐서
    ]

    # atomicAdd FP32 누적 오차를 고려해 atol을 1e-3으로 설정
    atol = 1e-3
    rtol = 1e-3
    all_passed = True

    for b, s, h in shapes:
        print(f"\n[Test Shape: ({b}, {s}, {h})]")

        torch.manual_seed(42)
        x_native = torch.randn(b, s, h, device="cuda", dtype=torch.float32, requires_grad=True)
        x_fused = x_native.clone().detach().requires_grad_(True)

        grad_output = torch.randn_like(x_native)

        native_model = NativeLNGELU(h).cuda()
        fused_model = FusedLNGELU(h).cuda()

        with torch.no_grad():
            fused_model.gamma.copy_(native_model.ln.weight)
            fused_model.beta.copy_(native_model.ln.bias)

        # Forward Pass
        y_native = native_model(x_native)
        y_fused = fused_model(x_fused)

        fwd_diff = (y_native - y_fused).abs().max().item()
        fwd_check = torch.allclose(y_native, y_fused, atol=atol, rtol=rtol)

        # Backward Pass
        loss_native = (y_native * grad_output).sum()
        loss_fused = (y_fused * grad_output).sum()

        loss_native.backward()
        loss_fused.backward()

        dx_diff = (x_native.grad - x_fused.grad).abs().max().item()
        dx_check = torch.allclose(x_native.grad, x_fused.grad, atol=atol, rtol=rtol)

        dgamma_diff = (native_model.ln.weight.grad - fused_model.gamma.grad).abs().max().item()
        dgamma_check = torch.allclose(native_model.ln.weight.grad, fused_model.gamma.grad, atol=atol, rtol=rtol)

        dbeta_diff = (native_model.ln.bias.grad - fused_model.beta.grad).abs().max().item()
        dbeta_check = torch.allclose(native_model.ln.bias.grad, fused_model.beta.grad, atol=atol, rtol=rtol)

        shape_passed = fwd_check and dx_check and dgamma_check and dbeta_check
        all_passed = all_passed and shape_passed

        status = "PASS" if shape_passed else "FAIL"
        print(f"  - Status          : [{status}]")
        print(f"  - Forward Max Diff : {fwd_diff:.2e} (Check: {'OK' if fwd_check else 'FAIL'})")
        print(f"  - dX Max Diff      : {dx_diff:.2e} (Check: {'OK' if dx_check else 'FAIL'})")
        print(f"  - dGamma Max Diff  : {dgamma_diff:.2e} (Check: {'OK' if dgamma_check else 'FAIL'})")
        print(f"  - dBeta Max Diff   : {dbeta_diff:.2e} (Check: {'OK' if dbeta_check else 'FAIL'})")

    print("\n" + "=" * 75)
    if all_passed:
        print("  [최종 검증 성공] 모든 Shape에서 PyTorch Native 결과와 일치합니다.")
    else:
        print("  [최종 검증 실패] 일부 테스트에서 허용 오차를 초과했습니다.")
    print("=" * 75)


if __name__ == "__main__":
    test_correctness()