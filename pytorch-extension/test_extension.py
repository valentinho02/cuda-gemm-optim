import torch
import torch.utils.benchmark as benchmark

try:
    import my_gemm_ext
except ImportError as exc:
    raise SystemExit(
        "my_gemm_ext is not installed. Run `pip install -e .` in this directory first."
    ) from exc


def require_cuda():
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is not available. Build and run this extension on a CUDA machine.")

def test_gemm_correctness():
    print("=== GEMM Forward Correctness Test ===")
    require_cuda()
    device = torch.device("cuda")

    cases = [
        (128, 128, 128),
        (511, 257, 333),
        (512, 512, 512),
    ]

    for M, K, N in cases:
        A = torch.randn(M, K, device=device, dtype=torch.float32)
        B = torch.randn(K, N, device=device, dtype=torch.float32)

        res_custom = my_gemm_ext.gemm(A, B)
        res_torch = torch.matmul(A, B)

        is_close = torch.allclose(res_custom, res_torch, atol=1e-3, rtol=1e-3)
        max_diff = (res_custom - res_torch).abs().max().item()
        print(f"{M}x{K} @ {K}x{N}: {'PASS' if is_close else 'FAIL'} (max diff: {max_diff:.6f})")
        if not is_close:
            raise AssertionError(f"correctness failed for {M}x{K} @ {K}x{N}")

def run_benchmark():
    print("\n=== Running GEMM Benchmark ===")
    require_cuda()
    device = torch.device("cuda")
    M, K, N = 2048, 2048, 2048
    A = torch.randn(M, K, device=device)
    B = torch.randn(K, N, device=device)

    t0 = benchmark.Timer(
        stmt="torch.matmul(A, B)",
        globals={"A": A, "B": B},
        label="GEMM Benchmark",
        sub_label="PyTorch Native Matmul"
    )

    t1 = benchmark.Timer(
        stmt="my_gemm_ext.gemm(A, B)",
        globals={"A": A, "B": B, "my_gemm_ext": my_gemm_ext},
        label="GEMM Benchmark",
        sub_label="Custom CUDA GEMM (v3)"
    )

    print(t0.blocked_autorange())
    print(t1.blocked_autorange())

if __name__ == "__main__":
    test_gemm_correctness()
    run_benchmark()
