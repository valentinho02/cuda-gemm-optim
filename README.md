# cuda-gemm-optim

**CUDA GEMM 커널을 naive부터 cuBLAS 근접 성능까지 단계적으로 최적화한 프로젝트.**
매 단계마다 "왜 빨라지는가"를 memory access 횟수, occupancy, throughput 관점에서 정량적으로 검증한다.

> (2026.07) 산출물

---

## 목차

- [프로젝트 개요](#프로젝트-개요)
- [최적화 여정](#최적화-여정)
- [벤치마크 결과](#벤치마크-결과)
- [프로젝트 구조](#프로젝트-구조)
- [빌드 및 실행](#빌드-및-실행)
- [배운 것](#배운-것)
- [다음 목표](#다음-목표)

---

## 프로젝트 개요

Single-precision GEMM(`C = A x B`, FP32)을 4단계로 최적화하며, 각 단계에서 GPU 아키텍처 지식(memory coalescing, shared memory, occupancy, ILP)이 실제 성능에 어떻게 반영되는지 직접 측정했다.

- **GPU**: NVIDIA T4 (Colab)
- **CUDA Toolkit**: 13.0
- **행렬 크기**: 128 / 512 / 1024 / 2048 / 4096 (정방 행렬)
- **정확도 검증**: 모든 커널은 cuBLAS 결과 대비 상대 오차 기준으로 correctness test 통과

## 최적화 여정

| 단계 | 기법 | 핵심 아이디어 | 파일 |
|---|---|---|---|
| 1 | Naive | 스레드당 출력 원소 1개, global memory 직접 접근 | [`src/v1_naive.cu`](src/v1_naive.cu) |
| 2 | Shared Memory Tiling | 타일 단위로 shared memory에 캐싱, global read 횟수 K/TILE_SIZE로 감소 | [`src/v2_tiling.cu`](src/v2_tiling.cu) |
| 3 | Register Blocking | 스레드 1개가 여러 출력 원소 계산 + `float4` 벡터화 로드로 대역폭 극대화 | [`src/v3_register_blocking.cu`](src/v3_register_blocking.cu) |
| 4 | cuBLAS | 비교 기준선 (`cublasSgemm`) | [`src/v4_cublas_baseline.cu`](src/v4_cublas_baseline.cu) |

## 벤치마크 결과

### 2048x2048 기준 성능 비교

| 커널 | GFLOPS | cuBLAS 대비 (%) | naive 대비 speedup |
|---|---|---|---|
| Naive | 563.73 | 18.38% | 1.0x |
| Shared Memory Tiling | 823.52 | 26.85% | 1.46x |
| Register Blocking | 1457.22 | 47.51% | 2.58x |
| cuBLAS | 3066.99 | 100% | 5.44x |

![speedup comparison](docs/images/speedup_vs_cublas.png)

### 행렬 크기별 스케일링

| 크기 | Naive (GFLOPS) | Tiling (GFLOPS) | Register Blocking (GFLOPS) | cuBLAS (GFLOPS) |
|---|---|---|---|---|
| 128 | | | | |
| 512 | | | | |
| 1024 | | | | |
| 2048 | 563.73 | 823.52 | 1457.22 | 3066.99 |
| 4096 | | | | |

![multi size scaling](docs/images/multi_size_scaling.png)

> 원본 CSV: [`benchmarks/results/`](benchmarks/results/)

## 프로젝트 구조

```
cuda-gemm-optim/
├── src/
│   ├── v1_naive.cu
│   ├── v2_tiling.cu
│   ├── v3_register_blocking.cu
│   ├── v4_cublas_baseline.cu
│   └── common/
│       ├── gemm_utils.cuh
│       └── benchmark_utils.cuh
├── benchmarks/
│   ├── run_all_benchmarks.sh
│   └── results/
├── pytorch_extension/          # pybind11 기반 PyTorch custom op
│   ├── setup.py
│   ├── gemm_binding.cpp
│   ├── gemm_kernel.cu
│   └── test_extension.py
├── fused_kernel/                # LayerNorm+GELU fused kernel
│   ├── layernorm_gelu_fused.cu
│   ├── layernorm_gelu_baseline.py
│   └── benchmark_fused.py
└── docs/
    └── images/
```

## 빌드 및 실행

```bash
# 개별 커널 컴파일 예시
nvcc -O3 -arch=sm_75 src/v3_register_blocking.cu -o v3_bench

# 전체 벤치마크 실행
bash benchmarks/run_all_benchmarks.sh

# PyTorch extension 빌드
cd pytorch_extension && pip install -e .
```

## 배운 것

- **Naive → Tiling**: memory-bound 커널에서 global memory 재접근 횟수를 줄이는 것이 성능에 가장 직접적인 영향을 준다. 원소당 K회 접근하던 것을 K/TILE_SIZE회로 줄이면서 1.46x 개선을 확인했다.
- **Tiling → Register Blocking**: shared memory bandwidth 자체가 다음 병목이 되고, 스레드당 여러 출력을 계산해 ILP를 높이고 `float4` 벡터화로 메모리 대역폭을 더 끌어올리면서 1.77x 추가 개선을 확인했다.
- **cuBLAS와의 격차**: 여전히 남아있는 52.49% 격차는 warp-level scheduling, register spilling 최적화, double buffering 등 더 세밀한 튜닝 영역이며 Phase 2(Triton, warp shuffle)에서 이어서 다룬다.

## 다음 목표

- [ ] `pytorch_extension/`: pybind11로 `torch.Tensor` 인터페이스 노출, `torch.matmul` 대비 벤치마크
- [ ] `fused_kernel/`: LayerNorm+GELU fused kernel, 순정 PyTorch 대비 forward/backward speedup 측정
- [ ] Triton 버전과의 비교는 별도 레포 [`triton-attention-study`](https://github.com/) 에서 진행

---

## 관련 글 (Velog)

- [코얼레싱했을 때 vs 안 했을 때 성능 차이](https://velog.io/@valentinho/CUDA-%EB%A9%94%EB%AA%A8%EB%A6%AC-coalescing)
- [gemm 최적화](https://velog.io/@valentinho/GEMM%EC%97%90-%EB%8C%80%ED%95%B4%EC%84%9C-%EC%95%8C%EC%95%84%EB%B3%B4%EC%9E%902)
- <!-- Week8: PyTorch Custom CUDA Extension으로 Fused Kernel 만들기 -->
