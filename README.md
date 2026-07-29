# cuda-gemm-optim

**CUDA GEMM 커널을 naive부터 cuBLAS 근접 성능까지 단계적으로 최적화한 프로젝트.**

매 단계마다 "왜 빨라지는가"를 memory access 횟수, occupancy, throughput 관점에서 정량적으로 검증한다.

> (2026.07) 산출물

---

## 목차

- [프로젝트 개요](#프로젝트-개요)
- [최적화 여정](#최적화-여정)
- [벤치마크 결과](#벤치마크-결과)
  - [CUDA Native Benchmark (2048x2048)](#cuda-native-benchmark-2048x2048)
  - [PyTorch Custom Extension Benchmark (2048x2048)](#pytorch-custom-extension-benchmark-2048x2048)
  - [행렬 크기별 스케일링](#행렬-크기별-스케일링)
- [메모리 접근 패턴 실험](#메모리-접근-패턴-실험)
- [프로젝트 구조](#프로젝트-구조)
- [빌드 및 실행](#빌드-및-실행)
- [배운 것](#배운-것)
- [다음 목표](#다음-목표)
- [관련 글 (Velog)](#관련-글-velog)

---

## 프로젝트 개요

Single-precision GEMM(`C = A x B`)을 4단계로 최적화하며, 각 단계에서 GPU 아키텍처 지식(memory coalescing, shared memory, occupancy, ILP)이 실제 성능에 어떻게 반영되는지 직접 측정했다.

- **GPU**: NVIDIA T4 (Colab)
- **CUDA Toolkit**: 13.0
- **행렬 크기**: 128 / 512 / 1024 / 2048 / 4096 (정방 행렬) 및 비정방 경계 조건 행렬
- **정확도 검증**: PyTorch Native Matmul 및 cuBLAS 결과 대비 상대 오차/절대 오차 기준으로 correctness test 통과 (비정방 경계 조건 모서리 처리 적용 완료)

---

## 최적화 여정

| 단계 | 기법 | 핵심 아이디어 | 파일 |
|---|---|---|---|
| 1 | Naive | 스레드당 출력 원소 1개, global memory 직접 접근 | [`src/v1_naive.cu`](src/v1_naive.cu) |
| 2 | Shared Memory Tiling | 타일 단위로 shared memory에 캐싱, global read 횟수 K/TILE_SIZE로 감소 | [`src/v2_tiling.cu`](src/v2_tiling.cu) |
| 3 | Register Blocking | 스레드 1개가 여러 출력 원소 계산 + `float4` 벡터화 로드로 대역폭 극대화 | [`src/v3_registerBlocking.cu`](src/v3_registerBlocking.cu) |
| 4 | cuBLAS | 비교 기준선 (`cublasSgemm`) | [`src/v4_cuBLAS.cu`](src/v4_cuBLAS.cu) |

---

## 벤치마크 결과

### CUDA Native Benchmark (2048x2048)

| 커널 | GFLOPS | cuBLAS 대비 (%) | naive 대비 speedup |
|---|---|---|---|
| Naive | 563.73 | 18.38% | 1.0x |
| Shared Memory Tiling | 823.52 | 26.85% | 1.46x |
| Register Blocking | 1457.22 | 47.51% | 2.58x |
| cuBLAS | 3066.99 | 100% | 5.44x |

### PyTorch Custom Extension Benchmark (2048x2048)

`torch.utils.benchmark`를 활용하여 PyTorch Native Matmul과 커스텀 구현(Register Blocking 적용 v3)의 실행 시간을 직접 비교 및 검증했다.

```text
=== GEMM Forward Correctness Test ===
128x128 @ 128x128: PASS (max diff: 0.000019)
511x257 @ 257x333: PASS (max diff: 0.000000)
512x512 @ 512x512: PASS (max diff: 0.000000)

=== Running GEMM Benchmark ===
GEMM Benchmark: PyTorch Native Matmul
  Median: 4.19 ms
  IQR:    0.04 ms (4.17 to 4.21)
  5 measurements, 10 runs per measurement, 1 thread

GEMM Benchmark: Custom CUDA GEMM (v3)
  Median: 9.93 ms
  3 measurements, 10 runs per measurement, 1 thread
```

- **PyTorch Native Matmul (cuBLAS 기반)**: Median **4.19 ms**
- **Custom CUDA GEMM (v3 Register Blocking)**: Median **9.93 ms** (Native 대비 약 2.37x 차이, CUDA Native 벤치마크의 GFLOPS 비율과 일치)
- 비정방 행렬(`511x257 @ 257x333`)을 포함한 다양한 스펙에 대해 **Out-of-Bounds 로딩 처리**를 통해 100% Correctness 검증 완료

### 행렬 크기별 스케일링

| 크기 | Naive (GFLOPS) | Tiling (GFLOPS) | Register Blocking (GFLOPS) | cuBLAS (GFLOPS) |
|---|---|---|---|---|
| 128 | 183.11 | 179.50 | 320.55 | 314.55 |
| 512 | 547.05 | 656.97 | 1511.00 | 3534.67 |
| 1024 | 401.63 | 690.01 | 1671.27 | 4426.56 |
| 2048 | 563.73 | 823.52 | 1457.22 | 3066.99 |
| 4096 | 610.31 | 921.19 | 1736.29 | 4425.91 |

> **Note (N=128)**: N=128의 총 연산량은 약 4.19 MFLOPs로, cuBLAS의 실측 peak(~4400 GFLOPS) 기준 이론상 실행 시간은 1µs 미만이다. 실측치로 역산한 GFLOPS(314.55)는 실제로는 알고리즘 성능이 아니라 커널 launch latency, cuBLAS algorithm heuristic dispatch 비용, `cudaEvent` 타이밍 해상도 같은 고정 오버헤드가 지배하는 구간임을 보여준다. 이 크기에서 cuBLAS(314.55)와 본 프로젝트의 register blocking 커널(320.55)이 거의 동일한 값을 보이는 것이 그 증거다 — 알고리즘 효율 차이가 아니라 "연산이 오버헤드에 묻히는 구간"에 진입했기 때문이다.
>
> **Note (N=512)**: 최초 측정 시 cuBLAS 값이 359.30 GFLOPS로 비정상적으로 낮게 나왔던 것은 벤치마크 코드에서 N을 512로 바꾸지 않고 실행한 실수였다. N=512로 정정 재측정한 결과 3534.67 GFLOPS로, register blocking 커널보다 우위를 보이는 정상적인 패턴을 확인했다.

---

## 메모리 접근 패턴 실험

Global memory access pattern이 성능에 미치는 영향을 확인하기 위해 `4096 x 4096` 행렬에서 coalesced access와 uncoalesced access를 비교했다.

| 접근 패턴 | 실행 시간 |
|---|---:|
| Coalesced Access | 0.63643 ms |
| Uncoalesced Access | 2.0925 ms |

Uncoalesced access가 coalesced access보다 약 3.29x 느렸다. 같은 연산량이어도 warp 내 스레드가 연속된 주소를 접근하는지 여부만으로 global memory throughput이 크게 달라진다는 것을 확인했다.

실험 코드: [`src/memory_coalescing.cu`](src/memory_coalescing.cu)

---

## 프로젝트 구조

```
cuda-gemm-optim/
├── src/
│   ├── v1_naive.cu
│   ├── v2_tiling.cu
│   ├── v3_registerBlocking.cu
│   ├── v4_cuBLAS.cu
│   └── memory_coalescing.cu
├── pytorch-extension/
│   ├── setup.py
│   ├── test_extension.py
│   ├── gemm_kernel.cu
│   └── gemm_binding.cpp
└── README.md
```

---

## 빌드 및 실행

### 개별 CUDA 커널 빌드 및 실행

```bash
# 개별 커널 컴파일 예시
nvcc -O3 -arch=sm_75 src/v3_registerBlocking.cu -o v3_bench

# memory coalescing 실험
nvcc -O3 -arch=sm_75 src/memory_coalescing.cu -o memory_coalescing
```

### PyTorch Custom Extension 빌드 및 벤치마크 실행

```bash
# C++/CUDA PyTorch Extension 빌드 및 설치
cd pytorch-extension
pip install -e .

# Correctness 및 PyTorch Benchmark 실행
python test_extension.py
```

---

## 배운 것

- **Naive → Tiling**: memory-bound 커널에서 global memory 재접근 횟수를 줄이는 것이 성능에 가장 직접적인 영향을 준다. 원소당 K회 접근하던 것을 K/TILE_SIZE회로 줄이면서 1.46x 개선을 확인했다.
- **Tiling → Register Blocking**: shared memory bandwidth 자체가 다음 병목이 되고, 스레드당 여러 출력을 계산해 ILP를 높이고 `float4` 벡터화로 메모리 대역폭을 더 끌어올리면서 1.77x 추가 개선을 확인했다.
- **Memory Coalescing**: `4096 x 4096` 메모리 접근 실험에서 coalesced access가 uncoalesced access보다 약 3.29x 빨랐다. GEMM에서도 warp 단위의 연속 주소 접근을 유지하는 것이 global memory 성능의 핵심임을 확인했다.
- **Out-of-Bounds Handling**: 공유 메모리 로딩 시 행렬 크기가 타일 규격으로 나누어떨어지지 않는 경우(`511x257 @ 257x333`), 경계 검사 조건문 및 `0.0f` 패딩 처리가 필수적임을 검증했다.
- **cuBLAS와의 격차**: 여전히 남아있는 52.49% 격차는 warp-level scheduling, register spilling 최적화, double buffering 등 더 세밀한 튜닝 영역이며 Phase 2(Triton, warp shuffle)에서 이어서 다룬다.

---

## 다음 목표

- [x] `pytorch-extension/`: C++/pybind11로 `torch.Tensor` 인터페이스 노출, `torch.matmul` 대비 벤치마크 수행
- [ ] `fused_kernel/`: LayerNorm+GELU fused kernel (forward + backward), 순정 PyTorch 대비 forward/backward speedup 측정
- [ ] Triton 버전과의 비교는 별도 레포 [`triton-attention-study`](https://github.com/) 에서 진행

---

## 관련 글 (Velog)

- [코얼레싱했을 때 vs 안 했을 때 성능 차이](https://velog.io/@valentinho/CUDA-%EB%A9%94%EB%AA%A8%EB%A6%AC-coalescing)
- [gemm 최적화를 해보자](https://velog.io/@valentinho/GEMM%EC%97%90-%EB%8C%80%ED%95%B4%EC%84%9C-%EC%95%8C%EC%95%84%EB%B3%B4%EC%9E%902)
- [fused kernel이 필요한 이유](https://velog.io/@valentinho/Cudapytorch-Fused-Kernel)