#include <iostream>
#include <vector>
#include <cstdlib>
#include <iomanip>
#include <cuda_runtime.h>

#define NUM_BINS 256
#define BLOCK_SIZE 256

// CUDA 오류 검출 매크로
#define CHECK_CUDA(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            std::cerr << "CUDA Error: " << cudaGetErrorString(err) \
                      << " at line " << __LINE__ << std::endl; \
            exit(EXIT_FAILURE); \
        } \
    } while (0)

// 1. Global Memory Naive Kernel
__global__ void histogram_global_naive(const unsigned char* data, unsigned int* histo, long N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) {
        atomicAdd(&histo[data[i]], 1);
    }
}

// 2. Privatized Kernel 
__global__ void histogram_privatized_kernel(const unsigned char* data, unsigned int* histo, long N) {
    __shared__ unsigned int s_histo[NUM_BINS];

    int tid = threadIdx.x;
    for (int bin = tid; bin < NUM_BINS; bin += blockDim.x) {
        s_histo[bin] = 0;
    }
    __syncthreads();

    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) {
        atomicAdd(&s_histo[data[i]], 1);
    }
    __syncthreads();

    for (int bin = tid; bin < NUM_BINS; bin += blockDim.x) {
        atomicAdd(&histo[bin], s_histo[bin]);
    }
}

// 3. Coarsened Kernel (Thread Coarsening + Grid-Strided Loop)
__global__ void histogram_coarsened_kernel(const unsigned char* data, unsigned int* histo, long N) {
    __shared__ unsigned int s_histo[NUM_BINS];

    int tid = threadIdx.x;
    for (int bin = tid; bin < NUM_BINS; bin += blockDim.x) {
        s_histo[bin] = 0;
    }
    __syncthreads();

    int g_tid = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    for (int i = g_tid; i < N; i += stride) {
        atomicAdd(&s_histo[data[i]], 1);
    }
    __syncthreads();

    for (int bin = tid; bin < NUM_BINS; bin += blockDim.x) {
        atomicAdd(&histo[bin], s_histo[bin]);
    }
}

// 검증용 CPU 히스토그램 함수
void histogram_cpu(const unsigned char* data, unsigned int* histo, long N) {
    std::fill(histo, histo + NUM_BINS, 0);
    for (long i = 0; i < N; ++i) {
        histo[data[i]]++;
    }
}

// 히스토그램 결과 비교 함수
bool verify_histogram(const unsigned int* ref, const unsigned int* test) {
    for (int i = 0; i < NUM_BINS; ++i) {
        if (ref[i] != test[i]) {
            std::cerr << "Mismatch at bin " << i << ": Ref=" << ref[i] << ", Test=" << test[i] << std::endl;
            return false;
        }
    }
    return true;
}

int main() {
    const long MB = 1024;

    // GPU 장치 정보 및 SM 개수 확인 (Coarsening 그리드 크기 결정용)
    int deviceId;
    cudaGetDevice(&deviceId);
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, deviceId);
    int numSMs = prop.multiProcessorCount;

    std::cout << "=========================================================================================\n";
    std::cout << " Running CUDA Histogram Benchmark on " << prop.name << "\n";
    std::cout << "=========================================================================================\n\n";

    long N = MB * 1024L * 1024L; // 바이트 단위 데이터 개수

    std::cout << "-----------------------------------------------------------------------------------------\n";
    std::cout << " Data Size: " << MB << " MB (" << N << " elements)\n";
    std::cout << "-----------------------------------------------------------------------------------------\n";

    // 입력 데이터 할당 및 초기화
    std::vector<unsigned char> h_data(N);
    for (long i = 0; i < N; ++i) {
        h_data[i] = static_cast<unsigned char>(rand() % NUM_BINS);
    }

    std::vector<unsigned int> h_histo_ref(NUM_BINS, 0);
    std::vector<unsigned int> h_histo_out(NUM_BINS, 0);

    // CPU 검증 데이터 계산 (소형 크기 우선 검증)
    histogram_cpu(h_data.data(), h_histo_ref.data(), N);

    // Device 메모리 할당
    unsigned char* d_data;
    unsigned int* d_histo;
    CHECK_CUDA(cudaMalloc(&d_data, N * sizeof(unsigned char)));
    CHECK_CUDA(cudaMalloc(&d_histo, NUM_BINS * sizeof(unsigned int)));

    CHECK_CUDA(cudaMemcpy(d_data, h_data.data(), N * sizeof(unsigned char), cudaMemcpyHostToDevice));

    // CUDA Event 기반 실행 시간 측정 설정
    cudaEvent_t start, stop;
    CHECK_CUDA(cudaEventCreate(&start));
    CHECK_CUDA(cudaEventCreate(&stop));

    float time_naive = 0.0f, time_privatized = 0.0f, time_coarsened = 0.0f;

    // -------------------------------------------------------------
    // 1. Naive Kernel
    // -------------------------------------------------------------
    CHECK_CUDA(cudaMemset(d_histo, 0, NUM_BINS * sizeof(unsigned int)));
    int grid_size_naive = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;

    CHECK_CUDA(cudaEventRecord(start));
    histogram_global_naive<<<grid_size_naive, BLOCK_SIZE>>>(d_data, d_histo, N);
    CHECK_CUDA(cudaEventRecord(stop));
    CHECK_CUDA(cudaEventSynchronize(stop));
    CHECK_CUDA(cudaEventElapsedTime(&time_naive, start, stop));

    // 검증
    CHECK_CUDA(cudaMemcpy(h_histo_out.data(), d_histo, NUM_BINS * sizeof(unsigned int), cudaMemcpyDeviceToHost));
    bool pass_naive = verify_histogram(h_histo_ref.data(), h_histo_out.data());

    // -------------------------------------------------------------
    // 2. Privatized Kernel
    // -------------------------------------------------------------
    CHECK_CUDA(cudaMemset(d_histo, 0, NUM_BINS * sizeof(unsigned int)));
    int grid_size_priv = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;

    CHECK_CUDA(cudaEventRecord(start));
    histogram_privatized_kernel<<<grid_size_priv, BLOCK_SIZE>>>(d_data, d_histo, N);
    CHECK_CUDA(cudaEventRecord(stop));
    CHECK_CUDA(cudaEventSynchronize(stop));
    CHECK_CUDA(cudaEventElapsedTime(&time_privatized, start, stop));

    // 검증
    CHECK_CUDA(cudaMemcpy(h_histo_out.data(), d_histo, NUM_BINS * sizeof(unsigned int), cudaMemcpyDeviceToHost));
    bool pass_priv = verify_histogram(h_histo_ref.data(), h_histo_out.data());

    // -------------------------------------------------------------
    // 3. Coarsened Kernel
    // -------------------------------------------------------------
    CHECK_CUDA(cudaMemset(d_histo, 0, NUM_BINS * sizeof(unsigned int)));
    // Grid-strided loop 커널에 맞춰 하드웨어를 충분히 활용할 고정 그리드 크기 설정 (SM당 32개 블록)
    int grid_size_coarsened = numSMs * 32;

    CHECK_CUDA(cudaEventRecord(start));
    histogram_coarsened_kernel<<<grid_size_coarsened, BLOCK_SIZE>>>(d_data, d_histo, N);
    CHECK_CUDA(cudaEventRecord(stop));
    CHECK_CUDA(cudaEventSynchronize(stop));
    CHECK_CUDA(cudaEventElapsedTime(&time_coarsened, start, stop));

    // 검증
    CHECK_CUDA(cudaMemcpy(h_histo_out.data(), d_histo, NUM_BINS * sizeof(unsigned int), cudaMemcpyDeviceToHost));
    bool pass_coarse = verify_histogram(h_histo_ref.data(), h_histo_out.data());

    // 결과 출력
    std::cout << std::fixed << std::setprecision(3);
    std::cout << "1. Global Naive     : " << std::setw(8) << time_naive << " ms | Speedup: 1.00x | Status: " << (pass_naive ? "PASS" : "FAIL") << "\n";
    std::cout << "2. Privatized       : " << std::setw(8) << time_privatized << " ms | Speedup: " << std::setw(4) << (time_naive / time_privatized) << "x | Status: " << (pass_priv ? "PASS" : "FAIL") << "\n";
    std::cout << "3. Coarsened        : " << std::setw(8) << time_coarsened << " ms | Speedup: " << std::setw(4) << (time_naive / time_coarsened) << "x | Status: " << (pass_coarse ? "PASS" : "FAIL") << "\n\n";

    // 메모리 해제
    CHECK_CUDA(cudaEventDestroy(start));
    CHECK_CUDA(cudaEventDestroy(stop));
    CHECK_CUDA(cudaFree(d_data));
    CHECK_CUDA(cudaFree(d_histo));

    return 0;
}

/*
copied result:
=========================================================================================
 Running CUDA Histogram Benchmark on Tesla T4
=========================================================================================

-----------------------------------------------------------------------------------------
 Data Size: 1024 MB (1073741824 elements)
-----------------------------------------------------------------------------------------
1. Global Naive     :  428.611 ms | Speedup: 1.00x | Status: PASS
2. Privatized       :   55.538 ms | Speedup: 7.717x | Status: PASS
3. Coarsened        :   12.184 ms | Speedup: 35.179x | Status: PASS

*/