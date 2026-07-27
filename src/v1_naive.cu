#include <iostream>
#include <cuda_runtime.h>
#include <iomanip>
#include <cublas_v2.h>

__global__ void sgemm_naive(const float* A, const float * B, float* C, int N){
    int row = blockDim.y * blockIdx.y + threadIdx.y;
    int col = blockDim.x * blockIdx.x + threadIdx.x;
    if (row < N && col < N){
        float acc = 0.0f;
        for (int k = 0; k < N; k++){
            acc += (A[row * N + k] * B[col + k * N]);
        }
        C[row * N + col] = acc;
    }
}

template<typename Kernel>
void run_benchmark(Kernel kernel, dim3 grid, dim3 block, const float* A, const float* B, float* C, int N, const std::string& name){
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    //warmup
    kernel<<<grid, block>>>(A, B, C, N);
    cudaDeviceSynchronize();
    
    int runs = 10;
    cudaEventRecord(start);
    for (int i = 0; i < runs; i++){
        kernel<<<grid, block>>>(A, B, C, N);
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float runtime = 0.0f;
    cudaEventElapsedTime(&runtime, start, stop);
    runtime /= 10.0f;

    double gflops = (2.0 * (double) N * N * N) / ((runtime / 1000.0) * 1e9);

    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    std::cout
    << std::left
    << std::setw(25) << name
    << ": "
    << std::fixed
    << std::setprecision(2)
    << runtime
    << " ms | "
    << std::setw(8)
    << gflops
    << " GFLOPS\n";
}

int main(){
    int N = 2048;
    size_t size = N * N * sizeof(float);
    float* hA = (float*)malloc(size);
    float* hB = (float*)malloc(size);

    for (int i = 0; i < N * N; i++){
        hA[i] = (float)(rand() % 100) / 100.0f;
        hB[i] = (float)(rand() % 100) / 100.0f;
    }
    float* dA= nullptr;
    float* dB = nullptr;
    float* dC = nullptr;
    cudaMalloc(&dA, size);
    cudaMalloc(&dB, size);
    cudaMalloc(&dC, size);
    cudaMemcpy(dA, hA, size, cudaMemcpyHostToDevice);
    cudaMemcpy(dB, hB, size, cudaMemcpyHostToDevice);
    cudaMemset(dC, 0, size);

    dim3 blockSize(32, 32);
    dim3 gridSize((N + blockSize.x - 1)/ blockSize.x, (N + blockSize.y - 1) / blockSize.y);

    run_benchmark(sgemm_naive, gridSize, blockSize, dA, dB, dC, N, "1. Sgemm naive ");
    
    cudaFree(dA); cudaFree(dB); cudaFree(dC);
    free(hA); free(hB);
}

/*
copied Result:
1. Sgemm naive           : 30.48 ms | 563.73   GFLOPS
*/