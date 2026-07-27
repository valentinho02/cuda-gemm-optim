#include <iostream>
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <iomanip>

int main(){
    int N = 2048;
    cublasHandle_t handle;
    cublasCreate(&handle);
    float alpha = 1.0f;
    float beta = 0.0f;

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    size_t size = N * N * sizeof(float); 
    float* hA = (float*)malloc(size);
    float* hB = (float*)malloc(size);
    
    float* dA = nullptr;
    float* dB = nullptr;
    float* dC = nullptr;
    cudaMalloc(&dA, size);
    cudaMalloc(&dB, size);
    cudaMalloc(&dC, size);
    cudaMemcpy(dA, hA, size, cudaMemcpyHostToDevice);
    cudaMemcpy(dB, hB, size, cudaMemcpyHostToDevice);
    cudaMemset(dC, 0, size);

    for (int i = 0; i < N * N; i++) {
        hA[i] = 1.0f;
        hB[i] = 1.0f;
    }

    //warmup
    cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, N, N, N, &alpha, dB, N, dA, N, &beta, dC, N);
    cudaDeviceSynchronize();

    int runs = 10;
    cudaEventRecord(start);
    for (int run = 0; run < runs; run++){
        cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, N, N, N, &alpha, dB, N, dA, N, &beta, dC, N);
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float runtime = 0.0f;
    cudaEventElapsedTime(&runtime, start, stop);
    runtime /= 10.0f;
    
    
    double gflops = (2.0 * (double) N * N * N) / ((runtime / 1000.0) * 1e9);
    
    std::cout
    << std::left
    << std::setw(25) << "cuBLAS"
    << ": "
    << std::fixed
    << std::setprecision(2)
    << runtime
    << " ms | "
    << std::setw(8)
    << gflops
    << " GFLOPS\n";

    cublasDestroy(handle);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    cudaFree(dA); cudaFree(dB); cudaFree(dC);
    free(hA); free(hB);
    return 0;    
}   

/*
copied Result:
cuBLAS                   : 5.60 ms | 3066.99  GFLOPS
*/