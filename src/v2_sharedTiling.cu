#include <iostream>
#include <cuda_runtime.h>
#include <iomanip>


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

#define TILESIZE 32
//presume that BlockDim is 32, 32
__global__ void sgemm_shared_tiling(const float* A, const float* B, float* C, int N){
    __shared__ float As[TILESIZE][TILESIZE];
    __shared__ float Bs[TILESIZE][TILESIZE];
    
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    float acc = 0.0f;

    for (int t = 0; t < (N + TILESIZE - 1) / TILESIZE; t++){
        //get a array info from A and B to Shared Memory
        if (row < N && t * TILESIZE + threadIdx.x < N)
            As[threadIdx.y][threadIdx.x] = A[row * N + t * TILESIZE + threadIdx.x];
        else 
            As[threadIdx.y][threadIdx.x] = 0.0f;
        if (t * TILESIZE + threadIdx.y < N && col < N)
            Bs[threadIdx.y][threadIdx.x] = B[col + t * TILESIZE * N + threadIdx.y * N];
        else 
            Bs[threadIdx.y][threadIdx.x] = 0.0f;
        __syncthreads();
        for (int k = 0; k < TILESIZE; k++){
            acc += (As[threadIdx.y][k] * Bs[k][threadIdx.x]);
        }
        __syncthreads();
    }
    if (row < N && col < N)
        C[row * N + col] = acc;
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

    run_benchmark(sgemm_shared_tiling, gridSize, blockSize, dA, dB, dC, N, "2. Sgemm Shared Memory Tiling");
    
    cudaFree(dA); cudaFree(dB); cudaFree(dC);
    free(hA); free(hB);
}

/*
copied Result:
2. Sgemm Shared Memory Tiling: 20.86 ms | 823.52   GFLOPS
*/