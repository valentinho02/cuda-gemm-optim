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

#define TILE_M 32
#define TILE_N 32
#define TILE_K 8
#define RSIZE 4
__global__ void sgemm_register_blocking(
    const float* A,
    const float* B,
    float* C,
    int N
){
    float regA[RSIZE];
    float regB[RSIZE];
    float regC[RSIZE][RSIZE] = {0.0f};

    int block_row = blockIdx.y * TILE_M;
    int block_col = blockIdx.x * TILE_N;

    __shared__ float As[TILE_M][TILE_K];
    __shared__ float Bs[TILE_K][TILE_N];
    for (int t = 0; t < N; t += TILE_K){
        // fill the shared memory
        int A_row = (threadIdx.y * 8 + threadIdx.x) / TILE_K;
        int A_col = (threadIdx.y * 8 + threadIdx.x) % TILE_K;
        As[A_row][A_col] = A[(block_row + A_row) * N + t + A_col];
        As[A_row + 8][A_col] = A[(block_row + A_row + 8) * N + t + A_col];
        As[A_row + 16][A_col] = A[(block_row + A_row + 16) * N + t + A_col];
        As[A_row + 24][A_col] = A[(block_row + A_row + 24) * N + t + A_col];

        int B_row = (threadIdx.y * 8 + threadIdx.x) / TILE_N;
        int B_col = (threadIdx.y * 8 + threadIdx.x) % TILE_N;
        Bs[B_row][B_col] = B[(t + B_row) * N + block_col + B_col];
        Bs[B_row + 2][B_col] = B[(t + B_row + 2) * N + block_col + B_col];
        Bs[B_row + 4][B_col] = B[(t + B_row + 4) * N + block_col + B_col];
        Bs[B_row + 6][B_col] = B[(t + B_row + 6) * N + block_col + B_col];
        // each threads fill 4, with sync all shared memory tile filled
        __syncthreads();

        for (int k = 0; k < TILE_K; k++){
            for (int i = 0; i < RSIZE; i++){
                regA[i] = As[threadIdx.y * RSIZE + i][k];
                regB[i] = Bs[k][threadIdx.x * RSIZE + i];
            }
            for (int i = 0; i < RSIZE; i++){
                for (int j = 0; j < RSIZE; j++){
                    regC[i][j] = fmaf(regA[i], regB[j], regC[i][j]);
                }
            }
        } 
        __syncthreads();  
    }
    for (int i = 0; i < RSIZE; i++){
        for (int j = 0; j < RSIZE; j++){
            int global_row = block_row + threadIdx.y * RSIZE + i;
            int global_col = block_col + threadIdx.x * RSIZE + j;
            if (global_row < N && global_col < N)
                C[global_row * N + global_col] = regC[i][j];
        }
    }
   
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

    dim3 blockSize(8, 8);
    dim3 gridSize((N + TILE_N - 1)/ TILE_N, (N + TILE_M - 1) / TILE_M);

    run_benchmark(sgemm_register_blocking, gridSize, blockSize, dA, dB, dC, N, "3. Register Blocking");
    
    cudaFree(dA); cudaFree(dB); cudaFree(dC);
    free(hA); free(hB);
}

/*
copied Result:
3. Register Blocking     : 11.79 ms | 1457.22  GFLOPS
*/