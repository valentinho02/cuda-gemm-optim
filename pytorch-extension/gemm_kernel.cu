#include <torch/extension.h>
#include <cuda_runtime.h>

#define TILE_M 32
#define TILE_N 32
#define TILE_K 8
#define REGSIZE 4

__global__ void 
gemm_register_blocking(
    const float* __restrict__ A, 
    const float* __restrict__ B, 
    float* __restrict__ C, 
    int M, int N, int K)
{
    __shared__ float As[TILE_M][TILE_K];
    __shared__ float Bs[TILE_K][TILE_N];

    float regA[REGSIZE];
    float regB[REGSIZE];
    float regC[REGSIZE][REGSIZE] = {0.0f};

    int block_row = blockIdx.y * TILE_M;
    int block_col = blockIdx.x * TILE_N;

    for (int t = 0; t < K; t += TILE_K){
        int tid = threadIdx.y * blockDim.x + threadIdx.x;
        
        int As_row = tid / TILE_K; 
        int As_col = tid % TILE_K; 
        
        int Bs_row = tid / TILE_N; 
        int Bs_col = tid % TILE_N; 

        #pragma unroll
        for (int i = 0; i < 4; ++i) {
            int cur_row = block_row + As_row + i * 8;
            int cur_col = t + As_col;
            if (cur_row < M && cur_col < K) {
                As[As_row + i * 8][As_col] = A[cur_row * K + cur_col];
            } else {
                As[As_row + i * 8][As_col] = 0.0f;
            }
        }

        #pragma unroll
        for (int i = 0; i < 4; ++i) {
            int cur_row = t + Bs_row + i * 2;
            int cur_col = block_col + Bs_col;
            if (cur_row < K && cur_col < N) {
                Bs[Bs_row + i * 2][Bs_col] = B[cur_row * N + cur_col];
            } else {
                Bs[Bs_row + i * 2][Bs_col] = 0.0f;
            }
        }

        __syncthreads();

        for (int k = 0; k < TILE_K; k++){
            for (int i = 0; i < REGSIZE; i++){
                regA[i] = As[threadIdx.y * REGSIZE + i][k];
            }
            for (int i = 0; i < REGSIZE; i++){
                regB[i] = Bs[k][threadIdx.x * REGSIZE + i];
            }
            for (int i = 0; i < REGSIZE; i++){
                for (int j = 0; j < REGSIZE; j++){
                    regC[i][j] += regA[i] * regB[j];
                }
            }
        }
        __syncthreads();
    }   

    for (int i = 0; i < REGSIZE; i++){
        for (int j = 0; j < REGSIZE; j++){
            int global_row = block_row + threadIdx.y * REGSIZE + i;
            int global_col = block_col + threadIdx.x * REGSIZE + j;
            if (global_row < M && global_col < N) {
                C[global_row * N + global_col] = regC[i][j];
            }
        }
    }
}

torch::Tensor gemm_cuda_forward(torch::Tensor A, torch::Tensor B){
    TORCH_CHECK(A.is_cuda(), "A must be a cuda tensor");
    TORCH_CHECK(B.is_cuda(), "B must be a cuda tensor");
    TORCH_CHECK(A.dim() == 2 && B.dim() == 2, "Tensor must be 2D");
    TORCH_CHECK(A.size(1) == B.size(0), "K dimension mismatch");
    
    int M = A.size(0);
    int N = B.size(1);
    int K = A.size(1);

    auto C = torch::zeros({M, N}, A.options());
    dim3 blockSize(8, 8);
    dim3 gridSize((N + TILE_N - 1) / TILE_N, (M + TILE_M - 1) / TILE_M);
    
    gemm_register_blocking<<<gridSize, blockSize>>>(
        A.data_ptr<float>(), 
        B.data_ptr<float>(), 
        C.data_ptr<float>(), 
        M, N, K
    );

    return C;
}