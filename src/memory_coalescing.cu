#include <iostream>
#include <vector>
#include <cuda_runtime.h>

const int WIDTH = 4096;
const int HEIGHT = 4096;
const int BLOCKSIZE = 32;

void CUDA_CHECK(cudaError_t err){
    if (err != cudaSuccess){
        std::cout << "error : " << cudaGetErrorString(err);
    }
}

__global__ void coalescedKernel(const float* input, float* output, int width, int height){
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (col < width && row < height){
        output[row * width + col] = input[row * width + col] * 2.0f;
    }
}


__global__ void uncoalescedKernel(const float * input, float* output, int width, int height){
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;

    if (row < height && col < width){
        output[col * height + row] = input[col * height + row] * 2.0f;
    }
}

int main(){
    size_t size = WIDTH * HEIGHT * sizeof(float);
    
    std::cout << "Coalescing vs Uncoalescing testing\n";
    std::cout << "Width: " << WIDTH << " Height : " << HEIGHT << "\n";

    std::vector<float> h_input(WIDTH * HEIGHT, 1.0f);
    std::vector<float> h_output(WIDTH * HEIGHT, 0.0f);

    float* d_input = nullptr;
    float* d_output = nullptr;
    CUDA_CHECK(cudaMalloc(&d_input, size));
    CUDA_CHECK(cudaMalloc(&d_output, size));

    CUDA_CHECK(cudaMemcpy(d_input, h_input.data(), size, cudaMemcpyHostToDevice));

    dim3 blockDim(BLOCKSIZE, BLOCKSIZE);
    dim3 gridDim((WIDTH + blockDim.x - 1) / blockDim.x, (HEIGHT + blockDim.y - 1) / blockDim.y);

    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    //warmup
    coalescedKernel <<<gridDim, blockDim>>>(d_input, d_output, WIDTH, HEIGHT);
    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaEventRecord(start));
    for (int i = 0; i < 100; i++){
        coalescedKernel <<<gridDim, blockDim>>>(d_input, d_output, WIDTH, HEIGHT);
    }
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));

    float coalescedTime = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&coalescedTime, start, stop));
    coalescedTime /= 100.0f;

    CUDA_CHECK(cudaEventRecord(start));
    for (int i = 0; i < 100; i++){
        uncoalescedKernel<<<gridDim, blockDim>>>(d_input, d_output, WIDTH, HEIGHT);
    }
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));

    float uncoalescedTime = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&uncoalescedTime, start, stop));
    uncoalescedTime /= 100.0f;

    std::cout << "\n[결과 비교]" << std::endl;
    std::cout << "1. Coalesced Access Time   : " << coalescedTime << " ms" << std::endl;
    std::cout << "2. Uncoalesced Access Time : " << uncoalescedTime << " ms" << std::endl;
    std::cout << "-> 속도 차이: 약 " << (uncoalescedTime / coalescedTime) << "배 차이!" << std::endl;

    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    CUDA_CHECK(cudaFree(d_input));
    CUDA_CHECK(cudaFree(d_output));

    return 0;
}

/*
copied result:
Coalescing vs Uncoalescing testing
Width: 4096 Height : 4096

[결과 비교]
1. Coalesced Access Time   : 0.63643 ms
2. Uncoalesced Access Time : 2.0925 ms
-> 속도 차이: 약 3.28787배 차이!
*/