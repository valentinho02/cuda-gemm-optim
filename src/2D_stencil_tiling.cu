#include <cuda.h>
#include <cuda_runtime.h>

#include <iostream>
#include <vector>
#include <cstdlib>
#include <ctime>

#define TILE_SIZE 32
#define RADIUS 1

// ======================================================
// Naive Kernel
// ======================================================

__global__ void stencil_2d_naive_kernel(
    const float* in,
    float* out,
    int width,
    int height,
    float c0,
    float c1)
{
    int gx = blockIdx.x * blockDim.x + threadIdx.x;
    int gy = blockIdx.y * blockDim.y + threadIdx.y;

    if (gx >= width || gy >= height)
        return;

    float center = in[gy * width + gx];

    float up =
        (gy > 0) ? in[(gy - 1) * width + gx] : 0.0f;

    float down =
        (gy < height - 1) ? in[(gy + 1) * width + gx] : 0.0f;

    float left =
        (gx > 0) ? in[gy * width + gx - 1] : 0.0f;

    float right =
        (gx < width - 1) ? in[gy * width + gx + 1] : 0.0f;

    out[gy * width + gx] =
        c0 * center +
        c1 * (up + down + left + right);
}

// ======================================================
// Shared Memory Kernel
// ======================================================

__global__ void stencil_2d_kernel(
    const float* __restrict__ in,
    float* __restrict__ out,
    int width,
    int height,
    float c0,
    float c1)
{
    __shared__ float s_data[TILE_SIZE + 2 * RADIUS][TILE_SIZE + 2 * RADIUS];

    int tx = threadIdx.x;
    int ty = threadIdx.y;

    int gx = blockIdx.x * TILE_SIZE + tx;
    int gy = blockIdx.y * TILE_SIZE + ty;

    int sm_x = tx + RADIUS;
    int sm_y = ty + RADIUS;

    // center

    if (gx < width && gy < height)
        s_data[sm_y][sm_x] = in[gy * width + gx];
    else
        s_data[sm_y][sm_x] = 0.0f;

    // left / right halo

    if (tx < RADIUS)
    {
        int left_gx = gx - RADIUS;
        int right_gx = gx + TILE_SIZE;

        if (left_gx >= 0 && gy < height)
            s_data[sm_y][sm_x - RADIUS] =
                in[gy * width + left_gx];
        else
            s_data[sm_y][sm_x - RADIUS] = 0.0f;

        if (right_gx < width && gy < height)
            s_data[sm_y][sm_x + TILE_SIZE] =
                in[gy * width + right_gx];
        else
            s_data[sm_y][sm_x + TILE_SIZE] = 0.0f;
    }

    // top / bottom halo

    if (ty < RADIUS)
    {
        int top_gy = gy - RADIUS;
        int bottom_gy = gy + TILE_SIZE;

        if (top_gy >= 0 && gx < width)
            s_data[sm_y - RADIUS][sm_x] =
                in[top_gy * width + gx];
        else
            s_data[sm_y - RADIUS][sm_x] = 0.0f;

        if (bottom_gy < height && gx < width)
            s_data[sm_y + TILE_SIZE][sm_x] =
                in[bottom_gy * width + gx];
        else
            s_data[sm_y + TILE_SIZE][sm_x] = 0.0f;
    }

    __syncthreads();

    if (gx < width && gy < height)
    {
        out[gy * width + gx] =
            c0 * s_data[sm_y][sm_x] +
            c1 * (
                s_data[sm_y - RADIUS][sm_x] +
                s_data[sm_y + RADIUS][sm_x] +
                s_data[sm_y][sm_x - RADIUS] +
                s_data[sm_y][sm_x + RADIUS]);
    }
}

// ======================================================
// v2 stencil kernel with shared memory
// ======================================================

__global__ void stencil_2d_kernel_v2(
    const float* __restrict__ in,
    float* __restrict__ out,
    int width,
    int height,
    float c0,
    float c1)
{
    __shared__ float s_data[TILE_SIZE + 2 * RADIUS][TILE_SIZE + 2 * RADIUS];
    int gx = blockIdx.x * TILE_SIZE + threadIdx.x;
    int gy = blockIdx.y * TILE_SIZE + threadIdx.y;
    
}

// ======================================================
// Benchmark
// ======================================================

void benchmark()
{
    std::vector<int> sizes =
    {
        512,
        1024,
        2048,
        4096,
        8192
    };

    const float c0 = 0.5f;
    const float c1 = 0.125f;

    const int repeat = 100;

    dim3 block(TILE_SIZE, TILE_SIZE);

    cudaEvent_t start, stop;

    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    std::cout << "\n===============================================\n";
    std::cout << "2D Stencil Benchmark\n";
    std::cout << "Repeat : " << repeat << "\n";
    std::cout << "===============================================\n\n";

    printf("%8s %15s %15s %12s\n",
           "Size",
           "Naive(ms)",
           "Shared(ms)",
           "Speedup");

    for (auto N : sizes)
    {
        size_t bytes = (size_t)N * N * sizeof(float);

        float* d_input;
        float* d_output;

        cudaMalloc(&d_input, bytes);
        cudaMalloc(&d_output, bytes);

        std::vector<float> h_input(N * N);

        for (int i = 0; i < N * N; i++)
            h_input[i] =
                static_cast<float>(rand()) / RAND_MAX;

        cudaMemcpy(
            d_input,
            h_input.data(),
            bytes,
            cudaMemcpyHostToDevice);

        cudaMemset(d_output, 0, bytes);

        dim3 grid(
            (N + TILE_SIZE - 1) / TILE_SIZE,
            (N + TILE_SIZE - 1) / TILE_SIZE);

        //--------------------------
        // Warm-up
        //--------------------------

        for (int i = 0; i < 10; i++)
        {
            stencil_2d_naive_kernel<<<grid, block>>>(
                d_input,
                d_output,
                N,
                N,
                c0,
                c1);

            stencil_2d_kernel<<<grid, block>>>(
                d_input,
                d_output,
                N,
                N,
                c0,
                c1);
        }

        cudaDeviceSynchronize();

        //--------------------------
        // Naive
        //--------------------------

        cudaEventRecord(start);

        for (int i = 0; i < repeat; i++)
        {
            stencil_2d_naive_kernel<<<grid, block>>>(
                d_input,
                d_output,
                N,
                N,
                c0,
                c1);
        }

        cudaEventRecord(stop);
        cudaEventSynchronize(stop);

        float naive_ms;

        cudaEventElapsedTime(
            &naive_ms,
            start,
            stop);

        naive_ms /= repeat;

        //--------------------------
        // Shared
        //--------------------------

        cudaEventRecord(start);

        for (int i = 0; i < repeat; i++)
        {
            stencil_2d_kernel<<<grid, block>>>(
                d_input,
                d_output,
                N,
                N,
                c0,
                c1);
        }

        cudaEventRecord(stop);
        cudaEventSynchronize(stop);

        float shared_ms;

        cudaEventElapsedTime(
            &shared_ms,
            start,
            stop);

        shared_ms /= repeat;

        printf("%8d %15.4f %15.4f %12.2fx\n",
               N,
               naive_ms,
               shared_ms,
               naive_ms / shared_ms);

        cudaFree(d_input);
        cudaFree(d_output);
    }

    cudaEventDestroy(start);
    cudaEventDestroy(stop);
}

// ======================================================

int main()
{
    srand((unsigned)time(nullptr));

    benchmark();

    return 0;
}

/*
v1: copied result
===============================================
2D Stencil Benchmark
Repeat : 100
===============================================

    Size       Naive(ms)      Shared(ms)      Speedup
     512          0.0097          0.0136         0.71x
    1024          0.0449          0.0640         0.70x
    2048          0.1628          0.2438         0.67x
    4096          0.6438          0.9686         0.66x
    8192          2.4425          3.2238         0.76x
*/

/*
v2: copied result

*/