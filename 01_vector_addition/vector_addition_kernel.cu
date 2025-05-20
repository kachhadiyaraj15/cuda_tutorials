#include <cuda_runtime.h>
#include <iostream>
#include <chrono>

__global__ void add_kernel(const int *a, const int *b, int *c, int N){
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
        // threadIdx.x	Index of thread within its block
        // blockIdx.x	Index of block within the grid
        // blockDim.x	Total number of threads per block
    if (idx < N)
        c[idx] = a[idx] + b[idx];
}

void run_gpu_add(const int *h_a, const int *h_b, int *h_c, int N, int warmups, int repeats, double &avg_time_ms){
    int *d_a, *d_b, *d_c;
    cudaMalloc(&d_a, N * sizeof(int));
    cudaMalloc(&d_b, N * sizeof(int));
    cudaMalloc(&d_c, N * sizeof(int));  

    cudaMemcpy(d_a, h_a, N * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, N * sizeof(int), cudaMemcpyHostToDevice);

    dim3 block(256);  // 256 threads per block 
    dim3 grid((N + block.x - 1) / block.x);  // Enough blocks to cover N elements.
        // (N + block.x - 1) / block.x correctly handles non-divisible cases — when N is not an exact multiple of block.x

    // warmup
    for (int i=0; i<warmups; i++){
        add_kernel<<<grid, block>>>(d_a, d_b, d_c, N);
        cudaDeviceSynchronize();
    }

    // Timing
    double total_time = 0.0;
    for (int i=0; i < repeats; i++){
        auto start = std::chrono::high_resolution_clock::now();
        add_kernel<<<grid, block>>>(d_a, d_b, d_c, N);
        cudaDeviceSynchronize();
        auto end = std::chrono::high_resolution_clock::now();
        total_time += std::chrono::duration<double, std::milli>(end - start).count();
    }

    avg_time_ms = total_time / repeats;
    cudaMemcpy(h_c, d_c, N * sizeof(int), cudaMemcpyDeviceToHost);

    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);
}