#pragma once

#include <cuda_runtime.h>

void sharedMemSgemm(
    const float* dA, 
    const float* dB,
    float* dC,
    int M, 
    int N, 
    int K,
    cudaStream_t stream
);