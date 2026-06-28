#pragma once

void sgemmResolveBankConflicts(
    const float* dA,
    const float* dB,
    float* dC,
    int M,
    int N,
    int K,
    cudaStream_t stream
);