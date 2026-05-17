#include "kernels/2_kernel_global_mem_coalesce.cuh"



namespace{

constexpr int kBlockSize= 32;

template <const int BLOCKSIZE>
__global__ void segmm_global_mem_coalease(
    int M, int N, int K,
    float alpha,
    const float* A,
    const float* B,
    float beta,
    float* C
){
    const int cRow = blockIdx.x * BLOCKSIZE + (threadIdx.x / BLOCKSIZE);
    const int cCol = blockIdx.y * BLOCKSIZE + (threadIdx.x % BLOCKSIZE);

    // if statemtent is necessary to makr things wok under tile quantization

    if (cRow < M && cCol < N){
        float tmp = 0.0;
        for ( int i=0; i<K; i++){
            tmp += A[ cRow * K + i] * B[i*N + cCol];
        }

        C[cRow * N + cCol] = tmp;
    }
}

} // namespace 

void coaleasceSgemm(
    const float* dA,
    const float* dB,
    float* dC,
    int M,
    int N, 
    int K,
    cudaStream_t stream
){
    const float alpha = 1.0f;
    const float beta = 0.0f;
    const dim3 block(kBlockSize * kBlockSize);
    const dim3 grid(
        (M + kBlockSize-1)/ kBlockSize,
        (N + kBlockSize - 1) / kBlockSize
    );

    segmm_global_mem_coalease<kBlockSize><<<grid, block, 0, stream>>>(M, N, K, alpha, dA, dB, beta, dC);
}