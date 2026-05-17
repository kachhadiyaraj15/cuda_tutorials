#include "kernels/3_kernel_shared_mem_blocking.cuh"


namespace{

constexpr int kBlockSize=32;

template <const int BLOCKSIZE>

__global__ void segmm_shared_mem_block(
    int M, int N, int K, 
    float alpha,
    const float* A,
    const float* B,
    float beta,
    float* C
){
    // the output blocj that we want to compute in this threadblock
    const int cRow = blockIdx.x;
    const int cCol = blockIdx.y;

    // allocate buffer for current block in fast shared mem 
    // shared mem is shared between all thread in a block 
    __shared__ float As[BLOCKSIZE * BLOCKSIZE];
    __shared__ float Bs[BLOCKSIZE * BLOCKSIZE];

    // the inner row & col that we're accessing in this thread
    const int threadCol = threadIdx.x % BLOCKSIZE;
    const int threadRow = threadIdx.x / BLOCKSIZE;

    // advance pointers to the starting positions 
    A += cRow * BLOCKSIZE * K; // row = cRow, col = 0
    B += cCol * BLOCKSIZE ; // row = 0 , col = cCOl
    C += cRow * BLOCKSIZE * N + cCol * BLOCKSIZE ; // row = cRow , col = cCol

    float tmp = 0.0;
    for (int bkIdx = 0; bkIdx < K; bkIdx += BLOCKSIZE){
        //Have each thread load one of the elements in A & B
        // Make the threadCol (=theradIdx.x) the consecutive index
        // to allow global memory access coalesing
        As[threadRow * BLOCKSIZE + threadCol] = A[threadRow * K + threadCol];
        Bs[threadRow * BLOCKSIZE + threadCol]= B[threadRow * N + threadCol];

        // block threads in this block until cache is fully populated 
        __syncthreads();
        A += BLOCKSIZE;
        B += BLOCKSIZE * N;

        // execute the dotproduct on the currently cached blcok 
        for (int dotIdx = 0 ; dotIdx < BLOCKSIZE; ++dotIdx){
            tmp += As[threadRow * BLOCKSIZE+ dotIdx] * Bs[dotIdx * BLOCKSIZE + threadCol];
        }

        // need to sync again at the end, to avoid faster threads
        // fetching the next block into the cache before slower threads are done 

        __syncthreads();
        C[threadRow * N + threadCol] = tmp;
    }
  }
}

void sharedMemSgemm(
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
        (M + kBlockSize - 1) / kBlockSize,
        (N + kBlockSize - 1) / kBlockSize
    );

    segmm_shared_mem_block<kBlockSize><<<grid, block, 0, stream>>>(M, N, K, alpha, dA, dB, beta, dC);
}