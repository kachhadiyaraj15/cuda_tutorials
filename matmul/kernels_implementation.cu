#include <cuda_runtime.h>

#include <cstdlib>
#include <iostream>
#include <vector>

#include "kernels/kernel_registry.cuh"
#include "utils/matmul_utils.cuh"

int main() {
    std::srand(0);

    const std::string result_file = "benchmark_results/03_kernel_shared_mem_blocking.csv";
    const int repeat = 20;

    CUDA_CHECK(cudaSetDevice(0));

    cudaEvent_t start = nullptr;
    cudaEvent_t stop = nullptr;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    const std::vector<KernelSpec> kernels = getRegisteredKernels();
    if (kernels.empty()) {
        std::cerr << "No kernels registered in kernels/kernel_registry.cu" << std::endl;
        return EXIT_FAILURE;
    }

    for (const GemmShape& shape : getTestShapes()) {
        const int M = shape.M;
        const int N = shape.N;
        const int K = shape.K;

        const size_t bytesA = static_cast<size_t>(M) * K * sizeof(float);
        const size_t bytesB = static_cast<size_t>(K) * N * sizeof(float);
        const size_t bytesC = static_cast<size_t>(M) * N * sizeof(float);

        std::vector<float> hA(M * K);
        std::vector<float> hB(K * N);
        std::vector<float> hC(M * N);

        randomizeMatrix(hA);
        randomizeMatrix(hB);
        zeroMatrix(hC);

        float* dA = nullptr;
        float* dB = nullptr;
        float* dC = nullptr;

        CUDA_CHECK(cudaMalloc(&dA, bytesA));
        CUDA_CHECK(cudaMalloc(&dB, bytesB));
        CUDA_CHECK(cudaMalloc(&dC, bytesC));

        CUDA_CHECK(cudaMemcpy(dA, hA.data(), bytesA, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(dB, hB.data(), bytesB, cudaMemcpyHostToDevice));

        for (const KernelSpec& kernel : kernels) {
            CUDA_CHECK(cudaMemset(dC, 0, bytesC));

            kernel.launch(dA, dB, dC, M, N, K, 0);
            CUDA_CHECK(cudaGetLastError());
            CUDA_CHECK(cudaDeviceSynchronize());

            CUDA_CHECK(cudaEventRecord(start));
            for (int i = 0; i < repeat; ++i) {
                kernel.launch(dA, dB, dC, M, N, K, 0);
            }
            CUDA_CHECK(cudaEventRecord(stop));
            CUDA_CHECK(cudaEventSynchronize(stop));
            CUDA_CHECK(cudaGetLastError());

            float total_ms = 0.0f;
            CUDA_CHECK(cudaEventElapsedTime(&total_ms, start, stop));

            const double avg_ms = static_cast<double>(total_ms) / static_cast<double>(repeat);
            const double gflops = computeGflops(M, N, K, avg_ms / 1000.0);

            appendCsvRow(result_file, kernel.name, M, N, K, avg_ms, gflops);
            std::cout << kernel.name
                      << " | M=" << M
                      << " N=" << N
                      << " K=" << K
                      << " | " << avg_ms << " ms"
                      << " | " << gflops << " GFLOP/s"
                      << std::endl;
        }

        CUDA_CHECK(cudaMemcpy(hC.data(), dC, bytesC, cudaMemcpyDeviceToHost));

        CUDA_CHECK(cudaFree(dA));
        CUDA_CHECK(cudaFree(dB));
        CUDA_CHECK(cudaFree(dC));
    }

    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    return 0;
}
