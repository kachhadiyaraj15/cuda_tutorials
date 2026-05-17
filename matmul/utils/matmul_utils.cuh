#pragma once

#include <cuda_runtime.h>

#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

struct GemmShape {
    int M;
    int N;
    int K;
};

using SgemmKernelLauncher = void (*)(
    const float* dA,
    const float* dB,
    float* dC,
    int M,
    int N,
    int K,
    cudaStream_t stream);

struct KernelSpec {
    std::string name;
    SgemmKernelLauncher launch;
};

inline void checkCuda(cudaError_t err, const char* file, int line) {
    if (err != cudaSuccess) {
        std::cerr << "CUDA error at " << file << ":" << line
                  << " -> " << cudaGetErrorString(err) << std::endl;
        std::exit(EXIT_FAILURE);
    }
}

#define CUDA_CHECK(call) checkCuda((call), __FILE__, __LINE__)

inline std::vector<GemmShape> getTestShapes() {
    return {
        {128, 128, 128},
        {256, 256, 256},
        {512, 256, 128},
        {1024, 512, 256},
        {512, 1024, 512},
        {1024, 1024, 1024},
        {2048,1024,512},
        {2048,2048,2048},
        {4096,512,4096},
        {4096,1024,2048},
        {4096,2048,4096},
        {4096,4096,4096}
    };
}

inline void randomizeMatrix(std::vector<float>& matrix) {
    for (float& value : matrix) {
        value = static_cast<float>(std::rand() % 5) +
                0.01f * static_cast<float>(std::rand() % 5);
        if (std::rand() % 2 == 0) {
            value = -value;
        }
    }
}

inline void zeroMatrix(std::vector<float>& matrix) {
    for (float& value : matrix) {
        value = 0.0f;
    }
}

inline double computeGflops(int M, int N, int K, double seconds) {
    const double flops =
        2.0 * static_cast<double>(M) * static_cast<double>(N) * static_cast<double>(K);
    return (flops * 1e-9) / seconds;
}

inline void appendCsvRow(
    const std::string& path,
    const std::string& kernel_name,
    int M,
    int N,
    int K,
    double avg_ms,
    double gflops) {
    const std::filesystem::path output_path(path);
    if (!output_path.parent_path().empty()) {
        std::filesystem::create_directories(output_path.parent_path());
    }

    const bool write_header = !std::filesystem::exists(output_path);
    std::ofstream out(path, std::ios::app);
    if (!out.is_open()) {
        std::cerr << "Failed to open result file: " << path << std::endl;
        std::exit(EXIT_FAILURE);
    }

    if (write_header) {
        out << "kernel,M,N,K,avg_ms,gflops\n";
    }

    out << kernel_name << ","
        << M << ","
        << N << ","
        << K << ","
        << avg_ms << ","
        << gflops << "\n";
}

