#include "kernels/kernel_registry.cuh"

#include "kernels/1_naive_sgemm.cuh"
#include "kernels/2_kernel_global_mem_coalesce.cuh"
#include "kernels/3_kernel_shared_mem_blocking.cuh"
#include "kernels/4_kernel_1D_blocktiling.cuh"
#include "kernels/5_kernel_2D_blocktiling.cuh"

std::vector<KernelSpec> getRegisteredKernels() {
    const int selected_kernel = 5;
    if constexpr (selected_kernel == 1){
        return {
            {"navie", launchNaiveSgemm}
        };
    }
    else if constexpr(selected_kernel == 2){
        return {
            {"coalesced", coaleasceSgemm}
        };
    }
    else if constexpr(selected_kernel == 3){
        return {
            {"sharedMemory", sharedMemSgemm}
        };
    }
    else if constexpr(selected_kernel == 4){
        return {
            {"1DBlockTiling", sgemm1DBlocktiling}
        };
    }
    else if constexpr(selected_kernel == 5){
        return {
            {"2DBlockTiling", sgemm2DBlocktiling}
        };
    }
    else{
        return {};
    }
}

