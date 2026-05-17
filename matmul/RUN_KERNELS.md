# Run SGEMM Kernels

Run these commands from PowerShell.

```powershell
# Go to the matmul project directory.
cd F:\cuda_tutorials\matmul

# Compile the benchmark program and kernel implementations into one executable.
nvcc -std=c++17 -I. kernels_implementation.cu kernels\1_naive_sgemm.cu kernels\2_kernel_global_mem_coalesce.cu kernels\kernel_registry.cu -o build\kernels_implementation.exe

# Run the compiled benchmark executable.
.\build\kernels_implementation.exe
```

Command details:

- `nvcc`: NVIDIA CUDA compiler.
- `-std=c++17`: compile using the C++17 standard.
- `-I.`: allow includes from the current `matmul` directory.
- `kernels_implementation.cu`: main benchmark file.
- `kernels\1_naive_sgemm.cu`: kernel 1 implementation.
- `kernels\2_kernel_global_mem_coalesce.cu`: kernel 2 implementation.
- `kernels\kernel_registry.cu`: selects/registers which kernel runs.
- `-o build\kernels_implementation.exe`: output executable path.

To choose the kernel, edit `kernels\kernel_registry.cu`:

```cpp
const int selected_kernel = 1; // run naive kernel
const int selected_kernel = 2; // run global memory coalesced kernel
```

Benchmark results are written to:

```text
benchmark_results\naive_kernel_results.csv
```
