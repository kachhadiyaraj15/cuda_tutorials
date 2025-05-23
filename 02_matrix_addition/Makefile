# Makefile for CUDA Matrix Addition
NVCC = nvcc

# SOurce File 
SRC = matrix_addition_kernel.cu

# Executable Name
EXEC = matrix_addition_kernel

# Desired GPU compute capability
# Find your GPU's compute capability:
# - On Linux: `nvidia-smi` to see your GPU model, then check NVIDIA's CUDA GPUs page:
#   https://developer.nvidia.com/cuda-gpus
# - Or run `/usr/local/cuda/bin/nvcc --list-gpu-code` or check your GPU's specs.
# Common values: 70 (Volta), 75 (Turing), 80 (Ampere), 86 (Ampere, RTX 30 series), 89 (Ada Lovelace, RTX 40 series)
# If unsure, pick one slightly lower than your exact capability or let nvcc choose (default behavior if -arch is omitted).
COMPUTE_CAPABILITY = 86 # Example: For RTX 30 series. Adjust as needed.

# CUDA compilaion flasgs 
NVCC_FLAGS = -arch=sm_$(COMPUTE_CAPABILITY) -O3 --std=c++17

# All target: Compile the CUDA code
all : $(EXEC)

$(EXEC): $(SRC)
	$(NVCC) $(SRC) -o $(EXEC) $(NVCC_FLAGS)

# Run target: Execute the compiled program
run: $(EXEC)
	./$(EXEC)

# Clean target: Remove compiled files
clean:
	# 'del' for Windows, 'rm -f' for Unix-like systems.
	# The '|| true' and '>NUL 2>NUL' suppress errors if the files don't exist.
	# This ensures the clean command always succeeds even if some files are already gone.
	# If you want to keep results.txt, explicitly DO NOT include it here.
	rm -f $(EXEC).exe $(EXEC).exp $(EXEC).lib $(EXEC).obj || true
	# For macOS debug symbols (harmless on Windows):
	rm -rf *.dSYM