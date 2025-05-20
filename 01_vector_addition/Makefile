TARGET = test_cuda
SRCS = vector_addition.cpp vector_addition_kernel.cu
CXXFLAGS = -O1
NVCC = nvcc

all:
	$(NVCC) $(CXXFLAGS) $(SRCS) -o $(TARGET)

clean:
	rm -f $(TARGET) results.txt