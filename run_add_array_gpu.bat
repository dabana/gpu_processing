DEL add_array_gpu.cu
COPY add_array_gpu.cpp add_array_gpu_copy.cpp
REN add_array_gpu_copy.cpp add_array_gpu.cu
nvcc add_array_gpu.cu -o add_array_gpu_cuda
nvprof ./add_array_gpu_cuda