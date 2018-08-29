#include <iostream>
#include <math.h>

__global__
void add(int n, float *x, float *y)
{

	//int index = threadIdx.x; // Thread index starts at zero
	int index = blockIdx.x * blockDim.x + threadIdx.x;
	int stride = blockDim.x * gridDim.x; // Number of threads per block * number of blocks
	for (int i = index; i < n; i += stride)
		y[i] = x[i] + y[i];
}

int main(void)
{
	int N = 1<<22; // about 1M elements ( 1 048 576 to be precise)

	float *x, float *y;

	//Allocate Unified Memory
	cudaMallocManaged(&x, N*sizeof(float));
	cudaMallocManaged(&y, N*sizeof(float));

	// intitialize x and y ont the host =
	for (int i = 0; i < N; ++i)
	{
		x[i] = 1.0f;
		y[i] = 2.0f;
	}

	// Run kernel on 1M elements on the GPU
	int Nthreads = 1<<10; // number of thread per block,i.e BlockDim, cannot exceed 1<<10
	int Nblocks = 1<<8;
	add<<<Nblocks, Nthreads>>>(N,x,y);

	//Wait for GPU to finish before accessing on host
	cudaDeviceSynchronize();

	// Free memory
	cudaFree(x);
	cudaFree(y);

	return 0;
}