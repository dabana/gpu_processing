//MATRIX MULTIPLICATION: used Cpp for better performances

//Part II: Parallelization using a cude GPU

#include <iostream>
#include <math.h>
#include <ctime>

__global__
//GPU version of matrice multiplication (did not have time to finish)
void matmult(int N, int M, int M2, float **a1, float **a2, float **out, int pitch)
{
	for (int i = 0; i < N; ++i)
	{
		for (int j = 0; j < M; ++j)
		{
			for (int k = 0; k < M2; ++k)
			{
				out[i][j] += a1[i][k] * a2[k][i];
			}
		}
	}
}

//GPU version of matrix multiplication (did not have time to finish)
void initialize_mat(int H, int W, float **array, float value, int pitch)
{
	for (int i = 0; i < H; ++i)
	{
		for (int j = 0; j < W; ++j)
		{
			array[i][j] = value;
		}
	}
}

int main(void)
{
	using namespace std;

	int H = 1<<10; // 1028 elements max! need dynamic allocation because I bust the stack
	int W = 1<<10; // 1028 elements max! need dynamic allocation because I bust the stack
	int W2 = 1<<10; // 1028 elements max! need dynamic allocation because I bust the stack

	//Memory allocation before GPU 
	//(to be fair in time calculating I should also time the memory allocation)
	/*
	float **array1;
	array1 = new float *[H];
	for (int i = 0; i < H; i++)
	{
		array1[i] = new float[W];
	}

	float **array2;
	array2 = new float *[W];
	for (int i = 0; i < W; i++)
	{
		array2[i] = new float[W2];
	}

	float **array3;
	array3 = new float *[H];
	for (int i = 0; i < H; i++)
	{
		array3[i] = new float[W2];
	}
	*/

	//Memory allocation for GPU (cuda)
	size_t pitch
	float* array1Ptr;
	float* array2Ptr;
	float* array3Ptr;
	cudaMallocPitch((void**) &array1Ptr, &pitch, W * sizeof(float), H);
	cudaMallocPitch((void**) &array2Ptr, &pitch, W2 * sizeof(float), W);
	cudaMallocPitch((void**) &array3Ptr, &pitch, W2 * sizeof(float), H);

	// Run initialization on the GPU
	//(to be fair in time calculation I should also time the initialization)
	int Nthreads = 1<<10; // number of thread per block,i.e BlockDim, cannot exceed 1<<10
	int Nblocks = 1<<8;
	initialize_mat<<<Nblocks, Nthreads>>>(H, W, array1Ptr, 1.0f, pitch); //need a pitch for device code memory access
	initialize_mat<<<Nblocks, Nthreads>>>(W, W2, array2Ptr, 2.0f, pitch); //need a pitch for device code memory access
	initialize_mat<<<Nblocks, Nthreads>>>(H, W2, array3Ptr, 0.0f, pitch); //need a pitch for device code memory access

	//Wait for GPU to finish before accessing on host
	//(I might need this, but it might also slow down the execution. Some tests need to be done but no time...)
	cudaDeviceSynchronize();

	
	// Run matrix multiplication on the CPU and calculate time
	clock_t begin = clock();
	matmult<<<Nblocks, Hthreads>>>(H, W, W2, array1, array2, array3, pitch);//need a pitch for device code memory access
	clock_t end = clock();
	double elapsed_secs = double(end - begin) / CLOCKS_PER_SEC;

	//Print the result (This might look different with the GPU implementation)
	cout << "The two first elements of array1 are " << array1[0][0] << " and " << array1[1][0] << endl;
	cout << "The two first elements of array2 are " << array2[0][0] << " and " << array2[1][0] << endl;
	cout << "The two first elements of array3 are " << array3[0][0] << " and " << array3[0][0] << endl;
	cout << "It took " << elapsed_secs << " seconds to multiply matrices." << endl;


	// Free memory on GPU
	cudaFree(array1Ptr);
	cudaFree(array2Ptr);
	cudaFree(array3Ptr);

	return 0;
}