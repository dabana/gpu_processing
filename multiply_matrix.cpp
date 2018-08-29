//MATRIX MULTIPLICATION: used Cpp for better performances

#include <iostream>
#include <math.h>
#include <ctime>

void matmult(int N, int M, int M2, float **a1, float **a2, float **out)
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

void initialize_mat(int H, int W, float **array, float value)
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

	//Allocate memory to arrays (to be fair in time calculating I should also time the memory allocation)
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


	// initialize matrices
	initialize_mat(H, W, array1, 1.0f);
	initialize_mat(W, W2, array2, 2.0f);
	initialize_mat(H, W2, array3, 0.0f);

	//Run and time multiplication of matrices
	clock_t begin = clock();
	matmult(H, W, W2, array1, array2, array3);
	clock_t end = clock();
	double elapsed_secs = double(end - begin) / CLOCKS_PER_SEC;

	cout << "The two first elements of array1 are " << array1[0][0] << " and " << array1[1][0] << endl;
	cout << "The two first elements of array2 are " << array2[0][0] << " and " << array2[1][0] << endl;
	cout << "The two first elements of array3 are " << array3[0][0] << " and " << array3[0][0] << endl;
	cout << "It took " << elapsed_secs << " seconds to multiply matrices." << endl;


	// Free memory
	delete [] array1;
	delete [] array2;
	delete [] array3;

	return 0;
}