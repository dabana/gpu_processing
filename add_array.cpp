#include <iostream>
#include <math.h>

void add(int n, float *x, float *y)
{
	for (int i = 0; i < n; i++)
		y[i] = x[i] + y[i];
}

int main(void)
{
	int N = 1<<20; // about 1M elements ( 1 048 576 to be precise)

	float *x = new float[N];
	float *y = new float[N];

	// initialize x and y on the host =
	for (int i = 0; i < N; ++i)
	{
		x[i] = 1.0f;
		y[i] = 2.0f;
	}

	// Run kernel on 1M elements on the CPU
	add(N,x,y);
	std::cout << "The two first elements of y are " << y[0] << " and " << y[1] << std::endl;

	// Free memory
	delete [] x;
	delete [] y;

	return 0;
}