#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <iostream>
#include <stdio.h>
#include <cstdlib>
#include <ctime>

#define max 20
#define min 0
#define dato 10

using namespace std;

__global__ void busqueda_bin(int* x, int *a, int* flag)
{
    int i = threadIdx.x;
	if (*(a + i) == *x)
		*(flag + i) = 1;
	else
		*(flag + i) = 0;
}

int main()
{
    int* a;
    int* x;
    int* flag;

	int* d_flag = 0;
	int* dev_a = 0;
	int* b = 0;

	x = new int[1];
	a = new int[max];
	flag = new int[max];

	*x = dato;

	cout << "Busqueda Binaria" << endl << "Dato: " << *x << endl << "Datos:\t";

	srand(time(0));

	for (int i = min; i < max; i++)
		* (a + i) = rand()%20;

	for (int i = min; i < max; i++)
		cout << *(a + i) << "\t";

	cudaError_t cudaStatus;
	cudaStatus = cudaSetDevice(0);
	    if (cudaStatus != cudaSuccess) {
	        fprintf(stderr, "cudasetdevice failed!  do you have a cuda-capable gpu installed?");
	        goto Error;
	    }

	//Reservar memoria en GPU
	cudaStatus = cudaMalloc((void**)& d_flag, max * sizeof(int));
		if (cudaStatus != cudaSuccess) {
		    fprintf(stderr, "cudaMalloc failed!");
		    goto Error;
		}

	cudaStatus = cudaMalloc((void**)& dev_a, max * sizeof(int));
		if (cudaStatus != cudaSuccess) {
		    fprintf(stderr, "cudaMalloc failed!");
		    goto Error;
		}

	cudaStatus = cudaMalloc((void**)& b, sizeof(int));
		if (cudaStatus != cudaSuccess) {
			fprintf(stderr, "cudaMalloc failed!");
			goto Error;
		}

	// Copy input vectors from host memory to GPU buffers.
	cudaStatus = cudaMemcpy(dev_a, a, max * sizeof(int), cudaMemcpyHostToDevice);
	    if (cudaStatus != cudaSuccess) {
	        fprintf(stderr, "cudaMemcpy failed!");
	        goto Error;
	    }

	cudaStatus = cudaMemcpy(b, x, sizeof(int), cudaMemcpyHostToDevice);
		if (cudaStatus != cudaSuccess) {
			fprintf(stderr, "cudaMemcpy failed!");
			goto Error;
		}
	
	//Launch a kernel on the GPU with one thread for each element.
	busqueda_bin<<<1, max>>>(b, dev_a, d_flag);

	// Check for any errors launching the kernel
    cudaStatus = cudaGetLastError();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "addKernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
        goto Error;
    }

	// cudaDeviceSynchronize waits for the kernel to finish, and returns
    // any errors encountered during the launch.
    cudaStatus = cudaDeviceSynchronize();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching addKernel!\n", cudaStatus);
        goto Error;
    }

    // Copy output vector from GPU buffer to host memory.
    cudaStatus = cudaMemcpy(flag, d_flag, max * sizeof(int), cudaMemcpyDeviceToHost);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }

Error:
	cudaFree(b);
	cudaFree(dev_a);
	cudaFree(d_flag);

	cout << endl << "Flag:\t";

	for (int i = min; i < max; i++)
		cout << *(flag + i) << "\t";

	delete[] a, flag;

    return 0;
}

