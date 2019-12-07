// finite_differences_sec.cpp : Este archivo contiene la función "main". La ejecución del programa comienza y termina ahí.
//
#pragma warning(disable:4996)

#include <math.h>
#include <stdio.h>
#include <iostream>

void ddParallel(float* out, float* in, int n, float h);
void diferencias(int N);
//void ddKernel(float* d_out, float* d_in, int size, float h);

int main()
{
	for(int N = 10; N < 1000000; N += 10)
		diferencias(N);
	/*
	FILE * outfile;
	outfile = fopen("ResultadosSecuencialGPU.csv", "w");
	for (int i = 1; i < N - 1; ++i) {
		fprintf(outfile, "%f,%f,%f,%f\n", x[i], u[i],
			result_parallel[i], result_parallel[i] + u[i]);
	}
	fclose(outfile);
	*/

	return 0;
}

void diferencias(int N){
	const float PI = 3.1415927;
	const float h = 2 * PI / N;
	float x[N] = { 0.0 };
	float u[N] = { 0.0 };
	float result_parallel[N] = { 0.0 };

	for (int i = 0; i < N; ++i) {
		x[i] = 2 * PI * i / N;
		u[i] = sinf(x[i]);
	}

	ddParallel(result_parallel, u, N, h);
}

__global__
void ddKernel(float* d_out, float* d_in, int size, float h){
	for(int i = 0 ; i < size; i++){
		if (i >= size) return;
		d_out[i] = (d_in[i - 1] - 2.f * d_in[i] + d_in[i + 1]) / (h * h);
	}
}

void ddParallel(float* out, float* in, int n, float h) {
	float* d_in = 0, *d_out = 0;
	
	float tiempo_computo;
	cudaEvent_t inicio, alto;

	cudaEventCreate(&inicio); cudaEventCreate(&alto); //Creamos los eventos
    cudaEventRecord(inicio); //Creamos una marca temporal, una especia de bandera 


	cudaMalloc(&d_in, n * sizeof(float));
	cudaMalloc(&d_out, n * sizeof(float));
	cudaMemcpy(d_in, in, n * sizeof(float), cudaMemcpyHostToDevice);

	ddKernel <<< 1, 1 >>> (d_out, d_in, n, h);

	cudaMemcpy(out, d_out, n * sizeof(float), cudaMemcpyDeviceToHost);
	cudaFree(d_in);
	cudaFree(d_out);

	cudaEventRecord(alto); // Creamos una marca temporal, otra bandera
	cudaEventSynchronize(alto); // Bloquea la CPU para evitar que se continue con el programa hasta que se completen los eventos
	cudaEventElapsedTime(&tiempo_computo, inicio, alto); //Calcula el tiempo entre los eventos
	cudaEventDestroy(inicio); cudaEventDestroy(alto); // Se liberan los espacios  de los eventos para poder medir de nuevo más tarde

	std::cout << tiempo_computo << "\n";
}
