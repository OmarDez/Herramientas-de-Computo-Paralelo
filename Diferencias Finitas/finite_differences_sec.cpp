// finite_differences_sec.cpp : Este archivo contiene la función "main". La ejecución del programa comienza y termina ahí.
//
#pragma warning(disable:4996)

#include <math.h>
#include <stdio.h>
#include <iostream>

void ddParallel(float* out, float* in, int n, float h);
void ddKernel(float* d_out, float* d_in, int size, float h, int i);

int main()
{
	const float PI = 3.1415927;
	const int N = 150;
	const float h = 2 * PI / N;
	float x[N] = { 0.0 };
	float u[N] = { 0.0 };
	float result_parallel[N] = { 0.0 };

	for (int i = 0; i < N; ++i) {
		x[i] = 2 * PI * i / N;
		u[i] = sinf(x[i]);
	}

	ddParallel(result_parallel, u, N, h);

	FILE * outfile;
	outfile = fopen("ResultadosSecuancial.csv", "w");
	for (int i = 1; i < N - 1; ++i) {
		fprintf(outfile, "%f,%f,%f,%f\n", x[i], u[i],
			result_parallel[i], result_parallel[i] + u[i]);
	}
	fclose(outfile);

	std::cout << "Hello World!\n";

	return 0;
}

void ddParallel(float* out, float* in, int n, float h) {

	for (int i = 0 ; i < n ; i++) {
		ddKernel(out, in, n, h, i);
	}
}

void ddKernel(float* d_out, float* d_in, int size, float h, int i){
	if (i >= size) return;
	d_out[i] = (d_in[i - 1] - 2.f * d_in[i] + d_in[i + 1]) / (h * h);
}