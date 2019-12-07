#include <iostream>
#include <stdio.h>
#include <math.h>

/*Finite Element Library*/
#include "finite_element.h"

/*Finite Element Solution to Example*/
void calculate_FE_sol (float* x, float* U, int N) {
	
	/*Matrix A 2D*/
	float** A;
	/*Zeros to all elements in the A Matrix*/
	A = zeros_2d(N, N);

	/*Zeros to b 1D array*/
	float* b =  new float [N];
	for (int i = 0; i < N ; i++)
		b[i] = 0;

	/*Caculating element length*/
	float h = 1.f / N;

	/*Local Contributions*/
	float A_local[2][2] = { {1 / h, -1 / h}, {-1 / h, 1 / h } };
	float b_local[2] = { h, h };


	/*Loop over the elements calculating local contributions
	and incrementing the global linear systems*/
	for (int k = 0; k < N; k++) {
		//Generate Tridiagonal from local contributions
		A[k][k] += A_local[0][0];
		A[k][k + 1] += A_local[0][1];
		if (k != N - 1) {
			A[k + 1][k] += A_local[1][0];
			A[k + 1][k + 1] += A_local[1][1];
			b[k + 1] += b_local[1];
		}
		b[k] += b_local[0];
	}

	/*Set Dirichlet boundary conditions*/
	for (int i = 0; i < N; i++) {
		A[0][i] = (i == 0) ? 1.f : 0.f;
		A[N - 1][i] = (i == N - 1 )? 1.f : 0.f;
	}

	b[0] = 0;
	b[N - 1] = 0;

	/*Print Matrix*/
#ifdef DEBUG
	std::cout << std::endl << "A:";
	printM(A, N, N);
	std::cout << std::endl << "B:" << std::endl;
	for (int i = 0; i < N; i++)
		std::cout << b[i] << "\t";
	std::cout << std::endl;
#endif

	/*Store linear solver values in an array*/
	float* xa = new float [N];
	/*Soleve linear ec.*/
	xa = linearSolver(A, b, N, N);

	std::cout << std::endl << "U:";
	std::cout << std::endl;
	for (int i = 0; i < N; i++)
		std::cout << xa[i] << "\t";
	std::cout << std::endl;
}

/*Linear Solver*/
float* linearSolver(float** A, float* d, int cols, int rows) {
	/*a diagonal*/
	float* a = new float[cols - 1];
	/*b diagonal*/
	float* b = new float[cols];
	/*c diagonal*/
	float* c = new float[cols - 1];

	for (int i = 0; i < cols; i++) {
		if (i < cols - 1)
			a[i] = A[i + 1][i];
		b[i] = A[i][i];
		if (i <= cols - 2)
			c[i] = A[i][i + 1];
	}

	return tdmaSolver(a, b, c, d, cols, cols);
}

/*Tri Diagonal Matrix Algorithm(a.k.a Thomas algorithm) solver*/
float* tdmaSolver(float* a, float* b, float* c, float* d, int nf, int bSize) {
	
	float mc;
	
	/*Initialization of diagonal sweep in 1*/
	for (int i = 1; i < nf; i++) {
		mc = a[i - 1] / b[i - 1];
		b[i] = b[i] - mc * c[i - 1];
		d[i] = d[i] - mc * d[i - 1];
	}

	/*New Variable for Computing*/
	float* x = new float[bSize];
	/*Copy b to x array*/
	for (int i = 0; i < bSize; i++)
		x[i] = b[i];

	x[bSize-1] = d[nf - 1] / b[bSize - 1];

	for (int il = nf - 2; il >= 0; il--)
		x[il] = (d[il] - c[il] * x[il + 1]) / b[il];

	return x;
}

/*Print Matrix function*/
void printM(float** matrix, int cols, int rows) {
	float m;
	printf("\n");
	for (int i = 0; i < rows; i++) {
		for (int j = 0; j < cols; j++) {
			m = matrix[i][j];
			if (m < 0)
				printf("%.5f \t", m);
			else
				printf(" %.5f \t", m);
		}
		printf("\n");
	}
}

/*Zeros for the 2D array*/
float** zeros_2d(int rows, int cols) {
	
	/*Dynamic Memory assignment*/
	float** A = new float* [rows];
	for (int i = 0; i < rows ; i++) 
		A[i] = new float [cols];

	/*Zeros to each element of the 2D array*/
	for (int i = 0; i < rows ; i++)
		for (int j = 0; j < cols ; j++)
			A[i][j] = 0 ;

	return A;
}
