#include <iostream>
#include <stdio.h>
#include <math.h>

/*Finite Element Library*/
#include "finite_element.h"
#define MTPB 512 //Threads allow per block (CUDA toolkit 10 allows 1024)

/*Finite Element Solution to Example*/

/*Linear Solver*/

    // this kernel will Parallely solves the tridiagonal sysytem
	__global__ 
	void pcr(float* a_s, float* b_s, float* c_s, float* d1, int k, int n)// k = ceil(log_2(n))
           {
    // i is the equation number
		int i = threadIdx.x + (blockDim.x *  blockIdx.x);
		
    // allocating memory in the shared memory for lower diag, diag, upper diag and right hand vector  
	extern __shared__ float memorySize[];
	
	float* a = memorySize;	
	float* b = (float*)&a[n];
	float* c = (float*)&b[n];
	float* d = (float*)&c[n];	

    // initialize the coffecient arrys from the globally define tridiagonal matrix       
			if(i == 0)
				a[i] = 0;
			else if(i == n - 1)
				a[i] = 0;
			else	
				a[i] = a_s[i-1];
            b[i] = b_s[i];
            c[i] = c_s[i];
            d[i] = d1[i];
    // waiting for every thread to finish above initialization
            __syncthreads();
			
    // executing all PCR steps by for loop  
            float alfa, beta, a1, c1, d2, a2, c2, d3;
            for(int j = 0; j<k ; j++) // k = ceil(log_2(n))
             {
				extern __shared__ float b_memorySize[];
				extern __shared__ float c_memorySize[];
				extern __shared__ float d_memorySize[];
			
    // claculating upper(p) and lower(q) equation numbers for each step by the current equation 
               int p = i - powf(2, j);
               int q = i + powf(2, j);

    // making one new equation from three equations and calculating new coefficients for new equation
               if(p>=0)
                 { 
                 alfa = -a[i]/b[p];
                  a1  =  alfa * a[p]; 
                  c1  =  alfa * c[p];
                  d2  =  alfa * d[p]; 
                 } 
    	   else
                 {
                 a1  = 0; 
                 c1  = 0;
                 d2  = 0;
                 }

              if(q<=n-1)
                 { 
                 beta = -c[i]/b[q];
                 a2   = beta * a[q];
                 c2   = beta * c[q];
                 d3   = beta * d[q];  
                 } 
    	   else
                 {
                  a2 = 0;
                  c2 = 0;
                  d3 = 0; 
                 }

    // waiting for each thread to finish the making of new equation 
              __syncthreads();

    // writing down the new coefficients in place of the coefficients of current equation
              a[i] = a1;
              b[i] = b[i] + c1 + a2;
              c[i] = c2;
              d[i] = d[i] + d3 + d2; 
	
    // waiting for each thread to finsh writing down the new coefficients before starting the new step                  
             __syncthreads();
             }
	
			d1[i] = d[i]/b[i]; 
			
           }
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
	/*
	std::cout << std::endl << "A:";
	printM(A, N, N);
	std::cout << std::endl << "B:" << std::endl;
	for (int i = 0; i < N; i++)
		std::cout << b[i] << "\t";
	std::cout << std::endl;
	*/
	/*Store linear solver values in an array*/
	float* xa = new float [N];
	
	xa = linearSolver(A, b, N, N);
	//std::cout.precision(2);
	/*
	std::cout << std::endl << "U:";
	std::cout << std::endl;
	for (int i = 0; i < N; i++)
		std::cout << xa[i] << "\t";
	std::cout << std::endl;*/
}

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

	float *a_device, *b_device, *c_device, *d_device; //Space to store on GPU
	float* x_host = new float[cols]; //Space to store solutions on host
	//Set time variables
	float tiempo_computo;
	cudaEvent_t inicio, alto;
	//Create temporal events
	cudaEventCreate(&inicio); cudaEventCreate(&alto); //Creamos los eventos
    cudaEventRecord(inicio); //Creamos una marca temporal, una especia de bandera 

	int k = ceil(log2f((float)cols)); //Number of steps on the linear solver
	const int NB = ceil( (float) cols/ (float)MTPB); //Number of blocks
	const int TPB = ceil( (float) cols / (float) NB); //Threads per block

	cudaMalloc((float**)&a_device, (cols -1) * sizeof(float*));
	cudaMalloc((float**)&b_device,  cols* sizeof(float*));
	cudaMalloc((float**)&c_device, (cols -1) * sizeof(float*));
	cudaMalloc((float**)&d_device, 	cols * sizeof(float*));
	unsigned sharedMemorySize = 4 * cols * sizeof(float);

	//Copy data from host to device
	cudaMemcpy(a_device, 	a, 		(cols - 1) * sizeof(float*),  	cudaMemcpyHostToDevice);
	cudaMemcpy(b_device, 	b, 		 cols * sizeof(float*), 		cudaMemcpyHostToDevice);
	cudaMemcpy(c_device, 	c, 		(cols - 1) * sizeof(float*), 	cudaMemcpyHostToDevice);
	cudaMemcpy(d_device, 	d, 		 cols * sizeof(float*), 		cudaMemcpyHostToDevice);

	/*PCR linear ec.*/

	pcr<<<NB, TPB, sharedMemorySize>>>(a_device, b_device, c_device, d_device, k, cols);

	//Copy data from device to host
	cudaMemcpy(x_host, d_device, cols * sizeof(float), cudaMemcpyDeviceToHost);

	//Free device memory
	cudaFree(a_device); cudaFree(b_device); cudaFree(c_device); cudaFree(d_device);

	cudaEventRecord(alto); // Creamos una marca temporal, otra bandera
	cudaEventSynchronize(alto); // Bloquea la CPU para evitar que se continue con el programa hasta que se completen los eventos
	cudaEventElapsedTime(&tiempo_computo, inicio, alto); //Calcula el tiempo entre los eventos
	cudaEventDestroy(inicio); cudaEventDestroy(alto); // Se liberan los espacios  de los eventos para poder medir de nuevo más tarde

	std::cout << tiempo_computo << "\n";

 	return x_host;
}

/*Print Matrix function*/	/*Print Matrix*/
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
