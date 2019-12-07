#pragma once

#define DEBUG

/*Finite Element Solution Function*/
void calculate_FE_sol(float* x, float* U, int N);

/*Zeros for the 2D array*/
float** zeros_2d(int rows, int cols);

/*Print Matrix function*/
void printM(float** matrix, int cols, int rows);

/*Tri Diagonal Matrix Algorithm(a.k.a Thomas algorithm) solver*/
//void tdmaSolver(float* a, float* b, float* c, float* d, int nf, int bSize, float* x_device);

/*Linear Solver*/
float* linearSolver(float** A, float* d, int cols, int rows);