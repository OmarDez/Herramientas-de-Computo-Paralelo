
#include <iostream>
#include <stdio.h>
#define sideMatrixSide 10
/*Finite Element Library*/
#include "finite_element.h"

int main(void) {

	float* x = new float[50];
	float* U = new float[50];
	int N = sideMatrixSide;
	calculate_FE_sol(x, U, N);

	return 0;
}