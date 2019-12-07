
#include <iostream>
#include <stdio.h>

/*Finite Element Library*/
#include "finite_element.h"

int main(void) {

	float* x = new float[50];
	float* U = new float[50];
	int N = 60;
	calculate_FE_sol(x, U, N);

	return 0;
}