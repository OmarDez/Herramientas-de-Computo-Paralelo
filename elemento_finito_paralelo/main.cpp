
#include <iostream>
#include <stdio.h>
/*Finite Element Library*/
#include "finite_element.h"

void times(int i) {

	float* x = new float[50];
	float* U = new float[50];
	int N = i;
	calculate_FE_sol(x, U, N);
}

int main(){
	for (int i = 10; i < 2300; i+=100)
	{
		times(i);
	}
	
}

