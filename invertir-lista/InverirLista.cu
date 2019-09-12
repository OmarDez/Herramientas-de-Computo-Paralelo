#include <stdio.h>
#define N 4
#define TPB 32

__global__ void invertir(int *salida, int *lista ){

}

int main()
{
    printf( "Entre un lista de 4 para invertir: ");
    int *lista[4];
    int *salida;
    int i;
    for( i = 0 ; i < 4; i++){
        scanf("%d", &lista[i]);
    }
    cudaMallaloc(&salida, N*sizeof(int));
    invertir<<<N/TPB, TPB>>>(salida, lista);
    printf("La lista es: ");

    for (i = 0; i < 4; i++){
        printf("%d", lista[i]);
    }
}

