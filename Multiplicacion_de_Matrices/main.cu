#include <stdio.h>
#include <math.h>
#include <iostream>

using namespace std;

#define TDM 3 // Tamño de la matriz (cuadrada)

__global__ void MultiplicarMatricesO1(float *matriz1_GPU, float *matriz2_GPU, float *matriz3_GPU, int pitch){
    unsigned int idx = threadIdx.x + (blockDim.x *  blockIdx.x);
    const unsigned index = idx;
    const unsigned int k = idx / (TDM * TDM);
    idx -= (k * TDM * TDM);
    const unsigned int j = idx / TDM;
    const unsigned int i = idx % TDM;

    float *elementos_matriz1 = (float *) ((char*)matriz1_GPU + (j + k) * pitch);
    float *elementos_matriz2 = (float *) ((char*)matriz2_GPU + i * pitch);
    float *elementos_matriz3 = (float *) ((char*)matriz3_GPU + j * pitch);
    elementos_matriz3[j] += elementos_matriz1[i] * elementos_matriz2[j+k]; 
    printf("\nthread: %i, Elemento 1: %i, Elemento2: %i",index, *(matriz2_GPU + i), matriz2_GPU[j]);
    printf("\nthread: %i, valor agregado: %i",index, elementos_matriz3[j]);
    printf("\nthread: %i, i: %i, j: %i, k: %i", index,i,j,k);
}
int main(){
    const unsigned int NDH = pow(TDM,3); //numero de hilos
    const unsigned int numero_bloques =  ceil( (float) NDH / (float) TDM ); 
    const unsigned int hilos_bloque = ceil( (float) NDH / (float) numero_bloques );
    
    
    float** matriz1_host = (float**)malloc(TDM * sizeof(float*));
    float** matriz2_host = (float**)malloc(TDM * sizeof(float*));

    for(int n = 0; n<TDM; n++){
        matriz1_host[n] = (float*)malloc(TDM * sizeof(float));
        matriz2_host[n] = (float*)malloc(TDM * sizeof(float));
    }
    
    for(int i = 0; i < TDM; i++){
        for(int j = 0; j < TDM; j++){
            *(*(matriz1_host + i) + j) = rand() % 10;
            *(*(matriz2_host + i) + j) = rand() % 10;
        }
    }
    cout << "Matrices a multiplicar \nMatriz 1" << endl;
    for(int i = 0; i < TDM; i++){
        for(int j = 0; j < TDM; j++){
            cout << *(*(matriz1_host + i) + j) << "\t";
        }
        cout << "\n";
    }
    cout << "\nMatriz 2" << endl;
    for(int i = 0; i < TDM; i++){
        for(int j = 0; j < TDM; j++){
            cout << *(*(matriz2_host + i) + j) << '\t';
        }
        cout << "\n";
    }

    size_t pitch;

    float *matriz1_GPU; cudaMallocPitch((float**)&matriz1_GPU, &pitch,TDM * sizeof(float), TDM );
    float *matriz2_GPU; cudaMallocPitch((float**)&matriz2_GPU, &pitch,TDM * sizeof(float), TDM );
    float *matriz3_GPU; cudaMallocPitch((float**)&matriz3_GPU, &pitch,TDM * sizeof(float), TDM );

    cudaMemcpy2D(matriz1_GPU, TDM * sizeof(float), matriz1_host, pitch, TDM * sizeof(float), TDM, cudaMemcpyHostToDevice);
    cudaMemcpy2D(matriz2_GPU, TDM * sizeof(float), matriz2_host, pitch, TDM * sizeof(float), TDM, cudaMemcpyHostToDevice);

    MultiplicarMatricesO1<<<numero_bloques, hilos_bloque>>>(matriz1_GPU, matriz2_GPU, matriz3_GPU, pitch);
    
    float matriz_salida[TDM][TDM];
    cudaMemcpy2D(matriz_salida, TDM * sizeof(float), matriz3_GPU, pitch, TDM * sizeof(float), TDM, cudaMemcpyDeviceToHost);
    
    cout.precision(3);

    cout << "\nMatriz Multiplicada" << endl;
    for(int i = 0; i < TDM; i++){
        for(int j = 0; j < TDM; j++){
            cout << *(*(matriz_salida + i) + j) << '\t';
        }
        cout << "\n";
    }
    free(matriz1_host); free(matriz2_host);
    cudaFree(matriz1_GPU); cudaFree(matriz2_GPU), cudaFree(matriz3_GPU);

    

    return 0;
}