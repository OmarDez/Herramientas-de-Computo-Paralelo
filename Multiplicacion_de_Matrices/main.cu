#include <stdio.h>
#include <math.h>
#include <iostream>

using namespace std;

#define TDM 3 // Tamño de la matriz (cuadrada)

__global__ void MultiplicarMatricesO1(float *matriz1_GPU, float *matriz2_GPU, float *matriz3_GPU, size_t pitch){   
    unsigned int idx = threadIdx.x + (blockDim.x *  blockIdx.x);
    const unsigned index = idx;
    const unsigned int k = idx / (TDM * TDM);
    idx -= (k * TDM * TDM);
    const unsigned int j = idx / TDM;
    const unsigned int i = idx % TDM;
    
    float *elementos_matriz1 = (float *) ((char*)matriz1_GPU + j * pitch);
    float *elementos_matriz2 = (float *) ((char*)matriz2_GPU + k * pitch);
    float *elementos_matriz3 = (float *) ((char*)matriz3_GPU + j * pitch);
    
    float a = elementos_matriz1[k] * elementos_matriz2[i]; 
    for(int x = 0; x < TDM; x ++){ 
        if(k == x)
            elementos_matriz3[i] += a;
        __syncthreads();
    }
    printf("\nthread: %i, i: %i, j: %i, k: %i", index,i,j,k);
    printf("\nthread: %i, Elemento1: %f \t Elemento2: %f",index, elementos_matriz1[k], elementos_matriz2[i]);
    printf("\nthread: %i, valor agregado: %f",index, a);
    
}
int main(){
    const unsigned int NDH = pow(TDM,3); //numero de hilos
    const unsigned int numero_bloques =  ceil( (float) NDH / (float) TDM ); 
    const unsigned int hilos_bloque = ceil( (float) NDH / (float) numero_bloques );
    
    float matriz1_host[TDM][TDM];
    float matriz2_host[TDM][TDM];

    for(int i = 0; i < TDM; i++){
        for(int j = 0; j < TDM; j++){
            matriz1_host[i][j] = (int)(i + j);//(float)(rand() % 10);
            matriz2_host[i][j] = (int)(i + j);//(float)(rand() % 10);
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

    float *matriz1_GPU; cudaMallocPitch(&matriz1_GPU, &pitch, TDM * sizeof(float), TDM );
    float *matriz2_GPU; cudaMallocPitch(&matriz2_GPU, &pitch, TDM * sizeof(float), TDM );
    float *matriz3_GPU; cudaMallocPitch(&matriz3_GPU, &pitch, TDM * sizeof(float), TDM );

    cudaMemcpy2D(matriz1_GPU, pitch, matriz1_host, TDM * sizeof(float), TDM * sizeof(float), TDM, cudaMemcpyHostToDevice);
    cudaMemcpy2D(matriz2_GPU, pitch, matriz2_host, TDM * sizeof(float), TDM * sizeof(float), TDM, cudaMemcpyHostToDevice);

    MultiplicarMatricesO1<<<numero_bloques, hilos_bloque>>>(matriz1_GPU, matriz2_GPU, matriz3_GPU, pitch);
    cudaDeviceSynchronize();
    
    float matriz_salida[TDM][TDM];
    cudaMemcpy2D(matriz_salida, TDM * sizeof(float), matriz3_GPU, pitch, TDM * sizeof(float), TDM, cudaMemcpyDeviceToHost);
    
    cout.precision(3);

    cout << "\nMatriz Multiplicada" << endl;
    for(int i = 0; i < TDM; i++){
        for(int j = 0; j < TDM; j++){
            cout << matriz_salida[i][j] << "\t";
        }
        cout << "\n";
    }
    //free(matriz1_host); free(matriz2_host);
    cudaFree(matriz1_GPU); cudaFree(matriz2_GPU), cudaFree(matriz3_GPU);

    

    return 0;
}