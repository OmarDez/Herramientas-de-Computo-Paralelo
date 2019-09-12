#include <stdio.h>
#include <math.h>
#include <iostream>
using namespace std;

#define TDB 1024  //Tamaño del bloque 
#define hy 0.34
#define hx 0.34
#define LT 1 //lado tuberia

__device__ double my_floor(double num) {
    if (num >= LLONG_MAX || num <= LLONG_MIN || num != num) {
        return num;
    }
    int n = (int)num;
    double d = (double)n;
    if (d == num || num >= 0)
        return d;
    else
        return d - 1;
}

__global__ void crearMalla(float *matriz, float *coeficientes_GPU, int nodos_x, int nodos_y, int pitch){
    const unsigned int idx = threadIdx.x;
    const unsigned int i = my_floor(idx / nodos_x);
    const unsigned int j = idx % nodos_x;
    const unsigned int n = nodos_x * nodos_y; 
    int k = 0, l = 0, columna = 0;
    printf("thread:%i \n", idx);

    float *row_a  = (float *) ((char*)matriz + idx * pitch);
    while(k < nodos_x && columna < n ){
      
        while(l < nodos_y){
            if( k == i - 1 && l == j ) 
                row_a[columna] = *(coeficientes_GPU + 0);
            else if( k == i + 1 && l == j ) 
                row_a[columna] = *(coeficientes_GPU + 1);
            else if( k == i && l == j ) 
                row_a[columna] = *(coeficientes_GPU + 2);    //
            else if( k == i && l == j - 1 ) 
                row_a[columna] = *(coeficientes_GPU + 3);
            else if( k == i && l == j + 1 ) 
                row_a[columna] = *(coeficientes_GPU + 4);
            else 
                row_a[columna] = 0;

            columna++;

            l++;
        }
        l = 0;

        if(k < nodos_x) k++;
        else k = 0;
    }

} 


int main(){
    const unsigned int nodos_x = ceil( (float)LT / (float)hx );
    const unsigned int nodos_y = ceil( (float)LT / (float)hy );
    const unsigned int NDH =  nodos_x * nodos_y; 
    const unsigned int numero_bloques =  ceil( (float) NDH / (float) TDB );
    const unsigned int hilos_bloque = ceil( (float) NDH / (float) numero_bloques );

    cout << " Se lanzaran " << numero_bloques << " bloque(s) de " << hilos_bloque << " hilos cada uno." << endl;

    float* coeficientes__HOST =  (float*) malloc(5);
    *(coeficientes__HOST + 0 )=  1/(pow(hx,2)); //(i-1, j)
    *(coeficientes__HOST + 1) =  1/(pow(hx,2)); //(i+1, j)
    *(coeficientes__HOST + 2) = -2 *( (1/(pow(hx,2))) + (1/(pow(hy,2)))); //(i, j)
    *(coeficientes__HOST + 3) =  1/(pow(hy,2)); //(i, j+1)
    *(coeficientes__HOST + 4) =  1/(pow(hy,2)); //(i, j+1)

    //for(unsigned x = 0 ; x < 5 ; ++x) cout << *(coeficientes__HOST + x) << "\n";
    
    size_t pitch;

    float *malla_salida_device; cudaMallocPitch((float**)&malla_salida_device, &pitch,NDH, NDH * sizeof(float));

    float *coeficientes_GPU; cudaMalloc((void**)&coeficientes_GPU, 5 * sizeof(float));

    cudaMemcpy(coeficientes_GPU, coeficientes__HOST, 5 * sizeof(float), cudaMemcpyHostToDevice);

    crearMalla<<<numero_bloques, hilos_bloque>>>(malla_salida_device, coeficientes_GPU, nodos_x, nodos_y, pitch);
    
    float malla_salida_host[NDH][NDH];
      
    cudaMemcpy2D(malla_salida_host, NDH * sizeof(float), malla_salida_device, pitch, NDH * sizeof(float), NDH, cudaMemcpyDeviceToHost);

    cudaFree(malla_salida_device); cudaFree(coeficientes_GPU);
    
    cout.precision(3);

    cout << "\n Nodos x: " << nodos_x << ", Nodos y: " << nodos_y << ".\n" <<endl; 
    for( int i = 0; i < NDH; i++){
        for(int j = 0 ; j < NDH ; j++)
            cout <<   malla_salida_host[i][j] << "\t";
        cout << "\n";
        }

    free(coeficientes__HOST); 

    return 0;
} 