#include <stdio.h>
#define TPB 1024

__global__ void invertirLista(int *in, int *out){
    const int idxIn = threadIdx.x;
    const int idxOut = blockDim.x - 1 - idxIn;
    out[idxOut] =  in[idxIn];
}

int main(){
    unsigned int size =  TPB * sizeof(int);
    
    int* h_in = (int*) malloc(size);
    
    int i;
    for( i = 0; i < TPB; i++){
        h_in[i] =  i;
    }

    int *d_in; cudaMalloc((void**)&d_in, size);
    int *d_out; cudaMalloc((void**)&d_out, size);

    cudaMemcpy(d_in, h_in, size, cudaMemcpyHostToDevice);
    
    invertirLista<<<1, TPB>>>(d_in, d_out);
    
    int* h_out = (int*) malloc(size);

    cudaMemcpy(h_out, d_out, size, cudaMemcpyDeviceToHost);
    
    cudaFree(d_in); cudaFree(d_out);

    printf(" IN / OUT \n");

    for(i = 0; i < TPB; i++){
        printf(" %d / %d \n", h_in[i], h_out[i]);
    }
    
    free(h_in); free(h_out);
    
    return 0;
}