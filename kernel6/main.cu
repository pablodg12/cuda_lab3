#include <stdio.h>
#include <math.h>

__constant__ int buff[10000];

__global__ void kernelCM(int *A, int *b, int N){
  int tId = threadIdx.x + blockIdx.x * blockDim.x;
  if(tId< N){
    for(int k=0; k < N; k++){
      b[tId] += A[(int)(tId*N+k)]*buff[k];
    }
  }
}

int main(int argc, char const *argv[])
{
  int n = 1e4;
  int block_size = 256;
  int grid_size = (int) ceil((float) n/ block_size);

  int *GPU_b;
  int *GPU_x;
  int *GPU_A;

  int *CPU_x = (int *) malloc(n * sizeof (int));
  int *CPU_A = (int *) malloc(2*n * sizeof (int));

  for(int k = 0; k < 2*n; k++){
    if(k < n){
      CPU_x[k] = 1;
    }
    CPU_A[k] = 1;
  }  

  cudaMalloc(&GPU_x , n * sizeof(int));
  cudaMalloc(&GPU_b , n * sizeof(int));
  cudaMalloc(&GPU_A , 2*n * sizeof(int));

  cudaMemcpy(GPU_A, CPU_A, 2*n * sizeof(int), cudaMemcpyHostToDevice);
  cudaMemcpyToSymbol(buff, CPU_x, n * sizeof(int), 0, cudaMemcpyHostToDevice);
  cudaMemset(GPU_b, 0, n * sizeof(int));

  kernelCM<<<grid_size, block_size, block_size*sizeof(int)>>>(GPU_A, GPU_b, n);

  cudaMemcpy(CPU_x, GPU_b, n * sizeof(int), cudaMemcpyDeviceToHost);

  //for(int k = 0; k< n; k++){
  //  printf("%d\n", CPU_x[k]);
  //}

  cudaFree(GPU_x);
  cudaFree(GPU_b);
  cudaFree(GPU_A);
  free(CPU_x);
  free(CPU_A);

  return(0);
}