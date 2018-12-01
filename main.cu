#include <stdio.h>
#include <math.h>


__global__ void kernelA(int *A, int *x, int *b, int N){
  int tId = threadIdx.x + blockIdx.x * blockDim.x;
  int mp = tId - 1e4*floor(tId/1e4);
  A[tId] = A[tId]*x[mp];
  if(tId < 2){
    printf("%d\n", A[tId]);
  }
  atomicAdd(&b[mp],A[tId]);
} 

int main(int argc, char const *argv[])
{
  int n = 1e8;
  int block_size = 256;
  int grid_size = (int) ceil((float) n/ block_size);

  int *GPU_b;
  int *GPU_x;
  int *GPU_A;

  int *CPU_x = (int *) malloc(n*0.5 * sizeof (int));
  int *CPU_A = (int *) malloc(n * sizeof (int));

  for(int k = 0; k < n; k++){
    if(k < n){
      CPU_x[k] = 1;
    }
    CPU_A[k] = 1;
  }  

  cudaMalloc(&GPU_x , n*0.5 * sizeof(int));
  cudaMalloc(&GPU_b , n*0.5 * sizeof(int));
  cudaMalloc(&GPU_A , n * sizeof(int));

  cudaMemcpy(GPU_A, CPU_A, n * sizeof(int), cudaMemcpyHostToDevice);
  cudaMemcpy(GPU_x, CPU_x, n*0.5 * sizeof(int), cudaMemcpyHostToDevice);
  cudaMemset(GPU_b,0,n*0.5 * sizeof(int));

  kernelA<<<grid_size, block_size>>>(GPU_A, GPU_x, GPU_b, n);

  cudaMemcpy(CPU_x, GPU_b, n*0.5 * sizeof(int), cudaMemcpyDeviceToHost);

  printf("%d\n", CPU_x[0]);

  cudaFree(GPU_x);
  cudaFree(GPU_b);
  cudaFree(GPU_A);
  free(CPU_x);
  free(CPU_x);

  return(0);
}
