#include <stdio.h>
#include <math.h>

__global__ void kernelA(int *A, int *x, int *b, int N){
  int tId = threadIdx.x + blockIdx.x * blockDim.x;
  for(int k=0;k<1e4;k++){
    atomicAdd(&b[k],A[(int)(k*1e4+tId)]*x[tId]);
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

  int *CPU_x = (int *) malloc(1e4 * sizeof (int));
  int *CPU_A = (int *) malloc(1e8 * sizeof (int));

  for(int k = 0; k < n; k++){
    if(k < 1e4){
      CPU_x[k] = 1;
    }
    CPU_A[k] = 1;
  }  

  cudaMalloc(&GPU_x , 1e4 * sizeof(int));
  cudaMalloc(&GPU_b , 1e4 * sizeof(int));
  cudaMalloc(&GPU_A , 1e8 * sizeof(int));

  cudaMemcpy(GPU_A, CPU_A, 1e8 * sizeof(int), cudaMemcpyHostToDevice);
  cudaMemcpy(GPU_x, CPU_x, 1e4 * sizeof(int), cudaMemcpyHostToDevice);
  cudaMemset(GPU_b,0,1e4 * sizeof(int));

  kernelA<<<grid_size, block_size>>>(GPU_A, GPU_x, GPU_b, n);

  cudaMemcpy(CPU_x, GPU_b, 1e4 * sizeof(int), cudaMemcpyDeviceToHost);

  printf("%d\n", CPU_x[0]);

  cudaFree(GPU_x);
  cudaFree(GPU_b);
  cudaFree(GPU_A);
  free(CPU_x);
  free(CPU_A);

  return(0);
}
