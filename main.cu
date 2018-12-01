#include <stdio.h>
#include <math.h>


__global__ void kernelA(int *A, int *x, int *b, int N){
  int tId = threadIdx.x + blockIdx.x * blockDim.x;
  int mp = tId - 1e4*floor(tId/1e4);
  A[tId] = A[tId]*x[mp];
  atomicAdd(&b[mp],A[tId]);
} 

int main(int argc, char const *argv[])
{
  int *GPU_b;
  int *GPU_x;
  int *GPU_A;

  

  int n = 1e8;
  int block_size = 256;
  int grid_size = (int) ceil((float) n/ block_size);

  int *CPU_b = (int *) malloc(n*0.5 * sizeof (int));  

  cudaMalloc(&GPU_x , n*0.5 * sizeof(int));
  cudaMalloc(&GPU_b , n*0.5 * sizeof(int));
  cudaMalloc(&GPU_A , n * sizeof(int));

  cudaMemset(GPU_x, 1, 1e4 * sizeof(int)); 
  cudaMemset(GPU_b, 0, 1e4 * sizeof(int));
  cudaMemset(GPU_A, 1, 1e8 * sizeof(int));

  kernelA<<<grid_size, block_size>>>(GPU_A, GPU_x, GPU_b, n);

  cudaMemcpy(CPU_b, GPU_b, 1e4 * sizeof(int), cudaMemcpyDeviceToHost);

  cudaFree(GPU_x);
  cudaFree(GPU_b);
  cudaFree(GPU_A);

  printf("%d\n", CPU_b[0]);

  return(0);
}
