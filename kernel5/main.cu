#include <stdio.h>
#include <math.h>

__global__ void kernelSM(int *A, int *x, int *b, int N){
  extern __shared__ int sm[];
  int tId = threadIdx.x + blockIdx.x * blockDim.x;
  if(tId < N){
    for(int k = 0; k < N/blockDim.x; k++){
      sm[threadIdx.x] = x[threadIdx.x + 256*k];
      __syncthreads();
      b[tId] += A[(int)(tId*N+(threadIdx.x+256*k))]*sm[threadIdx.x];
      __syncthreads();
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

  int *CPU_x = (int *) malloc(1e4 * sizeof (int));
  int *CPU_A = (int *) malloc(1e8 * sizeof (int));

  for(int k = 0; k < 1e8; k++){
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
  cudaMemset(GPU_b, 0, 1e4 * sizeof(int));

  kernelSM<<<grid_size, block_size, block_size*sizeof(int)>>>(GPU_A, GPU_x, GPU_b, n);

  cudaMemcpy(CPU_x, GPU_b, 1e4 * sizeof(int), cudaMemcpyDeviceToHost);

  for(int k = 0; k< 1e4; k++){
    printf("%d\n", CPU_x[k]);
  }

  cudaFree(GPU_x);
  cudaFree(GPU_b);
  cudaFree(GPU_A);
  free(CPU_x);
  free(CPU_A);

  return(0);
}