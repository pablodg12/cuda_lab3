#include <stdio.h>
#include <math.h>



int main(int argc, char const *argv[])
{
  int *GPU_x;
  int *GPU_b;
  int *GPU_A;
  cudaMalloc(&GPU_x , 1e4* sizeof(int));
  cudaMalloc(&GPU_b , 1e4* sizeof(int));
  cudaMalloc(&GPU_A , 1e8* sizeof(int));
  cudaMemset(GPU_x, 1, 1e4*sizeof(int));
  cudaMemset(GPU_A, 1, 1e8*sizeof(int));


  cudaFree(GPU_x);
  cudaFree(GPU_b);
  cudaFree(GPU_A);



  return(0);
}
