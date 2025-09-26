#include <cuda_runtime.h>

#include "../inc/util.cuh"
/*
Naive implementation cuz who cares about flops anyways
*/

__global__ void naive_matmul_kernel(
    float* matrix_a,
    float* matrix_b ,
    float* matrix_out
    
){
  
  int global_index_x = blockDim.x * blockIdx.x + threadIdx.x;
  int global_index_y = blockDim.y * blockIdx.y + threadIdx.y;

  // matrix a -> access -> matrix_a[global_index_x * K + k]
  // matrix b -> access -> matrix_b[k * N + global_index_y]
  // matrix c -> store -> matrix_c[global_index_x * N + k]
  if (global_index_x < M && global_index_y < N) { 
    float sum = 0.0f;
    for (int i = 0; i < K; i++) {
      sum += matrix_a[global_index_y * K + i] * matrix_b[i * N + global_index_x];
      // printf("exit");
    }
    matrix_out[global_index_y * N + global_index_x] = sum;
    }
}


void launch_naive_kernel(
    float* a,
    float* b,
    float* out
){
    dim3 block_dim(32 , 32);
    dim3 grid_dim((N + block_dim.x -1)/block_dim.x , (M+ block_dim.y -1) / block_dim.y);
    naive_matmul_kernel<<<grid_dim , block_dim>>>(a ,b , out);
    cudaDeviceSynchronize();
}


int main(){

  benchmark(launch_naive_kernel , "naive");
  
}