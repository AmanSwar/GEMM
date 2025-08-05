#include <cuda_runtime.h>
#include <iostream>
#include "../inc/util.cuh"

const int BM = 16;
const int BK = 16;
const int BN = 16;

__global__ void tiled_gemm_kernel(
    float* matrix_a ,
    float* matrix_b,
    float* matrix_out
){

    int global_index_x = blockDim.x * blockIdx.x + threadIdx.x;
    int global_index_y = blockDim.y * blockIdx.y + threadIdx.y;

    int local_index_x = threadIdx.x;
    int local_index_y = threadIdx.y;

    __shared__ float smem_a[BM][BK];
    __shared__ float smem_b[BK][BN];

    /*
    Total size of shared memory for each block in RTX 3050 -> 48kb / 48000 bytes
    so we can have block size as 64 x 64 -> 4096 fp32 elements per block -> 16384 bytes 
    and since we have 2 such blocks so 16384 * 2 -> 32768 bytes 
    but but but
    we want 1 tile to be in 1 block , if we have 64x64 dim then total elements = 4096 but the max amount of thread
    in a block is 1024 
    so 
    64 x 16 -> 1024
    */
    float sum = 0.0f;
    const int tile_size = BK;
    const int total_tiles = (K + BK - 1) / BK; 
    for(int tile = 0 ; tile < total_tiles ; tile++){
        /*
        tile jumping -> tile * tile_size;
        next row -> global_index_y * K
        */
        

        int a_row = global_index_y;
        int a_col = tile * tile_size + local_index_x;

        int b_col = global_index_x;
        int b_row = tile*tile_size + local_index_y;

        if(a_row < M && a_col < K && local_index_x < BK){
            smem_a[local_index_y][local_index_x] = matrix_a[a_row * K + a_col];
        }else{
            smem_a[local_index_y][local_index_x] = 0.0f;
        }

        if(b_row < K && b_col < N && local_index_y < BK){
            smem_b[local_index_y][local_index_x] = matrix_b[b_row * N + b_col];
        }else{
            smem_b[local_index_y][local_index_x] = 0.0f;
        }
        
        __syncthreads();

        for (int k = 0; k < min(BK, K - tile * BK); k++) {
          sum += smem_a[local_index_y][k] * smem_b[k][local_index_x];
        }

        __syncthreads();
    }

    if(global_index_x < N && global_index_y < M){
        matrix_out[global_index_y * N + global_index_x] = sum;
    }

}




void launch_tiled_kernel(
    float* a ,
    float* b,
    float *out
){
    
    dim3 block_dim(BN , BM);
    dim3 grid_dim(
        (N + block_dim.x -1)/  block_dim.x,
        (M + block_dim.y -1) / block_dim.y
    );
    tiled_gemm_kernel<<<grid_dim , block_dim>>>(a , b , out);
    
}



int main(){

  benchmark(launch_tiled_kernel , "TILED ");
  
}