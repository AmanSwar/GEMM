#pragma once

#include <cuda_runtime.h>
#include <mma.h>
#include <cuda_fp16.h>

using namespace nvcuda;

const int WMMA_M = 16;
const int WMMA_N = 16;
const int WMMA_K = 16;    

#define BLOCK_SIZE 128
#define WARP_SIZE 32

__global__ void wmma_naive_gemm(half *a, half *b, half *c, float alpha,float beta , int M , int N , int K) {

  int lda = K; 
  int ldb = N;   
  int ldc = N; 

  int warpM = (blockIdx.x * blockDim.x + threadIdx.x) / warpSize; // 0 , 1
  int warpN = (blockIdx.y * blockDim.y + threadIdx.y);

  //declare the fragements 
  //fragments of matrix A (row major) and matrixB (col major)
  wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, half, wmma::row_major> a_frag;
  wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, half, wmma::col_major> b_frag;

  //accumulator frags in fp32
  wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> acc_frag;
  wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, half> c_frag;

  //init frags
  wmma::fill_fragment(acc_frag, 0.0f);  


  for (int i = 0; i < K; i += WMMA_K) {
    int aRow = warpM * WMMA_M; 
    int aCol = i;

    int bRow = i;
    int bCol = warpN * WMMA_N;

    if (aRow < M && aCol < K && bRow < K && bCol < N) {
      wmma::load_matrix_sync(a_frag, a + aRow * lda + aCol, lda);
      wmma::load_matrix_sync(b_frag, b + bRow * ldb + bCol, ldb);

      wmma::mma_sync(acc_frag, a_frag, b_frag, acc_frag);
    }
  }
  
  int cRow = warpM * WMMA_M;
  int cCol = warpN * WMMA_N;

  if (cRow < M && cCol < N) {

    #pragma unroll
    for (int i = 0; i < c_frag.num_elements; i++) {
      c_frag.x[i] = acc_frag.x[i];
    }
    
    wmma::store_matrix_sync(c + cRow * ldc + cCol, c_frag, ldc,
                            wmma::mem_row_major);
  }
}


void launch_wmma_kernel(
    half* matrix_a,
    half* matrix_b,
    half* matrix_out,
    int M , int N , int K
){
    dim3 block_dim(32 , 4);
    dim3 grid_dim(
        (M + WMMA_M - 1) / WMMA_M / (block_dim.x / WARP_SIZE),
        (N + WMMA_N - 1) / WMMA_N / block_dim.y
    );
    wmma_naive_gemm<<<grid_dim , block_dim>>>(matrix_a , matrix_b , matrix_out , 1 , 0 , M , N , K);


}
