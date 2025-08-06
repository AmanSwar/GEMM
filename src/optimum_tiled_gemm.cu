#include <cuda_runtime.h>

#include "../inc/util.cuh"

const int BM = 64;
const int BK = 16;
const int BN = 64;

const int TM = 8;
const int TK = 8;
const int TN = 8;


__global__ void optimum_tiled_gemm_kernel(
    float* matrix_a,
    float* matrix_b,
    float* matrix_out
){

    //shared mem
    __shared__ float smem_a[BM][BK];
    __shared__ float smem_b[BK][BN];

    //indexing
    int global_index_x = blockIdx.x * blockDim.x + threadIdx.x;
    int global_index_y = blockIdx.y * blockDim.y + threadIdx.y;

    int local_index_x = threadIdx.x;
    int local_index_y = threadIdx.y;

    int row_start = global_index_y * TM;
    int col_start = global_index_x * TN;

    float reg_c[TM][TN] = {0.0f};

    int thread_block_dim_x = 8; // BN / TN (avoiding unnecessary division)
    int thread_block_dim_y = 8; // BM / TM 

    int total_blocks = (K + BK - 1) / BK; 
    for(int t = 0 ; t < total_blocks ; t++){ // loop over blocks along K dim
        #pragma unroll
        for(int m = 0 ; m < TM ; m++){
            for(int k = 0 ; k < BK ; k+= thread_block_dim_x){
                int m_idx = row_start + m;
                int k_idx = t * BK + k + local_index_x;

                if(m_idx < M && k_idx < K){
                    smem_a[local_index_y * TM + m][k + local_index_x] = matrix_a[m_idx * K + k_idx];
                }else{
                    smem_a[local_index_y * TM + m][k + local_index_x] = 0.0f;
                }
            }

        }

        #pragma unroll
        for(int k =0 ; k < BK ;k += thread_block_dim_y){
            for(int n = 0 ; n < TN ; n++){
                int k_idx = t * BK + k + local_index_y;
                int n_idx = col_start + n;

                if(k_idx < K && n_idx < N){
                    smem_b[k + local_index_y][local_index_x * TN + n] = matrix_b[k_idx * N + n_idx];
                }else{
                    smem_b[k+ local_index_y][local_index_x * TN + n] = 0.0f;
                }
            }
        }

        __syncthreads();   

        #pragma unroll
        for(int k = 0 ; k < BK ; k++){
            #pragma unroll
            for(int m = 0 ; m < TM ; m++){
                #pragma unroll
                for(int n = 0; n<TN; n++){
                    reg_c[m][n] += smem_a[local_index_y * TM + m][k] * smem_b[k][local_index_x * TN + n];
                }
            }
        }

        __syncthreads();
    }


    #pragma unroll
    for(int m = 0 ; m < TM ; m++){
        #pragma unroll
        for(int n = 0 ; n < TN ; n++){
            int r = row_start + m;
            int c = col_start + n;

            if(r < M && c < N){
                matrix_out[r * N + c] = reg_c[m][n];
            }
        }
    }
}

void launch_optim_tiled_kernel(
    float* matrix_a,
    float* matrix_b,
    float* matrix_out
){
    int total_threads_n = BN / TN;
    int total_threads_m = BM / TM;
    dim3 block_dim(total_threads_n , total_threads_m);
    dim3 grid_dim(
        (N + BN -1)/BN,
        (M + BM - 1) / BM
    );
    optimum_tiled_gemm_kernel<<<grid_dim , block_dim>>>(matrix_a , matrix_b , matrix_out);
    
}




int main(){

  benchmark(launch_optim_tiled_kernel , "REG PR TILED ");
  
}

