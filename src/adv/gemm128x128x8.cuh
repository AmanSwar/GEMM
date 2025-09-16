#include <__clang_cuda_builtin_vars.h>
#include <__clang_cuda_runtime_wrapper.h>
#include <cinttypes>
#include <cstdint>
#include <cuda_runtime.h>
#include "helper.cuh"



__global__
__launch_bounds__(256 , 2) void sgemm_128x128x8(
    int m , int n , int k ,
    const float alpha, const float beta,
    const float* A, // mat A -> row major
    const float* B, // mat B -> row major 
    const float* C, // mat c -> row major
    int lda , int ldb , int ldc
){
    const int smem_a_cols = 256; // for avoiding bank conflicts
    const int smem_a_size = smem_a_cols * 8; // total size of smem_a
    const int smem_a_ld = 132; //leading dim

    const int smem_b_cols = 128;
    const int smem_b_size = smem_b_cols * 8;
    const int smem_b_ld = 128;


    __shared__ float __align__(2 * smem_a_size * sizeof(float)) smem_ptr[2 * (smem_a_size + smem_b_size)];

    //C acc
    float acc[8][8]{};


    //register for global mem -> register -> shared mem | size= 4 for vectorized load
    float ldg_a_reg[4];
    float ldg_b_reg[4];

    //bit masks for boundary checks
    unsigned ldg_a_bitmask = 0x0;
    unsigned ldg_b_bitmask = 0x0;

    //shared mem ptrs
    float* smem_a_ptr = smem_ptr;
    float* smem_b_ptr = smem_a_ptr + 2 * smem_a_size;


    int warp_id = threadIdx.x / 32;
    int lane_id = threadIdx.x % 32;


    int ldg_a_start_x = threadIdx.x % 8; // ?
    int ldg_a_start_y = blockIdx.y * 128 + 4 * (threadIdx.x / 8); // ?
    int ldg_a_start = ldg_a_start_y * lda + ldg_a_start_x;

    const float* ldg_a_ptr = A + ldg_a_start;

    int ldg_a_offsets_y[4]; // row offset index
    int ldg_a_offsets[4]; // actual row offests

    #pragma unroll
    for(int i = 0 ; i < 4 ; i++){
        ldg_a_offsets_y[i] = i; // taking in value (0 , 1 ,2 , 3)
    }

    #pragma unroll
    for(int i = 0 ; i < 4 ; i++){
        ldg_a_offsets[i] = ldg_a_offsets_y[i] * lda;
    }

    #pragma unroll
    for(int i = 0 ; i < 4 ; i++){
        int m_idx = ldg_a_start_y + ldg_a_offsets_y[i];
        // if the global mem access is in bounds -> flip corresponding bit
        if(m_idx < m) {ldg_a_bitmask ^= (0x1 << i);}
    }


    int ldg_b_start_x = blockIdx.x * 128 + threadIdx.x % 32;
    int ldg_b_start_y = threadIdx.x / 32;

    int ldg_b_start = ldg_b_start_y * ldb + ldg_b_start_x;

    const float* ldg_b_ptr = B + ldg_b_start;

    int ldg_b_offsets_x[4];
    int ldg_b_offsets[4];

    #pragma unroll
    for(int  i = 0 ; i < 4 ; i++){
        ldg_b_offsets_x[i] = 32 * i;
    }

    #pragma unroll
    for(int i = 0 ; i < 4 ; i++){
        ldg_b_offsets[i] = ldg_b_offsets_x[i];
    }

    #pragma unroll
    for(int i = 0 ; i < 4 ; i++){
        int n_idx = ldg_b_start_x + ldg_b_offsets_x[i];

        if (n_idx < n) {ldg_b_bitmask ^= (0x1 << i);}

    }


    int sts_a_start_x = 4 * (threadIdx.x / 8);
    int sts_a_start_y = threadIdx.x % 8;
    int sts_a_start = sts_a_start_y * smem_a_ld + sts_a_start_x;

    float* sts_a_ptr = smem_a_ptr + sts_a_start;

    int sts_b_start_x = threadIdx.x % 32;
    int sts_b_start_y = threadIdx.x / 32;
    int sts_b_start = sts_b_start_y * smem_b_ld + sts_b_start_x;
    float* sts_b_ptr = smem_b_ptr + sts_b_start;
    int sts_b_offsets[4];

    #pragma unroll
    for(int i = 0 ; i < 4 ; i++){
        sts_b_offsets[i] = 32 * i;   
    }


    //convert gemeric to shared state space 
    uint64_t sts_a_addr = cvta_to_shared(sts_a_ptr);
    uint64_t sts_b_addr = cvta_to_shared(sts_b_ptr);

    int n_blocks_k = (k + 7) / 8 -1;
    int first_block_k_size = k - 8 * n_blocks_k;


    #pragma unroll
    for(int i = 0 ; i < 4 ; i++){
        bool guard_k = ldg_a_start_x < first_block_k_size;
        bool guard_m = ldg_a_bitmask & (0x1 << i);
        bool guard = guard_k && guard_m;

        ldg_a_reg[i] = ldg32_guard_mov0_ptx(ldg_a_ptr + ldg_a_offsets[i] , (unsigned)guard);
        
    }

    sts128_ptx(ldg_a_reg[0], ldg_a_reg[1], ldg_a_reg[2], ldg_a_reg[3], sts_a_addr);

    #pragma unroll
    for(int i = 0 ; i < 4 ; i++){
        bool guard_k = ldg_b_start_y < first_block_k_size;
        bool guard_n = ldg_b_bitmask & (0x1 << i);
        bool guard = guard_k && guard_n;

        ldg_b_reg[i] = ldg32_guard_mov0_ptx(ldg_b_ptr, (unsigned)guard);
    }

    #pragma unroll
    for(int i = 0 ; i < 4 ; i += 1){
        sts32_ptx(ldg_b_reg[i] , sts_b_addr + sts_b_offsets[i] * sizeof(float));
    }

}