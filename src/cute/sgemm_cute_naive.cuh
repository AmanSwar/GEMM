#include <cuda_runtime.h>
#include <cute/algorithm/gemm.hpp>
#include <cute/arch/cluster_sm90.hpp>
#include <cute/arch/copy_sm80.hpp>
#include <cute/layout.hpp>
#include <cute/pointer.hpp>
#include <cute/pointer_flagged.hpp>
#include <cute/tensor.hpp>
#include <cute/tensor_impl.hpp>

#include "../../inc/util.cuh"

constexpr int BLK_M = 128;
constexpr int BLK_N = 128;
constexpr int BLK_K = 32;

constexpr int TM = 8; 
constexpr int TN = 8; 
constexpr int TK = 8;

__global__ void naive_gemm_kernel(float const *A, float const *B, float *C) {
  using namespace cute;
  //layouts for tensors of A B and C
  auto layout_A = make_layout(make_shape(M, K), make_stride(K, _1{})); 
  auto layout_B = make_layout(make_shape(K, N), make_stride(N, _1{}));
  auto layout_C = make_layout(make_shape(M, N), make_stride(N, _1{}));

  //Tensors of all the matrix
  Tensor matrixA = make_tensor(make_gmem_ptr(A), layout_A);
  Tensor matrixB = make_tensor(make_gmem_ptr(B), layout_B);
  Tensor matrixC = make_tensor(make_gmem_ptr(C), layout_C);

  //declare coopraive thread array (CTA) -> basically block of threads
  auto cta_coords = make_coord(blockIdx.x, blockIdx.y, _); // cta coords for finding where the CTA sits in the grid
  auto bM = Int<BLK_M>{};
  auto bN = Int<BLK_N>{};
  auto bK = Int<BLK_K>{};

  // size of CTA
  auto cta_tiler = make_shape(bM, bN, bK);

  //tile where the CTA sits
  Tensor gA = local_tile(matrixA, cta_tiler, cta_coords, Step<_1, X, _1>{});
  Tensor gB = local_tile(matrixB, cta_tiler, cta_coords, Step<X, _1, _1>{});
  Tensor gC = local_tile(matrixC, cta_tiler, cta_coords, Step<_1, _1, X>{});

  //declare shared mem
  auto smemA_layout = make_layout(make_shape(bM, bK), make_stride(bK, _1{}));
  auto smemB_layout = make_layout(make_shape(bK, bN), make_stride(bN, _1{}));
  __shared__ float smemA[cosize_v<decltype(smemA_layout)>];
  __shared__ float smemB[cosize_v<decltype(smemB_layout)>];
  Tensor sA = make_tensor(make_smem_ptr(smemA), smemA_layout);
  Tensor sB = make_tensor(make_smem_ptr(smemB), smemB_layout);


  auto thr_layout = make_layout(make_shape(Int<16>{}, Int<8>{})); // 128 threads
  auto thr_idx = threadIdx.x;
  //now to increase the compute intensity -> we are going to assign multiple elements to a single thread
  // so divide the tile (CTA block) into sub tiles such that each threads looks after one sub tile
  // dim(sub tile) -> thr_layout
  Tensor tAsA = local_partition(sA, thr_layout, thr_idx); // tensor to store to shared memory
  Tensor tBsB = local_partition(sB, thr_layout, thr_idx);

  auto reg_layout = make_layout(make_shape(Int<TM>{}, Int<TN>{})); // layout for matrix C -> output matrix
  Tensor tCrC = make_tensor<float>(reg_layout);
  clear(tCrC);

  auto K_TILE_MAX = size<2>(gA);

  for (int k_tile = 0; k_tile < K_TILE_MAX; ++k_tile) {
    auto gA_slice = gA(_, _, k_tile);
    auto gB_slice = gB(_, _, k_tile);

    auto tAgA_slice = local_partition(gA_slice, thr_layout, thr_idx);
    auto tBgB_slice = local_partition(gB_slice, thr_layout, thr_idx);

    copy(tAgA_slice, tAsA);
    copy(tBgB_slice, tBsB);

    cp_async_fence();
    cp_async_wait<0>();
    __syncthreads();

    auto comp_thr_layout = make_layout(make_shape(Int<TM / 4>{}, Int<TN / 4>{}));
    auto tCsA = local_partition(sA, comp_thr_layout, thr_idx % size(comp_thr_layout));
    auto tCsB = local_partition(sB, comp_thr_layout, thr_idx % size(comp_thr_layout));

    auto tCrC_view =
        local_partition(tCrC, make_layout(make_shape(Int<4>{}, Int<4>{})), 0);

    gemm(tCsA, tCsB, tCrC_view);

    __syncthreads();
  }

  auto tCgC = local_partition(gC, reg_layout, thr_idx);
  axpby(1.0f, tCrC, 0.0f, tCgC);
}

void launch_gemm(float *A, float *B, float *C) {
  using namespace cute;

  dim3 dimBlock(128); // Fixed block size
  dim3 dimGrid((M + BLK_M - 1) / BLK_M, (N + BLK_N - 1) / BLK_N);

  naive_gemm_kernel<<<dimGrid, dimBlock>>>(A, B, C);
}

// __global__ void naive_gemm_kernel_manual(float const *A, float const *B,
//                                          float *C, int M, int N, int K) {
//   using namespace cute;

//   auto layout_A = make_layout(make_shape(M, K), make_stride(K, _1{}));
//   auto layout_B = make_layout(make_shape(K, N), make_stride(N, _1{}));
//   auto layout_C = make_layout(make_shape(M, N), make_stride(N, _1{}));

//   Tensor matrixA = make_tensor(make_gmem_ptr(A), layout_A);
//   Tensor matrixB = make_tensor(make_gmem_ptr(B), layout_B);
//   Tensor matrixC = make_tensor(make_gmem_ptr(C), layout_C);

//   auto cta_coords = make_coord(blockIdx.x, blockIdx.y, _);
//   auto bM = Int<BLK_M>{};
//   auto bN = Int<BLK_N>{};
//   auto bK = Int<BLK_K>{};

//   auto cta_tiler = make_shape(bM, bN, bK);
//   Tensor gA = local_tile(matrixA, cta_tiler, cta_coords, Step<_1, X, _1>{});
//   Tensor gB = local_tile(matrixB, cta_tiler, cta_coords, Step<X, _1, _1>{});
//   Tensor gC = local_tile(matrixC, cta_tiler, cta_coords, Step<_1, _1, X>{});

//   __shared__ float smemA[BLK_M * BLK_K];
//   __shared__ float smemB[BLK_K * BLK_N];

//   auto smemA_layout = make_layout(make_shape(bM, bK));
//   auto smemB_layout = make_layout(make_shape(bK, bN));

//   Tensor sA = make_tensor(make_smem_ptr(smemA), smemA_layout);
//   Tensor sB = make_tensor(make_smem_ptr(smemB), smemB_layout);

//   // Thread partitioning for loading
//   auto thr_layout = make_layout(make_shape(Int<16>{}, Int<8>{}));
//   int thr_idx = threadIdx.x;

//   Tensor tAsA = local_partition(sA, thr_layout, thr_idx);
//   Tensor tBsB = local_partition(sB, thr_layout, thr_idx);

//   // Register accumulator
//   float acc[TM * TN] = {0.0f};

//   auto K_TILE_MAX = size<2>(gA);

//   for (int k_tile = 0; k_tile < K_TILE_MAX; ++k_tile) {
//     auto gA_slice = gA(_, _, k_tile);
//     auto gB_slice = gB(_, _, k_tile);

//     auto tAgA_slice = local_partition(gA_slice, thr_layout, thr_idx);
//     auto tBgB_slice = local_partition(gB_slice, thr_layout, thr_idx);

//     copy(tAgA_slice, tAsA);
//     copy(tBgB_slice, tBsB);

//     __syncthreads();

//     // Manual computation - simple thread mapping
//     int tid_m = threadIdx.x / 16;
//     int tid_n = threadIdx.x % 16;

//     for (int tm = 0; tm < TM; ++tm) {
//       for (int tn = 0; tn < TN; ++tn) {
//         int m_idx = tid_m * TM + tm;
//         int n_idx = tid_n * TN + tn;

//         if (m_idx < BLK_M && n_idx < BLK_N) {
//           for (int k = 0; k < BLK_K; ++k) {
//             acc[tm * TN + tn] += sA(m_idx, k) * sB(k, n_idx);
//           }
//         }
//       }
//     }

//     __syncthreads();
//   }

//   int tid_m = threadIdx.x / 16;
//   int tid_n = threadIdx.x % 16;

//   for (int tm = 0; tm < TM; ++tm) {
//     for (int tn = 0; tn < TN; ++tn) {
//       int m_idx = blockIdx.x * BLK_M + tid_m * TM + tm;
//       int n_idx = blockIdx.y * BLK_N + tid_n * TN + tn;

//       if (m_idx < M && n_idx < N) {
//         matrixC(m_idx, n_idx) = acc[tm * TN + tn];
//       }
//     }
//   }
// }

// void launch_gemm_manual(float *A, float *B, float *C, int M, int N, int K) {
//   dim3 dimBlock(128);
//   dim3 dimGrid((M + BLK_M - 1) / BLK_M, (N + BLK_N - 1) / BLK_N);
//   naive_gemm_kernel_manual<<<dimGrid, dimBlock>>>(A, B, C, M, N, K);
// }

int main() {
  benchmark_fp32(launch_gemm, "CUTE NAIVE");
  return 0;
}