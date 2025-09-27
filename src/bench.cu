#include <cuda_runtime.h>
#include <cuda_fp16.h>

#include <string>
#include <iostream>
#include <iomanip>



#include "cublas/cublas_gemm.cuh"
#include "mma/hgemm_m16n8k16.cuh"
#include "wmma/wmma_gemm_naive_fp16.cuh"

#define M 2048
#define N 2048
#define K 2048

#define CUDA_CHECK(err)                                                        \
  {                                                                            \
    cudaError_t err_ = (err);                                                  \
    if (err_ != cudaSuccess) {                                                 \
      fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__,         \
              cudaGetErrorString(err_));                                       \
      exit(EXIT_FAILURE);                                                      \
    }                                                                          \
  }

template <class dtype> void init(dtype *arr, int size) {
  for (int i = 0; i < size; i++) {
    // arr[i] = static_cast<float>(std::rand()) / (static_cast<float>(RAND_MAX)
    // + 1.0f);
    arr[i] = float(i) / size;
  }
}



void benchmark_fp16(void (*func)(half *, half *, half * , int , int , int), std::string name , half* da , half* db , half* d_kernel){
  cudaEvent_t start, end;
  cudaEventCreate(&start);
  cudaEventCreate(&end);
  int ITER = 100;
  // init run
  func(da, db, d_kernel , M , N , K);
  CUDA_CHECK(cudaGetLastError());

  cudaEventRecord(start);
  
  for (int i = 0; i < ITER; i++) {
    func(da, db, d_kernel , M , N ,K);
  }

  cudaEventRecord(end);
  cudaEventSynchronize(end);

  float ms = 0.0f;
  cudaEventElapsedTime(&ms, start, end);

  std::cout << name << " TIME : " << std::fixed << std::setprecision(5)
            << (ms / ITER) / 1000 << std::endl;

  long long int FLOP = 2LL * M * N * K;
  std::cout << name << " GFLOPS : " << std::fixed << std::setprecision(5)
            << ((FLOP / ((ms / ITER) / 1000))) / 1e9 << std::endl;

  std::cout << "------------------------" << std ::endl;
}

int main(){
  half *ha = new half[M * K];
  half *hb = new half[K * N];
  half *h_kernel = new half[M * N]; // for kernel
  half *h_blas = new half[M * N];   // for cublas

  init(ha, M * K);
  init(hb, K * N);

  std::fill(h_kernel, h_kernel + M * N, 0.0f);
  std::fill(h_blas, h_blas + M * N, 0.0f);

  half *da, *db, *d_kernel, *d_blas;
  cudaMalloc(&da, sizeof(half) * M * K);
  cudaMalloc(&db, sizeof(half) * K * N);
  cudaMalloc(&d_kernel, sizeof(half) * M * N);
  cudaMalloc(&d_blas, sizeof(half) * M * N); // for cublas
  
  
  cudaMemcpy(da, ha, sizeof(half) * M * K, cudaMemcpyHostToDevice);
  cudaMemcpy(db, hb, sizeof(half) * K * N, cudaMemcpyHostToDevice);

  perf_cublas_tn(M , N , K);
  
  benchmark_fp16(hgemm_mma_m16n8k16_naive, "mma_m16n8k16_naive", da, db, d_kernel);
  benchmark_fp16(launch_wmma_kernel, "wmma_naive_kernel", da, db, d_kernel);

}