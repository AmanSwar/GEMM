#pragma once

#include <cublas_v2.h>
#include <cuda_runtime.h>

#include <iomanip>
#include <iostream>



void launch_cublass(int M, int N, int K, float *ha, float *hb, float *hc,
                    float *da, float *db, float *dc) {
      
  cublasHandle_t handle;
  cublasCreate(&handle);

  cudaEvent_t start, end;
  cudaEventCreate(&start);
  cudaEventCreate(&end);

  float alpha = 1.0f;
  float beta = 0.0f;
  cublasSetMatrix(M, K, sizeof(float), ha, M, da, M);
  cublasSetMatrix(K, N, sizeof(float), hb, K, db, K);
  cublasSetMatrix(M, N, sizeof(float), hc, M, dc, M);

  // init run
  cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, M, N, K, &alpha, da, M, db, K,
              &beta, dc, M);
  int iter = 100;

  cudaEventRecord(start);
  for (int i = 0; i < iter; i++) {
    cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, M, N, K, &alpha, da, M, db, K,
                &beta, dc, M);
  }

  cudaEventRecord(end);
  cudaEventSynchronize(end);

  float ms = 0.0f;
  cudaEventElapsedTime(&ms, start, end);
  
  std::cout << "CUBLAS TIME : "<< std::fixed << std::setprecision(5) << (ms / iter) / 1000 << std::endl;

  long long int FLOP = 2LL * M * N * K;
  std::cout << "CUBLAS GFLOPS : " <<std::fixed << std::setprecision(5) << ((FLOP / ((ms / iter)/1000))) / 1e9
            << std::endl;

  cublasGetMatrix(M, N, sizeof(float), dc, M, hc, M);
  cudaFree(da);
  cudaFree(db);
  cudaFree(dc);
  cublasDestroy(handle);

}


void cublas_tensor_op_tn_v2(cublasHandle_t handle, half *A, half *B, half *C,
                            size_t M, size_t N, size_t K) {
  half alpha = 1.0;
  half beta = 0.0;

  cublasGemmEx(handle, CUBLAS_OP_T, CUBLAS_OP_N, N, M, K, &alpha, B, CUDA_R_16F,
               K, A, CUDA_R_16F, K, &beta, C, CUDA_R_16F, N, CUBLAS_COMPUTE_16F,
               CUBLAS_GEMM_DEFAULT_TENSOR_OP);
}


void perf_cublas_tn(int M, int N, int K, int repeat=100) {
  size_t size_a = M * K * sizeof(half);
  size_t size_b = K * N * sizeof(half);
  size_t size_c = M * N * sizeof(half);

  half *d_a, *d_b;
  half *d_c;
  cudaMalloc(&d_a, size_a);
  cudaMalloc(&d_b, size_b);
  cudaMalloc(&d_c, size_c);

  cublasHandle_t handle = nullptr;
  cublasCreate(&handle);
  cublasSetMathMode(handle, CUBLAS_TENSOR_OP_MATH);

  // warmup
  for (int i = 0; i < 1; ++i) {
    cublas_tensor_op_tn_v2(handle, d_a, d_b, d_c, M, N, K);
  }
  cudaDeviceSynchronize();

  cudaEvent_t start, end;
  cudaEventCreate(&start);
  cudaEventCreate(&end);
  cudaEventRecord(start);

  for (int i = 0; i < repeat; i++) {
    cublas_tensor_op_tn_v2(handle, d_a, d_b, d_c, M, N, K);
  }

  cudaEventRecord(end);
  cudaDeviceSynchronize();
  cudaEventSynchronize(end);

  float msec, sec;
  cudaEventElapsedTime(&msec, start, end);
  sec = msec / 1000.0 / repeat;

  std::cout << "CUBLAS " << " TIME : " << std::fixed << std::setprecision(5)
            << sec << std::endl;

  long long int FLOP = 2LL * M * N * K;
  std::cout << "CUBLAS " << " GFLOPS : " << std::fixed << std::setprecision(5) << (FLOP / sec) / 1e9 << std::endl;

  std::cout << "------------------------" << std ::endl;
  cudaFree(d_a);
  cudaFree(d_b);
  cudaFree(d_c);
  cudaEventDestroy(start);
  cudaEventDestroy(end);
  cublasDestroy(handle);

}