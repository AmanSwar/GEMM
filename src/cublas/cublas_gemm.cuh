#include <cublas_v2.h>
#include <cuda_runtime.h>

#include <iomanip>
#include <iostream>

#include "../inc/launch.h"




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