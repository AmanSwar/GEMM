#pragma once

#include <cublas_v2.h>
#include <cuda_runtime.h>

#include <iostream>
#include <iomanip>
#include <string>

#define M 2048
#define N 2048
#define K 1024

bool verify(float *kernel_out, float *cublas_out, int size) {

  for (int i = 0; i < size; i++) {
    if (std::abs(kernel_out[i] - cublas_out[i]) > 1e-4) {
      std::cout << kernel_out[i] << " " << cublas_out[i] << std::endl;
      return false;
    }
  }
  return true;
}


void init(float *arr, int size) {
  for (int i = 0; i < size; i++) {
    arr[i] = static_cast<float>(std::rand()) / (static_cast<float>(RAND_MAX) + 1.0f);
  }
}

void launch_cublass(float *ha, float *hb, float *hc,
                    float *da, float *db, float *dc) {

  cublasHandle_t handle;
  cublasCreate(&handle);

  cudaEvent_t start, end;
  cudaEventCreate(&start);
  cudaEventCreate(&end);

  float alpha = 1.0f;
  float beta = 0.0f;
  // cublasSetMatrix(M, K, sizeof(float), ha, M, da, M);
  // cublasSetMatrix(K, N, sizeof(float), hb, K, db, K);
  // cublasSetMatrix(M, N, sizeof(float), hc, M, dc, M);

  int lda = K;
  int ldb = N;
  int ldc = N;
  // init run

  cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, N, M, K, &alpha, db, ldb, da, lda,
              &beta, dc, ldc);
  int iter = 100;

  cudaEventRecord(start);
  for (int i = 0; i < iter; i++) {
    cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, N, M, K, &alpha, db, ldb, da,
                lda, &beta, dc, ldc);
  }

  cudaEventRecord(end);
  cudaEventSynchronize(end);

  float ms = 0.0f;
  cudaEventElapsedTime(&ms, start, end);

  std::cout << "CUBLAS TIME : " << std::fixed << std::setprecision(5)
            << (ms / iter) / 1000 << std::endl;

  long long int FLOP = 2LL * M * N * K;
  std::cout << "CUBLAS GFLOPS : " << std::fixed << std::setprecision(5)
            << ((FLOP / ((ms / iter) / 1000))) / 1e9 << std::endl;

  cublasGetMatrix(M, N, sizeof(float), dc, M, hc, M);
  cudaFree(da);
  cudaFree(db);
  cudaFree(dc);
  cublasDestroy(handle);
}


void benchmark(void (*func)(float *, float *, float *),
               std::string name) {

  float *ha = new float[M * K];
  float *hb = new float[K * N];
  float *hc = new float[M * N];
  float *hd = new float[M * N]; // for cublas

  init(ha, M * K);
  init(hb, K * N);

  std::fill(hc, hc + M * N, 0.0f);
  std::fill(hd, hc + M * N, 0.0f);

  float *da, *db, *dc, *dd;
  cudaMalloc(&da, sizeof(float) * M * K);
  cudaMalloc(&db, sizeof(float) * K * N);
  cudaMalloc(&dc, sizeof(float) * M * N);
  cudaMalloc(&dd, sizeof(float) * M * N); // for cublas

  cudaMemcpy(da, ha, sizeof(float) * M * K, cudaMemcpyHostToDevice);
  cudaMemcpy(db, hb, sizeof(float) * K * N, cudaMemcpyHostToDevice);

  cudaEvent_t start, end;
  cudaEventCreate(&start);
  cudaEventCreate(&end);
  int ITER = 100;
  // init run
  func(da, db, dc);

  cudaEventRecord(start);
  for (int i = 0; i < ITER; i++) {
    func(da, db, dc);
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

  // std::cout << "##########################" << std :: endl;
  std::cout << "------------------------" << std :: endl;

  launch_cublass(ha, hb, hd, da, db, dd);

  cudaMemcpy(hc , dc , sizeof(float) * M * N , cudaMemcpyDeviceToHost);
  cudaMemcpy(hd, dd, sizeof(float) * M * N, cudaMemcpyDeviceToHost);

  bool ans = verify(hd, hc, M * N);
  std::string out = ans ? "PASS" : "FAIL";
  std::cout << "\n"<<out << std::endl; 
}