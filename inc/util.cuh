#pragma once

#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>

#include <iostream>
#include <iomanip>
#include <string>

#define M 2048
#define N 2048
#define K 1024

#define CUDA_CHECK(err)                                                        \
  {                                                                            \
    cudaError_t err_ = (err);                                                  \
    if (err_ != cudaSuccess) {                                                 \
      fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__,         \
              cudaGetErrorString(err_));                                       \
      exit(EXIT_FAILURE);                                                      \
    }                                                                          \
  }

template <class dtype>
bool verify(dtype *kernel_out, dtype *cublas_out, int size) {

  for (int i = 0; i < size; i++) {
    if (std::fabs(kernel_out[i] - cublas_out[i]) > 1e-3) {
      std::cout << static_cast<float>(kernel_out[i]) << " " << static_cast<float>(cublas_out[i]) << std::endl;
      // printf("kernel out : %f  || cublas out : %f"  , kernel_out[i] , cublas_out[i]);
      return false;
    }
  }
  std::cout << static_cast<float>(kernel_out[10]) << " "
            << static_cast<float>(cublas_out[10]) << std::endl;
  return true;
}

template <class dtype>
void init(dtype *arr, int size) {
  for (int i = 0; i < size; i++) {
    // arr[i] = static_cast<float>(std::rand()) / (static_cast<float>(RAND_MAX) + 1.0f);
    arr[i]= i / size;
  }
}

void launch_cublass_fp32(float *ha, float *hb, float *hc,
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

void launch_cublass_fp16(half *ha, half *hb, half *hc, half *da, half *db,
                         half *dc) {

  cublasHandle_t handle;
  cublasCreate(&handle);

  cudaEvent_t start, end;
  cudaEventCreate(&start);
  cudaEventCreate(&end);

  half alpha = __float2half(1.0f);
  half beta = __float2half(0.0f);
  // cublasSetMatrix(M, K, sizeof(float), ha, M, da, M);
  // cublasSetMatrix(K, N, sizeof(float), hb, K, db, K);
  // cublasSetMatrix(M, N, sizeof(float), hc, M, dc, M);

  int lda = K;
  int ldb = N;
  int ldc = N;
  // init run

  cublasHgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, N, M, K, &alpha, db, ldb, da,
              lda, &beta, dc, ldc);
  int iter = 100;

  cudaEventRecord(start);
  for (int i = 0; i < iter; i++) {
    cublasHgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, N, M, K, &alpha, db, ldb, da,
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

void benchmark_fp32(void (*func)(float *, float *, float *),
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
  CUDA_CHECK(cudaGetLastError());

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

  std::cout << "------------------------" << std :: endl;

  launch_cublass_fp32(ha, hb, hd, da, db, dd);

  cudaMemcpy(hc , dc , sizeof(float) * M * N , cudaMemcpyDeviceToHost);
  cudaMemcpy(hd, dd, sizeof(float) * M * N, cudaMemcpyDeviceToHost);

  bool ans = verify(hd, hc, M * N);
  std::string out = ans ? "PASS" : "FAIL";
  std::cout << "\n"<<out << std::endl; 
}

void benchmark_fp16(void (*func)(half *, half *, half *), std::string name) {

  half *ha = new half[M * K];
  half *hb = new half[K * N];
  half *hc = new half[M * N];
  half *hd = new half[M * N]; // for cublas

  init(ha, M * K);
  init(hb, K * N);

  std::fill(hc, hc + M * N, 0.0f);
  std::fill(hd, hc + M * N, 0.0f);

  half *da, *db, *dc, *dd;
  cudaMalloc(&da, sizeof(half) * M * K);
  cudaMalloc(&db, sizeof(half) * K * N);
  cudaMalloc(&dc, sizeof(half) * M * N);
  cudaMalloc(&dd, sizeof(half) * M * N); // for cublas

  cudaMemcpy(da, ha, sizeof(half) * M * K, cudaMemcpyHostToDevice);
  cudaMemcpy(db, hb, sizeof(half) * K * N, cudaMemcpyHostToDevice);

  cudaEvent_t start, end;
  cudaEventCreate(&start);
  cudaEventCreate(&end);
  int ITER = 100;
  // init run
  func(da, db, dc);
  CUDA_CHECK(cudaGetLastError());

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

  std::cout << "------------------------" << std ::endl;

  launch_cublass_fp16(ha, hb, hd, da, db, dd);

  cudaMemcpy(hc, dc, sizeof(float) * M * N, cudaMemcpyDeviceToHost);
  cudaMemcpy(hd, dd, sizeof(float) * M * N, cudaMemcpyDeviceToHost);

  bool ans = verify(hd, hc, M * N);
  std::string out = ans ? "PASS" : "FAIL";
  std::cout << "\n" << out << std::endl;
}