#pragma once

#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>

#include <iostream>
#include <iomanip>
#include <string>

#define M 1024
#define N 1024
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

bool verify_cpu(half* kernel_out , half* matrix_a ,half* matrix_b){
  float* c = new float[M*N];
  std::fill(c, c + M * N, 0.0f);

  for(int row = 0 ; row < M ; row++){
    for(int col = 0 ; col < N ; col++){
      float acc= 0.0f;
      for(int k = 0 ; k < K ; k++){
        acc += __half2float(matrix_a[row*K + k] * matrix_b[k * N + col]);
      }
      c[row * N + col] = acc;
    }
  }

  for(int i = 0 ; i < M * N ; i++){
    if(std::fabs(__half2float(kernel_out[i]) - c[i]) > 1e-2){
      std::cout << __half2float(kernel_out[i]) << " " << c[i];
      return false;
    }
  }
  return true;
}


template <class dtype1 , class dtype2>
bool verify(dtype1 *kernel_out, dtype2 *cublas_out, int size) {
  

  for (int i = 0; i < size; i++) {
    if (std::fabs(static_cast<float>(kernel_out[i]) - static_cast<float>(cublas_out[i])) > 1){
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
    arr[i]= float(i) / size;
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


void launch_cublass_fp16(__half* da , __half* db , __half* dc){
  /*
  Row major cublas with dtype fp16 using tensor cores
  by defualt cublas is column major hence implements 
  C(nxm) = B(nxk) . A(kxm)
  so we need to implement 
  C^T = B^T . A^T
  */
  //create handle
  cublasHandle_t handle;
  cublasCreate(&handle);

  //cuda events
  cudaEvent_t start, end;
  cudaEventCreate(&start);
  cudaEventCreate(&end);

  __half alpha = __float2half(1.0f);
  __half beta = __float2half(0.0f);

  cublasStatus_t status = cublasGemmEx(
    handle, // handle
    CUBLAS_OP_N, // no transpose for matrix b
    CUBLAS_OP_N, // no transpose for matrix a
    N, // cols of B and C
    M, // rows of A and C
    K, // common axis 
    &alpha, 
    db, // pointer to matrix B
    CUDA_R_16F,  // float16
    N, // leading dim of matrix B
    da, // pointer to matrix A 
    CUDA_R_16F,  // float16
    K, // leading dim of matrix A
    &beta, 
    dc,  // pointer to matrix C
    CUDA_R_16F,  // floa16 
    N,  // leading dim of matrix C
    CUBLAS_COMPUTE_16F, // accum dtype
    CUBLAS_GEMM_DEFAULT_TENSOR_OP
  );

  int iter = 100;
  cudaEventRecord(start);
  for (int i = 0; i < iter; i++) {
    cublasStatus_t status =
        cublasGemmEx(handle, CUBLAS_OP_N, CUBLAS_OP_N, N, M, K, &alpha, db,
                     CUDA_R_16F, N, da, CUDA_R_16F, K, &beta, dc, CUDA_R_16F, N,
                     CUBLAS_COMPUTE_16F, CUBLAS_GEMM_DEFAULT_TENSOR_OP);
  }

  cudaEventRecord(end);
  cudaEventSynchronize(end);

  float ms = 0.0f;
  cudaEventElapsedTime(&ms, start, end);

  std::cout << "CUBLAS TIME : " << std::fixed << std::setprecision(5)
            << (ms / iter) / 1000 << std::endl;

  long long int FLOP = 2LL * M * N * K;
  std::cout << "CUBLAS TFLOPS : " << std::fixed << std::setprecision(5)
            << ((FLOP / ((ms / iter) / 1000))) / 1e12 << std::endl;
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

void benchmark_fp16(void (*func)(half *, half *, float *), std::string name) {

  half *ha = new half[M * K];
  half *hb = new half[K * N];
  float *hc = new float[M * N]; // for kernel
  half *hd = new half[M * N]; // for cublas

  init(ha, M * K);
  init(hb, K * N);

  std::fill(hc, hc + M * N, 0.0f);
  std::fill(hd, hd + M * N, 0.0f);

  half *da, *db ,*dd;
  float* dc;
  cudaMalloc(&da, sizeof(half) * M * K);
  cudaMalloc(&db, sizeof(half) * K * N);
  cudaMalloc(&dc, sizeof(float) * M * N); // for kernel
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
  std::cout << name << " TFLOPS : " << std::fixed << std::setprecision(5)
            << ((FLOP / ((ms / ITER) / 1000))) / 1e12 << std::endl;

  std::cout << "------------------------" << std ::endl;

  launch_cublass_fp16(da, db, dd);

  cudaMemcpy(hc, dc, sizeof(float) * M * N, cudaMemcpyDeviceToHost); // kernel
  cudaMemcpy(hd, dd, sizeof(half) * M * N, cudaMemcpyDeviceToHost); // cublas
  // std::cout << __half2float(hd[10])  << std::endl;
  // bool ans = verify(hc, hd, M * N);
  bool ans = verify_cpu(hd , ha , hb);
  std::string out = ans ? "PASS" : "FAIL";
  std::cout << "\n" << out << std::endl;
}