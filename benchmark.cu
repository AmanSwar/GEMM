#include <cuda_runtime.h>
#include <iostream>

#include "inc/launch.h"

#define M 2048
#define N 2048
#define K 1024


int main() {
  float *ha = new float[M * K];
  float *hb = new float[K*N];
  float *hc = new float[M * N];
  float *hd = new float[M * N];

  gemm::init(ha, M * K);
  gemm::init(hb, K * N);

  std::fill(hc, hc + M * N, 0.0f);

  float *da, *db, *dc , *dd;
  cudaMalloc(&da, sizeof(float) * M * K);
  cudaMalloc(&db, sizeof(float) * K * N);
  cudaMalloc(&dc, sizeof(float) * M * N);
  cudaMalloc(&dd, sizeof(float) * M * N);

  cudaMemcpy(da , ha , sizeof(float) * M * K , cudaMemcpyHostToDevice);
  cudaMemcpy(db , hb , sizeof(float) * K * N , cudaMemcpyHostToDevice);

  launch_cublass(M , N , K , ha , hb , hc , da , db , dc);
  gemm::benchmark(launch_naive_kernel, "naive", da, db, dd, M, N, K);
  cudaDeviceSynchronize();

  cudaMemcpy(hd , dd , sizeof(float) * M * N , cudaMemcpyDeviceToHost);

  bool ans = gemm::verify(hd , hc , M*N);
  std::cout << ans << std::endl;

}