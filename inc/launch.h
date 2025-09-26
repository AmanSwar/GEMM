#pragma once
#include <cuda_runtime.h>
#include <string>

void launch_cublass(int M, int N, int K, float *ha, float *hb, float *hc,
                    float *da, float *db, float *dc);

void launch_naive_kernel(float *a, float *b, float *out, int M, int N, int K);

namespace gemm{
void init(float *arr, int N);
void benchmark(void (*func)(float *, float *, float *, int, int, int),
               std::string name, float *a, float *b, float *c, int M, int N,
               int K);

bool verify(float *kernel_out, float *cublas_out, int N);
}