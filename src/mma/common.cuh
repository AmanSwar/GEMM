#pragma once

#include <cuda_runtime.h>
#include <cuda_fp16.h>

//basic
#define WARP_SIZE 32

#define LDST32BITS(value) (reinterpret_cast<half2 *>(&(value))[0])
#define LDST64BITS(value) (reinterpret_cast<float2*>(&(value))[0])
#define LDST128BITS(value) (reinterpret_cast<float4*>(&(value))[0])


//load matrix

//load one matrix of size 8x8 and dtype = half from shared mem (addr) to registers (R)
#define LDMATRIX_X1(R, addr)\
  asm volatile("ldmatrix.sync.aligned.x1.m8n8.shared.b16 {%0}, [%1];\n" \
               : "=r"(R) \
               : "r"(addr))

//load 2 matrix of size 8x8 (half) from shared mem to registers (128 elements = 256 bytes)
#define LDMATRIX_X2(R0, R1, addr)\
  asm volatile("ldmatrix.sync.aligned.x2.m8n8.shared.b16 {%0, %1}, [%2];\n"\
               : "=r"(R0), "=r"(R1)\
               : "r"(addr))

// load 4 tiles of size 8x8 (half) from shared mem to registers
#define LDMATRIX_X4(R0, R1, R2, R3, addr)                                      \
  asm volatile(                                                                \
      "ldmatrix.sync.aligned.x4.m8n8.shared.b16 {%0, %1, %2, %3}, [%4];\n"     \
      : "=r"(R0), "=r"(R1), "=r"(R2), "=r"(R3)                                 \
      : "r"(addr))

// load one tile but transposed (shared mem (addr) -> registers (R) transposed)
#define LDMATRIX_X1_T(R, addr)                                                 \
  asm volatile("ldmatrix.sync.aligned.x1.trans.m8n8.shared.b16 {%0}, [%1];\n"  \
               : "=r"(R)                                                       \
               : "r"(addr))

#define LDMATRIX_X2_T(R0, R1, addr)                                            \
  asm volatile(                                                                \
      "ldmatrix.sync.aligned.x2.trans.m8n8.shared.b16 {%0, %1}, [%2];\n"       \
      : "=r"(R0), "=r"(R1)                                                     \
      : "r"(addr))

      
#define LDMATRIX_X4_T(R0, R1, R2, R3, addr)                                    \
  asm volatile(                                                                \
      "ldmatrix.sync.aligned.x4.trans.m8n8.shared.b16 {%0, %1, %2, %3}, "      \
      "[%4];\n"                                                                \
      : "=r"(R0), "=r"(R1), "=r"(R2), "=r"(R3)                                 \
      : "r"(addr))

// math utils
__device__ __host__ __forceinline__ int div_ceil(int a, int b) {
  return (a % b != 0) ? (a / b + 1) : (a / b);
}