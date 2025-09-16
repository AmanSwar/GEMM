#pragma once

// async copy PTX 
#include <__clang_cuda_runtime_wrapper.h>
#define COMMIT_GROUP_PTX asm volatile("cp.async.commit_group;"); //telling the GPU - "Hey I am done issuing async copies , you can start transfering them "

#define WAIT_GROUP_PTX(N) asm volatile("cp.async.wait_group %0;" : : "n"(N)) //blocks atleast N grps of async copy

#define WAIT_ALL_PTX asm volatile("cp.async.wait_all ;") // blocks until all outstanding async copies are completed.


__device__ __forceinline__ unsigned long long cvta_to_shared(float* generic_ptr){
    unsigned long long shared_addr;
    asm volatile("cvta.to.shared.u64 %0 %1" : "=l"(shared_addr) : "l"(generic_ptr)); // convert address to shared mem addr

    return shared_addr;
}


__device__ __forceinline__ float ldg32_guard_ptx(const float* ptr , int guard){
    float reg;

    asm volatile(
        "{.reg .pred p;\n\t" // declare a predicate register
        "setp.ne.u32 p , %2 , 0; \n\t" // p = 1 if guard != 0
        "@p ld.global.f32 %0 , [%1]; \n\t" // if p == 1 then load a fp32 value from ptr and store in register
        "}" : "=f"(reg) : "l"(ptr) , "r"(guard) // output -> reg | input  -> ptr and guard
    );
    return reg;
}


__device__ __forceinline__ float ldg32_guard_mov0_ptx(const float* ptr, int guard){
    float reg;

    asm volatile(
        "{;\n\t"
            ".reg .pred p;\n\t" //declare pred reg
            "setp.ne.u32 p, %2 , 0 ; \n\t" // if p != 0
            "@!p mov.b32 %0 , 0 ; \n\t" // if p == 0 then : reg = 0
            "@p ld.global.f32 %0 , [%1]; \n\t" // if p == 1 then : load fp32 value from ptr to reg
        "}" : "=f"(reg) : "l"(ptr) , "r"(guard) 
    );

    return reg;
}

__device__ __forceinline__ void sts128_ptx(
    float reg0, float reg1, float reg2 , float reg3,
    uint64_t addr
){
    asm volatile(
        "st.shared.v4.f32 [%0] , {%1 , %2 , %3 , %4}; \n\t" : : "l"(addr) , "f"(reg0), "f"(reg1), "f"(reg2), "f"(reg3) // store 4 float into 1 vector register
    );

}

__device__ __forceinline__ void lds128_ptx(float4* addr){
    float4 result;
    asm volatile(
        "ld.shared.v4.f32 {%0 , %1 , %2 , %3} , [%4];\n\t" : "=f"(result.x) , "=f"(result.y) , "=f"(result.z) , "=f"(result.w) : "l"(addr) // load d4 floats into 1 vectori register
    );
}


__device__ __forceinline__ void sts32_ptx(float reg ,uint64_t addr){
    asm volatile(
        "st.shared.f32 [%0] , %1;\n" : : "l"(addr) , "f"(reg) // store the value from reg to specified addr in shared mem
    );
}

__device__ __forceinline__ void stg32_guard_ptx(float reg , float* ptr,  int guard){
    asm volatile(
        "{;\n\t"
            ".reg .pred p; \n\t"  //declare pred reg
            "setp.ne.u32 p , %2 , 0; \n\t" // if guard == 0 -> pred = False
            "@p st.global.f32 [%0] , %1;\n\t" // if pred == True -> store value from reg to shared mem addr

        "}" : : "l"(ptr) , "f"(reg) , "r"(guard)
        );

}

__device__ __forceinline__ void cp_async_guard_ptx(float* addr , float* ptr , int guard){
    asm volatile(
        "{;\n\t"
            ".reg .pred p; \n\t" // declare pred reg
            "setp.ne.u32 p , %2 , 0;\n\t"
            "@p cp.async.ca.shared.global [%0] , [%1] , 4;" // if true -> async copy 4bytes from ptr to shared mem addr
            "}" : : "l"(addr) , "l"(ptr) , "r"(guard)
    );

}

__device__ __forceinline__ void cp_async_ignore_src_ptx(float* addr , float* ptr , int guard){
  asm volatile(
      "{;\n\t"
      ".reg .pred p;\n\t"
      "setp.ep.u32 p , %2 , 0; \n\t" // if guard == 0 then pred reg = true
      // if p==0 -> normal asyc copy from ptr -> addr
      // if p ==1 -> ignore the src fill shared mem addr with 0
      "cp.async.ca.shared.global [%0] , [%1] , 4 , p;}\n"
      :
      : "l"(addr), "l"(ptr), "r"(guard));
}