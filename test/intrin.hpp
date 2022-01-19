// Intrinsic for NEON instruction set
// Created by Dedong Xie
// On 2021-07-19
//
#include <arm_neon.h>

#define SIMD_WIDTH 16

#define SIMD_FMADD(a, b, c) vfmaq_f16(c, a, b)          // vmlaq(a, b, c): a * b + c -> dst
#define SIMD_SET1 vdupq_n_f16            // set1_ps(a): broadcast 32-bits value a to all elements of dst
#define SIMD_LOAD vldlq_dup_f16            // load_ps(const *mem_addr): load 128-bits from memory by mem_addr into dst.mem_addr (dst[511:0] := MEM[mem_addr+511:mem_addr])

#define SIMD_STORE vst1q_f16          // store_ps(mem_addr, a): store 128bits from a to memory at mem_addr
#define SIMD_STREAM vst1q_f16       // stream_ps(mem_addr, a): store 128-bits from a into memory through non-temporal memory
#define SIMD_ZERO() vdupq_f16(0.0)         // setzero_ps(): set dst to be all 0, return ector of type __m512 all elements zero

#define SIMD_FMSUB(a, b, c) vsubq_f16(vmulq_16(a, b),c)          // fmsub_ps(a, b, c): dst := a * b - c, returns dst
#define SIMD_FNMADD(a, b, c) vfmsq_f16(c, a, b)        // fnmadd_ps(a, b, c): dst := - a * b + c
#define SIMD_MUL vmulq_f16              // mul_ps(a, b): dst := a * b
#define SIMD_ADD vaddq_f16              // add_ps(a, b): dst := a + b
#define SIMD_SUB vsubq_f16              // sub_ps(a, b): dst := a - b

#define SIMD_MAX vmaxq_f16              // max_ps(a, b): dst[i+31:i] := MAX(a[i+31:i], b[i+31:i])

#define SIMD_FLOAT float16x8_t
#define FLOAT16 float16_t

#define SIMD_MAX_BLOCK 31
#define SIMD_W_BLOCK 12

#define SIMD_NUM_REGISTERS 16

#define SIMD_PREFETCH_L1(address)                                              \
    __prefetch(address)

#define SIMD_PREFETCH_L2(address)                                              \
    __prefetch(address)

#define CACHELINE_SIZE 16