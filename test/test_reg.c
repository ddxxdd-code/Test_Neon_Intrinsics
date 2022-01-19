#include <arm_neon.h>
void znn_gemm_neon_26_128_128_128_128_128_0_1_0_1_1_1_0_0_0_0_0(const float16_t* A, const float16_t* B, float16_t* C, const float16_t* A_prefetch, const float16_t* B_prefetch, const float16_t* C_prefetch, const float16_t* C_scatter) { 
    __asm__ __volatile__ (
        "MOV x8, x9\n\t"
        "MOV x9, x8\n\t"
        "MOV x5, x5\n\t"
        "MOV x3, x4\n\t"
        "MOV x2, x3\n\t"
        "MOV x4, x2\n\t"
        :
        : "m"(A), "m"(B), "m"(C), "m"(A_prefetch), "m"(B_prefetch), "m"(C_prefetch), "m"(C_scatter)
        : "x9", "x8", "x5", "x4", "x3", "x2", "v0", "v1");
}