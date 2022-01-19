// test ARM SIMD intrinsics on Graviton2 machine
// Dedong Xie
// 2021-07-18

#include <arm_neon.h>
#include <stdio.h>

int main(int argc, char *argv[]) {
    int16_t x = 10;
    int16_t y = 20;
    int16x8_t r = vdupq_n_s16(x);
    int16x8_t m = vdupq_n_s16(y);
    int16x8_t s = vaddq_s16(r, m);
    for (int i = 0; i < 8; i++) {
        printf("%d ", s[i]);
    }
    return 0;
}