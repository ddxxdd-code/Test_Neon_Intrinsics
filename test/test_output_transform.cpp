// Test output transform
// Ceated on 2021-12-07
// Dedong Xie

#include "intrin.hpp"
#include <stdio.h>

#define IS 1
#define OS 1

void printLines(SIMD_FLOAT *matrix, int height);

int main() {

    SIMD_FLOAT in[4] = {0};
    SIMD_FLOAT out[2] = {0};

    in[0] = SIMD_SET1(1.f);
    in[1] = SIMD_SET1(2.f);
    in[2] = SIMD_SET1(3.f);
    in[3] = SIMD_SET1(4.f);

    printLines(in, 2);

    out[0] = SIMD_ADD(in[0],in[IS]);
    out[0] = SIMD_ADD(out[0],in[IS * 2]);

    out[OS] = SIMD_SUB(in[IS],in[IS * 2]);
    out[OS] = SIMD_SUB(out[OS],in[IS * 3]);

    printLines(out, 2);

    return 0;
}

void printLines(SIMD_FLOAT *matrix, int height) {
    for (int i = 0; i < height; i++) {
        for (int j = 0; j < 8; j++) {
            printf("%f ", matrix[i][j]);
        }
        printf("\n");
    }
    printf("==============================================\n");
}