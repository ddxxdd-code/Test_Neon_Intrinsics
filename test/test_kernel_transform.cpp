// Test filter transform
// Ceated on 2021-12-06
// Dedong Xie

#include "intrin.hpp"
#include <stdio.h>

#define IS 1
#define OS 1

void printLines(SIMD_FLOAT *matrix, int height);

int main() {

    SIMD_FLOAT in[3] = {0};
    SIMD_FLOAT out[4] = {0};

    in[0] = SIMD_SET1(0.f);
    in[1] = SIMD_SET1(0.f);
    in[2] = SIMD_SET1(0.f);

    printLines(in, 3);

    SIMD_FLOAT C1D2 = SIMD_SET1(0.5f);

    out[0] = in[0];
    
    SIMD_FLOAT V12S = SIMD_MUL(in[0],C1D2);
    V12S = SIMD_FMADD(in[IS * 2],C1D2,V12S);

    out[OS] = SIMD_FMADD(C1D2,in[IS],V12S);

    out[OS * 2] = SIMD_FNMADD(C1D2,in[IS],V12S);

    out[OS * 3] = in[IS * 2];

    printLines(out, 4);

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
