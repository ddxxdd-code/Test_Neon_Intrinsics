// Test filter transform
// Ceated on 2021-07-26
// Dedong Xie

#include "intrin.hpp"
#include <stdio.h>

#define IS 1
#define OS 1

void printLines(SIMD_FLOAT *matrix, int height);

int main() {
    SIMD_FLOAT in[4] = {0};
    SIMD_FLOAT out[4] = {0};
    in[0] = SIMD_SET1(1.0);
    in[1] = SIMD_SET1(1.0);
    in[2] = SIMD_SET1(1.0);
    in[3] = SIMD_SET1(1.0);
    printLines(in, 4);

    out[0]      = SIMD_SUB(in[0], in[IS * 2]);
    out[OS * 1] = SIMD_ADD(in[IS], in[IS * 2]);
    out[OS * 2] = SIMD_SUB(in[IS * 2], in[IS]);
    out[OS * 3] = SIMD_SUB(in[IS], in[IS * 3]);

    // SIMD_FLOAT C1D2 = SIMD_SET1(0.5f);

    // out[0] = in[0];
 
    // SIMD_FLOAT V12S = SIMD_MUL(in[0],C1D2);
    // V12S = SIMD_FMADD(in[IS * 2],C1D2,V12S);

    // out[OS] = SIMD_FMADD(C1D2,in[IS],V12S);

    // out[OS * 2] = SIMD_FNMADD(C1D2,in[IS],V12S);

    // out[OS * 3] = in[IS * 2];

    printf("\n");
    printLines(out, 4);
    printf("\n");

    SIMD_FLOAT t[4];
    t[0] = SIMD_SET1(1.0f);
    t[0] = vsetq_lane_f16(2.0f, t[0], 1);
    t[1] = SIMD_SET1(2.0f);
    t[2] = SIMD_SUB(t[0], t[1]);
    t[3] = SIMD_ADD(t[0], t[1]);
    printLines(t, 4);
    return 0;
}

void printLines(SIMD_FLOAT*matrix, int height) {
    for (int i = 0; i < height; i++) {
        for (int j = 0; j < 8; j++) {
            //float result = static_cast<float>(*(matrix+i*8+j));
            printf("%f ", matrix[i][j]);
        }
        printf("\n");
    }
}