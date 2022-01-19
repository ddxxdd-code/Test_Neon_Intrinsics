// Test static_cast between fp16 and float
// Dedong Xie
// 2021-12-14
# include <stdio.h>
# include "intrin.hpp"

int main() {
    FLOAT16 a = 1.f;
    float b = static_cast<float>(a);
    printf("fp16 to fp32: %f to %f\n", a, b);
    float c = 1.f;
    FLOAT16 d = static_cast<FLOAT16>(c);
    printf("fp32 to fp16: %f to %f\n", c, d);
}