//
// Copyright (C) 2017 Aleksandar Zlateski <zlateski@mit.edu>
// Copyright (C) 2017 Zhen Jia <zhenj@princeton.edu>
// ---------------------------------------------------------------
//
// This program is free software: you can redistribute it and/or modify
// it under the terms of the GNU General Public License as published by
// the Free Software Foundation, either version 3 of the License, or
// (at your option) any later version.
//
// This program is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
// GNU General Public License for more details.
//
// You should have received a copy of the GNU General Public License
// along with this program.  If not, see <http://www.gnu.org/licenses/>.
//
#pragma once

#include "znn/asm/assembler.hpp"
#include "znn/asm/frame.hpp"
#include "znn/types.hpp"

#include <cstddef>
#include <dlfcn.h>
#include <fstream>
#include <mutex>
#include <string>

namespace znn
{
namespace win
{
namespace neon
{

inline std::pair<long_t, long_t> blocking(long_t n)             // typedef int64_t long_t;
{
    long_t besta = std::min(static_cast<long_t>(30), n);        // If n = 20*20, besta = 30
    long_t bestb = n % besta;                                   // bestb = 10

    for (long_t a = 30; a >= 16; --a)                           // for a = 30, 29, 28, ... 16
    {
        if (n % a == 0)                                         // If a | n
        {
            return std::make_pair(a, a);                        // return a, a
        }
        if (n % a > bestb)                                      // If n % a > n: we know that n < a
        {
            besta = a;                                          // besta = a
            bestb = n % a;                                      // bestb = n % besta
        }
    }                                                           // What is this fuction doing? Why they do it such way?

    return std::make_pair(besta, bestb);
}

inline void gemm_n_loop(frame& fr, long_t M, long_t N, long_t LDA, long_t LDB,
                        std::string A_reg, std::string B_reg, std::string C_reg,
                        std::string Bpf_reg, bool Apf0, bool Bpf0, bool Cpf0,
                        bool negate = false)
{
    auto f = fr.spawn();                // f: new frame, frame is for assembly, contains register files
    auto a = f.asmb();                  // a: frame.asmb

    f.mark_used(A_reg);                 // A_reg, B_reg are inputs, general regiaters
    f.mark_used(B_reg);

    if (Cpf0)
    {
        f.mark_used(C_reg);
    }

    if (Bpf_reg != no_reg)              // no_reg = ""
    {
        f.mark_used(Bpf_reg);
    }

    reg_t r_a_1 = no_reg;
    if (M > 1)
    {
        r_a_1 = f.get_register();       // get usable register
        a->mov(val(LDA * 4), r_a_1);    // load LDA * 4 to regiater r_a_1, each of r_a_k stores LDA * 4 * 2 values or 64 bits numbers
    }

    reg_t r_a_3 = no_reg;
    if (M > 3)
    {
        r_a_3 = f.get_register();
        a->mov(val(LDA * 4 * 3), r_a_3); // load LDA * 4 * 3 to r_a_3
    }

    reg_t r_a_5 = no_reg;
    if (M > 5)
    {
        r_a_5 = f.get_register();
        a->mov(val(LDA * 4 * 5), r_a_5); // load LDA * 4 * 5 to r_a_5
    }

    reg_t r_a_7 = no_reg;
    if (M > 7)
    {
        r_a_7 = f.get_register();
        a->mov(val(LDA * 4 * 7), r_a_7); // load LDA * 4 * 7 to r_a_7
    }

    reg_t A_reg_9 = no_reg;
    if (M > 9)
    {
        A_reg_9 = f.get_register();
        a->mov(A_reg, A_reg_9);
        a->add(val(LDA * 4 * 9), A_reg_9);
    }

    reg_t A_reg_18 = no_reg;
    if (M > 18)
    {
        A_reg_18 = f.get_register();
        a->mov(A_reg, A_reg_18);
        a->add(val(LDA * 4 * 18), A_reg_18);
    }

    reg_t A_reg_27 = no_reg;
    if (M > 27)
    {
        A_reg_27 = f.get_register();
        a->mov(A_reg, A_reg_27);
        a->add(val(LDA * 4 * 27), A_reg_27);
    }

#define W_BLOCK 16

    auto unrolled = [&](long_t i) {
        if (Bpf_reg != no_reg)
        {
            a->prefetcht1(ptr(i * LDB * 4, Bpf_reg));
        }
        if (M > 0)
        {
            a->vfmadd231ps(ptr(i * 4, A_reg), zmm(i % 2), zmm(2),
                           negate);
            if (Apf0 && i == 0)
            {
                a->prefetcht0(ptr(64 + i * 4, A_reg));
            }
        }
        if (M > 1)
        {
            a->add(r_a_1, A_reg);
            a->vfmadd231ps(ptr(i * 4, A_reg), zmm(i % 2),
                           zmm(3), negate);
            if (Apf0 && i == 1)
            {
                a->prefetcht0(ptr(i * 4 + 64, A_reg));
            }
        }
        if (M > 2)
        {
            a->add(r_a_1, A_reg);
            a->vfmadd231ps(ptr(i * 4, A_reg), zmm(i % 2),
                           zmm(4), negate);
            if (Apf0 && i == 2)
            {
                a->prefetcht0(ptr(i * 4 + 64, A_reg));
            }
        }
        if (M > 3)
        {
            a->sub(r_a_1, A_reg);
            a->sub(r_a_1, A_reg);
            a->add(r_a_3, A_reg);
            a->vfmadd231ps(ptr(i * 4, A_reg), zmm(i % 2),
                           zmm(5), negate);
            if (Apf0 && i == 3)
            {
                a->prefetcht0(ptr(i * 4 + 64, A_reg));
            }
        }

        if (Bpf0)
        {
            a->prefetcht0(ptr((i + W_BLOCK) * LDB * 4, B_reg));
        }

        if (M > 4)
        {
            a->sub(r_a_3, A_reg);
            a->add(r_a_1, A_reg);
            a->add(r_a_1, A_reg);
            a->add(r_a_1, A_reg);
            a->add(r_a_1, A_reg);
            a->vfmadd231ps(ptr(i * 4, A_reg), zmm(i % 2),
                           zmm(6), negate);
            if (Apf0 && i == 4)
            {
                a->prefetcht0(ptr(i * 4 + 64, A_reg));
            }
        }
        if (M > 5)
        {
            a->sub(r_a_1, A_reg);
            a->sub(r_a_1, A_reg);
            a->sub(r_a_1, A_reg);
            a->sub(r_a_1, A_reg);
            a->add(r_a_5, A_reg);
            a->vfmadd231ps(ptr(i * 4, A_reg), zmm(i % 2),
                           zmm(7), negate);
            if (Apf0 && i == 5)
            {
                a->prefetcht0(ptr(i * 4 + 64, A_reg));
            }
        }
        if (M > 6)
        {
            a->sub(r_a_5, A_reg);
            a->add(r_a_3, A_reg);
            a->add(r_a_3, A_reg);
            a->vfmadd231ps(ptr(i * 4, A_reg), zmm(i % 2),
                           zmm(8), negate);
            if (Apf0 && i == 6)
            {
                a->prefetcht0(ptr(i * 4 + 64, A_reg));
            }
        }
        if (M > 7)
        {
            a->sub(r_a_3, A_reg);
            a->sub(r_a_3, A_reg);
            a->add(r_a_7, A_reg);
            a->vfmadd231ps(ptr(i * 4, A_reg), zmm(i % 2),
                           zmm(9), negate);
            if (Apf0 && i == 7)
            {
                a->prefetcht0(ptr(i * 4 + 64, A_reg));
            }
        }

        if (Bpf0)
        {
            a->prefetcht0(ptr(i * LDB * 4 + 64, B_reg));
        }

        if (M > 8)
        {
            a->vfmadd231ps(ptr(i * 4, A_reg, r_a_1, 8), zmm(i % 2),
                           zmm(10), negate);
            if (Apf0 && i == 8)
            {
                a->prefetcht0(ptr(i * 4 + 64, A_reg, r_a_1, 8));
            }
        }
        if (M > 9)
        {
            a->vfmadd231ps(ptr(i * 4, A_reg_9), zmm(i % 2), zmm(11),
                           negate);
            if (Apf0 && i == 9)
            {
                a->prefetcht0(ptr(i * 4 + 64, A_reg_9));
            }
        }
        if (M > 10)
        {
            a->vfmadd231ps(ptr(i * 4, A_reg_9, r_a_1, 1), zmm(i % 2),
                           zmm(12), negate);
            if (Apf0 && i == 10)
            {
                a->prefetcht0(ptr(i * 4 + 64, A_reg_9, r_a_1, 1));
            }
        }
        if (M > 11)
        {
            a->vfmadd231ps(ptr(i * 4, A_reg_9, r_a_1, 2), zmm(i % 2),
                           zmm(13), negate);
            if (Apf0 && i == 11)
            {
                a->prefetcht0(ptr(i * 4 + 64, A_reg_9, r_a_1, 2));
            }
        }
        if (M > 12)
        {
            a->vfmadd231ps(ptr(i * 4, A_reg_9, r_a_3, 1), zmm(i % 2),
                           zmm(14), negate);
            if (Apf0 && i == 12)
            {
                a->prefetcht0(ptr(i * 4 + 64, A_reg_9, r_a_3, 1));
            }
        }
        if (M > 13)
        {
            a->vfmadd231ps(ptr(i * 4, A_reg_9, r_a_1, 4), zmm(i % 2),
                           zmm(15), negate);
            if (Apf0 && i == 13)
            {
                a->prefetcht0(ptr(i * 4 + 64, A_reg_9, r_a_1, 4));
            }
        }
        if (M > 14)
        {
            a->vfmadd231ps(ptr(i * 4, A_reg_9, r_a_5, 1), zmm(i % 2),
                           zmm(16), negate);
            if (Apf0 && i == 14)
            {
                a->prefetcht0(ptr(i * 4 + 64, A_reg_9, r_a_5, 1));
            }
        }
        if (M > 15)
        {
            a->vfmadd231ps(ptr(i * 4, A_reg_9, r_a_3, 2), zmm(i % 2),
                           zmm(17), negate);
            if (Apf0 && i == 15)
            {
                a->prefetcht0(ptr(i * 4 + 64, A_reg_9, r_a_3, 2));
            }
        }
        if (M > 16)
        {
            a->vfmadd231ps(ptr(i * 4, A_reg_9, r_a_7, 1), zmm(i % 2),
                           zmm(18), negate);
            if (Apf0 && i == 0)
            {
                a->prefetcht0(ptr(i * 4 + 64, A_reg_9, r_a_7, 1));
            }
        }
        if (M > 17)
        {
            a->vfmadd231ps(ptr(i * 4, A_reg_9, r_a_1, 8), zmm(i % 2),
                           zmm(19), negate);
            if (Apf0 && i == 1)
            {
                a->prefetcht0(ptr(i * 4 + 64, A_reg_9, r_a_1, 8));
            }
        }
        if (M > 18)
        {
            a->vfmadd231ps(ptr(i * 4, A_reg_18), zmm(i % 2), zmm(20),
                           negate);
            if (Apf0 && i == 2)
            {
                a->prefetcht0(ptr(i * 4 + 64, A_reg_18));
            }
        }
        if (M > 19)
        {
            a->vfmadd231ps(ptr(i * 4, A_reg_18, r_a_1, 1), zmm(i % 2),
                           zmm(21), negate);
            if (Apf0 && i == 3)
            {
                a->prefetcht0(ptr(i * 4 + 64, A_reg_18, r_a_1, 1));
            }
        }
        if (M > 20)
        {
            a->vfmadd231ps(ptr(i * 4, A_reg_18, r_a_1, 2), zmm(i % 2),
                           zmm(22), negate);
            if (Apf0 && i == 4)
            {
                a->prefetcht0(ptr(i * 4 + 64, A_reg_18, r_a_1, 2));
            }
        }
        if (M > 21)
        {
            a->vfmadd231ps(ptr(i * 4, A_reg_18, r_a_3, 1), zmm(i % 2),
                           zmm(23), negate);
            if (Apf0 && i == 5)
            {
                a->prefetcht0(ptr(i * 4 + 64, A_reg_18, r_a_3, 1));
            }
        }
        if (M > 22)
        {
            a->vfmadd231ps(ptr(i * 4, A_reg_18, r_a_1, 4), zmm(i % 2),
                           zmm(24), negate);
            if (Apf0 && i == 6)
            {
                a->prefetcht0(ptr(i * 4 + 64, A_reg_18, r_a_1, 4));
            }
        }
        if (M > 23)
        {
            a->vfmadd231ps(ptr(i * 4, A_reg_18, r_a_5, 1), zmm(i % 2),
                           zmm(25), negate);
            if (Apf0 && i == 7)
            {
                a->prefetcht0(ptr(i * 4 + 64, A_reg_18, r_a_5, 1));
            }
        }
        if (M > 24)
        {
            a->vfmadd231ps(ptr(i * 4, A_reg_18, r_a_3, 2), zmm(i % 2),
                           zmm(26), negate);
            if (Apf0 && i == 8)
            {
                a->prefetcht0(ptr(i * 4 + 64, A_reg_18, r_a_3, 2));
            }
        }
        if (M > 25)
        {
            a->vfmadd231ps(ptr(i * 4, A_reg_18, r_a_7, 1), zmm(i % 2),
                           zmm(27), negate);
            if (Apf0 && i == 9)
            {
                a->prefetcht0(ptr(i * 4 + 64, A_reg_18, r_a_7, 1));
            }
        }
        if (M > 26)
        {
            a->vfmadd231ps(ptr(i * 4, A_reg_18, r_a_1, 8), zmm(i % 2),
                           zmm(28), negate);
            if (Apf0 && i == 10)
            {
                a->prefetcht0(ptr(i * 4 + 64, A_reg_18, r_a_1, 8));
            }
        }
        if (M > 27)
        {
            a->vfmadd231ps(ptr(i * 4, A_reg_27), zmm(i % 2), zmm(29),
                           negate);
            if (Apf0 && i == 11)
            {
                a->prefetcht0(ptr(i * 4 + 64, A_reg_27));
            }
        }
        if (M > 28)
        {
            a->vfmadd231ps(ptr(i * 4, A_reg_27, r_a_1, 1), zmm(i % 2),
                           zmm(30), negate);
            if (Apf0 && i == 12)
            {
                a->prefetcht0(ptr(i * 4 + 64, A_reg_27, r_a_1, 1));
            }
        }
        /*
        if (M > 29)
        {
            a->vfmadd231ps(ptr(i * 4, A_reg_27, r_a_1, 2), zmm(i % 2),
                           zmm(31), negate);
            if (Apf0 && i == 13)
            {
                a->prefetcht0(ptr(i * 4 + 64, A_reg_27, r_a_1, 2));
            }
        }  
        */                                                                         // used zmm0-30
    };

    auto loop_reg = f.get_register();
    a->mov(val(0), loop_reg);
    auto loop_lab = a->label();
    a->add(val(1), loop_reg);

    // BODY

    // First unrolled
    a->vmov(ptr(0, B_reg), zmm(0));
    a->vmov(ptr(LDB * 4, B_reg), zmm(1));
    unrolled(0);

    // Mid unrolled
    for (long_t u = 1; u < W_BLOCK - 1; ++u)
    {
        a->vmov(ptr((u + 1) * 4 * LDB, B_reg), zmm((u + 1) % 2));                   // used zmm0, 1
        unrolled(u);
    }

    // Last unrolled
    unrolled(W_BLOCK - 1);

    a->add(val(W_BLOCK * 4), A_reg);
    if (M > 9)
    {
        a->add(val(W_BLOCK * 4), A_reg_9);
    }

    if (M > 18)
    {
        a->add(val(W_BLOCK * 4), A_reg_18);
    }
    if (M > 27)
    {
        a->add(val(W_BLOCK * 4), A_reg_27);
    }

    a->add(val(LDB * W_BLOCK * 4), B_reg);

    if (Bpf_reg != no_reg)
    {
        a->add(val(LDB * W_BLOCK * 4), Bpf_reg);
    }

    a->cmp(val(N / W_BLOCK), loop_reg);
    a->jl(loop_lab);

    // Revert A_reg, B_reg, Bpf_reg

    a->sub(val(N * 4), A_reg);
    a->sub(val(N * LDB * 4), B_reg);
    if (Bpf_reg != no_reg)
    {
        a->sub(val(N * LDB * 4), Bpf_reg);
    }
}

inline void gemm_k_loop(frame& fr, long_t M, long_t N, long_t K, long_t LDA,
                        long_t LDB, long_t LDC, long_t beta, std::string A_reg,
                        std::string B_reg, std::string C_reg,
                        std::string Apf_reg, std::string Bpf_reg,
                        std::string Cpf_reg, bool Apf0, bool Bpf0, bool Cpf0,
                        std::string Cscatter_reg, long_t SC_LD0, long_t SC_LD1,
                        bool negate)
{
    STRONG_ASSERT(K % 16 == 0);

    auto f = fr.spawn();
    auto a = f.asmb();

    f.mark_used(A_reg);
    f.mark_used(B_reg);
    f.mark_used(C_reg);

    if (Apf_reg != "")
        f.mark_used(Apf_reg);
    if (Bpf_reg != "")
        f.mark_used(Bpf_reg);
    if (Cpf_reg != "")
        f.mark_used(Cpf_reg);
    if (Cscatter_reg != "")
        f.mark_used(Cscatter_reg);

    auto body = [&]() {
        if (beta)
        {
            for (long_t z = 0; z < M; ++z)
            {
                a->vmov(ptr(LDC * 4 * z, C_reg), zmm(z + 2));
            }
        }
        else
        {
            a->set0(zmm(2));
            for (long_t z = 1; z < M; ++z)
            {
                a->vmov(zmm(2), zmm(z + 2));
            }
        }

        gemm_n_loop(f, M, N, LDA, LDB, A_reg, B_reg, C_reg, Bpf_reg, Apf0, Bpf0,
                    Cpf0, negate);

        for (long_t z = 0; z < M; ++z)
        {
            if (Cscatter_reg != no_reg)
            {
                a->vmovnt(zmm(z + 2), ptr(SC_LD0 * 4 * z, Cscatter_reg));                   // use at most zmm M + 2, what is the range of M?, non-temporal load
            }
            else
            {
                a->vmov(zmm(z + 2), ptr(LDC * 4 * z, C_reg));
            }
            if (Apf_reg != no_reg)
            {
                a->prefetcht1(ptr(LDA * z * 4, Apf_reg));
            }
            if (Cpf_reg != no_reg)
            {
                a->prefetcht1(ptr(LDC * z * 4, Cpf_reg));
            }
        }
    };

    if (K == 16)
    {
        body();
    }
    else
    {
        auto loop_reg = f.get_register();
        a->mov(val(0), loop_reg);
        auto loop_lab = a->label();
        a->add(val(1), loop_reg);

        body();

        a->add(val(16 * 4), C_reg);
        a->add(val(16 * 4), B_reg);

        if (Apf_reg != no_reg)
        {
            a->add(val(16 * 4), Apf_reg);
        }

        if (Cpf_reg != no_reg)
        {
            a->add(val(16 * 4), Cpf_reg);
        }

        if (Cscatter_reg != no_reg)
        {
            a->add(val(SC_LD1 * 4), Cscatter_reg);
        }

        a->cmp(val(K / 16), loop_reg);
        a->jl(loop_lab);

        a->sub(val(K * 4), C_reg);
        a->sub(val(K * 4), B_reg);

        if (Apf_reg != no_reg)
        {
            a->sub(val(K * 4), Apf_reg);
        }

        if (Cpf_reg != no_reg)
        {
            a->sub(val(K * 4), Cpf_reg);
        }

        if (Cscatter_reg != no_reg)
        {
            a->smart_sub((K / 16) * SC_LD1 * 4, Cscatter_reg);
        }
    }
}

inline std::pair<std::string, std::string>
znn_gemm(long_t M, long_t N, long_t K, long_t LDA, long_t LDB, long_t LDC,
         long_t beta, bool Apf, bool Bpf, bool Cpf, bool Apf0 = true,
         bool Bpf0 = true, bool Cpf0 = false, bool Cscatter = false,
         long_t SC_LD0 = 0, long_t SC_LD1 = 0, bool negate = false)
                                // Is called by As::tile_row_size, As::tile_col_size, Bs::tile_col_size,
                                // As::row_stride, Bs::row_stride, Cs::row_stride, 0, 1, 0,
                                // 1, al1_pf_, bl1_pf_, 0, 1, tile_stride, channel_stride
                                //
                                // M: tile_row_size, N: tile_col_size
                                // K: another's tile_col_size
                                // LDA: row_stride
                                // LDB: another's row stride
                                // LDC: c's row_stride
                                // beta: 0, Apf: 1, Bpf: 0, ...
{
    std::string name = "znn_gemm_neon_" + std::to_string(M) + "_" +
                       std::to_string(N) + "_" + std::to_string(K) + "_" +
                       std::to_string(LDA) + "_" + std::to_string(LDB) + "_" +
                       std::to_string(LDC) + "_" + std::to_string(beta) + "_" +
                       std::to_string(Apf) + "_" + std::to_string(Bpf) + "_" +
                       std::to_string(Cpf) + "_" + std::to_string(Apf0) + "_" +
                       std::to_string(Bpf0) + "_" + std::to_string(Cpf0) + "_" +
                       std::to_string(Cscatter) + "_" + std::to_string(SC_LD0) +
                       "_" + std::to_string(SC_LD1) + "_" +
                       std::to_string(negate);

    auto assm = std::make_shared<assembler>();

    // assm->jmp(name);
    // assm->put_label(name);

    frame main_frame(assm);

    STRONG_ASSERT(LDA % 16 == 0);
    STRONG_ASSERT(LDB % 16 == 0);
    STRONG_ASSERT(LDC % 16 == 0);
    STRONG_ASSERT(N % 16 == 0);
    STRONG_ASSERT(K % 16 == 0);

    reg_t A_reg = main_frame.get_register();
    reg_t B_reg = main_frame.get_register();
    reg_t C_reg = main_frame.get_register();

    assm->mov("%0", A_reg);                 // %0, %1, %2 are the input/output operand defined in your code
    assm->mov("%1", B_reg);
    assm->mov("%2", C_reg);

    reg_t Apf_reg = no_reg;
    reg_t Bpf_reg = no_reg;
    reg_t Cpf_reg = no_reg;

    if (Apf)
    {
        Apf_reg = main_frame.get_register();
        assm->mov("%3", Apf_reg);
    }

    if (Bpf)
    {
        Bpf_reg = main_frame.get_register();
        assm->mov("%4", Bpf_reg);
    }

    if (Cpf)
    {
        Cpf_reg = main_frame.get_register();
        assm->mov("%5", Cpf_reg);
    }

    reg_t Cscatter_reg = no_reg;
    if (Cscatter)
    {
        Cscatter_reg = main_frame.get_register();
        assm->mov("%6", Cscatter_reg);
    }

    auto blocks = blocking(M);                                  // blocking: return a pair, for min(30, blocka), size % blocka

    long_t whole_loops = (M - blocks.second) / blocks.first;

    if (whole_loops > 1)
    {
        auto loop_reg = main_frame.get_register();
        assm->mov(val(0), loop_reg);
        auto loop_lab = assm->label();

        assm->add(val(1), loop_reg);

        gemm_k_loop(main_frame, blocks.first, N, K, LDA, LDB, LDC, beta, A_reg,
                    B_reg, C_reg, Apf_reg, no_reg, Cpf_reg, Apf0, Bpf0, Cpf0,
                    Cscatter_reg, SC_LD0, SC_LD1, negate);

        assm->add(val(blocks.first * LDA * 4), A_reg);
        assm->add(val(blocks.first * LDC * 4), C_reg);

        if (Apf)
        {
            assm->add(val(blocks.first * LDA * 4), Apf_reg);
        }

        if (Cpf)
        {
            assm->add(val(blocks.first * LDC * 4), Cpf_reg);
        }

        if (Cscatter)
        {
            assm->add(val(blocks.first * SC_LD0 * 4), Cscatter_reg);
        }

        assm->cmp(val(whole_loops), loop_reg);
        assm->jl(loop_lab);
        main_frame.return_register(loop_reg);
    }
    else if (whole_loops == 1)
    {
        gemm_k_loop(main_frame, blocks.first, N, K, LDA, LDB, LDC, beta, A_reg,
                    B_reg, C_reg, Apf_reg, no_reg, Cpf_reg, Apf0, Bpf0, Cpf0,
                    Cscatter_reg, SC_LD0, SC_LD1, negate);

        assm->add(val(blocks.first * LDA * 4), A_reg);
        assm->add(val(blocks.first * LDC * 4), C_reg);

        if (Apf)
        {
            assm->add(val(blocks.first * LDA * 4), Apf_reg);
        }

        if (Cpf)
        {
            assm->add(val(blocks.first * LDC * 4), Cpf_reg);
        }

        if (Cscatter)
        {
            assm->add(val(blocks.first * SC_LD0 * 4), Cscatter_reg);
        }
    }

    gemm_k_loop(main_frame, blocks.second, N, K, LDA, LDB, LDC, beta, A_reg,
                B_reg, C_reg, Apf_reg, Bpf_reg, Cpf_reg, Apf0, Bpf0, Cpf0,
                Cscatter_reg, SC_LD0, SC_LD1, negate);

    return std::make_pair(name, assm->str());
}

extern "C" {
typedef void (*znn_gemm_t)(const float16_t*, const float16_t*, float16_t*, const float16_t*,
                           const float16_t*, const float16_t*, float16_t*);
}

struct znn_gemm_cache_t
{
    std::mutex                                          lock;
    std::map<std::string, std::pair<znn_gemm_t, void*>> cache;
};

inline static znn_gemm_cache_t znn_gemm_cache;

inline void clear_znn_gemms()
{
    std::unique_lock<std::mutex> guard(znn_gemm_cache.lock);
    for (auto& x : znn_gemm_cache.cache)
    {
        dlclose(x.second.second);
    }
    znn_gemm_cache.cache.clear();
}

inline znn_gemm_t get_znn_gemm(long_t M, long_t N, long_t K, long_t LDA,
                               long_t LDB, long_t LDC, long_t beta, bool Apf,
                               bool Bpf, bool Cpf, bool Apf0 = true,
                               bool Bpf0 = true, bool Cpf0 = false,
                               bool Cscatter = false, long_t SC_LD0 = 0,
                               long_t SC_LD1 = 0, bool negate = false) 
                                // Is called by As::tile_row_size, As::tile_col_size, Bs::tile_col_size,
                                // As::row_stride, Bs::row_stride, Cs::row_stride, 0, 1, 0,
                                // 1, al1_pf_, bl1_pf_, 0, 1, tile_stride, channel_stride
                                //
                                // M: tile_row_size, N: tile_col_size
                                // K: another's tile_col_size
                                // LDA: row_stride
                                // LDB: another's row stride
                                // LDC: c's row_stride
                                // beta: 0, Apf: 1, Bpf: 0, ...
{
    if (!Cscatter)
    {
        SC_LD0 = 0;
        SC_LD1 = 0;
    }

    std::unique_lock<std::mutex> guard(znn_gemm_cache.lock);

    std::string name = "znn_gemm_neon_" + std::to_string(M) + "_" +
                       std::to_string(N) + "_" + std::to_string(K) + "_" +
                       std::to_string(LDA) + "_" + std::to_string(LDB) + "_" +
                       std::to_string(LDC) + "_" + std::to_string(beta) + "_" +
                       std::to_string(Apf) + "_" + std::to_string(Bpf) + "_" +
                       std::to_string(Cpf) + "_" + std::to_string(Apf0) + "_" +
                       std::to_string(Bpf0) + "_" + std::to_string(Cpf0) + "_" +
                       std::to_string(Cscatter) + "_" + std::to_string(SC_LD0) +
                       "_" + std::to_string(SC_LD1) + "_" +
                       std::to_string(negate);

    if (znn_gemm_cache.cache.count(name))
    {
        return znn_gemm_cache.cache[name].first;
    }

    std::system("mkdir -p gen");

    auto fn = znn_gemm(M, N, K, LDA, LDB, LDC, beta, Apf, Bpf, Cpf, Apf0, Bpf0,
                       Cpf0, Cscatter, SC_LD0, SC_LD1, negate);

    if (znn_gemm_cache.cache.count(fn.first))
    {
        return znn_gemm_cache.cache[fn.first].first;
    }

    std::string fname = std::string("./gen/") + fn.first + ".c";

    std::ofstream ofs(fname.c_str());

    ofs << "void " + fn.first +
               "(const float16_t* A, const float16_t* B, float16_t* C, const "
               "float16_t* A_prefetch, const float16_t* B_prefetch, const float16_t* "
               "C_prefetch, const float16_t* C_scatter) { \n    __asm__ "
               "__volatile__ (\n";

    ofs << fn.second;

    ofs << "        :\n"
           "        : \"m\"(A), \"m\"(B), \"m\"(C), \"m\"(A_prefetch), "
           "\"m\"(B_prefetch), \"m\"(C_prefetch), \"m\"(C_scatter)\n"
           "        : \"X0\", \"X1\", \"X2\", \"X3\", "
           "\"X4\", \"X5\", \"X6\", \"X9\", \"X10\",\n"
           "\"X11\", \"X12\", \"X13\", \"X14\", \"X15\", "
           "\"V0\", \"V1\", \"V2\", \"V3\",\n"
           "\"V4\", \"V5\", \"V6\", \"V7\", \"V8\", "
           "\"V9\", \"V10\", \"V11\",\n"
           "\"V12\", \"V13\", \"V14\", \"V15\", \"V16\", \"V17\", \"V18\", \n"
           "\"V19\", \"V20\", \"V21\", \"V22\", \"V23\", \"V24\", \"V25\", \"V26\", \n"
           "\"V27\", \"V28\", \"V29\", \"V30\", \"V31\");\n}\n";

    std::string compile_command = std::string("gcc -shared -Wl,-soname,") +
                                  fn.first + ".so -O3 -DNDEBUG -fPIC " + fname +
                                  " -march=armv8.2-a+fp16 -o ./gen/" + fn.first +
                                  ".so";

    // std::string compile_command = std::string("gcc -shared -Wl,-soname,") +
    //                               fn.first + ".so -O3 -DNDEBUG -fPIC " +
    //                               fname +
    //                               " -mavx512f -mavx512pf -o ./gen/" +
    //                               fn.first +
    //                               ".so";

    ofs.flush();

    system(compile_command.c_str());

    std::string so_name = std::string("./gen/") + fn.first + ".so";

    void* myso = dlopen(so_name.c_str(), RTLD_NOW);

    if (!myso)
    {
        std::printf("%s\n", dlerror());
        exit(EXIT_FAILURE);
    }

    STRONG_ASSERT(myso);

    void* myfn = dlsym(myso, fn.first.c_str());

    STRONG_ASSERT(myfn);

    znn_gemm_cache.cache[fn.first] =
        std::make_pair(reinterpret_cast<znn_gemm_t>(myfn), myso);

    return reinterpret_cast<znn_gemm_t>(myfn);
}

} // namespace neon
} // namespace fft
} // namespace znn
