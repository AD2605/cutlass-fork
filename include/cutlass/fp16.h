/***************************************************************************************************
 * Copyright (c) 2024 - 2024 Codeplay Software Ltd. All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 * 1. Redistributions of source code must retain the above copyright notice, this
 * list of conditions and the following disclaimer.
 *
 * 2. Redistributions in binary form must reproduce the above copyright notice,
 * this list of conditions and the following disclaimer in the documentation
 * and/or other materials provided with the distribution.
 *
 * 3. Neither the name of the copyright holder nor the names of its
 * contributors may be used to endorse or promote products derived from
 * this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 * DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
 * FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
 * DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
 * SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
 * CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
 * OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 *
 **************************************************************************************************/
#pragma once

#if defined(CUTLASS_ENABLE_SYCL)
#include <sycl/sycl.hpp>
#else
#include <cuda_fp16.h>
#endif

// Add these definitions in the cutlass namespace, so they do not clash with the ones in cuda
namespace cutlass {

#if defined(CUTLASS_ENABLE_SYCL)
    using half = sycl::half;
    using half2 = sycl::half2;
    using __half2 = sycl::half2;
    using __half = sycl::half;
#else
    using half_raw = __half_raw;
    using half2 = __half2;
#endif

    CUTLASS_HOST_DEVICE
    float half2float (half const& flt) {
#if defined(CUTLASS_ENABLE_SYCL)
      return static_cast<float>(flt);
#else
      return __half2float(flt);
#endif
    }

    CUTLASS_HOST_DEVICE
    half float2half (float const& flt) {
#if defined(CUTLASS_ENABLE_SYCL)
      return static_cast<half>(flt);
#else
      return __float2half(flt);
#endif
    }

    CUTLASS_HOST_DEVICE
    half float2half_rn (float const& flt) {
#if defined(CUTLASS_ENABLE_SYCL)
      return static_cast<half>(flt);
#else
      return __float2half_rn(flt);
#endif
    }

    CUTLASS_HOST_DEVICE
    int int2half_rn (half const& flt) {
#if defined(CUTLASS_ENABLE_SYCL)
      return static_cast<int>(flt);
#else
      return __int2half_rn(flt);
#endif
    }

    CUTLASS_HOST_DEVICE
    half2 hsub2(const half2 a, const half2 b) {
#if defined(CUTLASS_ENABLE_SYCL)
      return a - b;
#else
      return __hsub2(a, b);
#endif
    }

#if defined (CUTLASS_ENABLE_SYCL)

// Move to SYCLCompat ?

#define HALF_HALF2_BINARY_OP(name, operation) \
      CUTLASS_HOST_DEVICE \
      __half __h##name(__half a, __half b) { \
            return a operation b;\
      } \
      CUTLASS_HOST_DEVICE\
      __half2 __h##name##2(__half2 a, __half2 b) {\
            return a operation b;\
      }

#define HALF_BINARY_OP(name, operation) \
      CUTLASS_HOST_DEVICE \
      __half __h##name(__half a, __half b) { \
            return a operation b;\
      }

#define HALF_HALF2_MINMAX_OP(name, operation) \
      CUTLASS_HOST_DEVICE \
      __half __h##name(__half a, __half b) { \
            return a operation b ? a: b;\
      } \
      CUTLASS_HOST_DEVICE\
      __half2 __h##name##2(__half2 a, __half2 b) {\
            return __half2( \
                  __h##name(a[0], b[0]), \
                  __h##name(a[1], b[1])\
            ); \
      }


HALF_HALF2_BINARY_OP(add, +)
HALF_HALF2_BINARY_OP(sub, -)
HALF_HALF2_BINARY_OP(mul, *)
HALF_HALF2_BINARY_OP(div, *)


HALF_BINARY_OP(eq, ==)
HALF_BINARY_OP(ne, !=)
HALF_BINARY_OP(lt, <)
HALF_BINARY_OP(le, <=)
HALF_BINARY_OP(gt, >)
HALF_BINARY_OP(ge, >=)

HALF_HALF2_MINMAX_OP(min, <)
HALF_HALF2_MINMAX_OP(max, >)

CUTLASS_HOST_DEVICE
__half __hneg(__half a) {return -a; }

CUTLASS_HOST_DEVICE
__half2 __hneg2(__half2 a) {return -a; }

CUTLASS_HOST_DEVICE
__half2 __half2half2(__half a) { return __half2(a, a); }

CUTLASS_HOST_DEVICE
__half2 __h2div(__half2 a, __half2 b) { return __hdiv2(a, b); }

CUTLASS_HOST_DEVICE
__half __hfma_relu(__half a, __half b, __half c) {
      #if (defined(__CUDA__ARCH__) || \
          defined(__SYCL_CUDA_ARCH__)) && \
          ((__CUDA_ARCH__ >= 800) || \
              (__SYCL_CUDA_ARCH__ >= 800))
          __half d;

          asm volatile("\n\t"
          "fma.rn.relu.f16 %0, %1, %2, %3; \n\t"
          "\n\t" : "=h"(*reinterpret_cast<unsigned short*>(&d)) : 
          "h"(*reinterpret_cast<unsigned short*>(&a)), 
          "h"(*reinterpret_cast<unsigned short*>(&b)), 
          "h"(*reinterpret_cast<unsigned short*>(&c)));
          return d;
      #else
            __half d;
            d = a * b + c;
            if (d < __half(0)) d = __half(0);
            return d;
      #endif
}


CUTLASS_HOST_DEVICE
__half2 __hfma2_relu(__half2 a, __half2 b, __half2 c) {
      #if (defined(__CUDA__ARCH__) || \
          defined(__SYCL_CUDA_ARCH__)) && \
          ((__CUDA_ARCH__ >= 800) || \
              (__SYCL_CUDA_ARCH__ >= 800))
          __half2 d;

          asm volatile("\n\t"
          "fma.rn.relu.f16x2 %0, %1, %2, %3; \n\t"
          "\n\t" : "=h"(*reinterpret_cast<unsigned short*>(&d)) : 
          "h"(*reinterpret_cast<unsigned int*>(&a)), 
          "h"(*reinterpret_cast<unsigned int*>(&b)), 
          "h"(*reinterpret_cast<unsigned int*>(&c)));
          return d;
      #else
            __half2 d;
            d = a * b + c;
            if (d[0] < 0) d[0] = __half(0);
            if (d[1] < 0) d[1] = __half(0);
            return d;
      #endif
}

CUTLASS_HOST_DEVICE
__half __hfma(__half a, __half b, __half c) {
      return a * b + c;
}

CUTLASS_HOST_DEVICE
__half2 __hfma2(__half2 a, __half2 b, __half2 c) {
      return a * b + c;
}

__half __uint2half_rn(const unsigned& n) {
      return __half(n);
}

#endif
}
