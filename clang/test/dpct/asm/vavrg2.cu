// UNSUPPORTED: cuda-8.0, cuda-9.0, cuda-9.1, cuda-9.2, cuda-10.0, cuda-10.1, cuda-10.2
// UNSUPPORTED: v8.0, v9.0, v9.1, v9.2, v10.0, v10.1, v10.2
// RUN: dpct --format-range=none -out-root %T/vavrg2 %s --cuda-include-path="%cuda-path/include" -- -std=c++14 -x cuda --cuda-host-only
// RUN: FileCheck %s --match-full-lines --input-file %T/vavrg2/vavrg2.dp.cpp
// RUN: %if build_lit %{icpx -c -fsycl %T/vavrg2/vavrg2.dp.cpp -o %T/vavrg2/vavrg2.dp.o %}

// clang-format off
#include <cuda_runtime.h>

__global__ void varvg2() {
  int a, b, c, d;
  // CHECK: d = dpct::extend_vavrg2<int32_t, uint32_t, int32_t>(a, b, c);
  // CHECK-NEXT: d = dpct::extend_vavrg2_sat<int32_t, uint32_t, int32_t>(a, b, c);
  // CHECK-NEXT: d = dpct::extend_vavrg2_add<int32_t, uint32_t, int32_t>(a, b, c);
  asm("vavrg2.s32.u32.s32 %0, %1, %2, %3;" : "=r"(d) : "r"(a), "r"(b), "r"(c));
  asm("vavrg2.s32.u32.s32.sat %0, %1, %2, %3;" : "=r"(d) : "r"(a), "r"(b), "r"(c));
  asm("vavrg2.s32.u32.s32.add %0, %1, %2, %3;" : "=r"(d) : "r"(a), "r"(b), "r"(c));
}

// clang-format on
