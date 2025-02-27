// RUN: dpct --format-range=none -out-root %T/cuda-get-error-string %s --cuda-include-path="%cuda-path/include" -- -std=c++14 -x cuda --cuda-host-only
// RUN: FileCheck %s --match-full-lines --input-file %T/cuda-get-error-string/cuda-get-error-string.dp.cpp
// RUN: %if build_lit %{icpx -c -fsycl %T/cuda-get-error-string/cuda-get-error-string.dp.cpp -o %T/cuda-get-error-string/cuda-get-error-string.dp.o %}

#include "cuda.h"

int printf(const char *format, ...);

// CHECK: /*
// CHECK-NEXT: DPCT1009:{{[0-9]+}}: SYCL reports errors using exceptions and does not use error codes. Please replace the "get_error_string_dummy(...)" with a real error-handling function.
// CHECK-NEXT: */
// CHECK-NEXT: #define PRINT_ERROR_STR(X) printf("%s\n", dpct::get_error_string_dummy(X))
#define PRINT_ERROR_STR(X) printf("%s\n", cudaGetErrorString(X))

// CHECK:  /*
// CHECK-NEXT:  DPCT1009:{{[0-9]+}}: SYCL reports errors using exceptions and does not use error codes. Please replace the "get_error_string_dummy(...)" with a real error-handling function.
// CHECK-NEXT:  */
// CHECK-NEXT: #define PRINT_ERROR_STR2(X)\
// CHECK-NEXT:  printf("%s\n", dpct::get_error_string_dummy(X))
#define PRINT_ERROR_STR2(X)\
  printf("%s\n", cudaGetErrorString(X))

// CHECK: /*
// CHECK-NEXT: DPCT1009:{{[0-9]+}}: SYCL reports errors using exceptions and does not use error codes. Please replace the "get_error_string_dummy(...)" with a real error-handling function.
// CHECK-NEXT: */
// CHECK-NEXT: #define PRINT_ERROR_STR3(X)\
// CHECK-NEXT:   printf("%s\
// CHECK-NEXT:          \n", dpct::get_error_string_dummy(X))
#define PRINT_ERROR_STR3(X)\
  printf("%s\
         \n", cudaGetErrorString(X))

// CHECK: /*
// CHECK-NEXT: DPCT1009:{{[0-9]+}}: SYCL reports errors using exceptions and does not use error codes. Please replace the "get_error_string_dummy(...)" with a real error-handling function.
// CHECK-NEXT: */
// CHECK-NEXT: #define PRINT_ERROR_NAME(X) printf("%s\n", dpct::get_error_string_dummy(X))
#define PRINT_ERROR_NAME(X) printf("%s\n", cudaGetErrorName(X))

// CHECK:   /*
// CHECK-NEXT:   DPCT1009:{{[0-9]+}}: SYCL reports errors using exceptions and does not use error codes. Please replace the "get_error_string_dummy(...)" with a real error-handling function.
// CHECK-NEXT:   */
// CHECK-NEXT: #define PRINT_ERROR_NAME2(X)\
// CHECK-NEXT:   printf("%s\n", dpct::get_error_string_dummy(X))
#define PRINT_ERROR_NAME2(X)\
  printf("%s\n", cudaGetErrorName(X))

// CHECK: /*
// CHECK-NEXT: DPCT1009:{{[0-9]+}}: SYCL reports errors using exceptions and does not use error codes. Please replace the "get_error_string_dummy(...)" with a real error-handling function.
// CHECK-NEXT: */
// CHECK-NEXT: #define PRINT_ERROR_NAME3(X)\
// CHECK-NEXT:   printf("%s\
// CHECK-NEXT:          \n", dpct::get_error_string_dummy(X))
#define PRINT_ERROR_NAME3(X)\
  printf("%s\
         \n", cudaGetErrorName(X))

// CHECK: /*
// CHECK-NEXT: DPCT1009:{{[0-9]+}}: SYCL reports errors using exceptions and does not use error codes. Please replace the "get_error_string_dummy(...)" with a real error-handling function.
// CHECK-NEXT: */
// CHECK-NEXT: #define PRINT_ERROR_STR_NAME(X)\
// CHECK-NEXT:   printf("%s\
// CHECK-NEXT:          %s\
// CHECK-NEXT:          \n", dpct::get_error_string_dummy(X),\
// CHECK-NEXT:          dpct::get_error_string_dummy(X))
#define PRINT_ERROR_STR_NAME(X)\
  printf("%s\
         %s\
         \n", cudaGetErrorString(X),\
         cudaGetErrorName(X))

const char *test_function() {
  // CHECK: /*
  // CHECK-NEXT: DPCT1010:{{[0-9]+}}: SYCL uses exceptions to report errors and does not use the error codes. The call was replaced with 0. You need to rewrite this code.
  // CHECK-NEXT: */
  PRINT_ERROR_STR(cudaGetLastError());
  // CHECK: /*
  // CHECK-NEXT: DPCT1010:{{[0-9]+}}: SYCL uses exceptions to report errors and does not use the error codes. The call was replaced with 0. You need to rewrite this code.
  // CHECK-NEXT: */
  PRINT_ERROR_NAME(cudaGetLastError());
  PRINT_ERROR_STR(cudaSuccess);
  PRINT_ERROR_NAME(cudaSuccess);

  // CHECK: /*
  // CHECK-NEXT: DPCT1010:{{[0-9]+}}: SYCL uses exceptions to report errors and does not use the error codes. The call was replaced with 0. You need to rewrite this code.
  // CHECK-NEXT: */
  PRINT_ERROR_STR2(cudaGetLastError());
  // CHECK: /*
  // CHECK-NEXT: DPCT1010:{{[0-9]+}}: SYCL uses exceptions to report errors and does not use the error codes. The call was replaced with 0. You need to rewrite this code.
  // CHECK-NEXT: */
  PRINT_ERROR_NAME2(cudaGetLastError());
  PRINT_ERROR_STR2(cudaSuccess);
  PRINT_ERROR_NAME2(cudaSuccess);

  // CHECK: /*
  // CHECK-NEXT: DPCT1010:{{[0-9]+}}: SYCL uses exceptions to report errors and does not use the error codes. The call was replaced with 0. You need to rewrite this code.
  // CHECK-NEXT: */
  PRINT_ERROR_STR3(cudaGetLastError());
  // CHECK: /*
  // CHECK-NEXT: DPCT1010:{{[0-9]+}}: SYCL uses exceptions to report errors and does not use the error codes. The call was replaced with 0. You need to rewrite this code.
  // CHECK-NEXT: */
  PRINT_ERROR_NAME3(cudaGetLastError());
  PRINT_ERROR_STR3(cudaSuccess);
  PRINT_ERROR_NAME3(cudaSuccess);

  // CHECK: /*
  // CHECK-NEXT: DPCT1010:{{[0-9]+}}: SYCL uses exceptions to report errors and does not use the error codes. The call was replaced with 0. You need to rewrite this code.
  // CHECK-NEXT: */
  PRINT_ERROR_STR_NAME(cudaGetLastError());
  PRINT_ERROR_STR_NAME(cudaSuccess);

//CHECK:/*
//CHECK-NEXT:DPCT1009:{{[0-9]+}}: SYCL reports errors using exceptions and does not use error codes. Please replace the "get_error_string_dummy(...)" with a real error-handling function.
//CHECK-NEXT:*/
//CHECK-NEXT:/*
//CHECK-NEXT:DPCT1010:{{[0-9]+}}: SYCL uses exceptions to report errors and does not use the error codes. The call was replaced with 0. You need to rewrite this code.
//CHECK-NEXT:*/
//CHECK-NEXT:  printf("%s\n", dpct::get_error_string_dummy(0));
  printf("%s\n", cudaGetErrorString(cudaGetLastError()));

//CHECK:  /*
//CHECK-NEXT:  DPCT1009:{{[0-9]+}}: SYCL reports errors using exceptions and does not use error codes. Please replace the "get_error_string_dummy(...)" with a real error-handling function.
//CHECK-NEXT:  */
//CHECK-NEXT:  printf("%s\n", dpct::get_error_string_dummy(0));
  printf("%s\n", cudaGetErrorString(cudaSuccess));

//CHECK:/*
//CHECK-NEXT:DPCT1009:{{[0-9]+}}: SYCL reports errors using exceptions and does not use error codes. Please replace the "get_error_string_dummy(...)" with a real error-handling function.
//CHECK-NEXT:*/
//CHECK-NEXT:printf("%s\n", dpct::get_error_string_dummy(0));
  printf("%s\n", cudaGetErrorName(cudaSuccess));
  CUresult e;
  const char *err_s;

//CHECK:  /*
//CHECK-NEXT:  DPCT1009:{{[0-9]+}}: SYCL reports errors using exceptions and does not use error codes. Please replace the "get_error_string_dummy(...)" with a real error-handling function.
//CHECK-NEXT:  */
//CHECK-NEXT:  err_s = dpct::get_error_string_dummy(e);
  cuGetErrorString(e, &err_s);

//CHECK:/*
//CHECK-NEXT:DPCT1009:{{[0-9]+}}: SYCL reports errors using exceptions and does not use error codes. Please replace the "get_error_string_dummy(...)" with a real error-handling function.
//CHECK-NEXT:*/
//CHECK-NEXT:  return dpct::get_error_string_dummy(0);
  return cudaGetErrorName(cudaSuccess);
}

//CHECK:void foo1(int err, const char *c) {
//CHECK-NEXT:  /*
//CHECK-NEXT:  DPCT1009:{{[0-9]+}}: SYCL reports errors using exceptions and does not use error codes. Please replace the "get_error_string_dummy(...)" with a real error-handling function.
//CHECK-NEXT:  */
//CHECK-NEXT:  c = dpct::get_error_string_dummy(err);
//CHECK-NEXT:  /*
//CHECK-NEXT:  DPCT1009:{{[0-9]+}}: SYCL reports errors using exceptions and does not use error codes. Please replace the "get_error_string_dummy(...)" with a real error-handling function.
//CHECK-NEXT:  */
//CHECK-NEXT:  c = dpct::get_error_string_dummy({{[0-9]+}});
//CHECK-NEXT:}
void foo1(CUresult err, const char *c) {
  cuGetErrorString(err, &c);
  cuGetErrorString(CUDA_ERROR_UNKNOWN, &c);
}

//CHECK:void foo2(dpct::err0 err) {
//CHECK-NEXT:  /*
//CHECK-NEXT:  DPCT1009:{{[0-9]+}}: SYCL reports errors using exceptions and does not use error codes. Please replace the "get_error_string_dummy(...)" with a real error-handling function.
//CHECK-NEXT:  */
//CHECK-NEXT:  dpct::get_error_string_dummy(err);
//CHECK-NEXT:  /*
//CHECK-NEXT:  DPCT1009:{{[0-9]+}}: SYCL reports errors using exceptions and does not use error codes. Please replace the "get_error_string_dummy(...)" with a real error-handling function.
//CHECK-NEXT:  */
//CHECK-NEXT:  dpct::get_error_string_dummy({{[0-9]+}});
//CHECK-NEXT:}
void foo2(cudaError_t err) {
  cudaGetErrorString(err);
  cudaGetErrorString(cudaErrorInvalidValue);
}

void report_cuda_error(const char *stmt, const char *func, const char *file,
                       int line, const char *msg) {}

#define __CUDA_CHECK__(err, success, error_fn)                                 \
  do {                                                                         \
    auto err_ = (err);                                                         \
    if (err_ != (success)) {                                                   \
      report_cuda_error(#err, __func__, __FILE__, __LINE__, error_fn(err_));   \
    }                                                                          \
  } while (0)
// CHECK: #define CUDA_CHECK(err) __CUDA_CHECK__(err, 0, dpct::get_error_string_dummy)
#define CUDA_CHECK(err) __CUDA_CHECK__(err, cudaSuccess, cudaGetErrorString)

int main() {
  float *f;
  // CHECK: /*
  // CHECK-NEXT: DPCT1009:{{[0-9]+}}: SYCL reports errors using exceptions and does not use error codes. Please replace the "get_error_string_dummy(...)" with a real error-handling function.
  // CHECK-NEXT: */
  // CHECK-NEXT: CUDA_CHECK(DPCT_CHECK_ERROR(f = sycl::malloc_device<float>(1, q_ct1)));
  // CHECK-NEXT: /*
  // CHECK-NEXT: DPCT1009:{{[0-9]+}}: SYCL reports errors using exceptions and does not use error codes. Please replace the "get_error_string_dummy(...)" with a real error-handling function.
  // CHECK-NEXT: */
  // CHECK-NEXT: CUDA_CHECK(DPCT_CHECK_ERROR(dpct::dpct_free(f, q_ct1)));
  CUDA_CHECK(cudaMalloc(&f, sizeof(float)));
  CUDA_CHECK(cudaFree(f));
  return 0;
}
