#include "cublas_v2.h"

void test(cublasHandle_t handle, cublasFillMode_t upper_lower,
          cublasOperation_t trans, int n, int k, const cuDoubleComplex *alpha,
          const cuDoubleComplex *a, int lda, const cuDoubleComplex *b, int ldb,
          const cuDoubleComplex *beta, cuDoubleComplex *c, int ldc) {
  // Start
  cublasZsyrkx(handle /*cublasHandle_t*/, upper_lower /*cublasFillMode_t*/,
               trans /*cublasOperation_t*/, n /*int*/, k /*int*/,
               alpha /*const cuDoubleComplex **/, a /*const cuDoubleComplex **/,
               lda /*int*/, b /*const cuDoubleComplex **/, ldb /*int*/,
               beta /*const cuDoubleComplex **/, c /*cuDoubleComplex **/,
               ldc /*int*/);
  // End
}
