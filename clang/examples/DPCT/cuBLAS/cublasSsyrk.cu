#include "cublas_v2.h"

void test(cublasHandle_t handle, cublasFillMode_t upper_lower,
          cublasOperation_t trans, int n, int k, const float *alpha,
          const float *a, int lda, const float *beta, float *c, int ldc) {
  // Start
  cublasSsyrk(handle /*cublasHandle_t*/, upper_lower /*cublasFillMode_t*/,
              trans /*cublasOperation_t*/, n /*int*/, k /*int*/,
              alpha /*const float **/, a /*const float **/, lda /*int*/,
              beta /*const float **/, c /*float **/, ldc /*int*/);
  // End
}
