#include "cublas_v2.h"

void test(cublasHandle_t handle, int64_t n, const cuComplex *x, int64_t incx,
          cuComplex *y, int64_t incy) {
  // Start
  cublasCcopy_64(handle /*cublasHandle_t*/, n /*int64_t*/,
                 x /*const cuComplex **/, incx /*int64_t*/, y /*cuComplex **/,
                 incy /*int64_t*/);
  // End
}
