// Option: --use-dpcpp-extensions=intel_device_math
// Option: --use-experimental-features=bfloat16_math_functions
#include "cuda_bf16.h"
#include "cuda_fp16.h"

__global__ void test(__half2 h, __nv_bfloat162 b) {
  // Start
  h2sqrt(h /*__half2*/);
  h2sqrt(b /*__nv_bfloat162*/);
  // End
}
