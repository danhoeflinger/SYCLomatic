// Option: --use-dpcpp-extensions=intel_device_math
__global__ void test(float f1, float f2) {
  // Start
  __fsub_rn(f1 /*float*/, f2 /*float*/);
  // End
}
