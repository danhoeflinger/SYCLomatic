// Option: --use-experimental-features=masked-sub-group-operation
__device__ void test(unsigned int r, unsigned int mask, unsigned int var,
                     unsigned int delta, int width) {
  // Start
  r = __shfl_up_sync(mask /*unsigned int*/, var /*unsigned int*/,
                     delta /*unsigned int*/, width /*int*/);
  // End
}