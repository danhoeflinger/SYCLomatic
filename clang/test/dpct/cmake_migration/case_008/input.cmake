find_package(CUDA)
find_package(CUDA REQUIRED)
find_package(CUDA ${REQUIRED_CUDA_VERSION} REQUIRED)
find_package(CUDAToolkit)
find_package(CUDAToolkit 11.4 REQUIRED)
find_package(CUDAToolkit REQUIRED curand cufft cusparse_static cublas cusolver)
find_package(CUDAToolkit REQUIRED nvToolsExt nppc_static nvptxcompiler_static)
find_package(CUB)
find_package(CUB REQUIRED)
#Current YAML rule comments find_package(MPI) as it changes CMAKE_CXX_COMPILER
find_package(MPI)
find_package(MPI REQUIRED)
#Current YAML rule comments find_package(OpenMP) as it changes CMAKE_CXX_COMPILER
FIND_PACKAGE(OpenMP)
FIND_PACKAGE(OpenMP REQUIRED)
#Current YAML rule comments find_package(NVJPEG) as no SYCL equivalent lib exists
find_package(NVJPEG)
find_package(NVJPEG 9.0 REQUIRED)
find_package( OpenMP REQUIRED)
find_package( MPI  )
find_package(  MPI REQUIRED )
FIND_PACKAGE( OpenMP)
FIND_PACKAGE(  OpenMP REQUIRED)

target_link_libraries(foo PRIVATE OpenMP::OpenMP_CXX)

target_link_libraries(foo PRIVATE
faiss_avx512
Python::Module
Python::NumPy
  OpenMP::OpenMP_CXX
)
