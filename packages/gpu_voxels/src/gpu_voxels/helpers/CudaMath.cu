// this is for emacs file handling -*- mode: c++; indent-tabs-mode: nil -*-

// -- BEGIN LICENSE BLOCK ----------------------------------------------
// -- END LICENSE BLOCK ------------------------------------------------

//----------------------------------------------------------------------
/*!\file
 *
 * \author  Sebastian Klemm
 * \date    2012-08-23
 *
 */
//----------------------------------------------------------------------
#include "CudaMath.h"
#include "cuda_handling.hpp"
#include "kernels/PointCloudOperations.h"

namespace gpu_voxels {


void transform(uint32_t nr_of_transformations, Matrix4f& base,
                         const Matrix4f* relatives, Matrix4f* absolutes)
{
  const uint32_t array_size_bytes = nr_of_transformations * sizeof(Matrix4f);

  Matrix4f* device_base;
  Matrix4f* device_relatives;
  Matrix4f* device_absolutes;

  // copy matrices to device
  HANDLE_CUDA_ERROR(cudaMalloc((void**)&device_base, sizeof(Matrix4f)));
  HANDLE_CUDA_ERROR(cudaMalloc((void**)&device_relatives, array_size_bytes));
  HANDLE_CUDA_ERROR(cudaMalloc((void**)&device_absolutes, array_size_bytes));
  HANDLE_CUDA_ERROR(cudaMemcpy(device_base, &base, sizeof(Matrix4f), cudaMemcpyHostToDevice));
  HANDLE_CUDA_ERROR(cudaMemcpy(device_relatives, relatives, array_size_bytes, cudaMemcpyHostToDevice));
  HANDLE_CUDA_ERROR(cudaMemcpy(device_absolutes, absolutes, array_size_bytes, cudaMemcpyHostToDevice));

  // transform
  uint32_t blocks;
  uint32_t threads_per_block;
  computeLinearLoad(nr_of_transformations, &blocks, &threads_per_block);
  HANDLE_CUDA_ERROR(cudaDeviceSynchronize());
  kernelMultiplyMatrixNbyOne<<< blocks, threads_per_block>>>
      (nr_of_transformations, device_base, device_relatives, device_absolutes);
  // copy results to host
  HANDLE_CUDA_ERROR(cudaMemcpy(absolutes, device_absolutes, array_size_bytes, cudaMemcpyDeviceToHost));
}

} // end of namespace
