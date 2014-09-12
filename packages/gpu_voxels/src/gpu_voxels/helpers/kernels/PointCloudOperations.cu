// this is for emacs file handling -*- mode: c++; indent-tabs-mode: nil -*-

// -- BEGIN LICENSE BLOCK ----------------------------------------------
// -- END LICENSE BLOCK ------------------------------------------------

//----------------------------------------------------------------------
/*!\file
 *
 * \author  Sebastian Klemm
 * \date    2012-08-23
 *
 * Point Cloud Kernel calls
 */
//----------------------------------------------------------------------
#include <cuda_runtime.h>
#include "PointCloudOperations.h"

namespace gpu_voxels {
__global__
void kernelMultiplyMatrixNbyOne(uint32_t nr_of_elements, Matrix4f* base, Matrix4f* relatives, Matrix4f* absolutes)
{
  const uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < nr_of_elements)
  {
    absolutes[i] = (*base) * relatives[i];
  }
}

} // end of namespace gpu_voxels
