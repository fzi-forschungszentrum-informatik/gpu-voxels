// this is for emacs file handling -*- mode: c++; indent-tabs-mode: nil -*-

// -- BEGIN LICENSE BLOCK ----------------------------------------------
// This file is part of the GPU Voxels Software Library.
//
// This program is free software licensed under the CDDL
// (COMMON DEVELOPMENT AND DISTRIBUTION LICENSE Version 1.0).
// You can find a copy of this license in LICENSE.txt in the top
// directory of the source code.
//
// © Copyright 2014 FZI Forschungszentrum Informatik, Karlsruhe, Germany
//
// -- END LICENSE BLOCK ------------------------------------------------

//----------------------------------------------------------------------
/*!\file
 *
 * \author  Andreas Hermann
 * \date    2016-06-05
 *
 * General Kernel calls
 */
//----------------------------------------------------------------------
#include <cuda_runtime.h>
#include "HelperOperations.h"
#include <gpu_voxels/helpers/common_defines.h>

namespace gpu_voxels {

__global__
void kernelCompareMem(const void* lhs, const void* rhs, uint32_t size_in_byte, bool *results)
{
  __shared__ bool cache[cMAX_THREADS_PER_BLOCK];
  uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
  uint32_t cache_index = threadIdx.x;
  cache[cache_index] = false;
  bool temp = true;

  while (i < size_in_byte)
  {
    temp &= ((char*)lhs)[i] == ((char*)rhs)[i];
    i += blockDim.x * gridDim.x;
  }

  cache[cache_index] = temp;
  __syncthreads();

  uint32_t j = blockDim.x / 2;

  while (j != 0)
  {
    if (cache_index < j)
    {
      cache[cache_index] = cache[cache_index] && cache[cache_index + j];
    }
    __syncthreads();
    j /= 2;
  }

  // copy results from this block to global memory
  if (cache_index == 0)
  {
    results[blockIdx.x] = cache[0];
  }
}

} // end of namespace gpu_voxels
