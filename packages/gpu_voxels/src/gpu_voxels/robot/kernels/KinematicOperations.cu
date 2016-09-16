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
 * \author  Sebastian Klemm
 * \date    2012-09-13
 *
 */
//----------------------------------------------------------------------
//#define LOCAL_DEBUG
#undef LOCAL_DEBUG

#include "KinematicOperations.h"
#include <stdio.h>

namespace gpu_voxels {


__global__
void kernelTransformPoseAlongChain(uint8_t chain_size, uint8_t joint_to_transform,
                                   const Matrix4f* basis_transformation, Matrix4f* dh_transformations,
                                   Vector3f* dev_point, Vector3f* dev_result)
{
  uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;

  if (i==0)
  {
    Matrix4f transformation = gpu_voxels::Matrix4f::createIdentity();

    for (uint32_t j=1; j<=joint_to_transform; j++)
    {
      transformation = transformation * dh_transformations[j-1];
      printf("executing for loop : %u\n", j);
    }
    transformation = (*basis_transformation) * transformation;

    (*dev_result) = transformation * (*dev_point);
  }
  else
  {
    printf("kernelTransformPoseAlongChain: This kernel should be configured to run with 1 thread only!\n");
  }

}




} // end of namespace gpu_voxels
#ifdef LOCAL_DEBUG
#undef LOCAL_DEBUG
#endif
