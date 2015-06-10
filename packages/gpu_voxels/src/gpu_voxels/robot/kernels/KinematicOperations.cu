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
void kernelKinematicChainTransform(uint8_t chain_size, uint8_t joint_to_transform, const Matrix4f* basis_transformation,
                                   const Matrix4f* dh_transformations, const Matrix4f* local_transformations,
                                   const uint32_t* point_cloud_sizes, const Vector3f** point_clouds, Vector3f** transformed_point_clouds)
{
	//this function is too slow, since every thread computes the transformation matrix
	//use the overloaded version with fewer parameters instead
  uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;

  Matrix4f transformation;

//  if (i==1) printf("transform (%u): (%f, %f, %f, %f)\n", sizeof(transformation), transformation.a11, transformation.a12, transformation.a13, transformation.a14);

  if (joint_to_transform > 0)   // not basis
  {
    // set d and theta part for current joint
    transformation = local_transformations[joint_to_transform-1];

    // append full dh parameters of other joints
    for (uint32_t j = joint_to_transform-1; j>=1; j--)
    {
       transformation = dh_transformations[j-1] * transformation;
  //      transformation.leftMultiply(dh_transformations[j-1]);
    }
  }
    else  // basis
  {
    transformation.setIdentity();
  }	
//  transformation.leftMultiply(*basis_transformation);
  transformation = (*basis_transformation) * transformation;

  __syncthreads();

  // if more than max_nr_of_blocks points are in ptcloud we need a loop
  while (i < point_cloud_sizes[joint_to_transform])
  {
//    applyTransform(transformation, point_clouds[joint_to_transform][i], transformed_point_clouds[joint_to_transform][i]);
    transformed_point_clouds[joint_to_transform][i] = transformation * point_clouds[joint_to_transform][i];
//    if (i==1)
//      printf("transforming (%f, %f, %f) --> (%f, %f, %f)",
//             point_clouds[joint_to_transform][i].x, point_clouds[joint_to_transform][i].y, point_clouds[joint_to_transform][i].z,
//             transformed_point_clouds[joint_to_transform][i].x, transformed_point_clouds[joint_to_transform][i].y, transformed_point_clouds[joint_to_transform][i].z);
//     transformed_point_clouds[joint_to_transform][i] = transformation * point_clouds[joint_to_transform][i];

    // increment by number of all threads that are running
    i += blockDim.x * gridDim.x;
  }
}

__global__
void kernelKinematicChainTransform(uint8_t joint_to_transform, const Matrix4f* transformation_,
                                   const MetaPointCloudStruct *point_clouds, MetaPointCloudStruct *transformed_point_clouds)
{
  // copying the transformation matrix to local memory might be faster than accessing it from the global memory
  Matrix4f transformation;
  transformation = *transformation_;

  uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;

//  if (i==1) printf("transform (%u): (%f, %f, %f, %f)\n", sizeof(transformation), transformation.a11, transformation.a12, transformation.a13, transformation.a14);

  // if more than max_nr_of_blocks points are in ptcloud we need a loop
  while (i < point_clouds->cloud_sizes[joint_to_transform])
  {

//    applyTransform(transformation, point_clouds[joint_to_transform][i], transformed_point_clouds[joint_to_transform][i]);
    transformed_point_clouds->clouds_base_addresses[joint_to_transform][i] = transformation * point_clouds->clouds_base_addresses[joint_to_transform][i];
//    if (i==1)
//      printf("transforming (%f, %f, %f) --> (%f, %f, %f)\n",
//             point_clouds->clouds_base_addresses[joint_to_transform][i].x,
//             point_clouds->clouds_base_addresses[joint_to_transform][i].y,
//             point_clouds->clouds_base_addresses[joint_to_transform][i].z,
//             transformed_point_clouds->clouds_base_addresses[joint_to_transform][i].x,
//             transformed_point_clouds->clouds_base_addresses[joint_to_transform][i].y,
//             transformed_point_clouds->clouds_base_addresses[joint_to_transform][i].z);

    // increment by number of all threads that are running
    i += blockDim.x * gridDim.x;
  }
}



__global__
void kernelTransformPoseAlongChain(uint8_t chain_size, uint8_t joint_to_transform,
                                   const Matrix4f* basis_transformation, Matrix4f* dh_transformations,
                                   Vector3f* dev_point, Vector3f* dev_result)
{
  uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;

  if (i==0)
  {
    Matrix4f transformation;
    transformation.setIdentity();

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
