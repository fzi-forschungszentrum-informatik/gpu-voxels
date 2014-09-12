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
#ifndef ICL_PLANNING_GPU_KERNELS_KINEMATIC_OPERATIONS_H_INCLUDED
#define ICL_PLANNING_GPU_KERNELS_KINEMATIC_OPERATIONS_H_INCLUDED

#include <cuda_runtime.h>
#include <gpu_voxels/helpers/cuda_datatypes.h>
#include <gpu_voxels/robot/KinematicLink.h>

namespace gpu_voxels {

/* helper functions */

//! Convert Denavit-Hartenberg parameters to transformation matrix
__device__ __forceinline__
void convertDHtoM(float theta, float d, float b, float a, float alpha, float q, uint8_t joint_type, Matrix4f& m)
{
//  printf("theta, d, a, alpha : \t%f, %f, %f, %f\n", theta, d, a, alpha);
  float ca = 0;
  float sa = 0;
  float ct = 0;
  float st = 0;

  if (joint_type == KinematicLink::PRISMATIC) /* Prismatic joint */
  {
    d += q;
  }
  else /* Revolute joint */
  {
    if (joint_type != KinematicLink::REVOLUTE)
    {
      printf("KinematicLink::convertDHtoM(): Error! Illegal joint type\n");
    }
    theta += q;
  }

  ca = (float) cos(alpha);
  sa = (float) sin(alpha);
  ct = (float) cos(theta);
  st = (float) sin(theta);

  m.a11 = ct;
  m.a12 = -st * ca;
  m.a13 = st * sa;
  m.a14 = a * ct - b * st;

  m.a21 = st;
  m.a22 = ct * ca;
  m.a23 = -ct * sa;
  m.a24 = a * st + b * ct;

  m.a31 = 0.0;
  m.a32 = sa;
  m.a33 = ca;
  m.a34 = d;

  m.a41 = 0.0;
  m.a42 = 0.0;
  m.a43 = 0.0;
  m.a44 = 1.0;
}


/* kernels */

/*!
 *  Update dh_transformations to new joint values.
 *  A modified transformation without a and without alpha will be stored in local_transformations
 */
__global__
void kernelUpdateTransformations(uint8_t chain_size, uint8_t* joint_types, KinematicLink::DHParameters* dh_parameters,
                                 Matrix4f* dh_transformations, Matrix4f* local_transformations);

/*!  Transformations of PointClouds along a kinematic chain.
 *  The kernel has to be called for each joint within a chain
 *  and should be configured by
 *  point_cloud_sizes[joint_to_transform].
 *
 *  This way all points within a point cloud are transformed along the
 *  kinematic chain stored in dh_transformations up to joint_to_transform
 */
__global__
void kernelKinematicChainTransform(uint8_t chain_size, uint8_t joint_to_transform, const Matrix4f* basis_transformation,
                                   const Matrix4f* dh_transformations, const Matrix4f* local_transformations,
                                   const uint32_t* point_cloud_sizes, const Vector3f** point_clouds, Vector3f** transformed_point_clouds);

/*!
 * Same as kernelUpdateTransformations, but the Transformation is not computed inside the kernel *
 */
__global__
void kernelKinematicChainTransform(uint8_t chain_size, uint8_t joint_to_transform, const Matrix4f* transformation, const MetaPointCloudStruct *point_clouds, MetaPointCloudStruct *transformed_point_clouds);





/*! Transform one point along kinematic chain.
 * The kernel has to be called for each joint within a chain
 * and should be configured << 1, 1 >>
 */
__global__
void kernelTransformPoseAlongChain(uint8_t chain_size, uint8_t joint_to_transform,
                                   const Matrix4f* basis_transformation, Matrix4f* dh_transformations,
                                   Vector3f* dev_point, Vector3f* dev_result);

} // end of namespace gpu_voxels
#endif
