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
 * \author  Florian Drews
 * \author  Christian Juelg
 * \date    2012-06-22
 *
 */
//----------------------------------------------------------------------
#ifndef GPU_VOXELS_CUDA_DATATYPES_H_INCLUDED
#define GPU_VOXELS_CUDA_DATATYPES_H_INCLUDED

#include <cuda_runtime.h>
#include "gpu_voxels/helpers/cuda_vectors.h"
#include "gpu_voxels/helpers/cuda_matrices.h"

// __ballot has been replaced by __ballot_sync in Cuda9
#if(__CUDACC_VER_MAJOR__ >= 9)
#define FULL_MASK 0xffffffff
#define BALLOT(PREDICATE) __ballot_sync(FULL_MASK, PREDICATE)
#else
#define BALLOT(PREDICATE) __ballot(PREDICATE)
#endif

namespace gpu_voxels {


/*************** Sensor ********************/

struct Sensor
{
  __host__ __device__ Sensor()
  {
  }

  __host__ __device__ Sensor(Vector3f _position, Matrix3f _orientation, uint32_t _data_width,
                             uint32_t _data_height) :
      position(_position), orientation(_orientation), data_width(_data_width), data_height(_data_height), data_size(
          _data_width * _data_height)
  {
  }

  __host__ __device__ Sensor(const Sensor& other) :
      position(other.position), orientation(other.orientation), data_width(other.data_width), data_height(
          other.data_height), data_size(other.data_size)
  {
  }

  Vector3f position;
  Matrix3f orientation;
  uint32_t data_width;
  uint32_t data_height;
  uint32_t data_size;
};


struct MetaPointCloudStruct
{
  uint16_t num_clouds;
  uint32_t accumulated_cloud_size;
  uint32_t *cloud_sizes;
  Vector3f** clouds_base_addresses;

  __device__ __host__
  MetaPointCloudStruct()
    : num_clouds(0),
      cloud_sizes(0),
      clouds_base_addresses(0)
    {
    }
};


/*!
 * \brief The OrientedBoxParams struct
 */
struct OrientedBoxParams
{
  gpu_voxels::Vector3f dim; //< half the side length
  gpu_voxels::Vector3f center; //< center of the cube
  gpu_voxels::Vector3f rot; //< rotation of the cube (Roll, Pitch, Yaw)
};

} // end of namespace
#endif
