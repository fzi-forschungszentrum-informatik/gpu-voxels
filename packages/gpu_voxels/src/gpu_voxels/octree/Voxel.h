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
 * \author  Florian Drews
 * \date    2013-11-07
 *
 */
//----------------------------------------------------------------------/*
#ifndef GPU_VOXELS_OCTREE_VOXEL_H_INCLUDED
#define GPU_VOXELS_OCTREE_VOXEL_H_INCLUDED

#include <thrust/functional.h>

#include <gpu_voxels/helpers/cuda_datatypes.h>
#include <gpu_voxels/helpers/common_defines.h>
#include <gpu_voxels/vis_interface/VisualizerInterface.h>

#include <gpu_voxels/octree/DataTypes.h>
#include <gpu_voxels/octree/Nodes.h>

#include <assert.h>

namespace gpu_voxels {
namespace NTree {

class Voxel
{
public:
  // TODO choose better memory layout; may use one byte of voxel_id for occ. probab.

  OctreeVoxelID voxelId;
  gpu_voxels::Vector3ui coordinates;

  __host__ __device__
  friend bool operator<(Voxel a, Voxel b)
  {
    return (a.voxelId < b.voxelId); // | (a.voxel_id == b.voxel_id & a.occupation < b.occupation);
  }

  __host__ __device__
  friend bool operator==(Voxel a, Voxel b)
  {
    return (a.voxelId == b.voxelId && a.coordinates == b.coordinates && a.occupancy == b.occupancy);
  }

private:
  Probability occupancy;

public:

  __host__ __device__
  Voxel()
  {
  }

  __host__ __device__
  Voxel(OctreeVoxelID voxelID, gpu_voxels::Vector3ui coordinates, Probability occupancy)
  {
    this->voxelId = voxelID;
    this->coordinates = coordinates;
    this->occupancy = occupancy;
  }

  __host__ __device__
  __forceinline__
  Probability getOccupancy() const
  {
    return occupancy;
  }

  __host__ __device__
  __forceinline__
  void setOccupancy(Probability value)
  {
    occupancy = value;
  }

};

struct count_per_size: public thrust::unary_function<Cube, voxel_count>
{
  OctreeVoxelID m_cube_side_length;

  __host__ __device__
  count_per_size(OctreeVoxelID cube_side_length)
  {
    m_cube_side_length = cube_side_length;
  }

  __host__ __device__
  inline voxel_count operator()(Cube value)
  {
    return (value.m_side_length == m_cube_side_length);
  }
};

}
}

#endif /* VOXEL_H_ */
