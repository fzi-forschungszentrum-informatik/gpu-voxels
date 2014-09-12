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
#ifndef GPU_VOXELS_VOXELMAP_BIT_VOXELMAP_H_INCLUDED
#define GPU_VOXELS_VOXELMAP_BIT_VOXELMAP_H_INCLUDED

#include <gpu_voxels/voxelmap/TemplateVoxelMap.h>
#include <gpu_voxels/voxelmap/BitVoxel.h>
#include <gpu_voxels/voxelmap/ProbabilisticVoxel.h>
#include <gpu_voxels/voxelmap/ProbVoxelMap.h>
#include <cstddef>

namespace gpu_voxels {
namespace voxelmap {

template<std::size_t length>
class BitVoxelMap: public TemplateVoxelMap<BitVoxel<length> >
{
public:
  typedef BitVoxel<length> Voxel;
  typedef TemplateVoxelMap<Voxel> Base;

  BitVoxelMap(const uint32_t dim_x, const uint32_t dim_y, const uint32_t dim_z, const float voxel_side_length, const MapType map_type);
  BitVoxelMap(Voxel* dev_data, const Vector3ui dim, const float voxel_side_length, const MapType map_type);

  virtual ~BitVoxelMap();

  virtual void clearBit(const uint32_t bit_index);

  virtual void clearBits(BitVector<length> bits);

  template<class Collider>
  BitVector<length> collisionCheckBitvector(ProbVoxelMap* other, Collider collider);

  virtual bool insertRobotConfiguration(const MetaPointCloud *robot_links, bool with_self_collision_test);

  virtual void clearVoxelType(VoxelType voxel_type);

protected:
  virtual void clearVoxelMapRemoteLock(const uint32_t bit_index);
};

} // end of namespace
} // end of namespace

#endif
