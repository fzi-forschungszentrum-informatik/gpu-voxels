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
 * \date    2014-07-08
 *
 */
//----------------------------------------------------------------------
#ifndef GPU_VOXELS_VOXELMAP_PROB_VOXELMAP_H_INCLUDED
#define GPU_VOXELS_VOXELMAP_PROB_VOXELMAP_H_INCLUDED

#include <gpu_voxels/voxelmap/TemplateVoxelMap.h>
#include <gpu_voxels/voxelmap/ProbabilisticVoxel.h>
#include <gpu_voxels/voxelmap/BitVoxel.h>
#include <gpu_voxels/helpers/common_defines.h>
#include <gpu_voxels/helpers/cuda_datatypes.h>

namespace gpu_voxels {
namespace voxelmap {

class ProbVoxelMap: public TemplateVoxelMap<ProbabilisticVoxel>
{
public:
  typedef ProbabilisticVoxel Voxel;
  typedef TemplateVoxelMap<Voxel> Base;

  ProbVoxelMap(const uint32_t dim_x, const uint32_t dim_y, const uint32_t dim_z, const float voxel_side_length, const MapType map_type);
  ProbVoxelMap(Voxel* dev_data, const Vector3ui dim, const float voxel_side_length, const MapType map_type);
  virtual ~ProbVoxelMap();

  template<std::size_t length>
  void insertSensorData(const Vector3f* points, const bool enable_raycasting, const bool cut_real_robot,
                        const uint32_t voxel_type, BitVoxel<length>* robot_map = NULL);

  virtual bool insertRobotConfiguration(const MetaPointCloud *robot_links, bool with_self_collision_test);

  virtual void clearVoxelType(VoxelType voxel_type);

  virtual void insertPointCloud(const std::vector<Vector3f> &points, const uint32_t voxel_type);
};

} // end of namespace
} // end of namespace

#endif
