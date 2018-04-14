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
 * \date    2014-07-09
 *
 */
//----------------------------------------------------------------------
#ifndef GPU_VOXELS_VOXELMAP_PROB_VOXELMAP_HPP_INCLUDED
#define GPU_VOXELS_VOXELMAP_PROB_VOXELMAP_HPP_INCLUDED

#include "ProbVoxelMap.h"
#include <gpu_voxels/voxelmap/TemplateVoxelMap.hpp>
#include <gpu_voxels/voxelmap/kernels/VoxelMapOperations.hpp>
#include <gpu_voxels/voxel/BitVoxel.hpp>
#include <gpu_voxels/voxel/ProbabilisticVoxel.hpp>

namespace gpu_voxels {
namespace voxelmap {

ProbVoxelMap::ProbVoxelMap(const Vector3ui dim, const float voxel_side_length, const MapType map_type) :
    Base(dim, voxel_side_length, map_type)
{

}

ProbVoxelMap::ProbVoxelMap(Voxel* dev_data, const Vector3ui dim, const float voxel_side_length, const MapType map_type) :
    Base(dev_data, dim, voxel_side_length, map_type)
{

}

ProbVoxelMap::~ProbVoxelMap()
{

}

template<std::size_t length>
void ProbVoxelMap::insertSensorData(const PointCloud &global_points, const Vector3f &sensor_pose, const bool enable_raycasting,
                                    const bool cut_real_robot, const BitVoxelMeaning robot_voxel_meaning,
                                    BitVoxel<length>* robot_map)
{
  lock_guard guard(this->m_mutex);

  computeLinearLoad(global_points.getPointCloudSize(), &m_blocks,
                           &m_threads);

  if (enable_raycasting)
  {
    kernelInsertSensorData<<<m_blocks, m_threads>>>(
        m_dev_data, m_voxelmap_size, m_dim, m_voxel_side_length, sensor_pose,
        global_points.getConstDevicePointer(), global_points.getPointCloudSize(), cut_real_robot, robot_map, robot_voxel_meaning, RayCaster());
    CHECK_CUDA_ERROR();
  }
  else
  {
    kernelInsertSensorData<<<m_blocks, m_threads>>>(
        m_dev_data, m_voxelmap_size, m_dim, m_voxel_side_length, sensor_pose,
        global_points.getConstDevicePointer(), global_points.getPointCloudSize(), cut_real_robot, robot_map, robot_voxel_meaning, DummyRayCaster());
    CHECK_CUDA_ERROR();
  }
  HANDLE_CUDA_ERROR(cudaDeviceSynchronize());
}

bool ProbVoxelMap::insertRobotConfiguration(const MetaPointCloud *robot_links, bool with_self_collision_test)
{
  LOGGING_ERROR_C(VoxelmapLog, ProbVoxelMap, GPU_VOXELS_MAP_OPERATION_NOT_SUPPORTED << endl);
  return false;
}

void ProbVoxelMap::clearBitVoxelMeaning(BitVoxelMeaning voxel_meaning)
{
  if(voxel_meaning != eBVM_OCCUPIED)
     LOGGING_ERROR_C(VoxelmapLog, ProbVoxelMap, GPU_VOXELS_MAP_ONLY_SUPPORTS_BVM_OCCUPIED << endl);
  else
    this->clearMap();
}

//Collsion Interface Implementations

size_t ProbVoxelMap::collideWith(const BitVectorVoxelMap *map, float coll_threshold, const Vector3i &offset)
{
  DefaultCollider collider(coll_threshold);
  return collisionCheckWithCounterRelativeTransform((TemplateVoxelMap*)map, collider, offset); //does the locking

}

size_t ProbVoxelMap::collideWith(const ProbVoxelMap *map, float coll_threshold, const Vector3i &offset)
{
  DefaultCollider collider(coll_threshold);
  return collisionCheckWithCounterRelativeTransform((TemplateVoxelMap*)map, collider, offset); //does the locking
}

} // end of namespace
} // end of namespace

#endif
