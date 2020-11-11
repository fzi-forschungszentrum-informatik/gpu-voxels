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
#include <gpu_voxels/helpers/PointCloud.h>

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
                                    const bool cut_real_robot, const BitVoxelMeaning robot_voxel_meaning, const Probability prob,
                                    BitVoxel<length>* robot_map)
{
  lock_guard guard(this->m_mutex);

  computeLinearLoad(global_points.getPointCloudSize(), &m_blocks,
                           &m_threads);

  if (enable_raycasting)
  {
    kernelInsertSensorData<<<m_blocks, m_threads>>>(
        m_dev_data, m_voxelmap_size, m_dim, m_voxel_side_length, sensor_pose,
        global_points.getConstDevicePointer(), global_points.getPointCloudSize(), cut_real_robot, robot_map, robot_voxel_meaning, prob, RayCaster());
    CHECK_CUDA_ERROR();
  }
  else
  {
    kernelInsertSensorData<<<m_blocks, m_threads>>>(
        m_dev_data, m_voxelmap_size, m_dim, m_voxel_side_length, sensor_pose,
        global_points.getConstDevicePointer(), global_points.getPointCloudSize(), cut_real_robot, robot_map, robot_voxel_meaning, prob, DummyRayCaster());
    CHECK_CUDA_ERROR();
  }
  HANDLE_CUDA_ERROR(cudaDeviceSynchronize());
}

/**
 * Convert a ProbVoxelMap as a pointcloud in ROS.
 *
 * In RViz: add the pointcloud, set render settings to Boxes, scale to voxel_side_length
 */
void ProbVoxelMap::publishPointcloud(sensor_msgs::PointCloud2* pointcloud_msg, const float occupancyThreshold)
{
  Vector3f *m_points;
  HANDLE_CUDA_ERROR( cudaMalloc((void** ) &m_points, sizeof(Vector3f)*(m_voxelmap_size-1) ));

  size_t cloudSize = 0;
  size_t *m_cloudSize;
  HANDLE_CUDA_ERROR( cudaMalloc((void** ) &m_cloudSize, sizeof(size_t)));
  HANDLE_CUDA_ERROR( cudaMemcpy(m_cloudSize, &cloudSize, sizeof(size_t), cudaMemcpyHostToDevice));

  HANDLE_CUDA_ERROR( cudaDeviceSynchronize());
  kernelGetProbabilisticPointCloud<<<m_blocks, m_threads>>>(this->m_dev_data, m_points, occupancyThreshold, m_voxelmap_size, m_voxel_side_length, m_dim, m_cloudSize);
  HANDLE_CUDA_ERROR( cudaDeviceSynchronize());
  HANDLE_CUDA_ERROR( cudaMemcpy(&cloudSize, m_cloudSize, sizeof(size_t), cudaMemcpyDeviceToHost));

  Vector3f *points = new Vector3f[cloudSize];;
  HANDLE_CUDA_ERROR( cudaMemcpy(points, m_points, sizeof(Vector3f)*cloudSize, cudaMemcpyDeviceToHost));
  PointCloud pointcloud(points, cloudSize);

  (*pointcloud_msg) = pointcloud.getPointCloud2();
  
  HANDLE_CUDA_ERROR(cudaFree(m_points));
  HANDLE_CUDA_ERROR(cudaFree(m_cloudSize));
}

bool ProbVoxelMap::insertMetaPointCloudWithSelfCollisionCheck(const MetaPointCloud *robot_links,
                                                              const std::vector<BitVoxelMeaning>& voxel_meanings,
                                                              const std::vector<BitVector<BIT_VECTOR_LENGTH> >& collision_masks,
                                                              BitVector<BIT_VECTOR_LENGTH>* colliding_meanings)
{
  LOGGING_ERROR_C(VoxelmapLog, ProbVoxelMap, GPU_VOXELS_MAP_OPERATION_NOT_SUPPORTED << endl);
  return true;
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

void ProbVoxelMap::move(Voxel* dest_data, const Voxel* src_data, const Vector3f offset) const
{
  HANDLE_CUDA_ERROR(cudaDeviceSynchronize());
  kernelMoveMap<<<m_blocks, m_threads>>>(dest_data, src_data, m_voxelmap_size, m_voxel_side_length, this->m_dim, offset);
  HANDLE_CUDA_ERROR(cudaDeviceSynchronize());
  CHECK_CUDA_ERROR();
}

void ProbVoxelMap::moveInto(ProbVoxelMap& dest, const Vector3f offset) const
{
  assert(this->m_dim == dest.m_dim);
  move(dest.m_dev_data, this->m_dev_data, offset);
}

} // end of namespace
} // end of namespace

#endif
