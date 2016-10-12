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
void ProbVoxelMap::insertSensorData(const Vector3f* points, const bool enable_raycasting,
                                    const bool cut_real_robot, const BitVoxelMeaning voxel_meaning,
                                    BitVoxel<length>* robot_map)
{
  this->lockSelf("ProbVoxelMap::insertSensorData");
  //  printf("got lock ----------------------------------------------------\n");
  //  if (enable_raycasting)
  //  {
  //    printf("inserting new SENSOR data WITH RAYCASTING ... ");
  //  }
  //  else
  //  {
  //    printf("inserting new SENSOR data ... ");
  //  }
  //  m_elapsed_time = 0;
  //  HANDLE_CUDA_ERROR(cudaEventRecord(m_start, 0));

  copySensorDataToDevice(points);
  transformSensorData();
  if (enable_raycasting)
  {
    // for debugging ray casting:
//    uint32_t blocks, threads;
//    computeLinearLoad(m_voxelmap_size, &blocks, &threads);
//    HANDLE_CUDA_ERROR(cudaDeviceSynchronize());
//    kernelClearVoxelMap<<<blocks, threads>>>(m_dev_data, m_voxelmap_size, eBVM_OCCUPIED);
    // ---
//    HANDLE_CUDA_ERROR(cudaDeviceSynchronize());
    kernelInsertSensorData<<<m_blocks_sensor_operations, m_threads_sensor_operations>>>(
        m_dev_data, m_voxelmap_size, m_dim, m_voxel_side_length, m_dev_sensor,
        m_dev_transformed_sensor_data, cut_real_robot, robot_map, voxel_meaning, RayCaster());
  }
  else
  {
    kernelInsertSensorData<<<m_blocks_sensor_operations, m_threads_sensor_operations>>>(
        m_dev_data, m_voxelmap_size, m_dim, m_voxel_side_length, m_dev_sensor,
        m_dev_transformed_sensor_data, cut_real_robot, robot_map, voxel_meaning, DummyRayCaster());
  }
  HANDLE_CUDA_ERROR(cudaDeviceSynchronize());

//  HANDLE_CUDA_ERROR(cudaEventRecord(m_stop, 0));
//  HANDLE_CUDA_ERROR(cudaEventSynchronize(m_stop));
//  HANDLE_CUDA_ERROR(cudaEventElapsedTime(&m_elapsed_time, m_start, m_stop));

//  printf(" ...done in %f ms!\n", m_elapsed_time);
//  printf("update counter: %u\n", m_update_counter);

//  printf("releasing lock ----------------------------------------------------\n");
  this->unlockSelf("ProbVoxelMap::insertSensorData");
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

void ProbVoxelMap::insertPointCloud(const std::vector<Vector3f> &points, const BitVoxelMeaning voxel_meaning)
{
  if(voxel_meaning != eBVM_OCCUPIED)
    LOGGING_ERROR_C(VoxelmapLog, ProbVoxelMap, GPU_VOXELS_MAP_ONLY_SUPPORTS_BVM_OCCUPIED << endl);
  else
    this->Base::insertPointCloud(points, voxel_meaning);
}

void ProbVoxelMap::insertPointCloud(const PointCloud &pointcloud, const BitVoxelMeaning voxel_meaning)
{
  if(voxel_meaning != eBVM_OCCUPIED)
    LOGGING_ERROR_C(VoxelmapLog, ProbVoxelMap, GPU_VOXELS_MAP_ONLY_SUPPORTS_BVM_OCCUPIED << endl);
  else
    this->Base::insertPointCloud(pointcloud, voxel_meaning);
}

void ProbVoxelMap::insertPointCloud(const Vector3f *points_d, uint32_t size, const BitVoxelMeaning voxel_meaning)
{
  if(voxel_meaning != eBVM_OCCUPIED)
    LOGGING_ERROR_C(VoxelmapLog, ProbVoxelMap, GPU_VOXELS_MAP_ONLY_SUPPORTS_BVM_OCCUPIED << endl);
  else
    this->Base::insertPointCloud(points_d, size, voxel_meaning);
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
