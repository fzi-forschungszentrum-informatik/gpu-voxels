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
 * \date    2014-05-16
 *
 */
//----------------------------------------------------------------------

#ifndef OCTOMAPPROVIDER_CPP_
#define OCTOMAPPROVIDER_CPP_


#include "OctomapProvider.h"
#include <gpu_voxels/octree/PointCloud.h>

#include <icl_core_performance_monitor/PerformanceMonitor.h>

#include <gpu_voxels/helpers/cuda_datatypes.h>

using namespace std;

namespace gpu_voxels {
namespace NTree {
namespace Provider {

OctomapProvider::OctomapProvider() :
    Provider()
{
  m_segment_name = "OctomapProvider";
}

OctomapProvider::~OctomapProvider()
{
  delete m_octree;
}

void OctomapProvider::visualize()
{
//  m_mutex.lock();
//  HANDLE_CUDA_ERROR(cudaIpcGetMemHandle(m_shm_memHandle, (void * ) m_voxelMap->getDeviceDataPtr()));
//  *m_shm_mapDim = m_voxelMap->getDimensions();
//  *m_shm_VoxelSize = m_voxelMap->getVoxelSideLength();
//  m_changed = false;
//  m_mutex.unlock();
}

void OctomapProvider::init(Provider_Parameter& parameter)
{
  const string prefix = "Octomap::" + string(__FUNCTION__);
  const string temp_timer = prefix + "_temp";

  m_mutex.lock();

  Provider::init(parameter);

  double resolution = parameter.resolution_tree / 1000.0; // resolution in m
  printf("res %f\n", resolution);
  m_octree = new octomap::OcTree(resolution);

  //octomap::Pointcloud* point_cloud = toOctoPointCloud(&parameter.points[0], parameter.points.size());

  PERF_MON_START(temp_timer);

  // manual insert
  for(uint32_t j = 0; j < parameter.points.size(); ++j)
  {
    gpu_voxels::Vector3f point = parameter.points[j];
    if (!isnan(point.x) && !isnan(point.y) && !isnan(point.z)){
      m_octree->setNodeValue(octomap::point3d(point.x, point.y, point.z), 1.0, true);
      //printf("x %f y %f z %f\n", point.x, point.y, point.z);
    }
    //point_cloud.push_back(h_point_cloud[i].x, h_point_cloud[i].y, h_point_cloud[i].z);
  }

  //octomap::point3d sensor(0, 0, 0);
  //m_octree->insertPointCloud(*point_cloud, sensor, -1, true, true);
  m_octree->updateInnerOccupancy();

  PERF_MON_PRINT_INFO_P(temp_timer, "Build", prefix);

  //delete point_cloud;

  PERF_MON_ADD_STATIC_DATA_P("Mem", m_octree->memoryUsage(), prefix);
  PERF_MON_ADD_STATIC_DATA_P("LeafNodes", m_octree->getNumLeafNodes(), prefix);
  PERF_MON_ADD_STATIC_DATA_P("NodeSize", m_octree->memoryUsageNode(), prefix);
  PERF_MON_ADD_STATIC_DATA_P("TotalNodes", m_octree->calcNumNodes(), prefix);

//  Provider::init(parameter);
//
//  // get shared memory pointer
//  m_shm_memHandle = m_segment.find_or_construct<cudaIpcMemHandle_t>(
//      std::string("handler_dev_pointer_" + m_shared_mem_id).c_str())(cudaIpcMemHandle_t());
//  m_shm_mapDim = m_segment.find_or_construct<Vector3ui>(
//      std::string("voxel_map_dimensions_" + m_shared_mem_id).c_str())(Vector3ui(0));
//  m_shm_VoxelSize = m_segment.find_or_construct<float>(
//      std::string("voxel_side_length_" + m_shared_mem_id).c_str())(0.0f);
//
//  //TODO Fix to handle m_shared_mem_id
//
////  m_shm_memHandle = m_segment.find_or_construct<cudaIpcMemHandle_t>(
////      std::string("handler_dev_pointer").c_str())(cudaIpcMemHandle_t());
////  m_shm_mapDim = m_segment.find_or_construct<Vector3ui>(std::string("voxel_map_dimensions").c_str())(
////      Vector3ui(0));
////  m_shm_VoxelSize = m_segment.find_or_construct<float>(std::string("voxel_side_length").c_str())(0.0f);
//
//  Vector3f offset;
//  Vector3ui map_dim = getMapDimensions(parameter.points, offset);
//  uint64_t map_voxel = uint64_t(map_dim.x) * uint64_t(map_dim.y) * uint64_t(map_dim.z);
//  printf("point cloud dimension %u %u %u\n", map_dim.x, map_dim.y, map_dim.z);
//
//  float scaling = 1.0;
//  uint64_t max_voxel = uint64_t(parameter.max_memory) / sizeof(gpu_voxels::Voxel);
//  printf("max_voxel %lu map_voxel %lu\n", max_voxel, map_voxel);
//  if (max_voxel <= map_voxel)
//    scaling = float(pow(max_voxel / double(map_voxel), 1.0 / 3));
//
//  printf("scaling %f\n", scaling);
//
//  std::vector<Vector3ui> points;
//  transformPointCloud(parameter.points, points, map_dim, scaling * 1000.0f);
//  printf("voxel map dimension %u %u %u\n", map_dim.x, map_dim.y, map_dim.z);
//
//  m_voxelMap = new gpu_voxels::VoxelMap(map_dim.x, map_dim.y, map_dim.z, 1);
//  m_voxelMap->insertPointCloud(points, gpu_voxels::Voxel::eC_EXECUTION, gpu_voxels::Voxel::eBVM_OCCUPIED);
//
//  m_sensor_position = gpu_voxels::Vector3f(map_dim.x / 2, map_dim.y / 2, map_dim.z / 2);
//  m_sensor_orientation = gpu_voxels::Vector3f(0, 0, 0);
//
//  printf("VoxelMap created!\n");

  m_mutex.unlock();
}

octomap::Pointcloud* OctomapProvider::toOctoPointCloud(gpu_voxels::Vector3f* h_point_cloud, const uint32_t num_points)
{
  octomap::Pointcloud* point_cloud = new octomap::Pointcloud();
  point_cloud->reserve(num_points);
  printf("num_points %u\n", num_points);
  for(uint32_t j = 0; j < num_points; ++j)
  {
    gpu_voxels::Vector3f point = h_point_cloud[j];
    if (!isnan(point.x) && !isnan(point.y) && !isnan(point.z)){
      point_cloud->push_back(octomap::point3d(point.x, point.y, point.z));
      //printf("x %f y %f z %f\n", point.x, point.y, point.z);
    }
    //point_cloud.push_back(h_point_cloud[i].x, h_point_cloud[i].y, h_point_cloud[i].z);
  }
  return point_cloud;
}

void OctomapProvider::newSensorData(gpu_voxels::Vector3f* h_point_cloud, const uint32_t num_points, const uint32_t width,
                             const uint32_t height)
{
  const string prefix = "Octomap::" + string(__FUNCTION__);
  const string temp_timer = prefix + "_temp";

  octomap::Pointcloud* point_cloud = toOctoPointCloud(h_point_cloud, num_points);
  octomap::point3d sensor(0, 0, 0);
  printf("size: %lu\n", point_cloud->size());

  PERF_MON_START(temp_timer);

  double max_range = -1;
  if(m_parameter->sensor_max_range > 0)
    max_range = m_parameter->sensor_max_range / 1000.0f;

  m_octree->insertPointCloud(*point_cloud, sensor, max_range, true, true);
  m_octree->updateInnerOccupancy();

  PERF_MON_PRINT_INFO_P(temp_timer, "OctomapInsert", prefix);
  PERF_MON_ADD_DATA_NONTIME_P("UsedMemOctomap", m_octree->memoryUsage(), prefix);

  delete point_cloud;
}


void OctomapProvider::newSensorData(const DepthData* h_depth_data, const uint32_t width, const uint32_t height)
{
  // not yet implemented
}

void OctomapProvider::collide()
{
  // not yet implemented
}

bool OctomapProvider::waitForNewData(volatile bool* stop)
{
  // wait till new data is required
  while (!*stop && !m_changed)
  {
    //boost::this_thread::yield();
    usleep(buffer_watch_delay);
  }
  return !*stop;
}

}
}
}


#endif /* OCTOMAPPROVIDER_CPP_ */
