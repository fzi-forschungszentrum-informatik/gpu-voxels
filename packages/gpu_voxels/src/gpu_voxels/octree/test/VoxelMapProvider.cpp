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
 * \date    2014-04-05
 *
 */
//----------------------------------------------------------------------
#include <gpu_voxels/octree/test/VoxelMapProvider.h>
#include <gpu_voxels/octree/PointCloud.h>
#include <gpu_voxels/octree/Sensor.h>
#include <gpu_voxels/octree/test/NTreeProvider.h>

#include <string>
#include <iostream>

#include <gpu_voxels/helpers/cuda_datatypes.h>
#include <gpu_voxels/helpers/common_defines.h>
//#include <gpu_voxels/voxelmap/Voxel.h>
//#include <gpu_voxels/voxelmap/VoxelMap.hpp>

#include <icl_core_performance_monitor/PerformanceMonitor.h>

using namespace std;

namespace gpu_voxels {
namespace NTree {
namespace Provider {

VoxelMapProvider::VoxelMapProvider() :
    Provider(), m_shm_memHandle(NULL), m_shm_mapDim(NULL), m_shm_VoxelSize(NULL)
{
  m_segment_name = shm_segment_name_voxelmaps;
}

VoxelMapProvider::~VoxelMapProvider()
{
  printf("VoxelMapProvider deconstructor called!\n");
//  m_segment.destroy_ptr(m_shm_memHandle);
//  m_segment.destroy_ptr(m_shm_mapDim);
//  m_segment.destroy_ptr(m_shm_VoxelSize);
  //shared_memory_object::remove(m_segment_name.c_str());

  delete m_voxelMap;
}

void VoxelMapProvider::visualize()
{
  m_mutex.lock();
  HANDLE_CUDA_ERROR(cudaIpcGetMemHandle(m_shm_memHandle, m_voxelMap->getVoidDeviceDataPtr()));
  *m_shm_mapDim = m_voxelMap->getDimensions();
  *m_shm_VoxelSize = m_voxelMap->getVoxelSideLength();
  m_changed = false;
  m_mutex.unlock();
}

void VoxelMapProvider::init(Provider_Parameter& parameter)
{
  m_mutex.lock();

  const string prefix = "VoxelMapProvider::" + string(__FUNCTION__);
  const string temp_timer = prefix + "_temp";
  PERF_MON_START(prefix);
  PERF_MON_START(temp_timer);

  Provider::init(parameter);

  // get shared memory pointer
  m_shm_memHandle = m_segment.find_or_construct<cudaIpcMemHandle_t>(
      std::string(shm_variable_name_voxelmap_handler_dev_pointer + m_shared_mem_id).c_str())(
      cudaIpcMemHandle_t());
  m_shm_mapDim = m_segment.find_or_construct<Vector3ui>(
      std::string(shm_variable_name_voxelmap_dimension + m_shared_mem_id).c_str())(Vector3ui(0));
  m_shm_VoxelSize = m_segment.find_or_construct<float>(
      std::string(shm_variable_name_voxel_side_length + m_shared_mem_id).c_str())(0.0f);

  // there should only be one segment of number_of_voxelmaps
  std::pair<uint32_t*, std::size_t> r = m_segment.find<uint32_t>(
      shm_variable_name_number_of_voxelmaps.c_str());
  if (r.second == 0)
  {
    // if it doesn't exist ..
    m_segment.construct<uint32_t>(shm_variable_name_number_of_voxelmaps.c_str())(1);
  }
  else
  {
    // if it exit increase it by one
    (*r.first)++;
  }

  Vector3ui map_dim;
  std::vector<Vector3f> insert_points;
  float voxel_map_res = 1.0f;
  if (parameter.points.size() != 0)
  {
    Vector3f offset;
    Vector3ui point_data_bounds = getMapDimensions(parameter.points, offset);
    map_dim = point_data_bounds;
    std::cout << "point cloud dimension: " << map_dim << std::endl;

    if (parameter.plan_size.x != 0.0f && parameter.plan_size.y != 0.0f && parameter.plan_size.z != 0.0f)
    {
      Vector3f tmp = parameter.plan_size * 1000.0f;
      map_dim = Vector3ui(uint32_t(tmp.x), uint32_t(tmp.y), uint32_t(tmp.z));
      std::cout << "dim in cm:" << map_dim << std::endl;
    }

    uint64_t map_voxel = uint64_t(map_dim.x) * uint64_t(map_dim.y) * uint64_t(map_dim.z);

    float scaling = 1.0;
    if (parameter.max_memory == 0)
    {
      // compute scaling factor based on voxel size
      scaling = 1.0f / parameter.resolution_tree;
    }
    else
    {
      // compute max scaling factor based on memory restriction
      uint64_t max_voxel = uint64_t(parameter.max_memory) / sizeof(ProbabilisticVoxel);
      printf("max_voxel %lu map_voxel %lu\n", max_voxel, map_voxel);
      if (max_voxel <= map_voxel)
        scaling = float(pow(max_voxel / double(map_voxel), 1.0 / 3));
    }

    printf("scaling %f\n", scaling);

    std::vector<Vector3ui> points;
    Vector3ui map_dim_tmp;
    transformPointCloud(parameter.points, points, map_dim_tmp, scaling * 1000.0f);

    map_dim = Vector3ui(uint32_t(ceil(map_dim.x * scaling)), uint32_t(ceil(map_dim.y * scaling)),
                        uint32_t(ceil(map_dim.z * scaling)));
    std::cout << "voxel map dimension: " <<  map_dim << std::endl;

    // center data at the middle of the map, just like for NTree
    point_data_bounds = Vector3ui(uint32_t(ceil(point_data_bounds.x * scaling)),
                                  uint32_t(ceil(point_data_bounds.y * scaling)),
                                  uint32_t(ceil(point_data_bounds.z * scaling)));
    Vector3ui tmp_offset = (map_dim - point_data_bounds) / Vector3ui(2);
    insert_points.resize(points.size());
    printf("scaling %f\n", scaling);

    voxel_map_res = (1.0f / scaling) / 1000.0f;;
    printf("mapres %f\n", voxel_map_res);
    for (int i = 0; i < int(points.size()); ++i)
    {
      points[i] = points[i] + tmp_offset;
      insert_points[i].x = points[i].x * voxel_map_res + voxel_map_res / 2;
      insert_points[i].y = points[i].y * voxel_map_res + voxel_map_res / 2;
      insert_points[i].z = points[i].z * voxel_map_res + voxel_map_res / 2;
    }
    PERF_MON_START(temp_timer);
  }
  else
  {
    // VoxelMap with same size as octree
    uint32_t dim = (uint32_t) pow(pow(BRANCHING_FACTOR, 1.0 / 3), parameter.resolution_tree);
    map_dim = Vector3ui(dim);
    voxel_map_res = parameter.resolution_tree * 0.001f; // voxel size in meter
  }

  switch(m_parameter->model_type)
  {
    case Provider_Parameter::eMT_Probabilistic:
    {
      m_voxelMap = new gpu_voxels::voxelmap::ProbVoxelMap(map_dim, voxel_map_res, MT_PROBAB_VOXELMAP);
      break;
    }
    case Provider_Parameter::eMT_BitVector:
    {
      m_voxelMap = new gpu_voxels::voxelmap::BitVectorVoxelMap(map_dim, voxel_map_res, MT_BITVECTOR_VOXELMAP);
      break;
    }
    default:
    {
      printf("ERROR: Unknown 'model_type'\n");
    }
  }
  m_segment.find_or_construct<MapType>(std::string(shm_variable_name_voxelmap_type + m_shared_mem_id).c_str())(m_voxelMap->getMapType());

  if (insert_points.size() != 0)
  {
    m_voxelMap->insertPointCloud(insert_points, gpu_voxels::eBVM_OCCUPIED);

//    if (m_parameter->model_type == Provider_Parameter::eMT_BitVector)
//    {
//      Vector3f offset(1, 1, 1);
//      for (uint32_t k = 1; k < 4; ++k)
//      {
//        std::vector<Vector3f> tmp = insert_points;
//        for (int i = 0; i < int(tmp.size()); ++i)
//          tmp[i] = tmp[i] + offset * k;
//
//        m_voxelMap->insertPointCloud(tmp, gpu_voxels::eBVM_UNDEFINED + k);
//      }
//    }

    PERF_MON_PRINT_INFO_P(temp_timer, "Build", prefix);
  }

  PERF_MON_ADD_DATA_NONTIME_P("UsedMemory", m_voxelMap->getMemoryUsage(), prefix);

  m_sensor_orientation = gpu_voxels::Vector3f(0, 0, 0);
  m_sensor_position = gpu_voxels::Vector3f(
      (m_voxelMap->getDimensions().x * m_voxelMap->getVoxelSideLength()) / 2,
      (m_voxelMap->getDimensions().y * m_voxelMap->getVoxelSideLength()) / 2,
      (m_voxelMap->getDimensions().z * m_voxelMap->getVoxelSideLength()) / 2) * 0.001f; // in meter

  printf("VoxelMap created!\n");

  m_mutex.unlock();
}

void VoxelMapProvider::newSensorData(gpu_voxels::Vector3f* h_point_cloud, const uint32_t num_points,
                                     const uint32_t width, const uint32_t height)
{
  const string prefix = "VoxelMapProvider::" + string(__FUNCTION__);
  const string temp_timer = prefix + "_temp";

  gpu_voxels::Matrix3f orientation;
  gpu_voxels::Vector3f temp = m_sensor_orientation;
#ifdef MODE_KINECT
  orientation = gpu_voxels::Matrix3f::createFromYPR(KINECT_ORIENTATION.z, KINECT_ORIENTATION.y, KINECT_ORIENTATION.x);
  temp.z *= -1; // invert to fix the incorrect positioning for ptu-mode
#endif

  Sensor sensor;
  sensor.pose = Matrix4f::createFromRotationAndTranslation(
        Matrix3f::createFromYPR(temp.z, temp.y, temp.x) * orientation, m_sensor_position);


// transform points in world corrdinates
  for (uint32_t i = 0; i < num_points; ++i)
  {
    gpu_voxels::Vector3f point = h_point_cloud[i];
    if (!isnan(point.x) && !isnan(point.y) && !isnan(point.z))
      h_point_cloud[i] = sensor.sensorCoordinatesToWorldCoordinates(point);
  }

  PointCloud tmp_point_cloud(h_point_cloud, num_points);

  PERF_MON_START(temp_timer);

  if (voxelmap::ProbVoxelMap* _voxelmap = dynamic_cast<voxelmap::ProbVoxelMap*>(m_voxelMap))
  {
    _voxelmap->insertSensorData<BIT_VECTOR_LENGTH>(tmp_point_cloud, m_sensor_position, true, false, eBVM_OCCUPIED, NULL);
  }
  else
  {
    printf("Voxelmap can't 'insertSensorData()'\n");
  }

  PERF_MON_PRINT_INFO_P(temp_timer, "InsertSensorData", prefix);
}

void VoxelMapProvider::newSensorData(const DepthData* h_depth_data, const uint32_t width,
                                     const uint32_t height)
{
// not yet implemented
}

void VoxelMapProvider::collide()
{
  if (m_collide_with != NULL)
  {
    m_mutex.lock();
    m_collide_with->lock();

    collide_wo_locking();

    m_collide_with->unlock();
    m_mutex.unlock();
  }
}

void VoxelMapProvider::collide_wo_locking()
{
  const string prefix = "VoxelMapProvider::" + string(__FUNCTION__);
  const string temp_timer = prefix + "_temp";

  if (m_collide_with != NULL)
  {
    voxel_count num_collisions = 0;

    if (VoxelMapProvider* _provider = dynamic_cast<VoxelMapProvider*>(m_collide_with))
    {
      PERF_MON_START(temp_timer);

      if (voxelmap::ProbVoxelMap* _voxelmap = dynamic_cast<voxelmap::ProbVoxelMap*>(_provider->getVoxelMap()))
      {
        gpu_voxels::DefaultCollider c;
        num_collisions = _voxelmap->collisionCheckWithCounter(_voxelmap, c);
      }
      else
      {
        printf("Voxelmap can't 'collisionCheckWithCounter()'\n");
      }

      PERF_MON_PRINT_INFO_P(temp_timer, "", prefix);
      PERF_MON_ADD_DATA_NONTIME_P("NumCollisions", num_collisions, prefix);
      m_changed = true;
      m_collide_with->setChanged(true);
    }
  }
}

bool VoxelMapProvider::waitForNewData(volatile bool* stop)
{
//  usleep(1000000);
//  return true;
// wait till new data is required
  while (!*stop && !m_changed)
  {
    //boost::this_thread::yield();
    usleep(buffer_watch_delay);
  }
  return !*stop;
}

voxelmap::AbstractVoxelMap *VoxelMapProvider::getVoxelMap()
{
  return (voxelmap::AbstractVoxelMap*) m_voxelMap;
}

}
}
}

