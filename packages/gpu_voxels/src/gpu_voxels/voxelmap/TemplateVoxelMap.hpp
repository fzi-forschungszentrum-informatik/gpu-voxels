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
 * \date    2014-07-10
 *
 */
//----------------------------------------------------------------------/*
#ifndef GPU_VOXELS_VOXELMAP_TEMPLATE_VOXELMAP_HPP_INCLUDED
#define GPU_VOXELS_VOXELMAP_TEMPLATE_VOXELMAP_HPP_INCLUDED

#include "TemplateVoxelMap.h"
#include <iostream>
#include <fstream>
#include <gpu_voxels/voxelmap/kernels/VoxelMapOperations.hpp>
#include <gpu_voxels/voxelmap/DefaultCollider.hpp>

// temp:
#include <time.h>

namespace gpu_voxels {
namespace voxelmap {

//#define ALTERNATIVE_CHECK
#undef  ALTERNATIVE_CHECK

#ifdef  ALTERNATIVE_CHECK
#define LOOP_SIZE       4
const uint32_t cMAX_NR_OF_THREADS_PER_BLOCK = 1024;
#endif

const uint32_t cMAX_NR_OF_BLOCKS = 65535;

const uint32_t cMAX_POINTS_PER_ROBOT_SEGMENT = 118000;

template<class Voxel>
TemplateVoxelMap<Voxel>::TemplateVoxelMap(const uint32_t dim_x, const uint32_t dim_y, const uint32_t dim_z,
                                          const float voxel_side_length, const MapType map_type) :
                                          m_dim(dim_x, dim_y, dim_z), m_limits(dim_x * voxel_side_length, dim_y * voxel_side_length,
                                         dim_z * voxel_side_length), m_voxel_side_length(voxel_side_length), m_voxelmap_size(
        getVoxelMapSize()), m_visualization_data(NULL), m_visualization_data_available(false), m_dev_data(
        NULL), m_dev_data_pointer(NULL), m_dev_dim(NULL), m_dev_limits(NULL), m_dev_voxelmap_size(NULL), m_collision_check_results(
        NULL),
    // Env Map specific stuff
    m_init_sensor(false), m_dev_raw_sensor_data(NULL), m_dev_sensor(NULL), m_dev_transformed_sensor_data(NULL)

{
  this->m_map_type = map_type;
  if (dim_x * dim_y * dim_z * sizeof(Voxel) > (pow(2, 32) - 1))
  {
    LOGGING_ERROR_C(VoxelmapLog, VoxelMap, "Map size limited to 32 bit addressing!" << endl);
    exit(-1);
  }

  if (getVoxelMapSize() * sizeof(Voxel) > (pow(2, 32) - 1))
  {
    LOGGING_ERROR_C(VoxelmapLog, VoxelMap, "Memory size is limited to 32 bit!" << endl);
    exit(-1);
  }
  HANDLE_CUDA_ERROR(cudaEventCreate(&m_start));
  HANDLE_CUDA_ERROR(cudaEventCreate(&m_stop));

  // the voxelmap
  HANDLE_CUDA_ERROR(cudaMalloc((void** )&m_dev_data, getMemorySizeInByte()));
  LOGGING_DEBUG_C(VoxelmapLog, VoxelMap, "Voxelmap base address is " << (void*) m_dev_data << endl);

  // a mirror of voxelmap that is updated when copyMapForVisualization() is called
  m_visualization_data = (Voxel*) malloc(getMemorySizeInByte());

  // copy of m_dev_data pointer
  HANDLE_CUDA_ERROR(cudaMalloc((void** )&m_dev_data_pointer, sizeof(Voxel*)));
  HANDLE_CUDA_ERROR(cudaMemcpy(m_dev_data_pointer, &m_dev_data, sizeof(Voxel*), cudaMemcpyHostToDevice));

  // copy of m_dim
  HANDLE_CUDA_ERROR(cudaMalloc((void** )&m_dev_dim, sizeof(Vector3ui)));
  HANDLE_CUDA_ERROR(cudaMemcpy(m_dev_dim, &m_dim, sizeof(Vector3ui), cudaMemcpyHostToDevice));

  // copy of m_limits
  HANDLE_CUDA_ERROR(cudaMalloc((void** )&m_dev_limits, sizeof(Vector3f)));
  HANDLE_CUDA_ERROR(cudaMemcpy(m_dev_limits, &m_limits, sizeof(Vector3f), cudaMemcpyHostToDevice));

  // voxelmap size
  HANDLE_CUDA_ERROR(cudaMalloc((void** )&m_dev_voxelmap_size, sizeof(uint32_t)));

  HANDLE_CUDA_ERROR(
      cudaMemcpy(m_dev_voxelmap_size, &m_voxelmap_size, sizeof(uint32_t), cudaMemcpyHostToDevice));

  // m_voxel_side_length
  HANDLE_CUDA_ERROR(cudaMalloc((void** )&m_dev_voxel_side_length, sizeof(float)));
  HANDLE_CUDA_ERROR(
      cudaMemcpy(m_dev_voxel_side_length, &m_voxel_side_length, sizeof(float), cudaMemcpyHostToDevice));

  m_math.computeLinearLoad(m_voxelmap_size, &m_blocks, &m_threads);
#ifdef ALTERNATIVE_CHECK
  m_math.computeLinearLoad((uint32_t) ceil((float) m_voxelmap_size / (float) LOOP_SIZE), &m_alternative_blocks, &m_alternative_threads);
#endif

  m_collision_check_results = new bool[cMAX_NR_OF_BLOCKS];
  m_collision_check_results_counter = new uint16_t[cMAX_NR_OF_BLOCKS];

  // initialize result arrays
  for (uint32_t i = 0; i < cMAX_NR_OF_BLOCKS; i++)
  {
    m_collision_check_results[i] = false;
    m_collision_check_results_counter[i] = 0;
  }

  HANDLE_CUDA_ERROR(cudaMalloc((void** )&m_dev_collision_check_results, cMAX_NR_OF_BLOCKS * sizeof(bool)));
  HANDLE_CUDA_ERROR(
      cudaMalloc((void** )&m_dev_collision_check_results_counter, cMAX_NR_OF_BLOCKS * sizeof(uint16_t)));

  // copy initialized arrays to device
  HANDLE_CUDA_ERROR(
      cudaMemcpy(m_dev_collision_check_results, m_collision_check_results, cMAX_NR_OF_BLOCKS * sizeof(bool),
                 cudaMemcpyHostToDevice));
  HANDLE_CUDA_ERROR(
      cudaMemcpy(m_dev_collision_check_results_counter, m_collision_check_results_counter,
                 cMAX_NR_OF_BLOCKS * sizeof(uint16_t), cudaMemcpyHostToDevice));
  clearMap();

  cuPrintDeviceMemoryInfo();
#ifndef ALTERNATIVE_CHECK
  // determine size of array for results of collision check
  if (m_voxelmap_size >= cMAX_NR_OF_BLOCKS)
  {
    m_result_array_size = cMAX_NR_OF_BLOCKS;
  }
  else
  {
    m_result_array_size = m_voxelmap_size;
  }
#else

  // determine size of array for results of collision check
  if (ceil((float) m_voxelmap_size/(float)LOOP_SIZE) >= cMAX_NR_OF_BLOCKS)
  {
    m_result_array_size = cMAX_NR_OF_BLOCKS;
  }
  else
  {
    m_result_array_size = ceil((float)m_voxelmap_size/((float)cMAX_NR_OF_THREADS_PER_BLOCK*(float)LOOP_SIZE));
  }

#endif

  // Robot Map specific:

  // allocate max point cloud for robot inserts -> ROB
  HANDLE_CUDA_ERROR(cudaMalloc((void** )&m_dev_point_data, cMAX_POINTS_PER_ROBOT_SEGMENT * sizeof(Vector3f)));

  // allocate memory for self-collision flag;
  HANDLE_CUDA_ERROR(cudaMalloc((void** )&m_dev_self_collision, sizeof(bool)));
  //syncSelfCollisionInfoToDevice();

  // END of Robot map specific

}
template<class Voxel>
TemplateVoxelMap<Voxel>::TemplateVoxelMap(Voxel* dev_data, const Vector3ui dim, const float voxel_side_length, const MapType map_type) :
  m_dim(dim), m_limits(dim.x * voxel_side_length, dim.y * voxel_side_length,
                                                 dim.z * voxel_side_length), m_voxel_side_length(
        voxel_side_length), m_voxelmap_size(getVoxelMapSize()), m_visualization_data(NULL), m_visualization_data_available(
        false), m_dev_data(dev_data), m_dev_data_pointer(NULL), m_dev_dim(NULL), m_dev_limits(NULL), m_dev_voxelmap_size(
        NULL), m_collision_check_results(NULL)
{
  this->m_map_type = map_type;

  // copy of m_dev_data pointer
  HANDLE_CUDA_ERROR(cudaMalloc((void** )&m_dev_data_pointer, sizeof(Voxel*)));
  HANDLE_CUDA_ERROR(cudaMemcpy(m_dev_data_pointer, &m_dev_data, sizeof(Voxel*), cudaMemcpyHostToDevice));

  // copy of m_dim
  HANDLE_CUDA_ERROR(cudaMalloc((void** )&m_dev_dim, sizeof(Vector3ui)));
  HANDLE_CUDA_ERROR(cudaMemcpy(m_dev_dim, &m_dim, sizeof(Vector3ui), cudaMemcpyHostToDevice));

  // copy of m_limits
  HANDLE_CUDA_ERROR(cudaMalloc((void** )&m_dev_limits, sizeof(Vector3f)));
  HANDLE_CUDA_ERROR(cudaMemcpy(m_dev_limits, &m_limits, sizeof(Vector3f), cudaMemcpyHostToDevice));

  // voxelmap size
  HANDLE_CUDA_ERROR(cudaMalloc((void** )&m_dev_voxelmap_size, sizeof(uint32_t)));
  HANDLE_CUDA_ERROR(
      cudaMemcpy(m_dev_voxelmap_size, &m_voxelmap_size, sizeof(uint32_t), cudaMemcpyHostToDevice));

  // m_voxel_side_length
  HANDLE_CUDA_ERROR(cudaMalloc((void** )&m_dev_voxel_side_length, sizeof(float)));
  HANDLE_CUDA_ERROR(
      cudaMemcpy(m_dev_voxel_side_length, &m_voxel_side_length, sizeof(float), cudaMemcpyHostToDevice));
  m_math.computeLinearLoad(m_voxelmap_size, &m_blocks, &m_threads);
}

template<class Voxel>
TemplateVoxelMap<Voxel>::~TemplateVoxelMap()
{
  if (m_dev_collision_check_results_counter)
  {
    HANDLE_CUDA_ERROR(cudaFree(m_dev_collision_check_results_counter));
  }
  if (m_dev_collision_check_results)
  {
    HANDLE_CUDA_ERROR(cudaFree(m_dev_collision_check_results));
  }
  if (m_collision_check_results_counter)
  {
    delete[] m_collision_check_results_counter;
  }
  if (m_collision_check_results)
  {
    delete[] m_collision_check_results;
  }

  if (m_dev_voxelmap_size)
  {
    HANDLE_CUDA_ERROR(cudaFree(m_dev_voxelmap_size));
  }
  if (m_dev_voxel_side_length)
  {
    HANDLE_CUDA_ERROR(cudaFree(m_dev_voxel_side_length));
  }

  if (m_dev_limits)
  {
    HANDLE_CUDA_ERROR(cudaFree(m_dev_limits));
  }
  if (m_dev_dim)
  {
    HANDLE_CUDA_ERROR(cudaFree(m_dev_dim));
  }
  if (m_dev_data_pointer)
  {
    HANDLE_CUDA_ERROR(cudaFree(m_dev_data_pointer));
  }
  if (m_visualization_data)
  {
    free(m_visualization_data);
  }

  if (m_dev_data)
  {
    HANDLE_CUDA_ERROR(cudaFree(m_dev_data));
  }

  HANDLE_CUDA_ERROR(cudaEventDestroy(m_start));
  HANDLE_CUDA_ERROR(cudaEventDestroy(m_stop));

  // Env Map specific stuff
  if (m_dev_transformed_sensor_data)
  {
    HANDLE_CUDA_ERROR(cudaFree(m_dev_transformed_sensor_data));
  }
  if (m_dev_sensor)
  {
    HANDLE_CUDA_ERROR(cudaFree(m_dev_sensor));
  }
  if (m_dev_raw_sensor_data)
  {
    HANDLE_CUDA_ERROR(cudaFree(m_dev_raw_sensor_data));
  }
  // End of Env Map specific

  // Robot Map specific destructor
  if (m_dev_self_collision)
  {
    HANDLE_CUDA_ERROR(cudaFree(m_dev_self_collision));
  }

  if (m_dev_point_data)
  {
    HANDLE_CUDA_ERROR(cudaFree(m_dev_point_data));
  }
  // End of Robot map specific
}

/* ======== VoxelMap operations  ======== */

template<class Voxel>
void TemplateVoxelMap<Voxel>::clearMap()
{
  while (!lockMutex())
  {
    boost::this_thread::yield();
  }
  // Clear occupancies
  HANDLE_CUDA_ERROR(cudaDeviceSynchronize());
  kernelClearVoxelMap<<<m_blocks, m_threads>>>(m_dev_data, m_voxelmap_size);

  // Clear result array
  for (uint32_t i = 0; i < cMAX_NR_OF_BLOCKS; i++)
  {
    m_collision_check_results[i] = false;
  }
  HANDLE_CUDA_ERROR(
      cudaMemcpy(m_dev_collision_check_results, m_collision_check_results, cMAX_NR_OF_BLOCKS * sizeof(bool),
                 cudaMemcpyHostToDevice));
  unlockMutex();
}

template<class Voxel>
void TemplateVoxelMap<Voxel>::printVoxelMapData()
{
  while (!lockMutex())
  {
    boost::this_thread::yield();
  }
  HANDLE_CUDA_ERROR(cuPrintDeviceArray(m_dev_data, m_voxelmap_size, "VoxelMap dump: "));
  unlockMutex();
}

//template<class Voxel>
//void TemplateVoxelMap<Voxel>::printVoxelMapDataFromDevice()
//{
//  while (!lockMutex())
//  {
//    boost::this_thread::yield();
//  }
//  HANDLE_CUDA_ERROR(cudaDeviceSynchronize());
//  kernelDumpVoxelMap<<< m_blocks, m_threads >>>
//  (m_dev_data, m_dev_dim, m_voxelmap_size);
//  HANDLE_CUDA_ERROR(cudaDeviceSynchronize());
//  unlockMutex();
//}

//template<class Voxel>
//void TemplateVoxelMap<Voxel>::copyMapForVisualization()
//{
////  printf("trying to get mutex\n");
//  while (!lockMutex())
//  {
//    boost::this_thread::yield();
//  }
////  printf("got mutex\n");
//
//  if (HANDLE_CUDA_ERROR(cudaMemcpy(m_visualization_data, m_dev_data, getMemorySizeInByte(), cudaMemcpyDeviceToHost)))
//  {
//    m_visualization_data_available = true;
//  }
//  else
//  {
//    m_visualization_data_available = false;
//  }
//  LOGGING_DEBUG_C(VoxelmapLog, VoxelMap, "Copied data for visualization." << endl);
//  unlockMutex();
//}

/* DUMPING OF MEASUREMENTS */

//template<class Voxel>
//bool TemplateVoxelMap<Voxel>::writeLog(std::string filename, uint32_t loop_size, bool reset_values)
//{
//  std::ofstream file(filename.c_str(), std::fstream::app);
//
//  if (!file)
//  {
//    LOGGING_ERROR_C(VoxelmapLog, VoxelMap, "Could not open file " << filename.c_str() << " !" << endl);
//    return false;
//  }
//  if (reset_values)
//  {
//    m_measured_data.clear();
//    file << "--------------------- Starting new Measurement ----------------------------------" << std::endl;
//  }
//  else if (m_measured_data.size() != 0)
//  {
//    double average = 0;
//#ifdef ALTERNATIVE_CHECK
//    loop_size = LOOP_SIZE;
//#endif
//    // header
//    file << "# map_size | average time.    loop_size = " << loop_size << "    dim: " << m_dim.x << " x "
//        << m_dim.y << " x " << m_dim.z << std::endl;
//    for (uint32_t i = 0; i < m_measured_data.size(); i++)
//    {
//      average += m_measured_data[i];
//    }
//    average /= m_measured_data.size();
//    file << m_dim.x * m_dim.y * m_dim.z << "  " << average << std::endl;
//
//    LOGGING_INFO_C(VoxelmapLog, VoxelMap, "Average written to " << filename.c_str() << endl);
//  }
//  else
//  {
//    LOGGING_ERROR_C(VoxelmapLog, VoxelMap, "No data to write!" << endl);
//    return false;
//  }
//  file.close();
//
//  return true;
//}
//template<class Voxel>
//bool TemplateVoxelMap<Voxel>::collisionCheckAlternative(const uint8_t threshold, VoxelMap* other,
//                                         const uint8_t other_threshold, uint32_t loop_size)
//{
////  bool locked_this = false;
////  bool locked_other = false;
////  uint32_t counter = 0;
////
////  while (!locked_this && !locked_other)
////  {
////    // lock mutexes
////    while (!locked_this)
////    {
////      locked_this = lockMutex();
////      boost::this_thread::yield();
////    }
////    while (!locked_other && (counter < 50))
////    {
////      locked_other = other->lockMutex();
////      boost::this_thread::yield();
////      counter++;
////    }
////
////    if (!locked_other)
////    {
////      counter = 0;
////      unlockMutex();
////    }
////  }
//  m_math.computeLinearLoad((uint32_t) ceil((float) m_voxelmap_size / (float) loop_size),
//                           &m_alternative_blocks, &m_alternative_threads);
////  printf("number of blocks: %i , number of threads: %i", m_alternative_blocks, m_alternative_threads);
//  HANDLE_CUDA_ERROR(cudaDeviceSynchronize());
//  m_elapsed_time = 0;
////  HANDLE_CUDA_ERROR(cudaEventRecord(m_start, 0));
//  kernelCollideVoxelMapsAlternative<<< m_alternative_blocks, m_alternative_threads >>>
//  (m_dev_data, m_voxelmap_size, threshold, other->getDeviceDataPtr(), other_threshold, loop_size, m_dev_collision_check_results);
////  HANDLE_CUDA_ERROR(cudaDeviceSynchronize());
//  HANDLE_CUDA_ERROR(
//      cudaMemcpy(m_collision_check_results, m_dev_collision_check_results, cMAX_NR_OF_BLOCKS * sizeof(bool),
//                 cudaMemcpyDeviceToHost));
//
////  bool test;
//  for (uint32_t i = 0; i < m_result_array_size; i++)
//  {
//    // collision as soon as first result is true
//    if (m_collision_check_results[i])
//    {
////      HANDLE_CUDA_ERROR(cudaEventRecord(m_stop, 0));
////      HANDLE_CUDA_ERROR(cudaEventSynchronize(m_stop));
////      HANDLE_CUDA_ERROR(cudaEventElapsedTime(&m_elapsed_time, m_start, m_stop));
////      printf(" ...done in %f ms!\n", m_elapsed_time);
////      m_measured_data.push_back(m_elapsed_time);
//      return true;
//    }
//  }
////  HANDLE_CUDA_ERROR(cudaEventRecord(m_stop, 0));
////  HANDLE_CUDA_ERROR(cudaEventSynchronize(m_stop));
////  HANDLE_CUDA_ERROR(cudaEventElapsedTime(&m_elapsed_time, m_start, m_stop));
////  printf(" ...done in %f ms!\n", m_elapsed_time);
////  m_measured_data.push_back(m_elapsed_time);
//  //HANDLE_CUDA_ERROR(cuPrintDeviceArray(m_dev_collision_check_results, cMAX_NR_OF_BLOCKS, " collision array on device "));
//  return false;
//}
template<class Voxel>
template<class OtherVoxel, class Collider>
bool TemplateVoxelMap<Voxel>::collisionCheck(TemplateVoxelMap<OtherVoxel>* other, Collider collider)
{
  bool locked_this = false;
  bool locked_other = false;
  uint32_t counter = 0;

  while (!locked_this && !locked_other)
  {
    // lock mutexes
    while (!locked_this)
    {
      locked_this = lockMutex();
      boost::this_thread::yield();
    }
    while (!locked_other && (counter < 50))
    {
      locked_other = other->lockMutex();
      boost::this_thread::yield();
      counter++;
    }

    if (!locked_other)
    {
      counter = 0;
      unlockMutex();
    }
  }
  //printf("collision check... ");

#ifndef ALTERNATIVE_CHECK
  HANDLE_CUDA_ERROR(cudaDeviceSynchronize());
//  m_elapsed_time = 0;
//  HANDLE_CUDA_ERROR(cudaEventRecord(m_start, 0));
//  printf("TemplateVoxelMap<Voxel>::collisionCheck\n");

  kernelCollideVoxelMaps<<<m_blocks, m_threads>>>(m_dev_data, m_voxelmap_size, other->getDeviceDataPtr(),
                                                  collider, m_dev_collision_check_results);

  HANDLE_CUDA_ERROR(cudaDeviceSynchronize());
  HANDLE_CUDA_ERROR(
      cudaMemcpy(m_collision_check_results, m_dev_collision_check_results, cMAX_NR_OF_BLOCKS * sizeof(bool),
                 cudaMemcpyDeviceToHost));

  for (uint32_t i = 0; i < m_blocks; i++)
  {
    //printf(" results[%d] = %s\n", i, m_collision_check_results[i]? "collision" : "no collision");
    // collision as soon as first result is true
    if (m_collision_check_results[i])
    {
//      HANDLE_CUDA_ERROR(cudaEventRecord(m_stop, 0));
//      HANDLE_CUDA_ERROR(cudaEventSynchronize(m_stop));
//      HANDLE_CUDA_ERROR(cudaEventElapsedTime(&m_elapsed_time, m_start, m_stop));
//      printf(" ...done in %f ms!\n", m_elapsed_time);
//      m_measured_data.push_back(m_elapsed_time);
//      printf("TemplateVoxelMap<Voxel>::collisionCheck finished\n");
      other->unlockMutex();
      unlockMutex();
      return true;
    }
  }
//  HANDLE_CUDA_ERROR(cudaEventRecord(m_stop, 0));
//  HANDLE_CUDA_ERROR(cudaEventSynchronize(m_stop));
//  HANDLE_CUDA_ERROR(cudaEventElapsedTime(&m_elapsed_time, m_start, m_stop));
  //printf(" ...done in %f ms!\n", m_elapsed_time);
  //HANDLE_CUDA_ERROR(cuPrintDeviceArray(m_dev_collision_check_results, cMAX_NR_OF_BLOCKS, " collision array on device "));
//  m_measured_data.push_back(m_elapsed_time);
//  printf("TemplateVoxelMap<Voxel>::collisionCheck finished\n");

  HANDLE_CUDA_ERROR(cudaDeviceSynchronize());

  // unlock mutexes
//  printf("unlocking other map's mutex\n");
  other->unlockMutex();
//  printf("unlocking this map's mutex\n");
  unlockMutex();
//  printf("done unlocking\n");
  return false;

#else

  HANDLE_CUDA_ERROR(cudaDeviceSynchronize());
  m_elapsed_time = 0;
//  HANDLE_CUDA_ERROR(cudaEventRecord(m_start, 0));
  kernelCollideVoxelMapsAlternative<<< m_alternative_blocks, m_alternative_threads >>>
  (m_dev_data, m_voxelmap_size, threshold, other->getDeviceDataPtr(), other_threshold, LOOP_SIZE, m_dev_collision_check_results);

  HANDLE_CUDA_ERROR(cudaDeviceSynchronize());
  HANDLE_CUDA_ERROR(cudaMemcpy(m_collision_check_results, m_dev_collision_check_results, cMAX_NR_OF_BLOCKS * sizeof(bool), cudaMemcpyDeviceToHost));

  for (uint32_t i=0; i<m_result_array_size; i++)
  {
    // collision as soon as first result is true
    if (m_collision_check_results[i])
    {
//      HANDLE_CUDA_ERROR(cudaEventRecord(m_stop, 0));
//      HANDLE_CUDA_ERROR(cudaEventSynchronize(m_stop));
//      HANDLE_CUDA_ERROR(cudaEventElapsedTime(&m_elapsed_time, m_start, m_stop));
//      printf(" ...done in %f ms!\n", m_elapsed_time);
//      m_measured_data.push_back(m_elapsed_time);
      return true;
    }
  }
//  HANDLE_CUDA_ERROR(cudaEventRecord(m_stop, 0));
//  HANDLE_CUDA_ERROR(cudaEventSynchronize(m_stop));
//  HANDLE_CUDA_ERROR(cudaEventElapsedTime(&m_elapsed_time, m_start, m_stop));
//  printf(" ...done in %f ms!\n", m_elapsed_time);
//  m_measured_data.push_back(m_elapsed_time);
  //HANDLE_CUDA_ERROR(cuPrintDeviceArray(m_dev_collision_check_results, cMAX_NR_OF_BLOCKS, " collision array on device "));
  return false;

#endif
}

//template<class Voxel>
//bool TemplateVoxelMap<Voxel>::collisionCheckBoundingBox(uint8_t threshold, VoxelMap* other, uint8_t other_threshold,
//                                         Vector3ui bounding_box_start, Vector3ui bounding_box_end)
//{
//  int number_of_blocks = bounding_box_end.y - bounding_box_start.y;
////	number_of_blocks = number_of_blocks * (bounding_box_end.x - bounding_box_start.x);
//  int number_of_threads = bounding_box_end.z - bounding_box_start.z;
//  int number_of_thread_runs = bounding_box_end.x - bounding_box_start.x;
//
//  bool* dev_result;
//  HANDLE_CUDA_ERROR(cudaEventRecord(m_start, 0));
//  //aquiring memory for results:
//  HANDLE_CUDA_ERROR(cudaMalloc((void** )&dev_result, sizeof(bool) * number_of_blocks * number_of_threads));
//
//  kernelCollideVoxelMapsBoundingBox<<<number_of_blocks, number_of_threads, number_of_threads>>>
//  (m_dev_data, m_voxelmap_size, threshold, other->m_dev_data, other_threshold,
//      dev_result, bounding_box_start.x, bounding_box_start.y, bounding_box_start.z,
//      number_of_thread_runs, m_dev_dim);
//
//  bool result_array[number_of_blocks * number_of_threads];
//  //Copying results back
//  HANDLE_CUDA_ERROR(
//      cudaMemcpy(&result_array[0], dev_result, sizeof(bool) * number_of_blocks * number_of_threads,
//                 cudaMemcpyDeviceToHost));
//
//  bool result = false;
//  //reducing result
//
//  for (int u = 0; u < number_of_blocks * number_of_threads; ++u)
//  {
////		printf("coll-check-bounding: block nr %i , result %i \n", u, result_array[u]);
//    if (result_array[u])
//    {
//      LOGGING_DEBUG_C(VoxelmapLog, VoxelMap, "Collision occurred!" << endl);
//      result = true;
//      break;
//    }
//  }
//  LOGGING_DEBUG_C(VoxelmapLog, VoxelMap, "No Collision occurred!" << endl);
//  HANDLE_CUDA_ERROR(cudaEventRecord(m_stop, 0));
//  HANDLE_CUDA_ERROR(cudaEventSynchronize(m_stop));
//  HANDLE_CUDA_ERROR(cudaEventElapsedTime(&m_elapsed_time, m_start, m_stop));
////	      printf(" ...done in %f ms!\n", m_elapsed_time);
//
//  //releasing memory
//
//  HANDLE_CUDA_ERROR(cudaFree(dev_result));
//
//  return result;
//  //
//  //
//  //
//  //	void kernelCollideVoxelMapsBoundingBox(Voxel* voxelmap, const uint32_t voxelmap_size,
//  //	                            const uint8_t threshold, Voxel* other_map,
//  //	                            const uint8_t other_threshold, bool* results,
//  //	                            uint32_t offset_x, uint32_t offset_y, uint32_t offset_z,
//  //	                            uint32_t size_x, Vector3ui* dimensions)
//
//}

//template<class Voxel>
//bool TemplateVoxelMap<Voxel>::collisionCheckIndices(uint8_t threshold, uint32_t* index_list, uint32_t index_size)
//{
//  uint32_t number_of_blocks;
//  uint32_t number_of_threads;
//  m_math.computeLinearLoad(index_size, &number_of_blocks, &number_of_threads);
//
//  HANDLE_CUDA_ERROR(cudaDeviceSynchronize());
//  m_elapsed_time = 0;
//  HANDLE_CUDA_ERROR(cudaEventRecord(m_start, 0));
//  //  printf("TemplateVoxelMap<Voxel>::collisionCheck\n");
//  kernelCollideVoxelMapsIndices<<<number_of_blocks, number_of_threads>>>
//  (m_dev_data, threshold, m_dev_collision_check_results, index_list, index_size);
//
//  HANDLE_CUDA_ERROR(
//      cudaMemcpy(m_collision_check_results, m_dev_collision_check_results, cMAX_NR_OF_BLOCKS * sizeof(bool),
//                 cudaMemcpyDeviceToHost));
//
////	printf("number of blocks: %i", number_of_blocks);
//  for (uint32_t i = 0; i < number_of_blocks; i++)
//  {
////		printf("---------------- i=1297: %i\n", m_collision_check_results[1297]);
////		if(i == 1297)
////		{
////			printf("----- 1297 ---- result: %i\n", m_collision_check_results[i]);
////		}
//    //printf(" results[%d] = %s\n", i, m_collision_check_results[i]? "collision" : "no collision");
//    // collision as soon as first result is true
//    if (m_collision_check_results[i])
//    {
//      LOGGING_DEBUG_C(VoxelmapLog, VoxelMap, "Collision occurred, i: " << i << endl);
//      HANDLE_CUDA_ERROR(cudaEventRecord(m_stop, 0));
//      HANDLE_CUDA_ERROR(cudaEventSynchronize(m_stop));
//      HANDLE_CUDA_ERROR(cudaEventElapsedTime(&m_elapsed_time, m_start, m_stop));
////	      printf(" ...done in %f ms!\n", m_elapsed_time);
//      //      m_measured_data.push_back(m_elapsed_time);
//      //      printf("TemplateVoxelMap<Voxel>::collisionCheck finished\n");
//      return true;
//    }
//
//  }
//  HANDLE_CUDA_ERROR(cudaEventRecord(m_stop, 0));
//  HANDLE_CUDA_ERROR(cudaEventSynchronize(m_stop));
//  HANDLE_CUDA_ERROR(cudaEventElapsedTime(&m_elapsed_time, m_start, m_stop));
////      printf(" ...done in %f ms!\n", m_elapsed_time);
//
//  return false;
//}

//template<class Voxel>
//uint64_t TemplateVoxelMap<Voxel>::collisionCheckIndicesBitmap(uint8_t threshold, uint32_t* index_list, uint32_t index_size,
//                                               uint64_t* bitmap_list, int64_t offset_x, int64_t offset_y)
//{
//
//  uint32_t number_of_blocks;
//  uint32_t number_of_threads;
//  m_math.computeLinearLoad(index_size, &number_of_blocks, &number_of_threads);
//  m_elapsed_time = 0;
//  //HANDLE_CUDA_ERROR(cudaEventRecord(m_start, 0));
//  //Allocating Memory on device for results
//  uint64_t* result_ptr_dev;
//  HANDLE_CUDA_ERROR(cudaMalloc((void** )&result_ptr_dev, sizeof(uint64_t) * number_of_blocks)); //Todo: Allocate this memory once in Constructor
//  //calculate the offset of the pointer:
//  uint64_t total_offset = offset_x;
//  total_offset += m_dim.x * offset_y;
//
//  kernelCollideVoxelMapsIndicesBitmap<<< number_of_blocks, number_of_threads>>>
//  (m_dev_data + total_offset, threshold, result_ptr_dev, index_list,
//      index_size, bitmap_list, m_dev_dim);
//
////    (Voxel* voxelmap, uint8_t threshold, uint64_t* results,
////    										uint32_t* index_list, uint32_t index_number, uint64_t* bitmap_list)
//
//  //copying result from device
//  uint64_t result_array[number_of_blocks];
//  for (uint32_t i = 0; i < number_of_blocks; ++i)
//  {
//    result_array[i] = 0;
//  }
//  HANDLE_CUDA_ERROR(cudaDeviceSynchronize());
//
//  HANDLE_CUDA_ERROR(
//      cudaMemcpy(&(result_array[0]), result_ptr_dev, sizeof(uint64_t) * number_of_blocks,
//                 cudaMemcpyDeviceToHost));
//  uint64_t result = 0;
//  for (uint32_t i = 0; i < number_of_blocks; ++i)
//  {
//    result |= result_array[i];
//
//  }
//
//  HANDLE_CUDA_ERROR(cudaFree(result_ptr_dev));
//
//  //HANDLE_CUDA_ERROR(cudaEventRecord(m_stop, 0));
//  //HANDLE_CUDA_ERROR(cudaEventSynchronize(m_stop));
//  //HANDLE_CUDA_ERROR(cudaEventElapsedTime(&m_elapsed_time, m_start, m_stop));
////      printf(" ...done in %f ms!\n", m_elapsed_time);
//  return result;
//
//}

template<class Voxel>
template<class OtherVoxel, class Collider>
uint32_t TemplateVoxelMap<Voxel>::collisionCheckWithCounter(TemplateVoxelMap<OtherVoxel>* other,
                                                            Collider collider)
{
  bool locked_this = false;
  bool locked_other = false;
  uint32_t counter = 0;

  while (!locked_this && !locked_other)
  {
    // lock mutexes
    while (!locked_this)
    {
      locked_this = lockMutex();
      boost::this_thread::yield();
    }
    while (!locked_other && (counter < 50))
    {
      locked_other = other->lockMutex();
      boost::this_thread::yield();
      counter++;
    }

    if (!locked_other)
    {
      counter = 0;
      unlockMutex();
    }
  }

  HANDLE_CUDA_ERROR(cudaDeviceSynchronize());
  kernelCollideVoxelMapsDebug<<<m_blocks, m_threads>>>(m_dev_data, m_voxelmap_size, other->getDeviceDataPtr(),
                                                       collider, m_dev_collision_check_results_counter);
  HANDLE_CUDA_ERROR(cudaDeviceSynchronize());
  HANDLE_CUDA_ERROR(
      cudaMemcpy(m_collision_check_results_counter, m_dev_collision_check_results_counter,
                 cMAX_NR_OF_BLOCKS * sizeof(uint16_t), cudaMemcpyDeviceToHost));

  uint32_t number_of_collisions = 0;
  for (uint32_t i = 0; i < m_blocks; i++)
  {
    number_of_collisions += m_collision_check_results_counter[i];
  }

  // unlock mutexes
//  printf("unlocking other map's mutex\n");
  other->unlockMutex();
//  printf("unlocking this map's mutex\n");
  unlockMutex();
//  printf("done unlocking\n");
  return number_of_collisions;
}

template<class Voxel>
size_t TemplateVoxelMap<Voxel>::collideWith(const GpuVoxelsMapSharedPtr other, float coll_threshold)
{
  size_t collisions = SSIZE_MAX;
  switch (other->getMapType())
  {
    case MT_PROBAB_VOXELMAP:
    {
      DefaultCollider collider(coll_threshold);
      VoxelMap* m = (VoxelMap*) other.get();
      collisions = collisionCheckWithCounter(m, collider);
      break;
    }
    case MT_BIT_VOXELMAP:
    {
      DefaultCollider collider(coll_threshold);
      BitVectorVoxelMap* m = (BitVectorVoxelMap*) other.get();
      collisions = collisionCheckWithCounter(m, collider);
      break;
    }
    case MT_OCTREE:
    {
      // Have to collide the octree with the voxel map the other way round
      // --> cyclic dependency between libs would be necessary

      LOGGING_ERROR_C(VoxelmapLog, TemplateVoxelMap, GPU_VOXELS_MAP_OPERATION_NOT_YET_SUPPORTED << " " << GPU_VOXELS_MAP_SWAP_FOR_COLLIDE << endl);

//      NTree::GvlNTreeDet* m = (NTree::GvlNTreeDet*) other.get();
//      GpuVoxelsMap* l = this;
//      GpuVoxelsMapSharedPtr tmp(l);
//      m->collideWith(tmp, coll_threshold);
      break;
    }
    default:
    {
      LOGGING_ERROR_C(VoxelmapLog, TemplateVoxelMap, GPU_VOXELS_MAP_OPERATION_NOT_YET_SUPPORTED << endl);
      break;
    }
  }
  return collisions;
}

template<class Voxel>
size_t TemplateVoxelMap<Voxel>::collideWithResolution(const GpuVoxelsMapSharedPtr other, float coll_threshold,
                                                      const uint32_t resolution_level)
{
  size_t collisions = SSIZE_MAX;
  switch (other->getMapType())
  {
    case MT_OCTREE:
    {
      LOGGING_ERROR_C(
          VoxelmapLog, TemplateVoxelMap,
          GPU_VOXELS_MAP_OPERATION_NOT_YET_SUPPORTED << " " << GPU_VOXELS_MAP_SWAP_FOR_COLLIDE << endl);
      break;
    }
    default:
    {
      if (resolution_level != 0)
        LOGGING_ERROR_C(VoxelmapLog, TemplateVoxelMap, GPU_VOXELS_MAP_OPERATION_NOT_SUPPORTED << endl);
      else
        collisions = collideWith(other, coll_threshold);
      break;
    }
  }
  return collisions;
}

template<class Voxel>
size_t TemplateVoxelMap<Voxel>::collideWithTypes(const GpuVoxelsMapSharedPtr other, BitVectorVoxel&  types_in_collision, float coll_threshold)
{
  LOGGING_ERROR_C(VoxelmapLog, TemplateVoxelMap, GPU_VOXELS_MAP_OPERATION_NOT_SUPPORTED << endl);
  return SSIZE_MAX;
}

//void TemplateVoxelMap<Voxel>::copyVoxelVectorToDevice(uint32_t index_list, uint32_t size, uint32_t* dev_voxel_list)
//{
//	HANDLE_CUDA_ERROR(cudaMalloc((void**)&dev_voxel_list, sizeof(uint32_t) * size));
//	HANDLE_CUDA_ERROR(cudaMemcpy(dev_voxel_list, &index_list[0], sizeof(uint32_t) * size, cudaMemcpyHostToDevice));
//}

//template<class Voxel>
//void TemplateVoxelMap<Voxel>::insertVoxelVector(uint32_t* dev_voxel_list, uint32_t size, bool with_bitvector, uint64_t mask)
//{
//  uint32_t blocks = 0;
//  uint32_t threads = 0;
//  m_math.computeLinearLoad(size, &blocks, &threads);
//  if (with_bitvector)
//  {
//    kernelInsertVoxelVectorBitmap<<< blocks, threads >>>
//    (m_dev_data, dev_voxel_list, size, mask);
//
////		void kernelInsertVoxelVectorBitmap(Voxel* destination_map, uint32_t* voxel_list,
////				uint32_t list_size, uint64_t mask);
//  }
//  else
//  {
//    kernelInsertVoxelVector<<< blocks, threads >>>
//    (m_dev_data, dev_voxel_list, size);
//  }
//}

//template<class Voxel>
//void TemplateVoxelMap<Voxel>::insertBitmapByIndices(uint32_t size, uint32_t* index_list, uint64_t* bitmaps)
//{
//  uint32_t blocks = 0;
//  uint32_t threads = 0;
//  m_math.computeLinearLoad(size, &blocks, &threads);
//
//  kernelInsertBitmapByIndices<<< blocks, threads >>>
//  (m_dev_data, index_list, size, bitmaps);
//
////	void kernelInsertBitmapByIndices(Voxel* destination_map, uint32_t* voxel_list,
////			uint32_t list_size, uint64_t* bitvector)
//}

//template<class Voxel>
//void TemplateVoxelMap<Voxel>::copyVoxelMapDifferentSize(VoxelMap* destination, VoxelMap* source, bool with_bitvector)
//{
//  //factor between the voxel side length of the destination and the source map.
//  //the Destination Map must alwas have a bigger or equal voxel side length as the
//  //source map
//  uint8_t factor = (uint8_t) (destination->getVoxelSideLength() / source->getVoxelSideLength());
//
//  uint32_t blocks = 0;
//  uint32_t threads = 0;
//
//  destination->m_math.computeLinearLoad(destination->m_voxelmap_size, &blocks, &threads);
//  if (with_bitvector)
//  {
//    kernelShrinkCopyVoxelMapBitvector<<<blocks, threads>>>(destination->m_dev_data,
//                                                           destination->m_voxelmap_size,
//                                                           destination->m_dev_dim, source->m_dev_data,
//                                                           source->m_voxelmap_size, source->m_dev_dim,
//                                                           factor);
//  }
//  else
//  {
//    kernelShrinkCopyVoxelMap<<<blocks, threads>>>(destination->m_dev_data, destination->m_voxelmap_size,
//                                                  destination->m_dev_dim, source->m_dev_data,
//                                                  source->m_voxelmap_size, source->m_dev_dim, factor);
//
//  }
//
//  HANDLE_CUDA_ERROR(cudaDeviceSynchronize());
////		void copyVoxelMapBitvector(Voxel* destination_map, const uint32_t destination_map_size, Vector3ui* dest_map_dim,
////				Voxel* source_map, const uint32_t source_map_size, Vector3ui* source_map_dim, uint8_t factor)
//}
/* ======== some functions for self testing ======== */

template<class Voxel>
bool TemplateVoxelMap<Voxel>::lockMutex()
{
  return m_mutex.try_lock();
}

template<class Voxel>
void TemplateVoxelMap<Voxel>::unlockMutex()
{
  m_mutex.unlock();
}

//template<class Voxel>
//void TemplateVoxelMap<Voxel>::insertBox(Vector3f cartesian_from, Vector3f cartesian_to, VoxelType voxeltype,
//                         uint8_t occupancy)
//{
//  Vector3ui integer_from;
//  Vector3ui integer_to;
//
//  // map cartesian coordinates to voxels:
//  integer_from.x = static_cast<uint32_t>(floor(cartesian_from.x / m_voxel_side_length));
//  integer_from.y = static_cast<uint32_t>(floor(cartesian_from.y / m_voxel_side_length));
//  integer_from.z = static_cast<uint32_t>(floor(cartesian_from.z / m_voxel_side_length));
//  integer_to.x = static_cast<uint32_t>(floor(cartesian_to.x / m_voxel_side_length));
//  integer_to.y = static_cast<uint32_t>(floor(cartesian_to.y / m_voxel_side_length));
//  integer_to.z = static_cast<uint32_t>(floor(cartesian_to.z / m_voxel_side_length));
//
//  insertBoxByIndices(integer_from, integer_to, voxeltype, occupancy);
//}

//template<class Voxel>
//void TemplateVoxelMap<Voxel>::insertBoxByIndices(Vector3ui indices_from, Vector3ui indices_to, VoxelType voxeltype,
//                                         uint8_t occupancy)
//{
//  if ((indices_from.x > m_dim.x) || (indices_from.y > m_dim.y) || (indices_from.z > m_dim.z)
//      || (indices_to.x > m_dim.x) || (indices_to.y > m_dim.y) || (indices_to.z > m_dim.z))
//  {
//    LOGGING_WARNING_C(VoxelmapLog, VoxelMap, "Indices for box out of map range!" << endl);
//    return;
//  }
//  if ((indices_from.x > indices_to.x) || (indices_from.y > indices_to.y) || (indices_from.z > indices_to.z))
//  {
//    LOGGING_WARNING_C(VoxelmapLog, VoxelMap,
//                      "from-indices for box should be smaller than to-indices!" << endl);
//    return;
//  }
//
////  uint32_t nr_of_voxels_in_box = (indices_to.x - indices_from.x + 1) * (indices_to.y - indices_from.y + 1)
////      * (indices_to.z - indices_from.z + 1);
////  printf("VoxelMap: inserting box from (%u, %u, %u) to (%u, %u, %u) with %u voxels.\n", indices_from.x,
////         indices_from.y, indices_from.z, indices_to.x, indices_to.y, indices_to.z, nr_of_voxels_in_box);
//
//  uint32_t box_blocks, box_threads;
//  m_math.computeLinearLoad(m_voxelmap_size, &box_blocks, &box_threads);
//  HANDLE_CUDA_ERROR(cudaDeviceSynchronize());
//  kernelInsertBox<<<box_blocks, box_threads>>>(m_dev_data, m_voxelmap_size, m_dev_dim, indices_from.x,
//                                               indices_from.y, indices_from.z, indices_to.x, indices_to.y,
//                                               indices_to.z, voxeltype, occupancy);
//
//  HANDLE_CUDA_ERROR(cudaDeviceSynchronize());
//}

/**
 * author: Matthias Wagner
 * Inserts a voxel at each point from the points list.
 */
template<class Voxel>
void TemplateVoxelMap<Voxel>::insertPointCloud(const std::vector<Vector3f> &points, const uint32_t voxel_type)
{
// copy points to the gpu
  Vector3f* d_points;
  HANDLE_CUDA_ERROR(cudaMalloc(&d_points, points.size() * sizeof(Vector3f)));
  HANDLE_CUDA_ERROR(
      cudaMemcpy(d_points, &points[0], points.size() * sizeof(Vector3f), cudaMemcpyHostToDevice));

  uint32_t num_blocks, threads_per_block;
  m_math.computeLinearLoad(points.size(), &num_blocks, &threads_per_block);
  HANDLE_CUDA_ERROR(cudaDeviceSynchronize());
  kernelInsertGlobalPointCloud<<<num_blocks, threads_per_block>>>(m_dev_data, m_dev_dim, m_voxel_side_length,
                                                                  d_points, points.size(), voxel_type);
  HANDLE_CUDA_ERROR(cudaDeviceSynchronize());
  HANDLE_CUDA_ERROR(cudaFree(d_points));
}

// ------ BEGIN Global API functions ------
template<class Voxel>
void TemplateVoxelMap<Voxel>::insertGlobalData(const std::vector<Vector3f> &point_cloud, VoxelType voxelType)
{
  insertPointCloud(point_cloud, voxelType);
}

template<class Voxel>
void TemplateVoxelMap<Voxel>::insertMetaPointCloud(const MetaPointCloud &meta_point_cloud,
                                                   VoxelType voxelType)
{
  LOGGING_INFO_C(VoxelmapLog, VoxelMap, "Inserting meta_point_cloud" << endl);
  m_math.computeLinearLoad(meta_point_cloud.getAccumulatedPointcloudSize(), &m_blocks_sensor_operations,
                           &m_threads_sensor_operations);
  kernelInsertMetaPointCloud<<<m_blocks_sensor_operations, m_threads_sensor_operations>>>(
      m_dev_data, meta_point_cloud.getDeviceConstPointer(), voxelType, m_dev_dim, m_voxel_side_length);
}

template<class Voxel>
size_t TemplateVoxelMap<Voxel>::getMemoryUsage()
{
  return getMemorySizeInByte();
}

template<class Voxel>
void TemplateVoxelMap<Voxel>::writeToDisk(const std::string path)
{
  LOGGING_ERROR_C(VoxelmapLog, TemplateVoxelMap, GPU_VOXELS_MAP_OPERATION_NOT_YET_SUPPORTED << endl);
}

template<class Voxel>
bool TemplateVoxelMap<Voxel>::readFromDisk(const std::string path)
{
  LOGGING_ERROR_C(VoxelmapLog, TemplateVoxelMap, GPU_VOXELS_MAP_OPERATION_NOT_YET_SUPPORTED << endl);
  return false;
}

// ------ END Global API functions ------

//template<class Voxel>
//float TemplateVoxelMap<Voxel>::getVoxelSideLength()
//{
//  return m_voxel_side_length;
//}

#ifdef ALTERNATIVE_CHECK
#undef ALTERNATIVE_CHECK
#endif

// Env map specific functions
template<class Voxel>
void TemplateVoxelMap<Voxel>::initSensorSettings(const Sensor& sensor)
{
  while (!lockMutex())
  {
    boost::this_thread::yield();
  }
  m_sensor = sensor;

  if (m_dev_raw_sensor_data)
  {
    HANDLE_CUDA_ERROR(cudaFree(m_dev_raw_sensor_data));
    HANDLE_CUDA_ERROR(cudaFree(m_dev_sensor));
    LOGGING_INFO_C(VoxelmapLog, VoxelMap, "Reinitialized sensor settings!" << endl);
  }

  HANDLE_CUDA_ERROR(cudaMalloc((void** )&m_dev_raw_sensor_data, m_sensor.data_size * sizeof(Vector3f)));
  HANDLE_CUDA_ERROR(
      cudaMalloc((void** )&m_dev_transformed_sensor_data, m_sensor.data_size * sizeof(Vector3f)));
  HANDLE_CUDA_ERROR(cudaMalloc((void** )&m_dev_sensor, sizeof(Sensor)));
  HANDLE_CUDA_ERROR(cudaMemcpy(m_dev_sensor, &m_sensor, sizeof(Sensor), cudaMemcpyHostToDevice));
  m_init_sensor = true;
  m_math.computeLinearLoad(m_sensor.data_size, &m_blocks_sensor_operations, &m_threads_sensor_operations);
  unlockMutex();
}

template<class Voxel>
void TemplateVoxelMap<Voxel>::updateSensorPose(const Sensor& sensor)
{
  while (!lockMutex())
  {
    boost::this_thread::yield();
  }
  if (m_init_sensor)
  {
    m_sensor.position = sensor.position;
    m_sensor.orientation = sensor.orientation;
    HANDLE_CUDA_ERROR(cudaMemcpy(m_dev_sensor, &m_sensor, sizeof(Sensor), cudaMemcpyHostToDevice));
  }
  else
  {
    LOGGING_ERROR_C(VoxelmapLog, VoxelMap, "Initialize Sensor first!" << endl);
  }
  unlockMutex();
}

template<class Voxel>
void TemplateVoxelMap<Voxel>::copySensorDataToDevice(const Vector3f* points)
{
  if (!m_init_sensor)
  {
    LOGGING_ERROR_C(VoxelmapLog, EnvironmentMap, "Call initSensorSettings() first!" << endl);
    exit(-1);
  }
//  m_elapsed_time = 0;
//  printf("copying SENSOR data... ");
//  HANDLE_CUDA_ERROR(cudaEventRecord(m_start, 0));
  HANDLE_CUDA_ERROR(
      cudaMemcpy(m_dev_raw_sensor_data, points, m_sensor.data_size * sizeof(Vector3f),
                 cudaMemcpyHostToDevice));
//  HANDLE_CUDA_ERROR(cudaEventRecord(m_stop, 0));
//  HANDLE_CUDA_ERROR(cudaEventSynchronize(m_stop));
//  HANDLE_CUDA_ERROR(cudaEventElapsedTime(&m_elapsed_time, m_start, m_stop));
//  printf(" ...done in %f ms!\n", m_elapsed_time);
}

template<class Voxel>
void TemplateVoxelMap<Voxel>::transformSensorData()
{
//  printf("transforming SENSOR data... ");
//  m_elapsed_time = 0;
//  HANDLE_CUDA_ERROR(cudaEventRecord(m_start, 0));
//  HANDLE_CUDA_ERROR(cudaDeviceSynchronize());
  kernelTransformSensorData<<<m_blocks_sensor_operations, m_threads_sensor_operations>>>(
      m_dev_sensor, m_dev_raw_sensor_data, m_dev_transformed_sensor_data);
//  HANDLE_CUDA_ERROR(cudaEventRecord(m_stop, 0));
//  HANDLE_CUDA_ERROR(cudaEventSynchronize(m_stop));
//  HANDLE_CUDA_ERROR(cudaEventElapsedTime(&m_elapsed_time, m_start, m_stop));
//  printf(" ...done in %f ms!\n", m_elapsed_time);
}

//template<class Voxel>
//void TemplateVoxelMap<Voxel>::insertSensorData(const Vector3f* points, const bool enable_raycasting,
//                                       const bool cut_real_robot, Voxel* robot_map)
//{
////  printf("trying to get lock\n");
//  while (!lockMutex())
//  {
////    printf("did not get lock\n");
//    boost::this_thread::yield();
//  }
////  printf("got lock ----------------------------------------------------\n");
////  if (enable_raycasting)
////  {
////    printf("inserting new SENSOR data WITH RAYCASTING ... ");
////  }
////  else
////  {
////    printf("inserting new SENSOR data ... ");
////  }
////  m_elapsed_time = 0;
////  HANDLE_CUDA_ERROR(cudaEventRecord(m_start, 0));
//
//  copySensorDataToDevice(points);
//  transformSensorData();
//  if (enable_raycasting)
//  {
//    // for debugging ray casting:
//    uint32_t blocks, threads;
//    m_math.computeLinearLoad(m_voxelmap_size, &blocks, &threads);
//    HANDLE_CUDA_ERROR(cudaDeviceSynchronize());
//    kernelClearVoxelMap<<<blocks, threads>>>(m_dev_data, m_voxelmap_size, eVT_OCCUPIED);
//    // ---
//    HANDLE_CUDA_ERROR(cudaDeviceSynchronize());
//    kernelInsertSensorDataWithRayCasting<<<m_blocks_sensor_operations, m_threads_sensor_operations>>>(
//        m_dev_data, m_voxelmap_size, m_dev_dim, m_voxel_side_length, m_dev_sensor,
//        m_dev_transformed_sensor_data, cut_real_robot, robot_map);
//  }
//  else
//  {
//    kernelInsertSensorData<<<m_blocks_sensor_operations, m_threads_sensor_operations>>>(
//        m_dev_data, m_voxelmap_size, m_dev_dim, m_voxel_side_length, m_dev_sensor,
//        m_dev_transformed_sensor_data, cut_real_robot, robot_map);
//  }
//  HANDLE_CUDA_ERROR(cudaDeviceSynchronize());
//
////  HANDLE_CUDA_ERROR(cudaEventRecord(m_stop, 0));
////  HANDLE_CUDA_ERROR(cudaEventSynchronize(m_stop));
////  HANDLE_CUDA_ERROR(cudaEventElapsedTime(&m_elapsed_time, m_start, m_stop));
//
////  printf(" ...done in %f ms!\n", m_elapsed_time);
////  printf("update counter: %u\n", m_update_counter);
//
////  printf("releasing lock ----------------------------------------------------\n");
//  unlockMutex();
//}

//template<class Voxel>
//uint32_t TemplateVoxelMap<Voxel>::getUpdateCounter()
//{
//  return m_update_counter;
//}

//template<class Voxel>
//void TemplateVoxelMap<Voxel>::increaseUpdateCounter()
//{
//  m_update_counter++;
//}

// END of Env Map specific functions

// BEGIN Robot Map specific functions
//template<class Voxel>
//void TemplateVoxelMap<Voxel>::insertData(const Vector3f* points, const uint32_t num_points)
//{
//  //printf("RobotMap:: inserting new data... ");
//  m_elapsed_time = 0;
//  HANDLE_CUDA_ERROR(cudaEventRecord(m_start, 0));
//
//  m_math.computeLinearLoad(num_points, &m_blocks_robot_operations, &m_threads_robot_operations);
//
//  HANDLE_CUDA_ERROR(
//      cudaMemcpy(m_dev_point_data, points, num_points * sizeof(Vector3f), cudaMemcpyHostToDevice));
//
//  kernelInsertStaticData<<< m_blocks_robot_operations, m_threads_robot_operations >>>
//  (m_dev_data, m_voxelmap_size, m_dev_dim, m_voxel_side_length, num_points, m_dev_point_data);
//
//  HANDLE_CUDA_ERROR(cudaEventRecord(m_stop, 0));
//  HANDLE_CUDA_ERROR(cudaEventSynchronize(m_stop));
//  HANDLE_CUDA_ERROR(cudaEventElapsedTime(&m_elapsed_time, m_start, m_stop));
//  //printf("RobotMap:: ...done in %f ms!\n", m_elapsed_time);
//}

//template<class Voxel>
//bool TemplateVoxelMap<Voxel>::insertRobotConfiguration(const MetaPointCloud *robot_links, bool with_self_collision_test)
//{
//  if (with_self_collision_test)
//  {
//    if (!m_selfcol_dependency_set)
//    {
//      LOGGING_ERROR_C(VoxelmapLog, RobotMap, "Self-collision check requested but self-collision dependencies not set!" << endl);
//      exit(-1);
//    }
//    //printf("RobotMap:: inserting configuration with self collision test ...\n");
//    m_self_collision = false;
//    syncSelfCollisionInfoToDevice();
//  }
////  else
////  {
////    printf("RobotMap:: inserting configuration ...\n");
////  }
////  m_elapsed_time = 0;
//
////  HANDLE_CUDA_ERROR(cudaEventRecord(m_start, 0));
//  uint32_t blocks = 0;
//  uint32_t threads = 0;
//  HANDLE_CUDA_ERROR(cudaDeviceSynchronize());
//
//  if (with_self_collision_test)
//  {
//    uint32_t selfcol_checks_done = 0;
//    for (uint32_t link_nr = 0; link_nr < robot_links->getNumberOfPointclouds(); link_nr++)
//    {
//      m_math.computeLinearLoad(robot_links->getPointcloudSize(link_nr), &blocks, &threads);
//      if (link_nr == m_links_to_enable_selfcol_check[selfcol_checks_done])
//      {
//        // perform insert with self-collision check
//        kernelInsertRobotKinematicLinkWithSelfCollisionCheck<<< blocks, threads >>>
//        (m_dev_data, m_voxelmap_size, m_dev_dim, m_voxel_side_length,
//            robot_links->getDeviceConstPointer(), link_nr, m_dev_self_collision);
//
//        syncSelfCollisionInfoToHost();
//
//        if (m_self_collision)
//        {
//          // there was a collision
//          return false;
//        }
//        selfcol_checks_done++;
//      }
//      else
//      {
//        // perform insert without self-collision check
//        kernelInsertRobotKinematicLink<<< blocks, threads >>>
//        (m_dev_data, m_voxelmap_size, m_dev_dim, m_voxel_side_length,
//            robot_links->getDeviceConstPointer(), link_nr);
//      }
//    }
//  }
//  else // self collision test disabled
//  {
//    for (uint32_t link_nr=0; link_nr < robot_links->getNumberOfPointclouds(); link_nr++)
//    {
//      m_math.computeLinearLoad(robot_links->getPointcloudSize(link_nr), &blocks, &threads);
//
//      kernelInsertRobotKinematicLink<<< blocks, threads >>>
//      (m_dev_data, m_voxelmap_size, m_dev_dim, m_voxel_side_length,
//          robot_links->getDeviceConstPointer(), link_nr);
//    }
//  }
//  HANDLE_CUDA_ERROR(cudaDeviceSynchronize());
////  HANDLE_CUDA_ERROR(cudaEventRecord(m_stop, 0));
////  HANDLE_CUDA_ERROR(cudaEventSynchronize(m_stop));
////  HANDLE_CUDA_ERROR(cudaEventElapsedTime(&m_elapsed_time, m_start, m_stop));
////  printf("RobotMap:: ...done in %f ms!\n", m_elapsed_time);
//
//  // insert a filled box according to given dimensions into robotmap (for debugging)
////  insertBox(Vector3f(1000, 1000, 800), Vector3f(3200, 3200, 2800),
////            Voxel::eC_EXECUTION, Voxel::eVT_OCCUPIED);
//
//  // there was no self-collision or no self-col check performed
//  return true;
//}

//template<class Voxel>
//bool TemplateVoxelMap<Voxel>::insertRobotConfiguration(const MetaPointCloud *robot_links)
//{
//
////  HANDLE_CUDA_ERROR(cudaEventRecord(m_start, 0));
//  uint32_t blocks = 0;
//  uint32_t threads = 0;
//  HANDLE_CUDA_ERROR(cudaDeviceSynchronize());
//
//  {
//    for (uint32_t link_nr = 0; link_nr < robot_links->getNumberOfPointclouds(); link_nr++)
//    {
//      m_math.computeLinearLoad(robot_links->getPointcloudSize(link_nr), &blocks, &threads);
//
//      kernelInsertRobotKinematicLink<<<blocks, threads>>>(m_dev_data, m_voxelmap_size, m_dev_dim,
//                                                          m_voxel_side_length,
//                                                          robot_links->getDeviceConstPointer(), link_nr);
//    }
//    HANDLE_CUDA_ERROR(cudaDeviceSynchronize());
////  HANDLE_CUDA_ERROR(cudaEventRecord(m_stop, 0));
////  HANDLE_CUDA_ERROR(cudaEventSynchronize(m_stop));
////  HANDLE_CUDA_ERROR(cudaEventElapsedTime(&m_elapsed_time, m_start, m_stop));
////  printf("RobotMap:: ...done in %f ms!\n", m_elapsed_time);
//
//    // insert a filled box according to given dimensions into robotmap (for debugging)
////  insertBox(Vector3f(1000, 1000, 800), Vector3f(3200, 3200, 2800),
////            Voxel::eC_EXECUTION, Voxel::eVT_OCCUPIED);
//
//    // there was no self-collision or no self-col check performed
//    return true;
//  }
//}

//template<class Voxel>
//void TemplateVoxelMap<Voxel>::insertConfigurationOverwritingSensorData(const MetaPointCloud *robot_links,
//                                                               VoxelMap* env_map)
//{
//  bool locked_this = false;
//  bool locked_other = false;
//  uint32_t counter = 0;
//
//  while (!locked_this && !locked_other)
//  {
//    // lock mutexes
//    while (!locked_this)
//    {
//      locked_this = lockMutex();
//      boost::this_thread::yield();
//    }
//    while (!locked_other && (counter < 50))
//    {
//      locked_other = env_map->remoteLock();
//      boost::this_thread::yield();
//      counter++;
//    }
//
//    if (!locked_other)
//    {
//      counter = 0;
//      unlockMutex();
//    }
//  }
//  uint32_t blocks = 0;
//  uint32_t threads = 0;
//  HANDLE_CUDA_ERROR(cudaDeviceSynchronize());
//
//  for (uint32_t link_nr = 0; link_nr < robot_links->getNumberOfPointclouds(); link_nr++)
//  {
//    m_math.computeLinearLoad(robot_links->getPointcloudSize(link_nr), &blocks, &threads);
//
//    kernelInsertRobotKinematicLinkOverwritingSensorData<<<blocks, threads>>>(
//        m_dev_data, m_voxelmap_size, m_dev_dim, m_voxel_side_length, robot_links->getDeviceConstPointer(),
//        link_nr, env_map->getDeviceDataPtr());
//  }
//  HANDLE_CUDA_ERROR(cudaDeviceSynchronize());
//
//  //  printf("unlocking environment map's mutex\n");
//  env_map->remoteUnlock();
//  //  printf("unlocking this map's mutex\n");
//  unlockMutex();
//  //  printf("done unlocking\n");
//}

//template<class Voxel>
//void TemplateVoxelMap<Voxel>::insertSweptVolumeConfiguration(uint32_t kinematic_chain_size,
//                                                     uint32_t* point_cloud_sizes,
//                                                     uint32_t* dev_point_cloud_sizes,
//                                                     Vector3f** dev_point_clouds, uint8_t swept_volume_index)
//{
////  printf("RobotMap:: inserting swept volume configuration ...\n");
////  m_elapsed_time = 0;
//
////  HANDLE_CUDA_ERROR(cudaEventRecord(m_start, 0));
//  uint32_t blocks = 0;
//  uint32_t threads = 0;
//
//  for (uint32_t link_nr = 0; link_nr < kinematic_chain_size; link_nr++)
//  {
//    HANDLE_CUDA_ERROR(cudaDeviceSynchronize());
//    m_math.computeLinearLoad(point_cloud_sizes[link_nr], &blocks, &threads);
//    kernelInsertSweptVolumeConfiguration<<<blocks, threads>>>(m_dev_data, m_voxelmap_size, m_dev_dim,
//                                                              m_voxel_side_length, link_nr,
//                                                              dev_point_cloud_sizes, dev_point_clouds,
//                                                              swept_volume_index);
//  }
////  HANDLE_CUDA_ERROR(cudaEventRecord(m_stop, 0));
////  HANDLE_CUDA_ERROR(cudaEventSynchronize(m_stop));
////  HANDLE_CUDA_ERROR(cudaEventElapsedTime(&m_elapsed_time, m_start, m_stop));
////  printf("RobotMap:: ...done in %f ms!\n", m_elapsed_time);
//
//  // insert a filled box according to given dimensions into robotmap (for debugging)
////  insertBox(Vector3f(1000, 1000, 800), Vector3f(3200, 3200, 2800),
////            Voxel::eC_EXECUTION, (Voxel::VoxelType)swept_volume_index);
//}
//
//template<class Voxel>
//void TemplateVoxelMap<Voxel>::removeSweptVolumeConfiguration(uint32_t kinematic_chain_size,
//                                                     uint32_t* point_cloud_sizes,
//                                                     uint32_t* dev_point_cloud_sizes,
//                                                     Vector3f** dev_point_clouds, uint8_t swept_volume_index)
//{
////  printf("RobotMap:: removing swept volume configuration ...\n");
////  m_elapsed_time = 0;
////
////  HANDLE_CUDA_ERROR(cudaEventRecord(m_start, 0));
//  uint32_t blocks = 0;
//  uint32_t threads = 0;
//
//  for (uint32_t link_nr = 0; link_nr < kinematic_chain_size; link_nr++)
//  {
//    HANDLE_CUDA_ERROR(cudaDeviceSynchronize());
//    m_math.computeLinearLoad(point_cloud_sizes[link_nr], &blocks, &threads);
//    kernelRemoveSweptVolumeConfiguration<<<blocks, threads>>>(m_dev_data, m_voxelmap_size, m_dev_dim,
//                                                              m_voxel_side_length, link_nr,
//                                                              dev_point_cloud_sizes, dev_point_clouds,
//                                                              swept_volume_index);
//  }
////  HANDLE_CUDA_ERROR(cudaEventRecord(m_stop, 0));
////  HANDLE_CUDA_ERROR(cudaEventSynchronize(m_stop));
////  HANDLE_CUDA_ERROR(cudaEventElapsedTime(&m_elapsed_time, m_start, m_stop));
////  printf("RobotMap:: ...done in %f ms!\n", m_elapsed_time);
//
//}
//
//template<class Voxel>
//void TemplateVoxelMap<Voxel>::setSelfCollisionDependencies(std::vector<uint32_t>& links_to_enable_check)
//{
//  m_links_to_enable_selfcol_check = links_to_enable_check;
//  m_selfcol_dependency_set = true;
//}
//
//template<class Voxel>
//void TemplateVoxelMap<Voxel>::syncSelfCollisionInfoToHost()
//{
//  HANDLE_CUDA_ERROR(
//      cudaMemcpy(&m_self_collision, m_dev_self_collision, sizeof(bool), cudaMemcpyDeviceToHost));
//
//}
//
//template<class Voxel>
//void TemplateVoxelMap<Voxel>::syncSelfCollisionInfoToDevice()
//{
//  HANDLE_CUDA_ERROR(
//      cudaMemcpy(m_dev_self_collision, &m_self_collision, sizeof(bool), cudaMemcpyHostToDevice));
//}

//template<class Voxel>
//void TemplateVoxelMap<Voxel>::clearBitvector(uint8_t bit_number)
//{
//  uint32_t blocks = 0;
//  uint32_t threads = 0;
//  m_math.computeLinearLoad(m_voxelmap_size, &blocks, &threads); //(*m_dev_voxelmap_size, &blocks, &threads);
//
//  kernelClearBitvector<<<blocks, threads>>>(m_dev_data, m_voxelmap_size, m_dev_dim, bit_number);
//
//}

//template<class Voxel>
//void TemplateVoxelMap<Voxel>::insertConfigurationIntoBitVector(uint32_t kinematic_chain_size,
//                                                       uint32_t* point_cloud_sizes,
//                                                       uint32_t* dev_point_cloud_sizes,
//                                                       Vector3f** dev_point_clouds,
//                                                       bool with_self_collision_test,
//                                                       const uint8_t bit_number)
//{
//  uint32_t blocks = 0;
//  uint32_t threads = 0;
//  uint64_t bitmap = 0x1;
//  bitmap = bitmap << bit_number;
//  for (uint32_t link_nr = 0; link_nr < kinematic_chain_size; link_nr++)
//  {
//    HANDLE_CUDA_ERROR(cudaDeviceSynchronize());
//    m_math.computeLinearLoad(point_cloud_sizes[link_nr], &blocks, &threads);
//
//    kernelInsertKinematicLinkBitvector<<<blocks, threads>>>(m_dev_data, m_voxelmap_size, m_dev_dim,
//                                                            m_voxel_side_length, link_nr,
//                                                            dev_point_cloud_sizes, dev_point_clouds, bitmap);
//  }
//
////	  }
//  //  HANDLE_CUDA_ERROR(cudaEventRecord(m_stop, 0));
//  //  HANDLE_CUDA_ERROR(cudaEventSynchronize(m_stop));
//  //  HANDLE_CUDA_ERROR(cudaEventElapsedTime(&m_elapsed_time, m_start, m_stop));
//  //  printf("RobotMap:: ...done in %f ms!\n", m_elapsed_time);
//
//  // insert a filled box according to given dimensions into robotmap (for debugging)
//  //  insertBox(Vector3f(1000, 1000, 800), Vector3f(3200, 3200, 2800),
//  //            Voxel::eC_EXECUTION, Voxel::eVT_OCCUPIED);
//
//}
// END of Robot map specific functions

}// end of namespace voxelmap
} // end of namespace gpu_voxels

#endif
