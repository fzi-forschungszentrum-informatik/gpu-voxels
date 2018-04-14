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
#include <gpu_voxels/voxel/DefaultCollider.hpp>
#include <gpu_voxels/voxel/SVCollider.hpp>

#include <thrust/fill.h>
#include <thrust/device_ptr.h>

// temp:
#include <time.h>

namespace gpu_voxels {
namespace voxelmap {

//#define ALTERNATIVE_CHECK
#undef  ALTERNATIVE_CHECK

#ifdef  ALTERNATIVE_CHECK
#define LOOP_SIZE       4
#endif



//const uint32_t cMAX_POINTS_PER_ROBOT_SEGMENT = 118000;

template<class Voxel>
TemplateVoxelMap<Voxel>::TemplateVoxelMap(const Vector3ui dim,
                                          const float voxel_side_length, const MapType map_type) :
                                          m_dim(dim),
                                          m_limits(dim.x * voxel_side_length, dim.y * voxel_side_length, dim.z * voxel_side_length),
                                          m_voxel_side_length(voxel_side_length), m_voxelmap_size(getVoxelMapSize()), m_dev_data(NULL),
                                          m_dev_points_outside_map(NULL),
                                          m_collision_check_results(NULL)
{
  this->m_map_type = map_type;
  if (dim.x * dim.y * dim.z * sizeof(Voxel) > (pow(2, 32) - 1))
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
  HANDLE_CUDA_ERROR(cudaMalloc((void** )&m_dev_data, getMemoryUsage()));
  LOGGING_DEBUG_C(VoxelmapLog, VoxelMap, "Voxelmap base address is " << (void*) m_dev_data << endl);

  HANDLE_CUDA_ERROR(cudaMalloc((void** )&m_dev_points_outside_map, sizeof(bool)));


  computeLinearLoad(m_voxelmap_size, &m_blocks, &m_threads);
#ifdef ALTERNATIVE_CHECK
  computeLinearLoad((uint32_t) ceil((float) m_voxelmap_size / (float) LOOP_SIZE), &m_alternative_blocks, &m_alternative_threads);
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
    m_result_array_size = ceil((float)m_voxelmap_size/((float)cMAX_THREADS_PER_BLOCK*(float)LOOP_SIZE));
  }

#endif

}
template<class Voxel>
TemplateVoxelMap<Voxel>::TemplateVoxelMap(Voxel* dev_data, const Vector3ui dim, const float voxel_side_length, const MapType map_type) :
  m_dim(dim), m_limits(dim.x * voxel_side_length, dim.y * voxel_side_length,
                                                 dim.z * voxel_side_length), m_voxel_side_length(
        voxel_side_length), m_voxelmap_size(getVoxelMapSize()), m_dev_data(dev_data), m_collision_check_results(NULL)
{
  this->m_map_type = map_type;

  computeLinearLoad(m_voxelmap_size, &m_blocks, &m_threads);
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
  if (m_dev_data)
  {
    HANDLE_CUDA_ERROR(cudaFree(m_dev_data));
  }
  if (m_dev_points_outside_map)
  {
    HANDLE_CUDA_ERROR(cudaFree(m_dev_points_outside_map));
  }

  HANDLE_CUDA_ERROR(cudaEventDestroy(m_start));
  HANDLE_CUDA_ERROR(cudaEventDestroy(m_stop));

}

/* ======== VoxelMap operations  ======== */

/*!
 * Specialized clearing function for Bitmap Voxelmaps.
 * As the bitmap for every voxel is empty,
 * it is sufficient to set the whole map to zeroes.
 * WATCH OUT: This sets the map to eBVM_FREE and not to eBVM_UNKNOWN!
 */
template<>
void TemplateVoxelMap<BitVectorVoxel>::clearMap()
{
  lock_guard guard(this->m_mutex);
  // Clear occupancies
  HANDLE_CUDA_ERROR(cudaDeviceSynchronize());
  HANDLE_CUDA_ERROR(
    cudaMemset(m_dev_data, 0, m_voxelmap_size*sizeof(gpu_voxels::BitVectorVoxel)));

  // Clear result array
  for (uint32_t i = 0; i < cMAX_NR_OF_BLOCKS; i++)
  {
    m_collision_check_results[i] = false;
  }
  HANDLE_CUDA_ERROR(
      cudaMemcpy(m_dev_collision_check_results, m_collision_check_results, cMAX_NR_OF_BLOCKS * sizeof(bool),
                 cudaMemcpyHostToDevice));
}

/*!
 * Specialized clearing function for probabilistic Voxelmaps.
 * As a ProbabilisticVoxel consists of only one byte, we can
 * memset the whole map to UNKNOWN_PROBABILITY
 */
template<>
void TemplateVoxelMap<ProbabilisticVoxel>::clearMap()
{
  lock_guard guard(this->m_mutex);
  // Clear occupancies
  HANDLE_CUDA_ERROR(cudaDeviceSynchronize());
  HANDLE_CUDA_ERROR(
    cudaMemset(m_dev_data, UNKNOWN_PROBABILITY, m_voxelmap_size*sizeof(gpu_voxels::ProbabilisticVoxel)));

  // Clear result array
  for (uint32_t i = 0; i < cMAX_NR_OF_BLOCKS; i++)
  {
    m_collision_check_results[i] = false;
  }
  HANDLE_CUDA_ERROR(
      cudaMemcpy(m_dev_collision_check_results, m_collision_check_results, cMAX_NR_OF_BLOCKS * sizeof(bool),
                 cudaMemcpyHostToDevice));
}

/*!
 * Specialized clearing function for DistanceVoxelmaps.
 * it is sufficient to set the whole map to PBA_UNINITIALISED_COORD.
 */
template<>
void TemplateVoxelMap<DistanceVoxel>::clearMap()
{
  lock_guard guard(this->m_mutex);
  HANDLE_CUDA_ERROR(cudaDeviceSynchronize());

  //  //deprecated: initialising voxels to all zero
  //  HANDLE_CUDA_ERROR(cudaMemset(m_dev_data, 0, m_voxelmap_size*sizeof(DistanceVoxel)));

  // Clear contents: distance of PBA_UNINITIALISED indicates uninitialized voxel
  DistanceVoxel pba_uninitialised_voxel;
  pba_uninitialised_voxel.setPBAUninitialised();
  thrust::device_ptr<DistanceVoxel> first(m_dev_data);

  thrust::fill(first, first+m_voxelmap_size, pba_uninitialised_voxel);

//  //TODO: adapt for distanceVoxel? eliminate?
//  // Clear result array
//  for (uint32_t i = 0; i < cMAX_NR_OF_BLOCKS; i++)
//  {
//    m_collision_check_results[i] = false;
//  }
//  HANDLE_CUDA_ERROR(
//      cudaMemcpy(m_dev_collision_check_results, m_collision_check_results, cMAX_NR_OF_BLOCKS * sizeof(bool),
//                 cudaMemcpyHostToDevice));

  HANDLE_CUDA_ERROR(cudaDeviceSynchronize());
}

template<class Voxel>
void TemplateVoxelMap<Voxel>::printVoxelMapData()
{
  lock_guard guard(this->m_mutex);
  HANDLE_CUDA_ERROR(cuPrintDeviceArray(m_dev_data, m_voxelmap_size, "VoxelMap dump: "));
}

//template<class Voxel>
//bool TemplateVoxelMap<Voxel>::collisionCheckAlternative(const uint8_t threshold, VoxelMap* other,
//                                         const uint8_t other_threshold, uint32_t loop_size)
//{
// Todo: DO LOCKING HERE!!
//  computeLinearLoad((uint32_t) ceil((float) m_voxelmap_size / (float) loop_size),
//                           &m_alternative_blocks, &m_alternative_threads);
////  printf("number of blocks: %i , number of threads: %i", m_alternative_blocks, m_alternative_threads);
//  HANDLE_CUDA_ERROR(cudaDeviceSynchronize());
//  m_elapsed_time = 0;
////  HANDLE_CUDA_ERROR(cudaEventRecord(m_start, 0));
//  kernelCollideVoxelMapsAlternative<<< m_alternative_blocks, m_alternative_threads >>>
//  (m_dev_data, m_voxelmap_size, threshold, other->getDeviceDataPtr(), other_threshold, loop_size, m_dev_collision_check_results);
//  CHECK_CUDA_ERROR();
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
  boost::lock(this->m_mutex, other->m_mutex);
  lock_guard guard(this->m_mutex, boost::adopt_lock);
  lock_guard guard2(other->m_mutex, boost::adopt_lock);
  //printf("collision check... ");

#ifndef ALTERNATIVE_CHECK
  HANDLE_CUDA_ERROR(cudaDeviceSynchronize());
//  m_elapsed_time = 0;
//  HANDLE_CUDA_ERROR(cudaEventRecord(m_start, 0));
//  printf("TemplateVoxelMap<Voxel>::collisionCheck\n");

  kernelCollideVoxelMaps<<<m_blocks, m_threads>>>(m_dev_data, m_voxelmap_size, other->getDeviceDataPtr(),
                                                  collider, m_dev_collision_check_results);
  CHECK_CUDA_ERROR();

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

  return false;

#else

  HANDLE_CUDA_ERROR(cudaDeviceSynchronize());
  m_elapsed_time = 0;
//  HANDLE_CUDA_ERROR(cudaEventRecord(m_start, 0));
  kernelCollideVoxelMapsAlternative<<< m_alternative_blocks, m_alternative_threads >>>
  (m_dev_data, m_voxelmap_size, threshold, other->getDeviceDataPtr(), other_threshold, LOOP_SIZE, m_dev_collision_check_results);
  CHECK_CUDA_ERROR();

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
//      number_of_thread_runs, m_dim);
//  CHECK_CUDA_ERROR();
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

template<class Voxel>
template<class OtherVoxel, class Collider>
uint32_t TemplateVoxelMap<Voxel>::collisionCheckWithCounter(TemplateVoxelMap<OtherVoxel>* other,
                                                            Collider collider)
{
  return collisionCheckWithCounterRelativeTransform(other, collider); //does the locking
}


template<class Voxel>
template<class OtherVoxel, class Collider>
uint32_t TemplateVoxelMap<Voxel>::collisionCheckWithCounterRelativeTransform(TemplateVoxelMap<OtherVoxel>* other,
                                                            Collider collider, const Vector3i &offset)
{
  boost::lock(this->m_mutex, other->m_mutex);
  lock_guard guard(this->m_mutex, boost::adopt_lock);
  lock_guard guard2(other->m_mutex, boost::adopt_lock);

  Voxel* dev_data_with_offset = NULL;
  if(offset != Vector3i())
  {
    // We take the base adress of this voxelmap and add the offset that we want to shift the other map.
    dev_data_with_offset = getVoxelPtrSignedOffset(m_dev_data, m_dim, offset);
  }else{
    dev_data_with_offset = m_dev_data;
  }
  HANDLE_CUDA_ERROR(cudaDeviceSynchronize());
  kernelCollideVoxelMapsDebug<<<m_blocks, m_threads>>>(dev_data_with_offset, m_voxelmap_size, other->getDeviceDataPtr(),
                                                       collider, m_dev_collision_check_results_counter);
  CHECK_CUDA_ERROR();
  HANDLE_CUDA_ERROR(cudaDeviceSynchronize());
  HANDLE_CUDA_ERROR(
      cudaMemcpy(m_collision_check_results_counter, m_dev_collision_check_results_counter,
                 cMAX_NR_OF_BLOCKS * sizeof(uint16_t), cudaMemcpyDeviceToHost));

  uint32_t number_of_collisions = 0;
  for (uint32_t i = 0; i < m_blocks; i++)
  {
    number_of_collisions += m_collision_check_results_counter[i];
  }

  return number_of_collisions;
}


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
//  destination->computeLinearLoad(destination->m_voxelmap_size, &blocks, &threads);
//  if (with_bitvector)
//  {
//    kernelShrinkCopyVoxelMapBitvector<<<blocks, threads>>>(destination->m_dev_data,
//                                                           destination->m_voxelmap_size,
//                                                           destination->m_dim, source->m_dev_data,
//                                                           source->m_voxelmap_size, source->m_dim,
//                                                           factor);
//    CHECK_CUDA_ERROR();
//  }
//  else
//  {
//    kernelShrinkCopyVoxelMap<<<blocks, threads>>>(destination->m_dev_data, destination->m_voxelmap_size,
//                                                  destination->m_dim, source->m_dev_data,
//                                                  source->m_voxelmap_size, source->m_dim, factor);
//    CHECK_CUDA_ERROR();
//
//  }
//
//  HANDLE_CUDA_ERROR(cudaDeviceSynchronize());
////		void copyVoxelMapBitvector(Voxel* destination_map, const uint32_t destination_map_size, Vector3ui* dest_map_dim,
////				Voxel* source_map, const uint32_t source_map_size, Vector3ui* source_map_dim, uint8_t factor)
//}

// ------ BEGIN Global API functions ------

/**
 * author: Matthias Wagner
 * Inserts a voxel at each point from the points list.
 */
template<class Voxel>
void TemplateVoxelMap<Voxel>::insertPointCloud(const std::vector<Vector3f> &points, const BitVoxelMeaning voxel_meaning)
{
// copy points to the gpu
  lock_guard guard(this->m_mutex);
  Vector3f* d_points;
  HANDLE_CUDA_ERROR(cudaMalloc(&d_points, points.size() * sizeof(Vector3f)));
  HANDLE_CUDA_ERROR(
      cudaMemcpy(d_points, &points[0], points.size() * sizeof(Vector3f), cudaMemcpyHostToDevice));

  insertPointCloud(d_points, points.size(), voxel_meaning);

  HANDLE_CUDA_ERROR(cudaFree(d_points));
}

template<class Voxel>
void TemplateVoxelMap<Voxel>::insertPointCloud(const PointCloud &pointcloud, const BitVoxelMeaning voxel_meaning)
{
  lock_guard guard(this->m_mutex);

  insertPointCloud(pointcloud.getConstDevicePointer(), pointcloud.getPointCloudSize(), voxel_meaning);

}

template<class Voxel>
void TemplateVoxelMap<Voxel>::insertPointCloud(const Vector3f* points_d, uint32_t size, const BitVoxelMeaning voxel_meaning)
{
  // reset warning indicator:
  HANDLE_CUDA_ERROR(cudaMemset((void*)m_dev_points_outside_map, 0, sizeof(bool)));
  bool points_outside_map;

  uint32_t num_blocks, threads_per_block;
  computeLinearLoad(size, &num_blocks, &threads_per_block);
  HANDLE_CUDA_ERROR(cudaDeviceSynchronize());
  kernelInsertGlobalPointCloud<<<num_blocks, threads_per_block>>>(m_dev_data, m_dim, m_voxel_side_length,
                                                                  points_d, size, voxel_meaning, m_dev_points_outside_map);
  CHECK_CUDA_ERROR();

  HANDLE_CUDA_ERROR(cudaMemcpy(&points_outside_map, m_dev_points_outside_map, sizeof(bool), cudaMemcpyDeviceToHost));
  HANDLE_CUDA_ERROR(cudaDeviceSynchronize());
  if(points_outside_map)
  {
    LOGGING_WARNING_C(VoxelmapLog, VoxelMap, "You tried to insert points that lie outside the map dimensions!" << endl);
  }
}

template<class Voxel>
void TemplateVoxelMap<Voxel>::insertMetaPointCloud(const MetaPointCloud &meta_point_cloud,
                                                   BitVoxelMeaning voxel_meaning)
{
  lock_guard guard(this->m_mutex);

  // reset warning indicator:
  HANDLE_CUDA_ERROR(cudaMemset((void*)m_dev_points_outside_map, 0, sizeof(bool)));
  bool points_outside_map;

  computeLinearLoad(meta_point_cloud.getAccumulatedPointcloudSize(), &m_blocks,
                           &m_threads);
  kernelInsertMetaPointCloud<<<m_blocks, m_threads>>>(
      m_dev_data, meta_point_cloud.getDeviceConstPointer(), voxel_meaning, m_dim, m_voxel_side_length,
      m_dev_points_outside_map);
  CHECK_CUDA_ERROR();

  HANDLE_CUDA_ERROR(cudaMemcpy(&points_outside_map, m_dev_points_outside_map, sizeof(bool), cudaMemcpyDeviceToHost));
  if(points_outside_map)
  {
    LOGGING_WARNING_C(VoxelmapLog, VoxelMap, "You tried to insert points that lie outside the map dimensions!" << endl);
  }
}

template<class Voxel>
void TemplateVoxelMap<Voxel>::insertMetaPointCloud(const MetaPointCloud& meta_point_cloud,
                                                   const std::vector<BitVoxelMeaning>& voxel_meanings)
{
  lock_guard guard(this->m_mutex);
  assert(meta_point_cloud.getNumberOfPointclouds() == voxel_meanings.size());

  // reset warning indicator:
  HANDLE_CUDA_ERROR(cudaMemset((void*)m_dev_points_outside_map, 0, sizeof(bool)));
  bool points_outside_map;

  computeLinearLoad(meta_point_cloud.getAccumulatedPointcloudSize(), &m_blocks,
                           &m_threads);

  BitVoxelMeaning* voxel_meanings_d;
  size_t size = voxel_meanings.size() * sizeof(BitVoxelMeaning);
  HANDLE_CUDA_ERROR(cudaMalloc((void**) &voxel_meanings_d, size));
  HANDLE_CUDA_ERROR(cudaMemcpy(voxel_meanings_d, &voxel_meanings[0], size, cudaMemcpyHostToDevice));

  kernelInsertMetaPointCloud<<<m_blocks, m_threads>>>(
      m_dev_data, meta_point_cloud.getDeviceConstPointer(), voxel_meanings_d, m_dim, m_voxel_side_length,
      m_dev_points_outside_map);
  CHECK_CUDA_ERROR();

  HANDLE_CUDA_ERROR(cudaDeviceSynchronize());
  HANDLE_CUDA_ERROR(cudaMemcpy(&points_outside_map, m_dev_points_outside_map, sizeof(bool), cudaMemcpyDeviceToHost));
  if(points_outside_map)
  {
    LOGGING_WARNING_C(VoxelmapLog, VoxelMap, "You tried to insert points that lie outside the map dimensions!" << endl);
  }
  HANDLE_CUDA_ERROR(cudaFree(voxel_meanings_d));
}

template<class Voxel>
bool TemplateVoxelMap<Voxel>::writeToDisk(const std::string path)
{
  lock_guard guard(this->m_mutex);
  std::ofstream out(path.c_str());
  if(!out.is_open())
  {
    LOGGING_ERROR_C(VoxelmapLog, VoxelMap, "Write to file " << path << " failed!" << endl);
    return false;
  }

  MapType map_type = this->getTemplateType();
  uint32_t buffer_size = this->getMemoryUsage();
  char* buffer = new char[buffer_size];

  LOGGING_INFO_C(VoxelmapLog, VoxelMap, "Dumping Voxelmap to disk: " <<
                 getVoxelMapSize() << " Voxels ==> " << (buffer_size * cBYTE2MBYTE) << " MB. ..." << endl);

  HANDLE_CUDA_ERROR(cudaMemcpy((void*)buffer, this->getConstVoidDeviceDataPtr(), buffer_size, cudaMemcpyDeviceToHost));

  bool bin_mode = true;
  // Write meta data and actual data
  if (bin_mode)
  {
    out.write((char*) &map_type, sizeof(MapType));
    out.write((char*) &m_voxel_side_length, sizeof(float));
    out.write((char*) &m_dim.x, sizeof(uint32_t));
    out.write((char*) &m_dim.y, sizeof(uint32_t));
    out.write((char*) &m_dim.z, sizeof(uint32_t));
    out.write((char*) &buffer[0], buffer_size);
  }
  else
  {
    out << map_type << "\n";
    out << m_voxel_side_length << "\n";
    out << m_dim.x << "\n";
    out << m_dim.y << "\n";
    out << m_dim.z << "\n";
    for (uint32_t i = 0; i < buffer_size; ++i)
      out << buffer[i] << "\n";
  }

  out.close();
  delete buffer;

  LOGGING_INFO_C(VoxelmapLog, VoxelMap, "... writing to disk is done." << endl);
  return true;
}


template<class Voxel>
bool TemplateVoxelMap<Voxel>::readFromDisk(const std::string path)
{
  lock_guard guard(this->m_mutex);
  MapType map_type;
  float voxel_side_length;
  uint32_t dim_x, dim_y, dim_z;


  std::ifstream in(path.c_str());
  if(!in.is_open())
  {
    LOGGING_ERROR_C(VoxelmapLog, VoxelMap, "Error in reading file " << path << endl);
    return false;
  }

  // Read meta data
  bool bin_mode = true;
  if(bin_mode)
  {
    in.read((char*)&map_type, sizeof(MapType));
    in.read((char*)&voxel_side_length, sizeof(float));
    in.read((char*)&dim_x, sizeof(uint32_t));
    in.read((char*)&dim_y, sizeof(uint32_t));
    in.read((char*)&dim_z, sizeof(uint32_t));
  }
  else
  {
    int tmp;
    in >> tmp;
    map_type = (MapType)tmp;
    in >> voxel_side_length;
    in >> dim_x;
    in >> dim_y;
    in >> dim_z;
  }

  Vector3ui dim(dim_x, dim_y, dim_z);

  // Check meta data
  if(map_type != this->getTemplateType())
  {
    LOGGING_ERROR_C(VoxelmapLog, VoxelMap, "Voxelmap type does not match!" << endl);
    return false;
  }
  if(voxel_side_length != this->m_voxel_side_length)
  {
    LOGGING_WARNING_C(VoxelmapLog, VoxelMap, "Read from file failed: Read Voxel side length (" << voxel_side_length <<
                      ") does not match current object (" << m_voxel_side_length << ")! Continuing though..." << endl);
  }else{
    if(map_type == MT_BITVECTOR_VOXELMAP)
    {
      //TODO: Check for size of bitvector, as that may change!
    }
  }
  if(dim != m_dim)
  {
    // after that check we may use the class size variables.
    LOGGING_ERROR_C(VoxelmapLog, VoxelMap, "Read from file failed: Read reference map dimension (" << dim << ") does not match current object (" << m_dim << ")!" << endl);

    return false;
  }

  // Read actual data
  LOGGING_INFO_C(VoxelmapLog, VoxelMap, "Reading Voxelmap from disk: " <<
                 getVoxelMapSize() << " Voxels ==> " << (getMemoryUsage() * cBYTE2MBYTE) << " MB. ..." << endl);
  char* buffer = new char[getMemoryUsage()];

  if(bin_mode)
  {
    in.read(buffer, getMemoryUsage());
  }else{
    for(uint32_t i = 0; i < getMemoryUsage(); ++i)
      in >> buffer[i];
  }

  // Copy data to device
  HANDLE_CUDA_ERROR(cudaMemcpy(this->getVoidDeviceDataPtr(), (void*)buffer, getMemoryUsage(), cudaMemcpyHostToDevice));

  in.close();
  delete buffer;
  LOGGING_INFO_C(VoxelmapLog, VoxelMap, "... reading from disk is done." << endl);
  return true;
}

template<class Voxel>
bool TemplateVoxelMap<Voxel>::merge(const GpuVoxelsMapSharedPtr other, const Vector3f &metric_offset, const BitVoxelMeaning* new_meaning)
{
  LOGGING_ERROR_C(VoxelmapLog, TemplateVoxelMap, GPU_VOXELS_MAP_OPERATION_NOT_YET_SUPPORTED << endl);
  return false;
}

template<class Voxel>
bool TemplateVoxelMap<Voxel>::merge(const GpuVoxelsMapSharedPtr other, const Vector3i &voxel_offset, const BitVoxelMeaning* new_meaning)
{
  LOGGING_ERROR_C(VoxelmapLog, TemplateVoxelMap, GPU_VOXELS_MAP_OPERATION_NOT_YET_SUPPORTED << endl);
  return false;
}

template<class Voxel>
Vector3ui TemplateVoxelMap<Voxel>::getDimensions() const
{
  return m_dim;
}

template<class Voxel>
Vector3f TemplateVoxelMap<Voxel>::getMetricDimensions() const
{
  return Vector3f(m_dim.x, m_dim.y, m_dim.z) * getVoxelSideLength();
}

// ------ END Global API functions ------


#ifdef ALTERNATIVE_CHECK
#undef ALTERNATIVE_CHECK
#endif



}// end of namespace voxelmap
} // end of namespace gpu_voxels

#endif
