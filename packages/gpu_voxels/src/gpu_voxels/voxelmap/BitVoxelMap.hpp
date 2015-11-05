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
#ifndef GPU_VOXELS_VOXELMAP_BIT_VOXELMAP_HPP_INCLUDED
#define GPU_VOXELS_VOXELMAP_BIT_VOXELMAP_HPP_INCLUDED

#include "BitVoxelMap.h"
#include <gpu_voxels/voxelmap/TemplateVoxelMap.hpp>
#include <gpu_voxels/voxel/BitVoxel.hpp>
#include <gpu_voxels/voxelmap/ProbVoxelMap.hpp>

#include <thrust/device_vector.h>
#include <thrust/device_ptr.h>

namespace gpu_voxels {
namespace voxelmap {

template<std::size_t length>
BitVoxelMap<length>::BitVoxelMap(const uint32_t dim_x, const uint32_t dim_y, const uint32_t dim_z,
                                 const float voxel_side_length, const MapType map_type) :
    Base(dim_x, dim_y, dim_z, voxel_side_length, map_type)
{

}

template<std::size_t length>
BitVoxelMap<length>::BitVoxelMap(Voxel* dev_data, const Vector3ui dim, const float voxel_side_length, const MapType map_type) :
    Base(dev_data, dim, voxel_side_length, map_type)
{

}

template<std::size_t length>
BitVoxelMap<length>::~BitVoxelMap()
{

}

template<std::size_t length>
void BitVoxelMap<length>::clearBit(const uint32_t bit_index)
{
  while (!this->lockMutex())
  {
    boost::this_thread::yield();
  }
  clearVoxelMapRemoteLock(bit_index);
  this->unlockMutex();
}

template<std::size_t length>
void BitVoxelMap<length>::clearVoxelMapRemoteLock(const uint32_t bit_index)
{
  kernelClearVoxelMap<<<this->m_blocks, this->m_threads>>>(this->m_dev_data, this->m_voxelmap_size,
                                                           bit_index);
  HANDLE_CUDA_ERROR(cudaDeviceSynchronize());
}

template<std::size_t length>
void BitVoxelMap<length>::clearBits(BitVector<length> bits)
{
  while (!this->lockMutex())
  {
    boost::this_thread::yield();
  }

  kernelClearVoxelMap<<<this->m_blocks, this->m_threads>>>(this->m_dev_data, this->m_voxelmap_size, bits);
  HANDLE_CUDA_ERROR(cudaDeviceSynchronize());

  this->unlockMutex();
}

template<std::size_t length>
template<class Collider>
uint32_t BitVoxelMap<length>::collisionCheckBitvector(BitVoxelMap<length>* other, Collider collider,
                                                      BitVector<length>& colliding_meanings,
                                                      const uint16_t sv_offset)
{
  uint32_t threads_per_block = 1024;
  //calculate number of blocks
  uint32_t number_of_blocks;
  number_of_blocks = (this->m_voxelmap_size + threads_per_block - 1) / threads_per_block;

  BitVector<length>* result_ptr_dev;
  HANDLE_CUDA_ERROR(cudaMalloc((void** )&result_ptr_dev, sizeof(BitVector<length> ) * number_of_blocks));

  uint16_t* num_collisions_dev;
  HANDLE_CUDA_ERROR(
      cudaMalloc((void** )&num_collisions_dev, number_of_blocks * sizeof(uint16_t)));

  kernelCollideVoxelMapsBitvector<<<number_of_blocks, threads_per_block,
                                    sizeof(BitVector<length> ) * threads_per_block>>>(
      this->m_dev_data, this->m_voxelmap_size, other->getDeviceDataPtr(), collider, result_ptr_dev, num_collisions_dev, sv_offset);

  //copying result from device
  uint16_t num_collisions_h[number_of_blocks];
  BitVector<length> result_array[number_of_blocks];
  for (uint32_t i = 0; i < number_of_blocks; ++i)
  {
    result_array[i] = BitVector<length>();
  }
  HANDLE_CUDA_ERROR(cudaDeviceSynchronize());

  HANDLE_CUDA_ERROR(
      cudaMemcpy(&(result_array[0]), result_ptr_dev, sizeof(BitVector<length> ) * number_of_blocks,
                 cudaMemcpyDeviceToHost));
  HANDLE_CUDA_ERROR(
      cudaMemcpy(&(num_collisions_h[0]), num_collisions_dev, sizeof(uint16_t) * number_of_blocks,
                 cudaMemcpyDeviceToHost));
  uint32_t result_num_collisions = 0;
  for (uint32_t i = 0; i < number_of_blocks; ++i)
  {
    colliding_meanings |= result_array[i];
    result_num_collisions += num_collisions_h[i];
  }

  HANDLE_CUDA_ERROR(cudaFree(result_ptr_dev));
  HANDLE_CUDA_ERROR(cudaFree(num_collisions_dev));
  return result_num_collisions;
}


template<std::size_t length>
size_t BitVoxelMap<length>::collideWithTypes(const GpuVoxelsMapSharedPtr other, BitVectorVoxel&  meanings_in_collision,
                                             float coll_threshold, const Vector3ui &offset)
{
  size_t num_collisions = SSIZE_MAX;
  switch (other->getMapType())
  {
    case MT_BITVECTOR_VOXELMAP:
    {
      SVCollider collider(coll_threshold);
      BitVectorVoxelMap* m = (BitVectorVoxelMap*) other.get();
      num_collisions = this->collisionCheckBitvector(m, collider, meanings_in_collision.bitVector());
      break;
    }
    case MT_BITVECTOR_VOXELLIST:
    {
      LOGGING_ERROR_C(VoxelmapLog, TemplateVoxelMap, GPU_VOXELS_MAP_SWAP_FOR_COLLIDE << endl);
      break;
    }
    case MT_BITVECTOR_OCTREE:
    {
      LOGGING_ERROR_C(VoxelmapLog, TemplateVoxelMap, GPU_VOXELS_MAP_OPERATION_NOT_YET_SUPPORTED << " " << GPU_VOXELS_MAP_SWAP_FOR_COLLIDE << endl);
      break;
    }
    default:
    {
      LOGGING_ERROR_C(VoxelmapLog, TemplateVoxelMap, GPU_VOXELS_MAP_OPERATION_NOT_YET_SUPPORTED << endl);
      break;
    }
  }
  return num_collisions;
}


template<std::size_t length>
bool BitVoxelMap<length>::insertRobotConfiguration(const MetaPointCloud *robot_links,
                                                   bool with_self_collision_test)
{
  LOGGING_ERROR_C(VoxelmapLog, BitVoxelMap, GPU_VOXELS_MAP_OPERATION_NOT_SUPPORTED << endl);
  return false;
}

template<std::size_t length>
void BitVoxelMap<length>::clearBitVoxelMeaning(BitVoxelMeaning voxel_meaning)
{
  this->clearBit(uint32_t(voxel_meaning));
}

template<std::size_t length>
void BitVoxelMap<length>::shiftLeftSweptVolumeIDs(uint8_t shift_size)
{
  if (shift_size > 63)
  {
    LOGGING_ERROR_C(VoxelmapLog, BitVoxelMap, "Maximum shift size is 63! Higher shift number requested. Not performing shift operation." << endl);
    return;
  }
  while (!this->lockMutex())
  {
    boost::this_thread::yield();
  }

  kernelShiftBitVector<<<this->m_blocks, this->m_threads>>>(this->m_dev_data, this->m_voxelmap_size, shift_size);
  HANDLE_CUDA_ERROR(cudaDeviceSynchronize());

  this->unlockMutex();
}

template<std::size_t length>
void BitVoxelMap<length>::triggerAddressingTest(Vector3ui dimensions, float voxel_side_length,
                                                    size_t nr_of_tests, bool *success)
{

  thrust::host_vector<Vector3f> testpoint_list(nr_of_tests);
  thrust::device_vector<Vector3f> dev_testpoint_list(nr_of_tests);

  srand(time(0));
  gpu_voxels::voxelmap::ProbVoxelMap* voxelmap_base_adress;
  voxelmap_base_adress = (gpu_voxels::voxelmap::ProbVoxelMap*)1234;

  bool* dev_success;
  HANDLE_CUDA_ERROR(cudaMalloc((void** )&dev_success, sizeof(bool)));
  HANDLE_CUDA_ERROR(cudaMemcpy(dev_success, success, sizeof(bool), cudaMemcpyHostToDevice));

  for (uint32_t i = 0; i < nr_of_tests; i++)
  {
    testpoint_list[i].x = (rand() / (double) RAND_MAX) * voxel_side_length * dimensions.x;
    testpoint_list[i].y = (rand() / (double) RAND_MAX) * voxel_side_length * dimensions.y;
    testpoint_list[i].z = (rand() / (double) RAND_MAX) * voxel_side_length * dimensions.z;
  }

  // Copy test data to device:
  dev_testpoint_list = testpoint_list;

  u_int32_t num_blocks;
  u_int32_t threads_per_block;
  computeLinearLoad(nr_of_tests, &num_blocks, &threads_per_block);
  kernelAddressingTest<<< num_blocks, threads_per_block >>> (voxelmap_base_adress, dimensions, voxel_side_length,
                                                             thrust::raw_pointer_cast(dev_testpoint_list.data()),
                                                             nr_of_tests, dev_success);
  HANDLE_CUDA_ERROR(cudaDeviceSynchronize());
  HANDLE_CUDA_ERROR(cudaMemcpy(success, dev_success, sizeof(bool), cudaMemcpyDeviceToHost));
}

} // end of namespace
} // end of namespace

#endif
