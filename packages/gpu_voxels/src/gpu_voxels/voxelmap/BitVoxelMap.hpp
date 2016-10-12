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
BitVoxelMap<length>::BitVoxelMap(const Vector3ui dim, const float voxel_side_length, const MapType map_type) :
    Base(dim, voxel_side_length, map_type)
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
  this->lockSelf("BitVoxelMap::clearBit");
  clearVoxelMapRemoteLock(bit_index);
  this->unlockSelf("BitVoxelMap::clearBit");
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
  this->lockSelf("BitVoxelMap::clearBits");

  kernelClearVoxelMap<<<this->m_blocks, this->m_threads>>>(this->m_dev_data, this->m_voxelmap_size, bits);
  HANDLE_CUDA_ERROR(cudaDeviceSynchronize());

  this->unlockSelf("BitVoxelMap::clearBits");
}

template<std::size_t length>
template<class Collider>
uint32_t BitVoxelMap<length>::collisionCheckBitvector(BitVoxelMap<length>* other, Collider collider,
                                                      BitVector<length>& colliding_meanings,
                                                      const uint16_t sv_offset)
{
  this->lockBoth(this, other, "BitVoxelMap::collisionCheckBitvector");
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
  this->unlockBoth(this, other, "BitVoxelMap::collisionCheckBitvector");
  return result_num_collisions;
}

template<std::size_t length>
size_t BitVoxelMap<length>::collideWith(const BitVectorVoxelMap *map, float coll_threshold, const Vector3i &offset)
{
  DefaultCollider collider(coll_threshold);
  return this->collisionCheckWithCounterRelativeTransform((TemplateVoxelMap<BitVectorVoxel>*)map, collider, offset);
}

template<std::size_t length>
size_t BitVoxelMap<length>::collideWith(const ProbVoxelMap *map, float coll_threshold, const Vector3i &offset)
{
  DefaultCollider collider(coll_threshold);
  return this->collisionCheckWithCounterRelativeTransform((TemplateVoxelMap<ProbabilisticVoxel>*)map, collider, offset);
}

template<std::size_t length>
size_t BitVoxelMap<length>::collideWithTypes(const BitVectorVoxelMap *map, BitVectorVoxel &types_in_collision, float coll_threshold, const Vector3i &offset)
{
  SVCollider collider(coll_threshold);
  return this->collisionCheckBitvector((BitVoxelMap*)map, collider, types_in_collision.bitVector());
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
  this->clearBit(uint32_t(voxel_meaning)); //does the locking
}

template<std::size_t length>
void BitVoxelMap<length>::shiftLeftSweptVolumeIDs(uint8_t shift_size)
{
  if (shift_size > 56) // restriction due to internal buffer
  {
    LOGGING_ERROR_C(VoxelmapLog, BitVoxelMap, "Maximum shift size is 56! Higher shift number requested. Not performing shift operation." << endl);
    return;
  }
  this->lockSelf("BitVoxelMap::shiftLeftSweptVolumeIDs");
  kernelShiftBitVector<<<this->m_blocks, this->m_threads>>>(this->m_dev_data, this->m_voxelmap_size, shift_size);
  HANDLE_CUDA_ERROR(cudaDeviceSynchronize());

  this->unlockSelf("BitVoxelMap::shiftLeftSweptVolumeIDs");
}

} // end of namespace
} // end of namespace

#endif
