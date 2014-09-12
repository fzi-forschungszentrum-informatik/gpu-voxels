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
#include <gpu_voxels/voxelmap/BitVoxel.hpp>
#include <gpu_voxels/voxelmap/ProbVoxelMap.hpp>

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
BitVector<length> BitVoxelMap<length>::collisionCheckBitvector(ProbVoxelMap* other, Collider collider)
{
  uint32_t threads_per_block = 1024;
  //calculate number of blocks
  uint32_t number_of_blocks;
  number_of_blocks = (this->m_voxelmap_size + threads_per_block - 1) / threads_per_block;
//    if(number_of_blocks > cMAX_NR_OF_BLOCKS)
//    {
//      number_of_blocks = cMAX_NR_OF_BLOCKS;
//    }
//    number_of_blocks = 80000;
//    number_of_blocks = 8000;
  //Allocating Memory on device for results
  uint64_t* result_ptr_dev;
  HANDLE_CUDA_ERROR(cudaMalloc((void** )&result_ptr_dev, sizeof(BitVector<length> ) * number_of_blocks));

  kernelCollideVoxelMapsBitvector<<<number_of_blocks, threads_per_block,
                                    sizeof(BitVector<length> ) * threads_per_block>>>(
      this->m_dev_data, this->m_voxelmap_size, other->getDeviceDataPtr(), 1, result_ptr_dev);

  //copying result from device
  BitVector<length> result_array[number_of_blocks];
  for (uint32_t i = 0; i < number_of_blocks; ++i)
  {
    result_array[i] = BitVector<length>();
  }
  HANDLE_CUDA_ERROR(cudaDeviceSynchronize());

  HANDLE_CUDA_ERROR(
      cudaMemcpy(&(result_array[0]), result_ptr_dev, sizeof(BitVector<length> ) * number_of_blocks,
                 cudaMemcpyDeviceToHost));
  uint64_t result = 0;
  for (uint32_t i = 0; i < number_of_blocks; ++i)
  {
    result |= result_array[i];
  }

  HANDLE_CUDA_ERROR(cudaFree(result_ptr_dev));
  return result;
}

template<std::size_t length>
bool BitVoxelMap<length>::insertRobotConfiguration(const MetaPointCloud *robot_links,
                                                   bool with_self_collision_test)
{
  LOGGING_ERROR_C(VoxelmapLog, BitVoxelMap, GPU_VOXELS_MAP_OPERATION_NOT_SUPPORTED << endl);
  return false;
}

template<std::size_t length>
void BitVoxelMap<length>::clearVoxelType(VoxelType voxel_type)
{
  this->clearBit(uint32_t(voxel_type));
}

} // end of namespace
} // end of namespace

#endif
