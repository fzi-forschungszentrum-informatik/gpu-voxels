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


namespace gpu_voxels {
namespace voxelmap {

template<std::size_t length>
BitVoxelMap<length>::BitVoxelMap(const Vector3ui dim, const float voxel_side_length, const MapType map_type) :
    Base(dim, voxel_side_length, map_type)
{
  m_num_self_collisions_checked_entities = 0;
  m_selfcolliding_subclouds_dev = NULL;
  m_collisions_masks_dev = NULL;
  m_subcloud_meanings_dev = NULL;
}

template<std::size_t length>
BitVoxelMap<length>::BitVoxelMap(Voxel* dev_data, const Vector3ui dim, const float voxel_side_length, const MapType map_type) :
    Base(dev_data, dim, voxel_side_length, map_type),
    m_num_self_collisions_checked_entities(0),
    m_selfcolliding_subclouds_dev(NULL),
    m_collisions_masks_dev(NULL),
    m_subcloud_meanings_dev(NULL)
{

}

template<std::size_t length>
BitVoxelMap<length>::~BitVoxelMap()
{
  if (m_selfcolliding_subclouds_dev) HANDLE_CUDA_ERROR(cudaFree(m_selfcolliding_subclouds_dev));
  if (m_collisions_masks_dev) HANDLE_CUDA_ERROR(cudaFree(m_collisions_masks_dev));
  if (m_subcloud_meanings_dev) HANDLE_CUDA_ERROR(cudaFree(m_subcloud_meanings_dev));
}

template<std::size_t length>
void BitVoxelMap<length>::clearBit(const uint32_t bit_index)
{
  lock_guard guard(this->m_mutex);
  clearVoxelMapRemoteLock(bit_index);
}

template<std::size_t length>
void BitVoxelMap<length>::clearVoxelMapRemoteLock(const uint32_t bit_index)
{
  kernelClearVoxelMap<<<this->m_blocks, this->m_threads>>>(this->m_dev_data, this->m_voxelmap_size,
                                                           bit_index);
  CHECK_CUDA_ERROR();
  HANDLE_CUDA_ERROR(cudaDeviceSynchronize());
}

template<std::size_t length>
void BitVoxelMap<length>::clearBits(BitVector<length> bits)
{
  lock_guard guard(this->m_mutex);

  kernelClearVoxelMap<<<this->m_blocks, this->m_threads>>>(this->m_dev_data, this->m_voxelmap_size, bits);
  CHECK_CUDA_ERROR();
  HANDLE_CUDA_ERROR(cudaDeviceSynchronize());
}

template<std::size_t length>
template<class Collider>
uint32_t BitVoxelMap<length>::collisionCheckBitvector(const BitVoxelMap<length>* other, Collider collider,
                                                      BitVector<length>& colliding_meanings,
                                                      const uint16_t sv_offset)
{
  boost::lock(this->m_mutex, other->m_mutex);
  lock_guard guard(this->m_mutex, boost::adopt_lock);
  lock_guard guard2(other->m_mutex, boost::adopt_lock);

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
      this->m_dev_data, this->m_voxelmap_size, other->getConstDeviceDataPtr(), collider, result_ptr_dev, num_collisions_dev, sv_offset);
  CHECK_CUDA_ERROR();

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
template<class Collider>
uint32_t BitVoxelMap<length>::collisionCheckBitvector(const voxelmap::ProbVoxelMap* other, Collider collider,
                                                      BitVector<length>& colliding_meanings,
                                                      const uint16_t sv_offset)
{
  boost::lock(this->m_mutex, other->m_mutex);
  lock_guard guard(this->m_mutex, boost::adopt_lock);
  lock_guard guard2(other->m_mutex, boost::adopt_lock);

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
      this->m_dev_data, this->m_voxelmap_size, other->getConstDeviceDataPtr(), collider, result_ptr_dev, num_collisions_dev, sv_offset);
  CHECK_CUDA_ERROR();

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
size_t BitVoxelMap<length>::collideWithTypes(const voxelmap::ProbVoxelMap* map, BitVectorVoxel& types_in_collision, float coll_threshold, const Vector3i &offset)
{
  SVCollider collider(coll_threshold);
  return this->collisionCheckBitvector(map, collider, types_in_collision.bitVector());
}


template<std::size_t length>
bool BitVoxelMap<length>::insertMetaPointCloudWithSelfCollisionCheck(const MetaPointCloud *meta_point_cloud,
                                                                     const std::vector<BitVoxelMeaning>& voxel_meanings_,
                                                                     const std::vector<BitVector<length> >& collision_masks_,
                                                                     BitVector<length>* colliding_meanings_)
{

  lock_guard guard(this->m_mutex);

  size_t num_links(meta_point_cloud->getNumberOfPointclouds());

  // check, if required mem was allocated on device already, otherwise allocate or reallocate:
  if (!m_selfcolliding_subclouds_dev)
  {
    HANDLE_CUDA_ERROR(cudaMalloc((void** )&m_selfcolliding_subclouds_dev, sizeof(BitVector<BIT_VECTOR_LENGTH>)));
  }
  if(m_num_self_collisions_checked_entities != num_links)
  {
    if (m_collisions_masks_dev) HANDLE_CUDA_ERROR(cudaFree(m_collisions_masks_dev));
    HANDLE_CUDA_ERROR(cudaMalloc((void** )&m_collisions_masks_dev, num_links * sizeof(BitVector<length>)));

    if (m_subcloud_meanings_dev) HANDLE_CUDA_ERROR(cudaFree(m_subcloud_meanings_dev));
    HANDLE_CUDA_ERROR(cudaMalloc((void** )&m_subcloud_meanings_dev, num_links * sizeof(BitVoxelMeaning)));

    m_num_self_collisions_checked_entities = num_links;
  }


  // as the params are const, we have to work internally with writable copies:
  std::vector<BitVoxelMeaning> voxel_meanings;
  std::vector<BitVector<length> > collision_masks;


  // Check if external meanings were given. Otherwise generate standard meanings:
  if(voxel_meanings_.size() == 0)
  {
    voxel_meanings.resize(num_links);
    for(size_t link = 0; link < num_links; ++link)
    {
      voxel_meanings[link] = BitVoxelMeaning(eBVM_SWEPT_VOLUME_START + link);
    }
  }else{
    assert(num_links == voxel_meanings_.size());
    voxel_meanings = voxel_meanings_;
  }


  // Check if external collision masks were given. Otherwise activate all checks:
  if(collision_masks_.size() == 0)
  {
    collision_masks.resize(num_links);
    BitVector<BIT_VECTOR_LENGTH> tmp_coll_mask;
    tmp_coll_mask = ~tmp_coll_mask;
    tmp_coll_mask.clearBit(eBVM_COLLISION); // mask out the colliding bit

    for(size_t mask_id = 0; mask_id < num_links; ++mask_id)
    {
      collision_masks[mask_id] = tmp_coll_mask; // set all to True
      collision_masks[mask_id].clearBit(eBVM_SWEPT_VOLUME_START + mask_id); // except own one
    }
  }else{
    assert(num_links == collision_masks_.size());
    collision_masks = collision_masks_;
  }

  HANDLE_CUDA_ERROR(cudaMemcpy(m_collisions_masks_dev, &collision_masks[0], collision_masks.size() * sizeof(BitVector<length>), cudaMemcpyHostToDevice));


  // reset out of map warning indicator:
  HANDLE_CUDA_ERROR(cudaMemset((void*)this->m_dev_points_outside_map, 0, sizeof(bool)));
  bool points_outside_map;

  // reset self collision indicator:
  HANDLE_CUDA_ERROR(cudaMemset((void*)m_selfcolliding_subclouds_dev, 0, sizeof(BitVector<length>)));

  // copy subcloud meanings
  HANDLE_CUDA_ERROR(cudaMemcpy(m_subcloud_meanings_dev, &voxel_meanings[0], num_links * sizeof(BitVoxelMeaning), cudaMemcpyHostToDevice));



  // To make sure, that no selfcollisions are missed, we have to launch a single kernel per subcloud.
  // This is the best way to ensure Block-Synchronization.
  // Otherwise it could be, that threads of two blocks insert different subclouds at the same time and miss collisions due to concurrent mem access
  std::vector<uint32_t> cloud_sizes = meta_point_cloud->getPointcloudSizes();
  for(size_t sub_cloud = 0; sub_cloud < num_links; ++sub_cloud)
  {
    computeLinearLoad(cloud_sizes[sub_cloud], &this->m_blocks,
                             &this->m_threads);

    kernelInsertMetaPointCloudSelfCollCheck<<<this->m_blocks, this->m_threads>>>(
        this->m_dev_data, meta_point_cloud->getDeviceConstPointer(), m_subcloud_meanings_dev, this->m_dim, sub_cloud, this->m_voxel_side_length,
        m_collisions_masks_dev, this->m_dev_points_outside_map, m_selfcolliding_subclouds_dev);
    CHECK_CUDA_ERROR();

    HANDLE_CUDA_ERROR(cudaDeviceSynchronize());
  }

  // check for out of map points:
  HANDLE_CUDA_ERROR(cudaMemcpy(&points_outside_map, this->m_dev_points_outside_map, sizeof(bool), cudaMemcpyDeviceToHost));
  if(points_outside_map)
  {
    LOGGING_WARNING_C(VoxelmapLog, VoxelMap, "You tried to insert points that lie outside the map dimensions!" << endl);
  }

  // copy back the results, either to param or temp variable
  if(colliding_meanings_ == NULL)
  {
    BitVector<length> colliding_meanings;
    HANDLE_CUDA_ERROR(cudaMemcpy(&colliding_meanings, m_selfcolliding_subclouds_dev, sizeof(BitVector<length>), cudaMemcpyDeviceToHost));
    return !colliding_meanings.isZero();
  }else{
    HANDLE_CUDA_ERROR(cudaMemcpy(colliding_meanings_, m_selfcolliding_subclouds_dev, sizeof(BitVector<length>), cudaMemcpyDeviceToHost));
    return !colliding_meanings_->isZero();
  }
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
  lock_guard guard(this->m_mutex);
  kernelShiftBitVector<<<this->m_blocks, this->m_threads>>>(this->m_dev_data, this->m_voxelmap_size, shift_size);
  CHECK_CUDA_ERROR();
  HANDLE_CUDA_ERROR(cudaDeviceSynchronize());
}

} // end of namespace
} // end of namespace

#endif
