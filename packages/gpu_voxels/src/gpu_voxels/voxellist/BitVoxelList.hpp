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
* \author  Felix Mauch <mauch@fzi.de>
* \date    2015-05-04
*
*/
//----------------------------------------------------------------------

#ifndef GPU_VOXELS_VOXELLIST_BITVOXELLIST_HPP_INCLUDED
#define GPU_VOXELS_VOXELLIST_BITVOXELLIST_HPP_INCLUDED

#include "BitVoxelList.h"
#include <gpu_voxels/voxellist/TemplateVoxelList.hpp>
//#include <gpu_voxels/voxelmap/ProbVoxelMap.hpp>
#include <gpu_voxels/logging/logging_voxellist.h>
#include <thrust/system_error.h>


namespace gpu_voxels{
namespace voxellist{
using namespace gpu_voxels::voxelmap;


template<std::size_t length, class VoxelIDType>
BitVoxelList<length, VoxelIDType>::BitVoxelList(const Vector3ui ref_map_dim, const float voxel_side_length, const MapType map_type)
  : TemplateVoxelList<BitVectorVoxel, VoxelIDType>(ref_map_dim, voxel_side_length, map_type)
{
  // We already resize the result vector for Bitvector Checks
  m_dev_colliding_bits_result_list.resize(cMAX_NR_OF_BLOCKS);
  m_colliding_bits_result_list.resize(cMAX_NR_OF_BLOCKS);

  // Allocate a BitVectorVoxel on the device to it use as bitmask for later coll-checks.
  cudaMalloc(&m_dev_bitmask, sizeof(BitVectorVoxel));
}


template<std::size_t length, class VoxelIDType>
BitVoxelList<length, VoxelIDType>::~BitVoxelList()
{
}

template<std::size_t length, class VoxelIDType>
void BitVoxelList<length, VoxelIDType>::clearBitVoxelMeaning(BitVoxelMeaning voxel_meaning)
{
  LOGGING_ERROR_C(VoxellistLog, BitVoxelList, GPU_VOXELS_MAP_OPERATION_NOT_YET_SUPPORTED << endl);
}

template<std::size_t length, class VoxelIDType>
size_t BitVoxelList<length, VoxelIDType>::collideWith(const ProbVoxelMap *map, float coll_threshold, const Vector3i &offset)
{
  DefaultCollider collider(coll_threshold);
  //voxelmap::ProbVoxelMap* m = (voxelmap::ProbVoxelMap*) other.get();
  return this->collisionCheckWithCollider((TemplateVoxelMap<ProbabilisticVoxel>*)map, collider, offset); // does the locking
}

template<std::size_t length, class VoxelIDType>
size_t BitVoxelList<length, VoxelIDType>::collideWith(const BitVectorVoxelMap *map, float coll_threshold, const Vector3i &offset)
{
  DefaultCollider collider(coll_threshold);
  //voxelmap::BitVectorVoxelMap* m = (voxelmap::BitVectorVoxelMap*) other.get();
  return this->collisionCheckWithCollider((TemplateVoxelMap<BitVectorVoxel>*)map, collider, offset); // does the locking
}

template<std::size_t length, class VoxelIDType>
size_t BitVoxelList<length, VoxelIDType>::collideWith(const BitVectorVoxelList *other, float coll_threshold, const Vector3i &offset)
{
  //BitVoxelList<BIT_VECTOR_LENGTH, VoxelIDType>* m = (BitVoxelList<BIT_VECTOR_LENGTH, VoxelIDType>*) other.get();
  size_t collisions = SSIZE_MAX;
  boost::lock(this->m_mutex, other->m_mutex);
  lock_guard guard(this->m_mutex, boost::adopt_lock);
  lock_guard guard2(other->m_mutex, boost::adopt_lock);

  thrust::device_vector<bool> collision_stencil(this->m_dev_id_list.size()); // Temporary data structure
  collisions = this->collideVoxellists((TemplateVoxelList<BitVectorVoxel, VoxelIDType>*)other, offset, collision_stencil);

  return collisions;
}

template<std::size_t length, class VoxelIDType>
size_t BitVoxelList<length, VoxelIDType>::collideWithTypes(const BitVectorVoxelList *map, BitVectorVoxel &types_in_collision, float coll_threshold, const Vector3i &offset)
{
  //TemplatedBitVectorVoxelList* other = dynamic_cast<TemplatedBitVectorVoxelList*>(map);
  TemplatedBitVectorVoxelList* other = (TemplatedBitVectorVoxelList*)map;
  boost::lock(this->m_mutex, other->m_mutex);
  lock_guard guard(this->m_mutex, boost::adopt_lock);
  lock_guard guard2(other->m_mutex, boost::adopt_lock);

  //========== Search for Voxels at the same spot in both lists: ==============
  TemplatedBitVectorVoxelList matching_voxels_list1(this->m_ref_map_dim, this->m_voxel_side_length, this->m_map_type);
  TemplatedBitVectorVoxelList matching_voxels_list2(this->m_ref_map_dim, this->m_voxel_side_length, this->m_map_type);
  findMatchingVoxels(other, 0, offset, &matching_voxels_list1, &matching_voxels_list2);

  //========== Now iterate over both shortened lists and inspect the Bitvectors =============
  thrust::device_vector< BitVectorVoxel> dev_merged_voxel_list(matching_voxels_list1.m_dev_id_list.size());
  thrust::transform(matching_voxels_list1.m_dev_list.begin(), matching_voxels_list1.m_dev_list.end(),
                    matching_voxels_list2.m_dev_list.begin(),
                    dev_merged_voxel_list.begin(), BitVectorVoxel::reduce_op());
  HANDLE_CUDA_ERROR(cudaDeviceSynchronize());

  types_in_collision = thrust::reduce(dev_merged_voxel_list.begin(), dev_merged_voxel_list.end(),
                                         BitVectorVoxel(), BitVectorVoxel::reduce_op());
  return matching_voxels_list1.m_dev_id_list.size();
}

template<std::size_t length, class VoxelIDType>
size_t BitVoxelList<length, VoxelIDType>::collideWithTypes(const BitVectorVoxelMap *other, BitVectorVoxel &types_in_collision, float coll_threshold, const Vector3i &offset)
{
  // Map Dims have to be equal to be able to compare pointer adresses!
  if(other->getDimensions() != this->m_ref_map_dim)
  {
    LOGGING_ERROR_C(VoxellistLog, BitVoxelList,
                    "The dimensions of the Voxellist reference map do not match the colliding voxel map dimensions. Not checking collisions!" << endl);
    return SSIZE_MAX;
  }
  boost::lock(this->m_mutex, other->m_mutex);
  lock_guard guard(this->m_mutex, boost::adopt_lock);
  lock_guard guard2(other->m_mutex, boost::adopt_lock);

  // get raw pointers to the thrust vectors data:
  BitVectorVoxel* dev_voxel_list_ptr = thrust::raw_pointer_cast(this->m_dev_list.data());
  VoxelIDType* dev_id_list_ptr = thrust::raw_pointer_cast(this->m_dev_id_list.data());
  BitVectorVoxel* m_dev_colliding_bits_result_list_ptr = thrust::raw_pointer_cast(m_dev_colliding_bits_result_list.data());

  uint32_t num_blocks, threads_per_block;
  computeLinearLoad(this->m_dev_list.size(), &num_blocks, &threads_per_block);
  HANDLE_CUDA_ERROR(cudaDeviceSynchronize());
  size_t dynamic_shared_mem_size = sizeof(BitVectorVoxel) * cMAX_THREADS_PER_BLOCK;
  kernelCollideWithVoxelMap<<<num_blocks, threads_per_block, dynamic_shared_mem_size>>>(dev_id_list_ptr, dev_voxel_list_ptr, (uint32_t)this->m_dev_list.size(),
                                                                  other->getConstDeviceDataPtr(), this->m_ref_map_dim, coll_threshold,
                                                                  offset, this->m_dev_collision_check_results_counter, m_dev_colliding_bits_result_list_ptr);
  CHECK_CUDA_ERROR();
  HANDLE_CUDA_ERROR(cudaDeviceSynchronize());

  // Copy back the results and reduce the block results:
  this->m_colliding_bits_result_list = this->m_dev_colliding_bits_result_list;
  HANDLE_CUDA_ERROR(
      cudaMemcpy(this->m_collision_check_results_counter, this->m_dev_collision_check_results_counter,
                 num_blocks * sizeof(uint16_t), cudaMemcpyDeviceToHost));

  size_t number_of_collisions = 0;
  types_in_collision.bitVector().clear();
  for (uint32_t i = 0; i < num_blocks; i++)
  {
    number_of_collisions += this->m_collision_check_results_counter[i];
    types_in_collision.bitVector() |= this->m_colliding_bits_result_list[i].bitVector();
  }
  return number_of_collisions;
}

template<std::size_t length, class VoxelIDType>
size_t BitVoxelList<length, VoxelIDType>::collideWithTypes(const ProbVoxelMap *map, BitVectorVoxel &types_in_collision, float coll_threshold, const Vector3i &offset)
{
  // Map Dims have to be equal to be able to compare pointer adresses!
  if(map->getDimensions() != this->m_ref_map_dim)
  {
    LOGGING_ERROR_C(VoxellistLog, BitVoxelList,
                    "The dimensions of the Voxellist reference map do not match the colliding voxel map dimensions. Not checking collisions!" << endl);
    return SSIZE_MAX;
  }

  //ProbVoxelMap* other = dynamic_cast<voxellist::ProbVoxelMap*>(map);
  ProbVoxelMap* other = (voxellist::ProbVoxelMap*)map;
  boost::lock(this->m_mutex, other->m_mutex);
  lock_guard guard(this->m_mutex, boost::adopt_lock);
  lock_guard guard2(other->m_mutex, boost::adopt_lock);

  // get raw pointers to the thrust vectors data:
  BitVectorVoxel* dev_voxel_list_ptr = thrust::raw_pointer_cast(this->m_dev_list.data());
  VoxelIDType* dev_id_list_ptr = thrust::raw_pointer_cast(this->m_dev_id_list.data());
  BitVectorVoxel* m_dev_colliding_bits_result_list_ptr = thrust::raw_pointer_cast(m_dev_colliding_bits_result_list.data());

  uint32_t num_blocks, threads_per_block;
  computeLinearLoad(this->m_dev_list.size(), &num_blocks, &threads_per_block);
  HANDLE_CUDA_ERROR(cudaDeviceSynchronize());
  size_t dynamic_shared_mem_size = sizeof(BitVectorVoxel) * cMAX_THREADS_PER_BLOCK;
  kernelCollideWithVoxelMap<<<num_blocks, threads_per_block, dynamic_shared_mem_size>>>(dev_id_list_ptr, dev_voxel_list_ptr, (uint32_t)this->m_dev_list.size(),
                                                                  other->getDeviceDataPtr(), this->m_ref_map_dim, coll_threshold,
                                                                  offset, this->m_dev_collision_check_results_counter, m_dev_colliding_bits_result_list_ptr);
  CHECK_CUDA_ERROR();
  HANDLE_CUDA_ERROR(cudaDeviceSynchronize());

  // Copy back the results and reduce the block results:
  this->m_colliding_bits_result_list = this->m_dev_colliding_bits_result_list;
  HANDLE_CUDA_ERROR(
      cudaMemcpy(this->m_collision_check_results_counter, this->m_dev_collision_check_results_counter,
                 num_blocks * sizeof(uint16_t), cudaMemcpyDeviceToHost));

  size_t number_of_collisions = 0;
  types_in_collision.bitVector().clear();
  for (uint32_t i = 0; i < num_blocks; i++)
  {
    number_of_collisions += this->m_collision_check_results_counter[i];
    types_in_collision.bitVector() |= this->m_colliding_bits_result_list[i].bitVector();
  }
  return number_of_collisions;
}

template<std::size_t length, class VoxelIDType>
template<class Voxel>
size_t BitVoxelList<length, VoxelIDType>::collideWithTypeMask(const TemplateVoxelMap<Voxel> *map,
                                                              const BitVectorVoxel& types_to_check, float coll_threshold, const Vector3i &offset)
{
  // Map Dims have to be equal to be able to compare pointer adresses!
  if(map->getDimensions() != this->m_ref_map_dim)
  {
    LOGGING_ERROR_C(VoxellistLog, BitVoxelList,
                    "The dimensions of the Voxellist reference map do not match the colliding voxel map dimensions. Not checking collisions!" << endl);
    return SSIZE_MAX;
  }
  boost::lock(this->m_mutex, map->m_mutex);
  lock_guard guard(this->m_mutex, boost::adopt_lock);
  lock_guard guard2(map->m_mutex, boost::adopt_lock);

  // get raw pointers to the thrust vectors data:
  BitVectorVoxel* dev_voxel_list_ptr = thrust::raw_pointer_cast(this->m_dev_list.data());
  VoxelIDType* dev_id_list_ptr = thrust::raw_pointer_cast(this->m_dev_id_list.data());

  cudaMemcpy(m_dev_bitmask, &types_to_check, sizeof(BitVectorVoxel), cudaMemcpyHostToDevice);

  uint32_t num_blocks, threads_per_block;
  computeLinearLoad(this->m_dev_list.size(), &num_blocks, &threads_per_block);
  HANDLE_CUDA_ERROR(cudaDeviceSynchronize());
  size_t dynamic_shared_mem_size = sizeof(uint16_t) * cMAX_THREADS_PER_BLOCK;

  kernelCollideWithVoxelMapBitMask<<<num_blocks, threads_per_block, dynamic_shared_mem_size>>>(dev_id_list_ptr, dev_voxel_list_ptr, (uint32_t)this->m_dev_list.size(),
                                                                                               map->getConstDeviceDataPtr(), this->m_ref_map_dim, coll_threshold,
                                                                                               offset, m_dev_bitmask, this->m_dev_collision_check_results_counter);
  CHECK_CUDA_ERROR();
  HANDLE_CUDA_ERROR(cudaDeviceSynchronize());

  // Copy back the results and reduce the block results:
  HANDLE_CUDA_ERROR(
      cudaMemcpy(this->m_collision_check_results_counter, this->m_dev_collision_check_results_counter,
                 num_blocks * sizeof(uint16_t), cudaMemcpyDeviceToHost));

  size_t number_of_collisions = 0;
  for (uint32_t i = 0; i < num_blocks; i++)
  {
    number_of_collisions += this->m_collision_check_results_counter[i];
  }

  return number_of_collisions;
}

template<std::size_t length, class VoxelIDType>
size_t BitVoxelList<length, VoxelIDType>::collideWithBitcheck(const BitVectorVoxelList *map, const u_int8_t margin, const Vector3i &offset)
{
  //TemplatedBitVectorVoxelList* other = dynamic_cast<TemplatedBitVectorVoxelList*>(map);
  TemplatedBitVectorVoxelList* other = (TemplatedBitVectorVoxelList*)map;

  boost::lock(this->m_mutex, other->m_mutex);
  lock_guard guard(this->m_mutex, boost::adopt_lock);
  lock_guard guard2(other->m_mutex, boost::adopt_lock);

  //========== Search for Voxels at the same spot in both lists: ==============
  TemplatedBitVectorVoxelList matching_voxels_list1(this->m_ref_map_dim, this->m_voxel_side_length, this->m_map_type);
  TemplatedBitVectorVoxelList matching_voxels_list2(this->m_ref_map_dim, this->m_voxel_side_length, this->m_map_type);
  findMatchingVoxels(other, margin, offset, &matching_voxels_list1, &matching_voxels_list2);

  //========== Now iterate over both shortened lists and inspect the Bitvectors =============
  thrust::device_vector<bool> dev_colliding_bits_list(matching_voxels_list1.m_dev_id_list.size());

  // only use the slower collision comperator, if a bitmarking was set!
  if(margin == 0)
  {
    thrust::transform(matching_voxels_list1.m_dev_list.begin(), matching_voxels_list1.m_dev_list.end(),
                      matching_voxels_list2.m_dev_list.begin(),
                      dev_colliding_bits_list.begin(), BitvectorCollision());
  }else{
    // TODO: Think about offset and add as a param
    thrust::transform(matching_voxels_list1.m_dev_list.begin(), matching_voxels_list1.m_dev_list.end(),
                      matching_voxels_list2.m_dev_list.begin(),
                      dev_colliding_bits_list.begin(), BitvectorCollisionWithBitshift(margin, 0));
  }
  HANDLE_CUDA_ERROR(cudaDeviceSynchronize());
  return thrust::count(dev_colliding_bits_list.begin(), dev_colliding_bits_list.end(), true);
}


template<std::size_t length, class VoxelIDType>
size_t BitVoxelList<length, VoxelIDType>::collideCountingPerMeaning(const GpuVoxelsMapSharedPtr other_,
                                                           std::vector<size_t>&  collisions_per_meaning,
                                                           const Vector3i &offset_)
{
  try
  {
    switch (other_->getMapType())
    {
      case MT_BITVECTOR_VOXELLIST:
      {
        TemplatedBitVectorVoxelList* other = dynamic_cast<TemplatedBitVectorVoxelList*>(other_.get());

        boost::lock(this->m_mutex, other->m_mutex);
        lock_guard guard(this->m_mutex, boost::adopt_lock);
        lock_guard guard2(other->m_mutex, boost::adopt_lock);

        //========== Search for Voxels at the same spot in both lists: ==============
        TemplatedBitVectorVoxelList matching_voxels_list1(this->m_ref_map_dim, this->m_voxel_side_length, this->m_map_type);
        TemplatedBitVectorVoxelList matching_voxels_list2(this->m_ref_map_dim, this->m_voxel_side_length, this->m_map_type);
        findMatchingVoxels(other, 0, offset_, &matching_voxels_list1, &matching_voxels_list2);

        // matching_voxels_list1 now contains all Voxels that lie in collision
        // Copy it to the host, iterate over all voxels and count the Meanings:

        size_t summed_colls = 0;
        thrust::host_vector<BitVectorVoxel> h_colliding_voxels;
        h_colliding_voxels = matching_voxels_list1.m_dev_list;

        assert(collisions_per_meaning.size() == BIT_VECTOR_LENGTH);

        // TODO: Put this in a kernel!
        for(size_t i = 0; i < h_colliding_voxels.size(); i++)
        {
          for(size_t j = 0; j < BIT_VECTOR_LENGTH; j++)
          {
            if(h_colliding_voxels[i].bitVector().getBit(j))
            {
              collisions_per_meaning[j]++;
              summed_colls++;
            }
          }
        }
        return summed_colls;
      }
      case MT_COUNTING_VOXELLIST:
      {
        CountingVoxelList* other = dynamic_cast<CountingVoxelList*>(other_.get());

        boost::lock(this->m_mutex, other->m_mutex);
        lock_guard guard(this->m_mutex, boost::adopt_lock);
        lock_guard guard2(other->m_mutex, boost::adopt_lock);

        //========== Search for Voxels at the same spot in both lists: ==============
        TemplatedBitVectorVoxelList matching_voxels_list1(this->m_ref_map_dim, this->m_voxel_side_length, this->m_map_type);
        CountingVoxelList matching_voxels_list2(this->m_ref_map_dim, this->m_voxel_side_length, this->m_map_type);
        findMatchingVoxels(other, offset_, &matching_voxels_list1);

        // matching_voxels_list1 now contains all Voxels that lie in collision
        // Copy it to the host, iterate over all voxels and count the Meanings:

        size_t summed_colls = 0;
        thrust::host_vector<BitVectorVoxel> h_colliding_voxels;
        h_colliding_voxels = matching_voxels_list1.m_dev_list;

        assert(collisions_per_meaning.size() == BIT_VECTOR_LENGTH);

        // TODO: Put this in a kernel!
        for(size_t i = 0; i < h_colliding_voxels.size(); i++)
        {
          for(size_t j = 0; j < BIT_VECTOR_LENGTH; j++)
          {
            if(h_colliding_voxels[i].bitVector().getBit(j))
            {
              collisions_per_meaning[j]++;
              summed_colls++;
            }
          }
        }
        return summed_colls;
      }
      default:
      {
        LOGGING_ERROR_C(VoxellistLog, BitVoxelList, GPU_VOXELS_MAP_OPERATION_NOT_SUPPORTED << endl);
        return SSIZE_MAX;
      }
    }
  }
  catch(thrust::system_error &e)
  {
    LOGGING_ERROR_C(VoxellistLog, BitVoxelList, "Caught Thrust exception " << e.what() << endl);
    exit(-1);
  }

}

template<std::size_t length, class VoxelIDType>
void BitVoxelList<length, VoxelIDType>::findMatchingVoxels(const TemplatedBitVectorVoxelList *list2,
                                              const u_int8_t margin, const Vector3i &offset,
                                              TemplatedBitVectorVoxelList* matching_voxels_list1,
                                              TemplatedBitVectorVoxelList* matching_voxels_list2,
                                              bool omit_coords) const
{
  if(offset != Vector3i(0))
  {
    LOGGING_ERROR_C(VoxellistLog, BitVoxelList, "Offset for VoxelList operation not supported! Result is undefined." << endl);
    return;
  }

  const TemplatedBitVectorVoxelList *list1 = this;

  boost::lock(list1->m_mutex, list2->m_mutex);
  lock_guard guard(list1->m_mutex, boost::adopt_lock);
  lock_guard guard2(list2->m_mutex, boost::adopt_lock);

  bool wasSwapped = false;
  size_t num_hits1 = 0;
  size_t num_hits2 = 0;
  thrust::device_vector<bool> output;

  const TemplatedBitVectorVoxelList* biggerList;
  const TemplatedBitVectorVoxelList* smallerList;

  // For performance reasons choose the list in an order,
  // so that we search the longer list for the elemensts
  // of the shorter list in the first processing step.
  if(list1->m_dev_id_list.size() < list2->m_dev_id_list.size())
  {
    biggerList = list2;
    smallerList = list1;
    wasSwapped = true;
  }
  else
  {
    biggerList = list1;
    smallerList = list2;
  }

  //========== First search: Search for IDs from smaller list in bigger list. =================================================
  try
  {
    output = thrust::device_vector<bool>(smallerList->m_dev_id_list.size());
    thrust::binary_search(thrust::device,
                          biggerList->m_dev_id_list.begin(), biggerList->m_dev_id_list.end(),
                          smallerList->m_dev_id_list.begin(), smallerList->m_dev_id_list.end(),
                          output.begin());
  }
  catch(thrust::system_error &e)
  {
    LOGGING_ERROR_C(VoxellistLog, BitVoxelList, "Caught Thrust exception " << e.what() << endl);
    exit(-1);
  }

  //========== Copy matching IDs (and the voxeldata) from smallerList to matching_voxels_list2.
  try
  {
    num_hits1 = thrust::count(output.begin(), output.end(), true);
    matching_voxels_list2->m_dev_id_list.resize(num_hits1);
    matching_voxels_list2->m_dev_list.resize(num_hits1);

    if (omit_coords) 
    {
      // we use the "output" list as stencil
      thrust::copy_if( thrust::make_zip_iterator(thrust::make_tuple(smallerList->m_dev_id_list.begin(), smallerList->m_dev_list.begin())),
                       thrust::make_zip_iterator(thrust::make_tuple(smallerList->m_dev_id_list.end(), smallerList->m_dev_list.end())),
                       output.begin(),
                       thrust::make_zip_iterator(thrust::make_tuple(matching_voxels_list2->m_dev_id_list.begin(), matching_voxels_list2->m_dev_list.begin())),
                       thrust::identity<bool>());
    } else
    {
      // resize m_dev_coord and include coords in copy_if zip iterators
      matching_voxels_list2->m_dev_coord_list.resize(num_hits1);

      // we use the "output" list as stencil to copy the IDs (and the voxeldata) from the biggerList to matching_voxels_list1
      thrust::copy_if( thrust::make_zip_iterator(thrust::make_tuple(
                           smallerList->m_dev_id_list.begin(), 
                           smallerList->m_dev_list.begin(), 
                           smallerList->m_dev_coord_list.begin())),
                       thrust::make_zip_iterator(thrust::make_tuple(
                           smallerList->m_dev_id_list.end(), 
                           smallerList->m_dev_list.end(),
                           smallerList->m_dev_coord_list.end())),
                       output.begin(),
                       thrust::make_zip_iterator(thrust::make_tuple(
                           matching_voxels_list2->m_dev_id_list.begin(), 
                           matching_voxels_list2->m_dev_list.begin(),
                           matching_voxels_list2->m_dev_coord_list.begin())),
                       thrust::identity<bool>());
    }
  }
  catch(thrust::system_error &e)
  {
    LOGGING_ERROR_C(VoxellistLog, BitVoxelList, "Caught Thrust exception " << e.what() << endl);
    exit(-1);
  }

  //========== Second search: Search for matched IDs in biggerList. =================================================
  try
  {
    output = thrust::device_vector<bool>(biggerList->m_dev_id_list.size()); // the result as vector of bools
    thrust::binary_search(thrust::device,
                          matching_voxels_list2->m_dev_id_list.begin(), matching_voxels_list2->m_dev_id_list.end(),
                          biggerList->m_dev_id_list.begin(), biggerList->m_dev_id_list.end(),
                          output.begin());

    //========== Resize matching_voxels_list1 to the number of matching entries found in previous step.
    num_hits2 = thrust::count(output.begin(), output.end(), true);
    matching_voxels_list1->m_dev_id_list.resize(num_hits2);
    matching_voxels_list1->m_dev_list.resize(num_hits2);

    if (omit_coords) 
    {
      // we use the "output" list as stencil to copy the IDs (and the voxeldata) from the biggerList to matching_voxels_list1
      thrust::copy_if( thrust::make_zip_iterator(thrust::make_tuple(biggerList->m_dev_id_list.begin(), biggerList->m_dev_list.begin())),
                       thrust::make_zip_iterator(thrust::make_tuple(biggerList->m_dev_id_list.end(), biggerList->m_dev_list.end())),
                       output.begin(),
                       thrust::make_zip_iterator(thrust::make_tuple(matching_voxels_list1->m_dev_id_list.begin(), matching_voxels_list1->m_dev_list.begin())),
                       thrust::identity<bool>());
    } else
    {
      // resize m_dev_coord and include coords in copy_if zip iterators
      matching_voxels_list1->m_dev_coord_list.resize(num_hits2);

      // we use the "output" list as stencil to copy the IDs (and the voxeldata) from the biggerList to matching_voxels_list1
      thrust::copy_if( thrust::make_zip_iterator(thrust::make_tuple(
                           biggerList->m_dev_id_list.begin(), 
                           biggerList->m_dev_list.begin(), 
                           biggerList->m_dev_coord_list.begin())),
                       thrust::make_zip_iterator(thrust::make_tuple(
                           biggerList->m_dev_id_list.end(), 
                           biggerList->m_dev_list.end(),
                           biggerList->m_dev_coord_list.end())),
                       output.begin(),
                       thrust::make_zip_iterator(thrust::make_tuple(
                           matching_voxels_list1->m_dev_id_list.begin(), 
                           matching_voxels_list1->m_dev_list.begin(),
                           matching_voxels_list1->m_dev_coord_list.begin())),
                       thrust::identity<bool>());
    }

  }
  catch(thrust::system_error &e)
  {
    LOGGING_ERROR_C(VoxellistLog, BitVoxelList, "Caught Thrust exception " << e.what() << endl);
    exit(-1);
  }

  // If we swapped in the beginning we have to also swap the matching_voxels_lists
  // So that matching_voxels_list1 contains voxeldata from input list1
  // and matching_voxels_list2 contains voxeldata from input list2
  if(wasSwapped)
  {
    matching_voxels_list1->m_dev_id_list.swap(matching_voxels_list2->m_dev_id_list);
    matching_voxels_list1->m_dev_list.swap(matching_voxels_list2->m_dev_list);

    if (!omit_coords)
    {
      matching_voxels_list1->m_dev_coord_list.swap(matching_voxels_list2->m_dev_coord_list);
    }
  }

  // Do a sanity check:
  if(num_hits1 != num_hits2)
  {
    LOGGING_ERROR_C(VoxellistLog, BitVoxelList, "ASSERT 'num_hits1 == num_hits2' failed! " << num_hits1 << " != " << num_hits2 << endl);
  }
  assert(num_hits1 == num_hits2);

  HANDLE_CUDA_ERROR(cudaDeviceSynchronize());
}

template<std::size_t length, class VoxelIDType>
void BitVoxelList<length, VoxelIDType>::findMatchingVoxels(const CountingVoxelList *list2,
                                              const Vector3i &offset,
                                              TemplatedBitVectorVoxelList* matching_voxels_list, bool omit_coords) const
{
  if(offset != Vector3i(0))
  {
    LOGGING_ERROR_C(VoxellistLog, BitVoxelList, "Offset for VoxelList operation not supported! Result is undefined." << endl);
    return;
  }

  const TemplatedBitVectorVoxelList *list1 = this;

  boost::lock(list1->m_mutex, list2->m_mutex);
  lock_guard guard(list1->m_mutex, boost::adopt_lock);
  lock_guard guard2(list2->m_mutex, boost::adopt_lock);

  size_t num_hits1 = 0;
  thrust::device_vector<bool> output;

  //========== Search for IDs from CVL in BVL =================================================
  try
  {
    output = thrust::device_vector<bool>(list1->m_dev_id_list.size());
    thrust::binary_search(thrust::device,
                          list2->m_dev_id_list.begin(), list2->m_dev_id_list.end(),
                          list1->m_dev_id_list.begin(), list1->m_dev_id_list.end(),
                          output.begin());
  }
  catch(thrust::system_error &e)
  {
    LOGGING_ERROR_C(VoxellistLog, BitVoxelList, "Caught Thrust exception " << e.what() << endl);
    exit(-1);
  }

  //========== Copy matching IDs (and the voxeldata) from smallerList to matching_voxels_list.
  try
  {
    num_hits1 = thrust::count(output.begin(), output.end(), true);
    matching_voxels_list->m_dev_id_list.resize(num_hits1);
    matching_voxels_list->m_dev_list.resize(num_hits1);

    if (omit_coords) 
    {
      // we use the "output" list as stencil
      thrust::copy_if( thrust::make_zip_iterator(thrust::make_tuple(list1->m_dev_id_list.begin(), list1->m_dev_list.begin())),
                       thrust::make_zip_iterator(thrust::make_tuple(list1->m_dev_id_list.end(), list1->m_dev_list.end())),
                       output.begin(),
                       thrust::make_zip_iterator(thrust::make_tuple(matching_voxels_list->m_dev_id_list.begin(), matching_voxels_list->m_dev_list.begin())),
                       thrust::identity<bool>());
    } else
    {
      // resize m_dev_coords and include coords in copy_if zip iterators
      matching_voxels_list->m_dev_coord_list.resize(num_hits1);

      // we use the "output" list as stencil
      thrust::copy_if( thrust::make_zip_iterator(thrust::make_tuple(list1->m_dev_id_list.begin(), list1->m_dev_list.begin(), list1->m_dev_coord_list.begin())),
                       thrust::make_zip_iterator(thrust::make_tuple(list1->m_dev_id_list.end(), list1->m_dev_list.end(), list1->m_dev_coord_list.end())),
                       output.begin(),
                       thrust::make_zip_iterator(thrust::make_tuple(
                           matching_voxels_list->m_dev_id_list.begin(), 
                           matching_voxels_list->m_dev_list.begin(), 
                           matching_voxels_list->m_dev_coord_list.begin())),
                       thrust::identity<bool>());
    }
  }
  catch(thrust::system_error &e)
  {
    LOGGING_ERROR_C(VoxellistLog, BitVoxelList, "Caught Thrust exception " << e.what() << endl);
    exit(-1);
  }

  HANDLE_CUDA_ERROR(cudaDeviceSynchronize());
}

template<std::size_t length, class VoxelIDType>
void BitVoxelList<length, VoxelIDType>::shiftLeftSweptVolumeIDs(uint8_t shift_size)
{
  if (shift_size > 56) // restriction due to internal buffer
  {
    LOGGING_ERROR_C(VoxellistLog, BitVoxelList, "Maximum shift size is 56! Higher shift number requested. Not performing shift operation." << endl);
    return;
  }

  lock_guard guard(this->m_mutex);

  try
  {
    thrust::transform(this->m_dev_list.begin(), this->m_dev_list.end(),
                      this->m_dev_list.begin(),
                      ShiftBitvector(shift_size));
    HANDLE_CUDA_ERROR(cudaDeviceSynchronize());
  }
  catch(thrust::system_error &e)
  {
    LOGGING_ERROR_C(VoxellistLog, BitVoxelList, "Caught Thrust exception during shiftLeftSweptVolumeIDs: " << e.what() << endl);
    exit(-1);
  }
}

struct is_not_in_swept_volume_steps
{
  is_not_in_swept_volume_steps(BitVectorVoxel* voxellist, BitVoxelMeaning min_step, BitVoxelMeaning max_step)
    : voxellist(voxellist)
    , min_step(min_step)
    , max_step(max_step)
  {
  }

  __device__
  bool operator()(size_t voxel_listindex)
  {
    BitVectorVoxel& voxel = voxellist[voxel_listindex];
    for (uint32_t meaning = min_step; meaning <= max_step; meaning++)
    {
      if (voxel.bitVector().getBit(meaning))
      {
        return false;
      }
    }
    return true;
  }

  BitVectorVoxel* voxellist;
  BitVoxelMeaning min_step;
  BitVoxelMeaning max_step;
};

struct listindex_to_coordinate
{
  listindex_to_coordinate(Vector3ui* coordinates)
    : coordinates(coordinates)
  {
  }

  __device__
  Vector3ui operator()(size_t listindex)
  {
    return coordinates[listindex];
  }

  Vector3ui* coordinates;
};

template<std::size_t length, class VoxelIDType>
void BitVoxelList<length, VoxelIDType>::copyCoordsToHostBvmBounded(std::vector<Vector3ui>& host_vec, BitVoxelMeaning min_step, BitVoxelMeaning max_step)
{
  thrust::device_vector<size_t> list_indices(this->m_dev_list.size());

  // Generate list index sequence from 1..n
  thrust::sequence(list_indices.begin(), list_indices.end());

  // Remove all indices whose voxels are not in swept volume steps
  size_t new_size = thrust::remove_if(list_indices.begin(), list_indices.end(), is_not_in_swept_volume_steps(this->m_dev_list.data().get(), min_step, max_step)) - list_indices.begin();

  // Transform index list to coordinate list
  thrust::device_vector<Vector3ui> coordinates(new_size);
  thrust::transform(list_indices.begin(), list_indices.begin() + new_size, coordinates.begin(), listindex_to_coordinate(this->m_dev_coord_list.data().get()));
  
  // Copy to host
  host_vec.resize(new_size);
  thrust::copy(coordinates.begin(), coordinates.end(), host_vec.begin());
}

} // end namespace voxellist
} // end namespace gpu_voxels

#endif // GPU_VOXELS_VOXELLIST_BITVOXELLIST_HPP_INCLUDED
