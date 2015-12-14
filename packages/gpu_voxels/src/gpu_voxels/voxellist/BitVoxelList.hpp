// this is for emacs file handling -*- mode: c++; indent-tabs-mode: nil -*-

// -- BEGIN LICENSE BLOCK ----------------------------------------------
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
BitVoxelList<length, VoxelIDType>::BitVoxelList(const Vector3ui ref_map_dim, const float voxel_sidelength, const MapType map_type)
  : TemplateVoxelList<BitVectorVoxel, VoxelIDType>(ref_map_dim, voxel_sidelength, map_type)
{
  // We already resize the result vector for Bitvector Checks
  m_dev_colliding_bits_result_list.resize(cMAX_NR_OF_BLOCKS);
  m_colliding_bits_result_list.resize(cMAX_NR_OF_BLOCKS);
}


template<std::size_t length, class VoxelIDType>
BitVoxelList<length, VoxelIDType>::~BitVoxelList()
{
}

template<std::size_t length, class VoxelIDType>
bool BitVoxelList<length, VoxelIDType>::insertRobotConfiguration(const MetaPointCloud *robot_links, bool with_self_collision_test)
{
  LOGGING_ERROR_C(VoxellistLog, BitVoxelList, GPU_VOXELS_MAP_OPERATION_NOT_YET_SUPPORTED << endl);
  return false;
}

template<std::size_t length, class VoxelIDType>
void BitVoxelList<length, VoxelIDType>::clearBitVoxelMeaning(BitVoxelMeaning voxel_meaning)
{
  LOGGING_ERROR_C(VoxellistLog, BitVoxelList, GPU_VOXELS_MAP_OPERATION_NOT_YET_SUPPORTED << endl);
}

template<std::size_t length, class VoxelIDType>
size_t BitVoxelList<length, VoxelIDType>::collideWithBitcheck(const GpuVoxelsMapSharedPtr other_, const u_int8_t margin, const Vector3ui &offset)
{
  // Map locking for the lists is performed in "findMatchingVoxels"
  // TODO: Implement locking for the maps seperately!
  try
  {
    switch (other_->getMapType())
    {
      case MT_BITVECTOR_VOXELLIST:
      {
        TemplatedBitVectorVoxelList* other = dynamic_cast<TemplatedBitVectorVoxelList*>(other_.get());

        //========== Search for Voxels at the same spot in both lists: ==============
        TemplatedBitVectorVoxelList matching_voxels_list1(this->m_ref_map_dim, this->m_voxel_side_length, this->m_map_type);
        TemplatedBitVectorVoxelList matching_voxels_list2(this->m_ref_map_dim, this->m_voxel_side_length, this->m_map_type);
        findMatchingVoxels(this, other, margin, offset, &matching_voxels_list1, &matching_voxels_list2);

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

        return thrust::count(dev_colliding_bits_list.begin(), dev_colliding_bits_list.end(), true);
      }

      case MT_BITVECTOR_OCTREE:
      case MT_BITVECTOR_VOXELMAP:
      case MT_BITVECTOR_MORTON_VOXELLIST:
      {
        LOGGING_ERROR_C(VoxellistLog, BitVoxelList, GPU_VOXELS_MAP_OPERATION_NOT_YET_SUPPORTED << endl);
        break;
      }

      default:
      {
        LOGGING_ERROR_C(VoxellistLog, BitVoxelList, GPU_VOXELS_MAP_OPERATION_NOT_SUPPORTED << endl);
        break;
      }
    }
  }
  catch(thrust::system_error &e)
  {
    LOGGING_ERROR_C(VoxellistLog, BitVoxelList, "Caught Thrust exception " << e.what() << endl);
    exit(-1);
  }
  return SSIZE_MAX;
}


template<std::size_t length, class VoxelIDType>
size_t BitVoxelList<length, VoxelIDType>::collideWithTypes(const GpuVoxelsMapSharedPtr other_,
                                                           BitVectorVoxel&  meanings_in_collision,
                                                           float coll_threshold, const Vector3ui &offset_)
{
  // Map locking for the lists is performed in "findMatchingVoxels"
  // TODO: Implement locking for the maps seperately!
  try
  {
    switch (other_->getMapType())
    {
      case MT_BITVECTOR_VOXELLIST:
      {
        TemplatedBitVectorVoxelList* other = dynamic_cast<TemplatedBitVectorVoxelList*>(other_.get());

        //========== Search for Voxels at the same spot in both lists: ==============
        TemplatedBitVectorVoxelList matching_voxels_list1(this->m_ref_map_dim, this->m_voxel_side_length, this->m_map_type);
        TemplatedBitVectorVoxelList matching_voxels_list2(this->m_ref_map_dim, this->m_voxel_side_length, this->m_map_type);
        findMatchingVoxels(this, other, 0, offset_, &matching_voxels_list1, &matching_voxels_list2);

        //========== Now iterate over both shortened lists and inspect the Bitvectors =============
        thrust::device_vector< BitVectorVoxel> dev_merged_voxel_list(matching_voxels_list1.m_dev_id_list.size());
        thrust::transform(matching_voxels_list1.m_dev_list.begin(), matching_voxels_list1.m_dev_list.end(),
                          matching_voxels_list2.m_dev_list.begin(),
                          dev_merged_voxel_list.begin(), BitVectorVoxel::reduce_op());


        meanings_in_collision = thrust::reduce(dev_merged_voxel_list.begin(), dev_merged_voxel_list.end(),
                                               BitVectorVoxel(), BitVectorVoxel::reduce_op());

        return matching_voxels_list1.m_dev_id_list.size();
      }
      case MT_PROBAB_VOXELMAP:
      {
        // Map Dims have to be equal to be able to compare pointer adresses!
        if(other_->getDimensions() != this->m_ref_map_dim)
        {
          LOGGING_ERROR_C(VoxellistLog, BitVoxelList,
                          "The dimensions of the Voxellist reference map do not match the colliding voxel map dimensions. Not checking collisions!" << endl);
          return SSIZE_MAX;
        }


        ProbVoxelMap* other = dynamic_cast<voxellist::ProbVoxelMap*>(other_.get());

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
                                                                        offset_, this->m_dev_collision_check_results_counter, m_dev_colliding_bits_result_list_ptr);
        HANDLE_CUDA_ERROR(cudaDeviceSynchronize());

        // Copy back the results and reduce the block results:
        this->m_colliding_bits_result_list = this->m_dev_colliding_bits_result_list;
        HANDLE_CUDA_ERROR(
            cudaMemcpy(this->m_collision_check_results_counter, this->m_dev_collision_check_results_counter,
                       num_blocks * sizeof(uint16_t), cudaMemcpyDeviceToHost));

        size_t number_of_collisions = 0;
        meanings_in_collision.bitVector().clear();
        for (uint32_t i = 0; i < num_blocks; i++)
        {
          number_of_collisions += this->m_collision_check_results_counter[i];
          meanings_in_collision.bitVector() |= this->m_colliding_bits_result_list[i].bitVector();
        }

        return number_of_collisions;
      }
      case MT_BITVECTOR_OCTREE:
      case MT_BITVECTOR_VOXELMAP:
      {
        LOGGING_ERROR_C(VoxellistLog, BitVoxelList, GPU_VOXELS_MAP_SWAP_FOR_COLLIDE << endl);
        return SSIZE_MAX;
      }
      case MT_BITVECTOR_MORTON_VOXELLIST:
      {
        LOGGING_ERROR_C(VoxellistLog, BitVoxelList, GPU_VOXELS_MAP_OPERATION_NOT_YET_SUPPORTED << endl);
        return SSIZE_MAX;
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
size_t BitVoxelList<length, VoxelIDType>::collideCountingPerMeaning(const GpuVoxelsMapSharedPtr other_,
                                                           std::vector<size_t>&  collisions_per_meaning,
                                                           const Vector3ui &offset_)
{
  // Map locking for the lists is performed in "findMatchingVoxels"
  // TODO: Implement locking for the maps seperately!
  try
  {
    switch (other_->getMapType())
    {
      case MT_BITVECTOR_VOXELLIST:
      {
        TemplatedBitVectorVoxelList* other = dynamic_cast<TemplatedBitVectorVoxelList*>(other_.get());

        //========== Search for Voxels at the same spot in both lists: ==============
        TemplatedBitVectorVoxelList matching_voxels_list1(this->m_ref_map_dim, this->m_voxel_side_length, this->m_map_type);
        TemplatedBitVectorVoxelList matching_voxels_list2(this->m_ref_map_dim, this->m_voxel_side_length, this->m_map_type);
        findMatchingVoxels(this, other, 0, offset_, &matching_voxels_list1, &matching_voxels_list2);

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
              summed_colls ++;
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

/*!
 * \brief BitVoxelList<length, VoxelIDType>::findMatchingVoxels
 * \param list1 Const input
 * \param list2 Const input
 * \param margin
 * \param offset
 * \param matching_voxels_list1 Contains all Voxels from list1 whose position matches a Voxel from list2
 * \param matching_voxels_list2 Contains all Voxels from list2 whose position matches a Voxel from list1
 */
template<std::size_t length, class VoxelIDType>
void BitVoxelList<length, VoxelIDType>::findMatchingVoxels(TemplatedBitVectorVoxelList* list1, TemplatedBitVectorVoxelList* list2,
                                              const u_int8_t margin, const Vector3ui &offset,
                                              TemplatedBitVectorVoxelList* matching_voxels_list1, TemplatedBitVectorVoxelList* matching_voxels_list2)
{
  bool locked_list_1 = false;
  bool locked_list_2 = false;
  uint32_t counter = 0;

  while (!locked_list_1 && !locked_list_2)
  {
    // lock mutexes
    while (!locked_list_1)
    {
      locked_list_1 = list1->lockMutex();
      if(!locked_list_1) boost::this_thread::yield();
    }
    while (!locked_list_2 && (counter < 50))
    {
      locked_list_2 = list2->lockMutex();
      if(!locked_list_2) boost::this_thread::yield();
      counter++;
    }
    if (!locked_list_2)
    {
      LOGGING_WARNING_C(VoxellistLog, BitVoxelList, "Could not lock second list since 50 trials!" << endl);
      counter = 0;
      list1->unlockMutex();
      boost::this_thread::yield();
    }
  }



  //std::cout << "List1: ";
  //for(size_t i = 0; i < list1->m_dev_id_list.size(); i++)
  //    std::cout << list1->m_dev_id_list[i] << " ";
  //std::cout << std::endl;

  //std::cout << "List2: ";
  //for(size_t i = 0; i < list2->m_dev_id_list.size(); i++)
  //    std::cout << list2->m_dev_id_list[i] << " ";
  //std::cout << std::endl;

  size_t num_hits1 = 0;
  size_t num_hits2 = 0;
  thrust::device_vector<bool> output;

  //========== First search: Search for list1 in list2. =================================================
  try
  {
    output = thrust::device_vector<bool>(list2->m_dev_id_list.size());
    // if offset is given, we need our own comparison opperator!
    if(offset != Vector3ui(0))
    {
      LOGGING_WARNING_C(VoxellistLog, BitVoxelList, "Offset for VoxelList collision was given. Thrust performace is not optimal with that!" << endl);
      thrust::binary_search(thrust::device,
                            list1->m_dev_id_list.begin(), list1->m_dev_id_list.end(),
                            list2->m_dev_id_list.begin(), list2->m_dev_id_list.end(),
                            output.begin(), voxellist::offsetLessOperator<VoxelIDType>(this->m_ref_map_dim, offset));
    }else{
      //todo: tests what performs better: shorter vec search in longer vec or vice versa
      thrust::binary_search(thrust::device,
                            list1->m_dev_id_list.begin(), list1->m_dev_id_list.end(),
                            list2->m_dev_id_list.begin(), list2->m_dev_id_list.end(),
                            output.begin());
    }
  }
  catch(thrust::system_error &e)
  {
    LOGGING_ERROR_C(VoxellistLog, BitVoxelList, "Caught Thrust exception " << e.what() << endl);
    exit(-1);
  }

  //========== Shorten list2 to those entries found in this list.
  try
  {
    num_hits1 = thrust::count(output.begin(), output.end(), true);
    matching_voxels_list2->m_dev_id_list.resize(num_hits1);
    matching_voxels_list2->m_dev_list.resize(num_hits1);

    // we use the "output" list as stencil
    thrust::copy_if( thrust::make_zip_iterator(thrust::make_tuple(list2->m_dev_id_list.begin(), list2->m_dev_list.begin())),
                     thrust::make_zip_iterator(thrust::make_tuple(list2->m_dev_id_list.end(),   list2->m_dev_list.end())),
                     output.begin(),
                     thrust::make_zip_iterator(thrust::make_tuple(matching_voxels_list2->m_dev_id_list.begin(), matching_voxels_list2->m_dev_list.begin())),
                     thrust::identity<bool>());
  }
  catch(thrust::system_error &e)
  {
    LOGGING_ERROR_C(VoxellistLog, BitVoxelList, "Caught Thrust exception " << e.what() << endl);
    exit(-1);
  }

  //========== Second search: Search for shortened list2 in this list. =================================================
  try
  {
    output = thrust::device_vector<bool>(list1->m_dev_id_list.size()); // the result as vector of bools
    // if offset is given, we need our own comparison opperator!
    if(offset != Vector3ui(0))
    {
      LOGGING_WARNING_C(VoxellistLog, BitVoxelList, "Offset for VoxelList collision was given. Thrust performace is not optimal with that!" << endl);
      thrust::binary_search(thrust::device,
                            matching_voxels_list2->m_dev_id_list.begin(), matching_voxels_list2->m_dev_id_list.end(),
                            list1->m_dev_id_list.begin(), list1->m_dev_id_list.end(),
                            output.begin(), voxellist::offsetLessOperator<VoxelIDType>(this->m_ref_map_dim, offset));
    }else{
      //todo: tests what performs better: shorter vec search in longer vec or vice versa
      thrust::binary_search(thrust::device,
                            matching_voxels_list2->m_dev_id_list.begin(), matching_voxels_list2->m_dev_id_list.end(),
                            list1->m_dev_id_list.begin(), list1->m_dev_id_list.end(),
                            output.begin());
    }

    //========== Shorten list1 to those entries found in shortened list2.
    num_hits2 = thrust::count(output.begin(), output.end(), true);
    matching_voxels_list1->m_dev_id_list.resize(num_hits2);
    matching_voxels_list1->m_dev_list.resize(num_hits2);

    // we use the "output" list as stencil
    thrust::copy_if( thrust::make_zip_iterator(thrust::make_tuple(list1->m_dev_id_list.begin(), list1->m_dev_list.begin())),
                     thrust::make_zip_iterator(thrust::make_tuple(list1->m_dev_id_list.end(),   list1->m_dev_list.end())),
                     output.begin(),
                     thrust::make_zip_iterator(thrust::make_tuple(matching_voxels_list1->m_dev_id_list.begin(), matching_voxels_list1->m_dev_list.begin())),
                     thrust::identity<bool>());

  }
  catch(thrust::system_error &e)
  {
    LOGGING_ERROR_C(VoxellistLog, BitVoxelList, "Caught Thrust exception " << e.what() << endl);
    exit(-1);
  }

  list1->unlockMutex();
  list2->unlockMutex();

  //for(size_t i = 0; i < num_hits1; i++)
  //{
  //  std::cout << "Keys: " << matching_voxels_list1->m_dev_id_list[i] << " | " << matching_voxels_list1->m_dev_id_list[i] << std::endl;
  //  std::cout << matching_voxels_list1->m_dev_list[i] << std::endl;
  //  std::cout << matching_voxels_list2->m_dev_list[i] << std::endl;
  //}
  assert(num_hits1 == num_hits2);
}


template<std::size_t length, class VoxelIDType>
void BitVoxelList<length, VoxelIDType>::shiftLeftSweptVolumeIDs(uint8_t shift_size)
{
  if (shift_size > 63)
  {
    LOGGING_ERROR_C(VoxellistLog, BitVoxelList, "Maximum shift size is 63! Higher shift number requested. Not performing shift operation." << endl);
    return;
  }
  size_t counter = 1;
  while (!this->lockMutex())
  {
    boost::this_thread::yield();
    if(counter % 50 == 0)
    {
      LOGGING_WARNING_C(VoxellistLog, BitVoxelList, "Could not lock list since 50 trials!" << endl);
    }
    counter++;
  }

  try
  {
    thrust::transform(this->m_dev_list.begin(), this->m_dev_list.end(),
                      this->m_dev_list.begin(),
                      ShiftBitvector(shift_size));
  }
  catch(thrust::system_error &e)
  {
    LOGGING_ERROR_C(VoxellistLog, BitVoxelList, "Caught Thrust exception during shiftLeftSweptVolumeIDs: " << e.what() << endl);
    exit(-1);
  }

  this->unlockMutex();
}


} // end namespace voxellist
} // end namespace gpu_voxels

#endif // GPU_VOXELS_VOXELLIST_BITVOXELLIST_HPP_INCLUDED
