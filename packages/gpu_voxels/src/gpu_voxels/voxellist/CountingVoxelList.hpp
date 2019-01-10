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
* \author  Christian Juelg <juelg@fzi.de>
* \date    2017-10-10
*
*/
//----------------------------------------------------------------------

#ifndef GPU_VOXELS_VOXELLIST_COUNTINGVOXELLIST_HPP_INCLUDED
#define GPU_VOXELS_VOXELLIST_COUNTINGVOXELLIST_HPP_INCLUDED

#include "CountingVoxelList.h"
#include <gpu_voxels/logging/logging_voxellist.h>

#include <thrust/device_vector.h>
#include <thrust/transform.h>
#include <thrust/remove.h>
#include <thrust/system_error.h>

namespace gpu_voxels {
namespace voxellist {
// using namespace gpu_voxels::voxelmap;


CountingVoxelList::CountingVoxelList(const Vector3ui ref_map_dim,
                                     const float voxel_side_length,
                                     const MapType map_type)
  : TemplateVoxelList<CountingVoxel, MapVoxelID>(ref_map_dim, voxel_side_length, map_type)
{
  // We already resize the result vector for Bitvector Checks
  // m_dev_colliding_bits_result_list.resize(cMAX_NR_OF_BLOCKS);
  // m_colliding_bits_result_list.resize(cMAX_NR_OF_BLOCKS);

  // Allocate a BitVectorVoxel on the device to it use as bitmask for later coll-checks.
  // cudaMalloc(&m_dev_bitmask, sizeof(CountingVoxel));

  //TODO: check memory allocation for bitmask, colliding_bits, dev_colliding_bits
}


CountingVoxelList::~CountingVoxelList()
{
}


void CountingVoxelList::clearBitVoxelMeaning(BitVoxelMeaning voxel_meaning)
{
  LOGGING_ERROR_C(
    VoxellistLog, CountingVoxelList, GPU_VOXELS_MAP_OPERATION_NOT_YET_SUPPORTED << endl);
}

struct is_collision_candidate
{
  int8_t threshold;
  is_collision_candidate(int8_t th) : threshold(th) {}

  __host__ __device__
  bool operator()(CountingVoxel v) const
  {
    return v.getCount() >= threshold;
  }
};

struct is_underpopulated
{
  int8_t threshold;
  is_underpopulated(int8_t th) : threshold(th) {}

  __host__ __device__
  bool operator()(thrust::tuple<MapVoxelID, Vector3ui, CountingVoxel> triple_it) const
  {
    CountingVoxel cv = thrust::get<2>(triple_it);
    //printf("voxel count was: %d, id: %d\n", cv.getCount(), thrust::get<0>(triple_it));
    return cv.getCount() < threshold;
  }
};

size_t CountingVoxelList::collideWith(const voxellist::BitVectorVoxelList* map, float coll_threshold, const Vector3i &offset)
{
  size_t collisions = SSIZE_MAX;

  boost::lock(this->m_mutex, map->m_mutex);
  lock_guard guard(this->m_mutex, boost::adopt_lock);
  lock_guard guard2(map->m_mutex, boost::adopt_lock);

  thrust::device_vector<bool> collision_stencil(this->m_dev_id_list.size()); // Temporary data structure

  //after transform the collision_stencil will have a true in every element that should be considered for collision checking
  is_collision_candidate filter(static_cast<int>(coll_threshold));
  thrust::transform(this->m_dev_list.begin(), this->m_dev_list.end(), collision_stencil.begin(), filter);

  collisions = this->collideVoxellists(map, offset, collision_stencil);
  return collisions;
}

void CountingVoxelList::remove_underpopulated(const int8_t threshold)
{
  //this->screendump(true); // DEBUG

  lock_guard guard(this->m_mutex);

  // find the overlapping voxels:

  keyCoordVoxelZipIterator new_end;

  is_underpopulated filter(threshold);

  // remove voxels below threshold
  new_end = thrust::remove_if(this->getBeginTripleZipIterator(),
                              this->getEndTripleZipIterator(),
                              filter);

  size_t new_length = thrust::distance(m_dev_id_list.begin(), thrust::get<0>(new_end.get_iterator_tuple()));
  this->resize(new_length);

  //this->screendump(true); // DEBUG

  return;
}

} // end namespace voxellist
} // end namespace gpu_voxels

#endif // GPU_VOXELS_VOXELLIST_COUNTINGVOXELLIST_HPP_INCLUDED
