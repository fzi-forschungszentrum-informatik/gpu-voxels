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

#ifndef GPU_VOXELS_VOXELLIST_COUNTINGVOXELLIST_H
#define GPU_VOXELS_VOXELLIST_COUNTINGVOXELLIST_H

#include <gpu_voxels/voxel/CountingVoxel.h>
//#include <gpu_voxels/voxel/SVCollider.h>
#include <gpu_voxels/voxellist/TemplateVoxelList.h>
#include <gpu_voxels/voxellist/BitVoxelList.h>
#include <cstddef>

namespace gpu_voxels {
namespace voxellist {


class CountingVoxelList : public TemplateVoxelList<CountingVoxel, MapVoxelID>,
    public CollidableWithBitVectorVoxelList
{
public:
  // This can either represent a MORTON or Voxelmap Bitvector Voxel List:
  //  typedef CountingVoxelList<VoxelIDType> TemplatedCountingVoxelList;

  CountingVoxelList(const gpu_voxels::Vector3ui ref_map_dim,
                    const float voxel_sidelength,
                    const gpu_voxels::MapType map_type);

  ~CountingVoxelList();

  virtual void clearBitVoxelMeaning(BitVoxelMeaning voxel_meaning);

  virtual MapType getTemplateType() { return this->m_map_type; }

  size_t collideWith(const voxellist::BitVectorVoxelList* map, float coll_threshold = 1.0, const Vector3i &offset = Vector3i());

  void remove_underpopulated(const int8_t threshold);

private:
  //  /**
  //   * @brief findMatchingVoxels Assure to lock the input maps before calling this function.
  //   * \param list1 Const input
  //   * \param list2 Const input
  //   * \param margin
  //   * \param offset
  //   * \param matching_voxels_list1 Contains all Voxels from list1 whose position matches a Voxel
  //   from
  //   * list2
  //   * \param matching_voxels_list2 Contains all Voxels from list2 whose position matches a Voxel
  //   from
  //   * list1
  //   */
  //  void findMatchingVoxels(const TemplatedBitVectorVoxelList* list1,
  //                          const TemplatedBitVectorVoxelList* list2,
  //                          const u_int8_t margin,
  //                          const Vector3i& offset,
  //                          TemplatedBitVectorVoxelList* matching_voxels_list1,
  //                          TemplatedBitVectorVoxelList* matching_voxels_list2) const;

  // thrust::device_vector<CountingVoxel> m_dev_colliding_bits_result_list;
  // thrust::host_vector<CountingVoxel> m_colliding_bits_result_list;
  // CountingVoxel* m_dev_bitmask;
};

} // end namespace voxellist
} // end namespace gpu_voxels

#endif // GPU_VOXELS_VOXELLIST_COUNTINGVOXELLIST_H
