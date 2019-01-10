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

#ifndef GPU_VOXELS_VOXELLIST_BITVOXELLIST_H
#define GPU_VOXELS_VOXELLIST_BITVOXELLIST_H

#include <gpu_voxels/voxel/BitVoxel.h>
#include <gpu_voxels/voxel/SVCollider.h>
#include <gpu_voxels/voxellist/TemplateVoxelList.h>
#include <gpu_voxels/voxellist/CountingVoxelList.h>
#include <cstddef>

namespace gpu_voxels {
namespace voxellist {

/*!
 * \brief The BitvectorCollision struct
 * Thrust operator that does an AND operation between two bitvector voxels and checks, if any bits are set
 */
struct BitvectorCollision : public thrust::binary_function<BitVectorVoxel,BitVectorVoxel,bool >
{
  __host__ __device__
  bool operator()(const BitVectorVoxel &lhs, const BitVectorVoxel &rhs) const
  {
    BitVector<BIT_VECTOR_LENGTH> both_set = lhs.bitVector() & rhs.bitVector();
    return !both_set.isZero();
  }
};


/*!
 * \brief The BitvectorCollisionWithBitshift struct
 * Same as BitvectorCollision but uses the slower variant that also evaluates a margin of bits around
 * the checked bit while doing the AND operation.
 */
struct BitvectorCollisionWithBitshift : public thrust::binary_function<BitVectorVoxel,BitVectorVoxel,bool >
{
  u_int8_t bit_margin;
  uint32_t sv_offset;

  BitvectorCollisionWithBitshift(u_int8_t bit_margin_, uint32_t sv_offset_)
  {
    bit_margin = bit_margin_;
    sv_offset = sv_offset_;
  }

  __host__ __device__
  bool operator()(const BitVectorVoxel &lhs, const BitVectorVoxel &rhs)
  {
    BitVector<BIT_VECTOR_LENGTH> collision_result; // TODO: Get rid of this temp variable
    return bitMarginCollisionCheck<BIT_VECTOR_LENGTH>(lhs.bitVector(), rhs.bitVector(), &collision_result, bit_margin, sv_offset);
  }
};

/*!
 * \brief The BitvectorOr struct
 * Thrust operator that calculated the OR operation on two BitVectorVoxels
 */
struct BitvectorOr : public thrust::binary_function<BitVectorVoxel,BitVectorVoxel,BitVector<BIT_VECTOR_LENGTH> >
{
  __host__ __device__
  BitVector<BIT_VECTOR_LENGTH> operator()(const BitVectorVoxel &lhs, const BitVectorVoxel &rhs) const
  {
    return lhs.bitVector() | rhs.bitVector();
  }
};



struct ShiftBitvector : public thrust::unary_function<BitVectorVoxel,BitVectorVoxel>
{
  uint8_t shift_size;

  ShiftBitvector(uint8_t shift_size_)
  {
    shift_size = shift_size_;
  }

  __host__ __device__
  BitVectorVoxel operator()(const BitVectorVoxel &input_voxel) const
  {
    BitVectorVoxel ret(input_voxel);
    performLeftShift(ret.bitVector(), shift_size);
    return ret;
  }
};


template<std::size_t length, class VoxelIDType>
class BitVoxelList : public TemplateVoxelList<BitVectorVoxel, VoxelIDType>,
    public CollidableWithBitVectorVoxelMap, public CollidableWithBitVectorVoxelList, public CollidableWithProbVoxelMap,
    public CollidableWithTypesBitVectorVoxelList, public CollidableWithTypesProbVoxelMap,
    public CollidableWithTypesBitVectorVoxelMap,
    public CollidableWithBitcheckBitVectorVoxelList
{
public:

  // This can either represent a MORTON or Voxelmap Bitvector Voxel List:
  typedef BitVoxelList<BIT_VECTOR_LENGTH, VoxelIDType> TemplatedBitVectorVoxelList;

  BitVoxelList(const Vector3ui ref_map_dim, const float voxel_side_length, const MapType map_type);

  virtual ~BitVoxelList();

//  virtual void clearBit(const uint32_t bit_index);

//  virtual void clearBits(BitVector<length> bits);

//  template<class Collider>
//  BitVector<length> collisionCheckBitvector(ProbVoxelMap* other, Collider collider);

  virtual void clearBitVoxelMeaning(BitVoxelMeaning voxel_meaning);

  virtual MapType getTemplateType() { return this->m_map_type; }

  //Collision Interface
  size_t collideWith(const voxelmap::ProbVoxelMap* map, float coll_threshold = 1.0, const Vector3i &offset = Vector3i());
  size_t collideWith(const voxelmap::BitVectorVoxelMap* map, float coll_threshold = 1.0, const Vector3i &offset = Vector3i());
  size_t collideWith(const voxellist::BitVectorVoxelList* map, float coll_threshold = 1.0, const Vector3i &offset = Vector3i());
  size_t collideWithTypes(const voxelmap::ProbVoxelMap* map, BitVectorVoxel& types_in_collision, float coll_threshold = 1.0, const Vector3i &offset = Vector3i());
  size_t collideWithTypes(const voxellist::BitVectorVoxelList* map, BitVectorVoxel& types_in_collision, float coll_threshold = 1.0, const Vector3i &offset = Vector3i());
  size_t collideWithTypes(const voxelmap::BitVectorVoxelMap *map, BitVectorVoxel &types_in_collision, float coll_threshold = 1.0, const Vector3i &offset = Vector3i());
  template< class Voxel>
  size_t collideWithTypeMask(const voxelmap::TemplateVoxelMap<Voxel> *map, const BitVectorVoxel& types_to_check, float coll_threshold = 1.0, const Vector3i &offset = Vector3i());
  size_t collideWithBitcheck(const voxellist::BitVectorVoxelList* map, const u_int8_t margin = 0, const Vector3i &offset = Vector3i());

  size_t collideCountingPerMeaning(const GpuVoxelsMapSharedPtr other, std::vector<size_t>&  collisions_per_meaning, const Vector3i &offset_ = Vector3i());
  
  /**
   * @brief findMatchingVoxels 
   * \param other Const second input list, first is this
   * \param margin as in collideWithBitcheck, currently ignored
   * \param offset
   * \param matching_voxels_list1 Contains all Voxels from this whose position matches a Voxel from other
   * \param matching_voxels_list2 Contains all Voxels from other whose position matches a Voxel from this
   * \param omit_coords Controls whether the matching_voxels_lists will have an empty m_dev_coord_list
   */
  void findMatchingVoxels(const TemplatedBitVectorVoxelList *other,
                          const u_int8_t margin, const Vector3i &offset,
                          TemplatedBitVectorVoxelList* matching_voxels_list1, 
                          TemplatedBitVectorVoxelList* matching_voxels_list2, 
                          bool omit_coords = true) const;

  /**
   * @brief findMatchingVoxels 
   * \param other Const second input list, first is this
   * \param offset
   * \param matching_voxels_list Contains all Voxels from this whose position matches a Voxel from other
   * \param omit_coords Controls whether the matching_voxels_list will have an empty m_dev_coord_list
   */
  void findMatchingVoxels(const CountingVoxelList *other,
                          const Vector3i &offset, 
                          TemplatedBitVectorVoxelList* matching_voxels_list, 
                          bool omit_coords = true) const;

  /**
   * @brief Shifts all swept-volume-IDs by shift_size towards lower IDs.
   * Currently this is limited to a shift size <64
   * @param shift_size Shift size of bitshift
   */
  void shiftLeftSweptVolumeIDs(uint8_t shift_size);
  
  virtual void copyCoordsToHostBvmBounded(std::vector<Vector3ui>& host_vec, BitVoxelMeaning min_step, BitVoxelMeaning max_step);

protected:
//  virtual void clearVoxelMapRemoteLock(const uint32_t bit_index);

private:

  thrust::device_vector< BitVectorVoxel > m_dev_colliding_bits_result_list;
  thrust::host_vector< BitVectorVoxel > m_colliding_bits_result_list;
  BitVectorVoxel* m_dev_bitmask;


};

} // end namespace voxellist
} // end namespace gpu_voxels

#endif // GPU_VOXELS_VOXELLIST_BITVOXELLIST_H
