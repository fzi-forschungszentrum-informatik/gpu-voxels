// this is for emacs file handling -*- mode: c++; indent-tabs-mode: nil -*-

// -- BEGIN LICENSE BLOCK ----------------------------------------------
// -- END LICENSE BLOCK ------------------------------------------------

//----------------------------------------------------------------------
/*!\file
*
* \author  Felix Mauch <mauch@fzi.de>
* \date    2015-6-16
*
*/
//----------------------------------------------------------------------

#ifndef GPU_VOXELS_VOXEL_SVCOLLIDER_H
#define GPU_VOXELS_VOXEL_SVCOLLIDER_H

#include <gpu_voxels/voxel/BitVoxel.h>
#include <gpu_voxels/voxel/ProbabilisticVoxel.h>

namespace gpu_voxels {


class SVCollider
{
public:
  __host__ __device__
  SVCollider();

  __host__ __device__
  SVCollider(const float coll_threshold, const size_t window_size = 0);

  __host__ __device__
  SVCollider(const probability threshold1, const probability threshold2, const size_t window_size = 0);

  __host__ __device__
  bool collide(const ProbabilisticVoxel& v1, const ProbabilisticVoxel& v2) const;

  __host__ __device__
  bool collide(const ProbabilisticVoxel& v1) const;

  template<std::size_t length>
  __host__ __device__
  bool collide(const ProbabilisticVoxel& v1, const BitVoxel<length>& v2) const;

  template<std::size_t length>
  __host__ __device__
  bool collide(const BitVoxel<length>& v1, const ProbabilisticVoxel& v2) const;

  template<std::size_t length>
  __host__ __device__
  bool collide(const BitVoxel<length>& v1, const BitVoxel<length>& v2) const;

  template<std::size_t length>
  __host__ __device__
  bool collide(const BitVoxel<length>& v1, const BitVoxel<length>& v2, BitVector<length>* collisions, const uint32_t sv_offset) const;

protected:
  __host__ __device__
  static probability floatToProbability(const float val);

protected:
  probability m_threshold1, m_threshold2;
  size_t m_type_range;
};

} // end of ns

#endif // GPU_VOXELS_VOXELMAP_SVCOLLIDER_H
