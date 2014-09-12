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
 * \author  Florian Drews
 * \date    2014-07-08
 *
 */
//----------------------------------------------------------------------/*
#ifndef GPU_VOXELS_VOXELMAP_BIT_VOXEL_H_INCLUDED
#define GPU_VOXELS_VOXELMAP_BIT_VOXEL_H_INCLUDED

#include <gpu_voxels/voxelmap/AbstractVoxel.h>
#include <gpu_voxels/helpers/BitVector.h>
#include <cstddef>

namespace gpu_voxels {
namespace voxelmap {

/**
 * @brief Voxel which represents a bit vector.
 */
template<std::size_t length>
class BitVoxel : public AbstractVoxel
{
public:
  __host__ __device__
  BitVoxel();

  __host__ __device__
  ~BitVoxel();

  __host__ __device__
  BitVector<length>& bitVector();

  __host__ __device__
  const BitVector<length>& bitVector() const;

  __host__ __device__
  void insert(const uint32_t voxel_type);

  __host__ __device__
  static BitVoxel<length> reduce(const BitVoxel<length> voxel, const BitVoxel<length> other_voxel);

  struct reduce_op //: public thrust::binary_function<VoxelTypeFlags, VoxelTypeFlags, VoxelTypeFlags>
  {
    __host__ __device__
    BitVoxel operator()(const BitVoxel& a, const BitVoxel& b) const
    {
      BitVoxel res;
      res.bitVector() = a.bitVector() | b.bitVector();
      return res;
    }
  };

protected:
  BitVector<length> m_bit_vector;
};

} // end of ns
} // end of ns

#endif
