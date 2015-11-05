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
#ifndef GPU_VOXELS_VOXEL_BIT_VOXEL_HPP_INCLUDED
#define GPU_VOXELS_VOXEL_BIT_VOXEL_HPP_INCLUDED

#include "BitVoxel.h"
#include <gpu_voxels/helpers/BitVector.h>

namespace gpu_voxels {

template<std::size_t length>
__host__ __device__
BitVoxel<length>::BitVoxel() :
    m_bit_vector()
{

}

template<std::size_t length>
__host__ __device__
BitVoxel<length>::~BitVoxel()
{

}

template<std::size_t length>
__host__ __device__
BitVector<length>& BitVoxel<length>::bitVector()
{
  return m_bit_vector;
}

template<std::size_t length>
__host__ __device__
const BitVector<length>& BitVoxel<length>::bitVector() const
{
  return m_bit_vector;
}

template<std::size_t length>
__host__ __device__
void BitVoxel<length>::insert(const BitVoxelMeaning voxel_meaning)
{
  m_bit_vector.setBit(voxel_meaning);
}

template<std::size_t length>
__host__ __device__
BitVoxel<length> BitVoxel<length>::reduce(const BitVoxel<length> voxel, const BitVoxel<length> other_voxel)
{
  BitVoxel<length> res;
  BitVector<length>& b = res.bitVector();
  b = voxel.bitVector() | other_voxel.bitVector();
  return res;
}

} // end of ns

#endif
