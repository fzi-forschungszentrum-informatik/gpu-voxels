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
* \date    2015-06-16
*
*/
//----------------------------------------------------------------------

#ifndef GPU_VOXELS_VOXEL_DEFAULT_SVCOLLIDER_HPP
#define GPU_VOXELS_VOXEL_DEFAULT_SVCOLLIDER_HPP

#include "SVCollider.h"
#include <gpu_voxels/voxel/BitVoxel.hpp>
#include <gpu_voxels/voxel/ProbabilisticVoxel.hpp>

namespace gpu_voxels {

SVCollider::SVCollider() :
    m_threshold1(100), m_threshold2(100),
    m_type_range(10)
{

}

SVCollider::SVCollider(const float coll_threshold, const size_t window_size) :
    m_threshold1(floatToProbability(coll_threshold)), m_threshold2(floatToProbability(coll_threshold)),
    m_type_range(window_size)
{

}

SVCollider::SVCollider(const Probability threshold1, const Probability threshold2, const size_t window_size) :
    m_threshold1(threshold1), m_threshold2(threshold2),
    m_type_range(window_size)
{

}

bool SVCollider::collide(const ProbabilisticVoxel& v1, const ProbabilisticVoxel& v2) const
{
  return v1.getOccupancy() >= m_threshold1 && v2.getOccupancy() > m_threshold2;
}

bool SVCollider::collide(const ProbabilisticVoxel& v1) const
{
  return v1.getOccupancy() >= m_threshold1;
}

template<std::size_t length>
__host__ __device__
bool SVCollider::collide(const ProbabilisticVoxel& v1, const BitVoxel<length>& v2) const
{
  return v1.getOccupancy() >= m_threshold1 && !v2.bitVector().noneButEmpty();
}

template<std::size_t length>
__host__ __device__
bool SVCollider::collide(const BitVoxel<length>& v1, const ProbabilisticVoxel& v2) const
{
  return collide(v2, v1);
}

//template<std::size_t length>
//__host__ __device__
//bool SVCollider::collide(const BitVoxel<length>& v1, const BitVoxel<length>& v2) const
//{
//  BitVector<length> collisions;

//  return collide(v1, v2, &collisions);
//}

template<std::size_t length>
__host__ __device__
bool SVCollider::collide(const BitVoxel<length>& v1, const BitVoxel<length>& v2,
                         BitVector<length>* collisions, const uint32_t sv_offset) const
{
  return bitMarginCollisionCheck<length>(v1.bitVector(), v2.bitVector(), collisions, m_type_range, sv_offset);
}


template<std::size_t length>
__host__ __device__
bool SVCollider::collide(const BitVoxel<length>& v1, const ProbabilisticVoxel& v2, BitVector<length>* collisions, const uint32_t sv_offset) const
{
    if((v2.getOccupancy() >= m_threshold2) && (!v1.bitVector().noneButEmpty()))
    {
        *collisions |= v1.bitVector();
        return true;
    }
    return false;
}

template<std::size_t length>
__host__ __device__
bool SVCollider::collide(const ProbabilisticVoxel& v1, const BitVoxel<length>& v2, BitVector<length>* collisions, const uint32_t sv_offset) const
{
    if((v1.getOccupancy() >= m_threshold1) && (!v2.bitVector().noneButEmpty()))
    {
        *collisions |= v2.bitVector();
        return true;
    }
    return false;
}


template<class OtherVoxel>
__host__ __device__
bool SVCollider::collide(const DistanceVoxel& v1, const OtherVoxel& v2) const {
    return false; // has no meaning
}

template<class OtherVoxel>
__host__ __device__
bool SVCollider::collide(const OtherVoxel& v1, const DistanceVoxel& v2) const {
    return false; // has no meaning
}

Probability SVCollider::floatToProbability(const float val)
{
  float tmp = (val * (float(MAX_PROBABILITY) - float(MIN_PROBABILITY))) + MIN_PROBABILITY;
  return Probability(tmp);
}

} // end of ns
#endif // GPU_VOXELS_VOXEL_DEFAULT_SVCOLLIDER_HPP
