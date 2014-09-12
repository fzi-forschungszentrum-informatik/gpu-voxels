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
 * \date    2014-07-09
 *
 */
//----------------------------------------------------------------------/*
#ifndef GPU_VOXELS_VOXELMAP_DEFAULT_COLLIDER_HPP_INCLUDED
#define GPU_VOXELS_VOXELMAP_DEFAULT_COLLIDER_HPP_INCLUDED

#include "DefaultCollider.h"
#include <gpu_voxels/voxelmap/BitVoxel.hpp>

namespace gpu_voxels {
namespace voxelmap {

DefaultCollider::DefaultCollider() :
    m_threshold1(100), m_threshold2(100)
{

}

DefaultCollider::DefaultCollider(const float coll_threshold) :
    m_threshold1(floatToProbability(coll_threshold)), m_threshold2(floatToProbability(coll_threshold))
{

}

DefaultCollider::DefaultCollider(const probability threshold1, const probability threshold2) :
    m_threshold1(threshold1), m_threshold2(threshold2)
{

}

bool DefaultCollider::collide(const ProbabilisticVoxel& v1, const ProbabilisticVoxel& v2) const
{
  return v1.getOccupancy() >= m_threshold1 && v2.getOccupancy() > m_threshold2;
}

bool DefaultCollider::collide(const ProbabilisticVoxel& v1) const
{
  return v1.getOccupancy() >= m_threshold1;
}

template<std::size_t length>
__host__ __device__
bool DefaultCollider::collide(const ProbabilisticVoxel& v1, const BitVoxel<length>& v2) const
{
  return v1.getOccupancy() >= m_threshold1 && !v2.bitVector().isZero();
}

template<std::size_t length>
__host__ __device__
bool DefaultCollider::collide(const BitVoxel<length>& v1, const ProbabilisticVoxel& v2) const
{
  return collide(v2, v1);
}

template<std::size_t length>
__host__ __device__
bool DefaultCollider::collide(const BitVoxel<length>& v1, const BitVoxel<length>& v2) const
{
  return !v1.bitVector().isZero() && !v2.bitVector().isZero();
}

probability DefaultCollider::floatToProbability(const float val)
{
  float tmp = (val * (float(MAX_PROBABILITY) - float(MIN_PROBABILITY))) + MIN_PROBABILITY;
  return probability(tmp);
}

} // end of ns
} // end of ns

#endif
