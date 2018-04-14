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
 * WARNING: This class is copied by value into many kernels!
 * Therefore it may not have any complex member variables.
 */
//----------------------------------------------------------------------/*
#ifndef GPU_VOXELS_VOXEL_DEFAULT_COLLIDER_HPP_INCLUDED
#define GPU_VOXELS_VOXEL_DEFAULT_COLLIDER_HPP_INCLUDED

#include "DefaultCollider.h"
#include <gpu_voxels/voxel/BitVoxel.hpp>
#include <gpu_voxels/voxel/ProbabilisticVoxel.hpp>
#include <gpu_voxels/voxel/CountingVoxel.hpp>

namespace gpu_voxels {

DefaultCollider::DefaultCollider() :
    m_threshold1(100), m_threshold2(100)
{
}

DefaultCollider::DefaultCollider(const float coll_threshold) :
    m_threshold1(ProbabilisticVoxel::floatToProbability(coll_threshold)), m_threshold2(ProbabilisticVoxel::floatToProbability(coll_threshold))
{

}

DefaultCollider::DefaultCollider(const Probability threshold1, const Probability threshold2) :
    m_threshold1(threshold1), m_threshold2(threshold2)
{

}

bool DefaultCollider::collide(const ProbabilisticVoxel& v1, const ProbabilisticVoxel& v2) const
{
  return v1.getOccupancy() >= m_threshold1 && v2.getOccupancy() >= m_threshold2;
}

bool DefaultCollider::collide(const ProbabilisticVoxel& v1) const
{
  return v1.getOccupancy() >= m_threshold1;
}

template<std::size_t length>
__host__ __device__
bool DefaultCollider::collide(const ProbabilisticVoxel& v1, const BitVoxel<length>& v2) const
{
  return v1.getOccupancy() >= m_threshold1 && !(v2.bitVector().noneButEmpty());
}

template<std::size_t length>
__host__ __device__
bool DefaultCollider::collide(const BitVoxel<length>& v1, const ProbabilisticVoxel& v2) const
{
  return v2.getOccupancy() >= m_threshold2 && !(v1.bitVector().noneButEmpty());
}

template<std::size_t length>
__host__ __device__
bool DefaultCollider::collide(const BitVoxel<length>& v1, const BitVoxel<length>& v2) const
{
  return !(v1.bitVector().noneButEmpty()) && !(v2.bitVector().noneButEmpty());
}

template<class OtherVoxel>
__host__ __device__
bool DefaultCollider::collide(const DistanceVoxel& v1, const OtherVoxel& v2) const {
    return false; // has no meaning
}

template<class OtherVoxel>
__host__ __device__
bool DefaultCollider::collide(const OtherVoxel& v1, const DistanceVoxel& v2) const {
    return false; // has no meaning
}

} // end of ns

#endif
