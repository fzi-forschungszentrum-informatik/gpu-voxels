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
 * \author  Herbert Pietrzyk
 * \date    2017-10-25
 *
 */
//----------------------------------------------------------------------/*
#ifndef GPU_VOXELS_VOXEL_COUNTING_VOXEL_HPP_INCLUDED
#define GPU_VOXELS_VOXEL_COUNTING_VOXEL_HPP_INCLUDED

#include "CountingVoxel.h"


namespace gpu_voxels {

__host__ __device__
CountingVoxel::CountingVoxel() :
    m_count(0)
{

}

__host__ __device__
CountingVoxel::~CountingVoxel()
{

}

__host__ __device__
bool CountingVoxel::isOccupied(uint8_t occ_threshold) const
{
    return m_count >= occ_threshold;
}

__host__ __device__
int8_t CountingVoxel::getCount() const
{
    return m_count;
}

__host__ __device__
int8_t& CountingVoxel::count()
{
    return m_count;
}

__host__ __device__
const int8_t& CountingVoxel::count() const
{
    return m_count;
}

__host__ __device__
void CountingVoxel::insert(const uint32_t voxel_meaning)
{
    m_count++;
}

__host__ __device__ 
CountingVoxel CountingVoxel::reduce(const CountingVoxel voxel, const CountingVoxel other_voxel)
{
    CountingVoxel res = voxel;
    res.m_count += other_voxel.m_count;
    return res;
}

} // end of ns

#endif
