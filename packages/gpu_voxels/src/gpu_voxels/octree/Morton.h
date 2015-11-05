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
 * \date    2014-04-15
 *
 */
//----------------------------------------------------------------------/*
#ifndef GPU_VOXELS_OCTREE_MORTON_H_INCLUDED
#define GPU_VOXELS_OCTREE_MORTON_H_INCLUDED

#include <gpu_voxels/octree/DataTypes.h>
#include <cuda_runtime.h>
#include <time.h>

namespace gpu_voxels {
namespace NTree {

// TODO: have a look at this to speed up morton code computation: http://www.forceflow.be/2013/10/07/morton-encodingdecoding-through-bit-interleaving-implementations/

// The following definition and function is taken from FCL
/*
 * Software License Agreement (BSD License)
 *
 *  Copyright (c) 2011, Willow Garage, Inc.
 *  All rights reserved.
 *
 *  Redistribution and use in source and binary forms, with or without
 *  modification, are permitted provided that the following conditions
 *  are met:
 *
 *   * Redistributions of source code must retain the above copyright
 *     notice, this list of conditions and the following disclaimer.
 *   * Redistributions in binary form must reproduce the above
 *     copyright notice, this list of conditions and the following
 *     disclaimer in the documentation and/or other materials provided
 *     with the distribution.
 *   * Neither the name of Willow Garage, Inc. nor the names of its
 *     contributors may be used to endorse or promote products derived
 *     from this software without specific prior written permission.
 *
 *  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
 *  "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
 *  LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS
 *  FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE
 *  COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,
 *  INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
 *  BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
 *  LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
 *  CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
 *  LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN
 *  ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 *  POSSIBILITY OF SUCH DAMAGE.
 */
// function call to morton_code60() in operator() of struct transform_to_morton not possible
#define MORTON_TRAFO(VAL)           \
uint32_t lo_x = VAL x & 1023u;      \
uint32_t lo_y = VAL y & 1023u;      \
uint32_t lo_z = VAL z & 1023u;      \
uint32_t hi_x = VAL x >> 10u;       \
uint32_t hi_y = VAL y >> 10u;       \
uint32_t hi_z = VAL z >> 10u;       \
return (uint64_t(morton_code(hi_x, hi_y, hi_z)) << 30) | uint64_t(morton_code(lo_x, lo_y, lo_z));

__host__  __device__
 static __forceinline__ uint32_t morton_code(uint32_t x, uint32_t y, uint32_t z)
{
  x = (x | (x << 16)) & 0x030000FF;
  x = (x | (x << 8))  & 0x0300F00F;
  x = (x | (x << 4))  & 0x030C30C3;
  x = (x | (x << 2))  & 0x09249249;

  y = (y | (y << 16)) & 0x030000FF;
  y = (y | (y << 8))  & 0x0300F00F;
  y = (y | (y << 4))  & 0x030C30C3;
  y = (y | (y << 2))  & 0x09249249;

  z = (z | (z << 16)) & 0x030000FF;
  z = (z | (z << 8))  & 0x0300F00F;
  z = (z | (z << 4))  & 0x030C30C3;
  z = (z | (z << 2))  & 0x09249249;

  return x | (y << 1) | (z << 2);
}
// #######################################################################################################

__host__  __device__
 static __forceinline__ uint16_t morton_code12(uint16_t x, uint16_t y, uint16_t z)
{
  x = (x | (x << 4)) & 0xC30C3;
  x = (x | (x << 2)) & 0x9249;

  y = (y | (y << 4)) & 0x30C3;
  y = (y | (y << 2)) & 0x9249;

  z = (z | (z << 4)) & 0x30C3;
  z = (z | (z << 2)) & 0x9249;

  return x | (y << 1) | (z << 2);
}

//__host__ __device__
//static __forceinline__ uint64_t morton_code60___(uint64_t x, uint64_t y, uint64_t z)
//{
//  uint64_t lo_x = x & 1023u;
//  uint64_t lo_y = y & 1023u;
//  uint64_t lo_z = z & 1023u;
//  uint64_t hi_x = x >> 10u;
//  uint64_t hi_y = y >> 10u;
//  uint64_t hi_z = z >> 10u;
//
//  return (uint64_t(morton_code(hi_x, hi_y, hi_z)) << 30) | uint64_t(morton_code(lo_x, lo_y, lo_z));
//}

// http://fgiesen.wordpress.com/2009/12/13/decoding-morton-codes/
// Inverse of Part1By2 - "delete" all bits not at positions divisible by 3
__host__  __device__
 static __forceinline__ uint32_t Compact1By2(uint32_t x)
{
  x &= 0x09249249;                  // x = ---- 9--8 --7- -6-- 5--4 --3- -2-- 1--0
  x = (x ^ (x >> 2)) & 0x030c30c3;  // x = ---- --98 ---- 76-- --54 ---- 32-- --10
  x = (x ^ (x >> 4)) & 0x0300f00f;  // x = ---- --98 ---- ---- 7654 ---- ---- 3210
  x = (x ^ (x >> 8)) & 0xff0000ff;  // x = ---- --98 ---- ---- ---- ---- 7654 3210
  x = (x ^ (x >> 16)) & 0x000003ff; // x = ---- ---- ---- ---- ---- --98 7654 3210
  return x;
}

__host__ __device__
static __forceinline__ void inv_morton_code(uint32_t zcode, uint32_t& x, uint32_t& y, uint32_t& z)
{
  assert(zcode <= ((1 << 30) - 1));
  x = Compact1By2(zcode);
  y = Compact1By2(zcode >> 1);
  z = Compact1By2(zcode >> 2);
}

__host__ __device__
static __forceinline__ void inv_morton_code60(uint64_t zcode, uint32_t& x, uint32_t& y, uint32_t& z)
{
  assert(zcode <= ((uint64_t(1) << 60) - 1));
  uint32_t low_30 = uint32_t(zcode) & ((1 << 30) - 1);
  uint32_t high_30 = (uint32_t) (zcode >> 30);

  x = (Compact1By2(high_30) << 10) | Compact1By2(low_30);
  y = (Compact1By2(high_30 >> 1) << 10) | Compact1By2(low_30 >> 1);
  z = (Compact1By2(high_30 >> 2) << 10) | Compact1By2(low_30 >> 2);
}

__host__ __device__
static __forceinline__ void inv_morton_code60(uint64_t zcode, gpu_voxels::Vector3ui& coordinates)
{
  inv_morton_code60(zcode, coordinates.x, coordinates.y, coordinates.z);
}

__host__  __device__
 static __forceinline__ uint64_t morton_code60(uint32_t x, uint32_t y, uint32_t z)
{
  MORTON_TRAFO()
}

__host__  __device__
 static __forceinline__ uint64_t morton_code60(gpu_voxels::Vector3ui coordinates)
{
  return morton_code60(coordinates.x, coordinates.y, coordinates.z);
}

template<std::size_t branching_factor>
__host__  __device__
 __forceinline__ OctreeVoxelID getVoxelSideLength(uint32_t level)
{
  return powf(powf(branching_factor, 1.0f / 3.0f), level);
}

template<typename T>
struct transform_to_morton: public thrust::unary_function<T, OctreeVoxelID>
{
  __host__  __device__
   __forceinline__ OctreeVoxelID operator()(T value)
  {
    MORTON_TRAFO(value.)
  }
};

template<std::size_t branching_factor>
__host__  __device__
 __forceinline__ OctreeVoxelID getZOrderPrefix(OctreeVoxelID value, uint32_t level)
{
  return value >> ((level + 1) * (uint32_t) log2(float(branching_factor)));
}

template<std::size_t branching_factor>
__host__  __device__
 __forceinline__ OctreeVoxelID getZOrderNodeId(const OctreeVoxelID value, const uint32_t level)
{
  return (value >> (level * (uint32_t) log2(float(branching_factor)))) & (branching_factor - 1);
}

/*
 *  Returns the ID of the level from where both leafs have the same path to the root.
 */
template<std::size_t branching_factor>
__device__
 __forceinline__ uint8_t getCommonLevel(OctreeVoxelID idA, OctreeVoxelID idB)
{
  assert(sizeof(OctreeVoxelID) == 8); // otherwise use __clz() instead of __clzll()

#ifdef __CUDACC__
#define op __clzll
#else
#define op ffs
#endif
  const int numLeadingZeros = op(idA xor idB);
#undef op

  return (uint8_t) ceil((sizeof(OctreeVoxelID) * 8 - numLeadingZeros) / log2(float(branching_factor)));
}

template<std::size_t branching_factor>
__host__  __device__
 __forceinline__ OctreeVoxelID getNextSubTree(OctreeVoxelID value, uint32_t level)
{
  return value >> ((level + 1) * (OctreeVoxelID) log2(float(branching_factor)));
}

template<std::size_t branching_factor>
__host__  __device__
 __forceinline__ uint64_t getTree(OctreeVoxelID id, uint32_t splitLevel)
{
  return id << (splitLevel * (OctreeVoxelID) log2(float(branching_factor)));
}

template<std::size_t branching_factor>
__host__  __device__
 __forceinline__ OctreeVoxelID getZOrderNextLevel(OctreeVoxelID prefix, uint32_t child)
{
  return (prefix << (uint32_t) log2(float(branching_factor))) + child;
}

template<std::size_t branching_factor>
__host__  __device__
 __forceinline__ OctreeVoxelID getZOrderLastLevel(OctreeVoxelID prefix, uint32_t level)
{
  return prefix << (level * uint32_t(log2(float(branching_factor))));
}

/**
 * Data structure to identify any cube/voxel in morton space
 */
struct MortonCube{
public:
  OctreeVoxelID m_voxel_id; // morton code of the voxel, only the prefix according to the level matters
  uint8_t m_level; // level of the voxel in the tree and therefore the length of the morton prefix

  __host__  __device__
  MortonCube() : m_voxel_id(0), m_level(255)
  {
  }

  __host__  __device__
  MortonCube(const OctreeVoxelID voxel_id, const uint8_t level) : m_voxel_id(voxel_id), m_level(level)
  {
  }
};

} // end of ns
} // end of ns

#endif /* MORTON_H_ */
