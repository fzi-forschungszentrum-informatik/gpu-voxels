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
 * \date    2014-04-16
 *
 */
//----------------------------------------------------------------------/*
#ifndef GPU_VOXELS_OCTREE_VOXELTYPEFLAGS_H_INCLUDED
#define GPU_VOXELS_OCTREE_VOXELTYPEFLAGS_H_INCLUDED

#include <gpu_voxels/octree/DataTypes.h>

#include <thrust/functional.h>

#include <iostream>

namespace gpu_voxels {
namespace NTree {

/**
 * Class for handling the bit flags to distinguish between different voxel types / swept volumes
 */
template<int SIZE>
class VoxelTypeFlags
{
public:
  uint32_t m_flags[SIZE];
//
//  __host__ __device__ VoxelTypeFlags()
//  {
//#pragma unroll
//    for (int i = 0; i < SIZE; ++i)
//      m_flags[i] = 0;
//  }

  __host__ __device__
  void clear()
  {
#if defined(__CUDACC__) && !defined(__GNUC__)
# pragma unroll
#endif
    for (int i = 0; i < SIZE; ++i)
      m_flags[i] = 0;
  }

  __host__    __device__
  VoxelTypeFlags<SIZE> operator|(const VoxelTypeFlags<SIZE>& o) const
  {
    VoxelTypeFlags<SIZE> res;
#if defined(__CUDACC__) && !defined(__GNUC__)
# pragma unroll
#endif
    for (int i = 0; i < SIZE; ++i)
      res.m_flags[i] = m_flags[i] | o.m_flags[i];
    return res;
  }

  __host__ __device__
  void operator|=(const VoxelTypeFlags& o)
  {
#if defined(__CUDACC__) && !defined(__GNUC__)
# pragma unroll
#endif
    for (int i = 0; i < SIZE; ++i)
      m_flags[i] |= o.m_flags[i];
  }

  __host__ __device__
  bool isZero() const
  {
    bool result = true;
#if defined(__CUDACC__) && !defined(__GNUC__)
# pragma unroll
#endif
    for (int i = 0; i < SIZE; ++i)
      result &= m_flags[i] == 0;
    return result;
  }

#ifdef __CUDACC__
  __device__
  static void reduce(VoxelTypeFlags<SIZE>& flags, const int thread_id, const int num_threads,
                     uint32_t* shared_mem)
  {
#if defined(__CUDACC__) && !defined(__GNUC__)
# pragma unroll
#endif
    for (int i = 0; i < SIZE; ++i)
    {
      shared_mem[thread_id] = flags.m_flags[i];
      __syncthreads();
      REDUCE(shared_mem, thread_id, num_threads, |)
      if (thread_id == 0)
        flags.m_flags[i] = shared_mem[0];
      __syncthreads();
    }
  }
#endif

#ifdef __CUDACC__
  __device__
  static void reduceAtomic(VoxelTypeFlags<SIZE>& flags, VoxelTypeFlags<SIZE>& global_flags)
  {
#if defined(__CUDACC__) && !defined(__GNUC__)
# pragma unroll
#endif
    for (int i = 0; i < SIZE; ++i)
    {
      atomicOr(&global_flags.m_flags[i], flags.m_flags[i]);
    }
  }
#endif

  struct reduce_op //: public thrust::binary_function<VoxelTypeFlags, VoxelTypeFlags, VoxelTypeFlags>
  {
    __host__    __device__ VoxelTypeFlags operator()(const VoxelTypeFlags& a, const VoxelTypeFlags& b) const
    {
      return a | b;
    }
  };

  __host__ __device__
   friend std::ostream& operator<<(std::ostream& out, const VoxelTypeFlags& p)
  {
    for (uint32_t i = 0; i < SIZE; ++i)
      out << p.m_flags[i] << " ";
    out << std::endl;
    return out;
  }

}
;

//template<int SIZE>
//__host__     __device__ VoxelTypeFlags<SIZE> operator|(VoxelTypeFlags<SIZE>& a, VoxelTypeFlags<SIZE>& b)
//{
//  VoxelTypeFlags<SIZE> res;
//#pragma unroll
//  for (int i = 0; i < SIZE; ++i)
//    res.m_flags[i] = a.m_flags[i] | b.m_flags[i];
//  return res;
//}

}  // end of ns
}  // end of ns

#endif
