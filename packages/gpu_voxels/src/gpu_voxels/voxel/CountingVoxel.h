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
 * \author  Christian Jülg
 * \date    2017-10-17
 *
 */
//----------------------------------------------------------------------/*
#ifndef GPU_VOXELS_VOXEL_COUNTING_VOXEL_H_INCLUDED
#define GPU_VOXELS_VOXEL_COUNTING_VOXEL_H_INCLUDED

#include <gpu_voxels/helpers/common_defines.h>
#include <gpu_voxels/voxel/AbstractVoxel.h>

namespace gpu_voxels {

/**
 * @brief Counting voxel type for filtering noise data with density threshold
 */
class CountingVoxel : public AbstractVoxel
{
public:
  /**
   * @brief CountingVoxel
   */
  __host__ __device__
  CountingVoxel();

  __host__ __device__
  ~CountingVoxel();

  __host__ __device__
  bool isOccupied(uint8_t occ_threshold) const;

  __host__ __device__
  int8_t getCount() const;

  __host__ __device__
  int8_t& count();

  __host__ __device__
  const int8_t& count() const;

  __host__ __device__
  void insert(const uint32_t voxel_meaning);

  __host__ __device__
  static CountingVoxel reduce(const CountingVoxel voxel, const CountingVoxel other_voxel);

  struct reduce_op //: public thrust::binary_function<BitVoxelMeaningFlags, BitVoxelMeaningFlags, BitVoxelMeaningFlags>
  {
    __host__ __device__
    CountingVoxel operator()(const CountingVoxel& a, const CountingVoxel& b) const
    {
      CountingVoxel tmp = a;
      tmp.m_count += b.m_count;
      return tmp;
    }
  };

  template <typename T>
  __host__
  friend T& operator<<(T& os, const CountingVoxel& dt)
  {
    os << int(dt.m_count);
    return os;
  }

  __host__
  friend std::istream& operator>>(std::istream& in, CountingVoxel& dt)
  {
    uint8_t tmp;
    in >> tmp;
    dt.m_count = tmp;
    return in;
  }

protected:
  int8_t m_count;
};

} // end of ns

#endif // GPU_VOXELS_VOXEL_COUNTING_VOXEL_H_INCLUDED
