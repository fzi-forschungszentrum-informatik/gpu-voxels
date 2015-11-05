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
#ifndef GPU_VOXELS_VOXEL_PROBABILISTIC_VOXEL_H_INCLUDED
#define GPU_VOXELS_VOXEL_PROBABILISTIC_VOXEL_H_INCLUDED

#include <gpu_voxels/voxel/AbstractVoxel.h>
#include <gpu_voxels/helpers/common_defines.h>

namespace gpu_voxels {

/**
 * @brief Type for holding the occupation probability
 */
typedef int8_t probability;
static const probability UNKNOWN_PROBABILITY = probability(-128);
static const probability MIN_PROBABILITY = probability(-127);
static const probability MAX_PROBABILITY = probability(127);

/**
 * @brief Probabilistic voxel type with probability in log-odd representation
 */
class ProbabilisticVoxel: public AbstractVoxel
{
public:
  /**
   * @brief ProbabilisticVoxel
   */
  __host__ __device__
  ProbabilisticVoxel();

  __host__ __device__
  ~ProbabilisticVoxel();

  /**
   * @brief updateOccupancy Updates the occupancy of this voxel based on the log-odd representation.
   * @param occupancy A new occupancy measurement.
   * @return Returns the updated occupancy.
   */
  __host__   __device__
  probability updateOccupancy(const probability occupancy);

  /**
   * @brief occupancy Write reference.
   * @return
   */
  __host__   __device__
  probability& occupancy();

  /**
   * @brief occupancy Read-only reference.
   * @return
   */
  __host__   __device__
   const probability& occupancy() const;

  /**
   * @brief getOccupancy Read-only access per copy
   * @return
   */
  __host__   __device__
  probability getOccupancy() const;

  __host__   __device__
  void insert(const uint32_t voxel_meaning);

  __host__ __device__
  static ProbabilisticVoxel reduce(const ProbabilisticVoxel voxel, const ProbabilisticVoxel other_voxel);

  struct reduce_op //: public thrust::binary_function<BitVoxelMeaningFlags, BitVoxelMeaningFlags, BitVoxelMeaningFlags>
  {
    __host__ __device__
    ProbabilisticVoxel operator()(const ProbabilisticVoxel& a, const ProbabilisticVoxel& b) const
    {
      ProbabilisticVoxel tmp = a;
      tmp.updateOccupancy(b.getOccupancy());
      return tmp;
    }
  };


  __host__
  friend std::ostream& operator<<(std::ostream& os, const ProbabilisticVoxel& dt)
  {
    os << dt.getOccupancy();
    return os;
  }

  __host__
  friend std::istream& operator>>(std::istream& in, ProbabilisticVoxel& dt)
  {
    probability tmp;
    in >> tmp;
    dt.occupancy() = tmp;
    return in;
  }

protected:
  probability m_occupancy;
};

} // end of ns

#endif
