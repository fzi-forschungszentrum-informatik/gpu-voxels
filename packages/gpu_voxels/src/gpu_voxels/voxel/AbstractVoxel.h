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
#ifndef GPU_VOXELS_VOXEL_ABSTRACT_VOXEL_H_INCLUDED
#define GPU_VOXELS_VOXEL_ABSTRACT_VOXEL_H_INCLUDED

#include <cuda_runtime.h>
#include <stdint.h>
#include <gpu_voxels/helpers/common_defines.h>

namespace gpu_voxels {

/**
 * @brief Interface for different voxel subclasses.
 */
class AbstractVoxel
{
  // ##### No virtual inheritance possible since this would blow up the the voxels by the vptr #####

  /**
   * @brief insert Inserts new data into this voxel
   * @param voxel_meaning Meaning of the voxel to insert data into
   */
  __host__ __device__
  void insert(const BitVoxelMeaning voxel_meaning);

  /**
   * @brief reduce Reduces 'this' and 'other_voxel' into a single voxel
   * @param other_voxel
   * @return Reduced voxel
   */
  __host__ __device__
  AbstractVoxel reduce(const AbstractVoxel other_voxel);

  __host__ __device__
  bool isOccupied(float col_threshold) const;

};

} // end of ns

#endif
