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
#ifndef GPU_VOXELS_VOXEL_DEFAULT_COLLIDER_H_INCLUDED
#define GPU_VOXELS_VOXEL_DEFAULT_COLLIDER_H_INCLUDED

#include <gpu_voxels/voxel/BitVoxel.h>
#include <gpu_voxels/voxel/ProbabilisticVoxel.h>

namespace gpu_voxels {

class DefaultCollider
{
public:
  __host__ __device__
  DefaultCollider();

  __host__ __device__
  DefaultCollider(const float coll_threshold);

  __host__ __device__
  DefaultCollider(const probability threshold1, const probability threshold2);

  __host__ __device__
  bool collide(const ProbabilisticVoxel& v1, const ProbabilisticVoxel& v2) const;

  __host__ __device__
  bool collide(const ProbabilisticVoxel& v1) const;

  template<std::size_t length>
  __host__ __device__
  bool collide(const ProbabilisticVoxel& v1, const BitVoxel<length>& v2) const;

  template<std::size_t length>
  __host__ __device__
  bool collide(const BitVoxel<length>& v1, const ProbabilisticVoxel& v2) const;

  template<std::size_t length>
  __host__ __device__
  bool collide(const BitVoxel<length>& v1, const BitVoxel<length>& v2) const;

protected:
  __host__ __device__
  static probability floatToProbability(const float val);

protected:
  probability m_threshold1, m_threshold2;
};

} // end of ns

#endif
