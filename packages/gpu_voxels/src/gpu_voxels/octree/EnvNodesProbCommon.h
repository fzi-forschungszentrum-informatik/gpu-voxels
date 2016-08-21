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
 * \date    2013-12-11
 *
 */
//----------------------------------------------------------------------/*
// remove include guards
//#ifndef GPU_VOXELS_OCTREE_ENV_NODES_COMMON_H_INCLUDED
//#define GPU_VOXELS_OCTREE_ENV_NODES_COMMON_H_INCLUDED

// Common code of classes LeafNodeProb and InnerNodeProb to solve the diamond problem of multi-inheritance without virtual inheritance

  __device__ __host__ __forceinline__
  bool isOccupied() const
  {
    return (getOccupancy() != UNKNOWN_PROBABILITY) && (getOccupancy() >= THRESHOLD_OCCUPANCY);
  }

  __device__ __host__ __forceinline__
  bool isUnknown() const
  {
    return getOccupancy() == UNKNOWN_PROBABILITY;
  }

  __device__ __host__ __forceinline__
  bool isFree() const
  {
    return (getOccupancy() != UNKNOWN_PROBABILITY) && (getOccupancy() < THRESHOLD_OCCUPANCY);
  }

  __device__ __host__ __forceinline__
  bool isInConflict(const LeafNodeProb env_LeafNode) const
  {
    return isOccupied() & env_LeafNode.isOccupied();
  }

//#endif
