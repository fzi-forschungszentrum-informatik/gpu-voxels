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
 * \date    2014-06-20
 *
 */
//----------------------------------------------------------------------
#ifndef DEFAULTCOLLIDER_CU_
#define DEFAULTCOLLIDER_CU_

#include <gpu_voxels/octree/EnvironmentNodes.h>
#include <gpu_voxels/octree/EnvNodesProbabilistic.h>

namespace gpu_voxels {
namespace NTree {

class DefaultCollider
{
public:
  __host__ __device__
  DefaultCollider() :
      m_occupancy_threshold(THRESHOLD_OCCUPANCY)
  {

  }

  __host__ __device__
  DefaultCollider(const float coll_theshold) :
      m_occupancy_threshold(floatToProbability(coll_theshold))
  {

  }

  __host__ __device__
  DefaultCollider(Probability threshold) :
      m_occupancy_threshold(threshold)
  {

  }

  // ##### Deterministic tree nodes #####
  __host__ __device__ __forceinline__
  bool collideNode(const Environment::Node& a, const Environment::Node& b) const
  {
    return a.hasStatus(ns_OCCUPIED) && b.hasStatus(ns_OCCUPIED);
  }

  __host__ __device__ __forceinline__
  bool collide(const Environment::InnerNode& a, const Environment::InnerNode& b) const
  {
    return collideNode(a, b);
  }

  __host__ __device__ __forceinline__
  bool collide(const Environment::LeafNode& a, const Environment::LeafNode& b) const
  {
    return collideNode(a, b);
  }

  __host__ __device__ __forceinline__
  bool collide(const Environment::LeafNode& a, const Environment::InnerNode& b) const
  {
    return collideNode(a, b);
  }

  __host__ __device__ __forceinline__
  bool collide(const Environment::InnerNode& a, const Environment::LeafNode& b) const
  {
    return collideNode(a, b);
  }
  //######################################

  // ##### Probabilistic tree nodes #####
  __host__ __device__ __forceinline__
  bool isOccupied(const Environment::NodeProb& a) const
  {
    return (a.getOccupancy() != UNKNOWN_PROBABILITY && a.getOccupancy() >= m_occupancy_threshold);
  }

  __host__ __device__ __forceinline__
  bool collideNodeProb(const Environment::NodeProb& a, const Environment::NodeProb& b) const
  {
    return isOccupied(a) && isOccupied(b);
  }

  __host__ __device__ __forceinline__
  bool collide(const Environment::InnerNodeProb& a, const Environment::InnerNodeProb& b) const
  {
    return collideNodeProb(a, b);
  }

  __host__ __device__ __forceinline__
  bool collide(const Environment::LeafNodeProb& a, const Environment::LeafNodeProb& b) const
  {
    return collideNodeProb(a, b);
  }

  __host__ __device__ __forceinline__
  bool collide(const Environment::InnerNodeProb& a, const Environment::LeafNodeProb& b) const
  {
    return collideNodeProb(a, b);
  }

  __host__ __device__ __forceinline__
  bool collide(const Environment::LeafNodeProb& a, const Environment::InnerNodeProb& b) const
  {
    return collideNodeProb(a, b);
  }
  //######################################

  __host__ __device__
  static Probability floatToProbability(const float val)
  {
    float tmp = (val * (float(MAX_PROBABILITY) - float(MIN_PROBABILITY))) + MIN_PROBABILITY;
    return Probability(tmp);
  }

//private:
  Probability m_occupancy_threshold;
};

} // end of ns
} // end of ns
#endif /* DEFAULTCOLLIDER_CU_ */
