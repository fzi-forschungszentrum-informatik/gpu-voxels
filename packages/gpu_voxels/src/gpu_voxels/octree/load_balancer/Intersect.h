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
 * \date    2014-07-23
 *
 */
//----------------------------------------------------------------------/*
#ifndef GPU_VOXELS_OCTREE_LOAD_BALANCER_INTERSECT_H_INCLUDED
#define GPU_VOXELS_OCTREE_LOAD_BALANCER_INTERSECT_H_INCLUDED

#include <gpu_voxels/octree/load_balancer/AbstractLoadBalancer.h>
#include <gpu_voxels/octree/DefaultCollider.h>

namespace gpu_voxels {
namespace NTree {
namespace LoadBalancer {

typedef RunConfig<128> IntersectRunConfig; // Experimental founding

template<std::size_t branching_factor, std::size_t level_count, class InnerNode, class LeafNode,
    class InnerNode2, class LeafNode2, class Collider, bool mark_collisions>
class Intersect : public AbstractLoadBalancer<branching_factor, level_count, InnerNode, LeafNode,
    WorkItemIntersect<InnerNode, InnerNode2>, IntersectRunConfig>
{
public:
  typedef WorkItemIntersect<InnerNode, InnerNode2> WorkItem;
  typedef IntersectRunConfig RunConfig;
  typedef AbstractLoadBalancer<branching_factor, level_count, InnerNode, LeafNode, WorkItem, RunConfig> Base;

  Intersect(
      NTree<branching_factor, level_count, InnerNode, LeafNode>* ntree_a,
      NTree<branching_factor, level_count, InnerNode2, LeafNode2>* ntree_b,
      const uint32_t min_level = 0,
      Collider collider = DefaultCollider());
  virtual ~Intersect();

protected:
  // ---------- Abstract method implementation --------------
  virtual void doWork();
  virtual void doPostCalculations();
  // --------------------------------------------------------

  virtual bool doPreparations();
  virtual void doCleanup();

protected:
  /**
   * Holds the number of detected collisions.
   */
  std::size_t* m_dev_num_collisions;

  /**
   * Object which defines collisions between the different node types.
   */
  const Collider m_collider;

  const uint32_t m_min_level;

  NTree<branching_factor, level_count, InnerNode, LeafNode>* m_ntree_a;
  NTree<branching_factor, level_count, InnerNode2, LeafNode2>* m_ntree_b;

  public:
  /**
   * Holds the number of detected collisions.
   */
    std::size_t m_num_collisions;
};

}
}
}

#endif
