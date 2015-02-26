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

namespace gpu_voxels {
namespace NTree {

// Forward declaration
class DefaultCollider;

namespace LoadBalancer {

typedef RunConfig<128> IntersectRunConfig; // Number of threads due to experimental founding

/**
 * @brief Load balanced collision check between two \code NTree \endcode
 *
 * @tparam branching_factor Branching factor of the corresponding \code NTree \endcode
 * @tparam level_count Number of levels of the corresponding \code NTree \endcode
 * @tparam InnerNode Inner node type of the corresponding first \code NTree \endcode
 * @tparam LeafNode Leaf node type of the corresponding first \code NTree \endcode
 * @tparam InnerNode2 Inner node type of the corresponding second \code NTree \endcode
 * @tparam LeafNode2 Leaf node type of the corresponding second \code NTree \endcode
 * @tparam Collider Type of collider which defines a collision
 * @tparam mark_collisions \code true \endcode to set the collision flag if necessary
 */
template<std::size_t branching_factor, std::size_t level_count, class InnerNode, class LeafNode,
    class InnerNode2, class LeafNode2, class Collider, bool mark_collisions>
class Intersect : public AbstractLoadBalancer<branching_factor, level_count, InnerNode, LeafNode,
    WorkItemIntersect<InnerNode, InnerNode2>, IntersectRunConfig>
{
public:
  typedef WorkItemIntersect<InnerNode, InnerNode2> WorkItem;
  typedef IntersectRunConfig RunConfig;
  typedef AbstractLoadBalancer<branching_factor, level_count, InnerNode, LeafNode, WorkItem, RunConfig> Base;

   /**
   * @brief Intersect constructor.
   * @param ntree_a First \code NTree \endcode to collide with.
   * @param ntree_b Second \code NTree \endcode to collide with.
   * @param min_level Traverse the tree down to this level for collision checking if necessary. Defines the resolution of this collision check.
   * @param collider Collider object which defines what is a collision.
   */
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
   * @brief Holds the number of detected collisions.
   */
  std::size_t* m_dev_num_collisions;

  /**
   * @brief Object which defines collisions between the different node types.
   */
  const Collider m_collider;

  const uint32_t m_min_level;

  NTree<branching_factor, level_count, InnerNode, LeafNode>* m_ntree_a;
  NTree<branching_factor, level_count, InnerNode2, LeafNode2>* m_ntree_b;

  public:
  /**
    * @brief Holds the number of detected collisions.
    */
   std::size_t m_num_collisions;
};

}
}
}

#endif
