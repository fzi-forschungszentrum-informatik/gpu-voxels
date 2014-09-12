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
 * \date    2014-08-31
 *
 */
//----------------------------------------------------------------------/*
#ifndef GPU_VOXELS_OCTREE_LOAD_BALANCER_PROPAGATE_H_INCLUDED
#define GPU_VOXELS_OCTREE_LOAD_BALANCER_PROPAGATE_H_INCLUDED

#include <gpu_voxels/octree/load_balancer/AbstractLoadBalancer.h>
#include <gpu_voxels/octree/DefaultCollider.h>

namespace gpu_voxels {
namespace NTree {
namespace LoadBalancer {

typedef RunConfig<128> PropagateRunConfig; // Experimental founding

template<std::size_t branching_factor,
    std::size_t level_count,
    class InnerNode,
    class LeafNode>
class Propagate: public AbstractLoadBalancer<branching_factor, level_count, InnerNode, LeafNode,
    WorkItemPropagate<InnerNode>, PropagateRunConfig>
{
public:
  typedef WorkItemPropagate<InnerNode> WorkItem;
  typedef PropagateRunConfig RunConfig;
  typedef AbstractLoadBalancer<branching_factor, level_count, InnerNode, LeafNode, WorkItem, RunConfig> Base;

  Propagate(NTree<branching_factor, level_count, InnerNode, LeafNode>* ntree, const uint32_t num_tasks = 1024);
  virtual ~Propagate();

protected:
  // ---------- Abstract method implementation --------------
  virtual void doWork();
  virtual void doPostCalculations();
  // --------------------------------------------------------

  virtual bool doPreparations();
  virtual void doCleanup();

protected:
  NTree<branching_factor, level_count, InnerNode, LeafNode>* m_ntree;
};

}
}
}

#endif
