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
 * \date    2014-08-28
 *
 */
//----------------------------------------------------------------------/*
#ifndef GPU_VOXELS_OCTREE_LOAD_BALANCER_INTERSECT_VMAP_H_INCLUDED
#define GPU_VOXELS_OCTREE_LOAD_BALANCER_INTERSECT_VMAP_H_INCLUDED

#include <gpu_voxels/octree/load_balancer/AbstractLoadBalancer.h>
#include <gpu_voxels/octree/DefaultCollider.h>
#include <gpu_voxels/octree/NTree.h>

namespace gpu_voxels {
namespace NTree {
namespace LoadBalancer {

typedef RunConfig<128> IntersectVMapRunConfig; // Experimental founding

template<std::size_t branching_factor,
    std::size_t level_count,
    class InnerNode,
    class LeafNode,
    int vft_size,
    bool set_collision_flag,
    bool compute_voxelTypeFlags,
    class VoxelType>
class IntersectVMap: public AbstractLoadBalancer<branching_factor, level_count, InnerNode, LeafNode,
    WorkItemIntersectVoxelMap<InnerNode>, IntersectVMapRunConfig>
{
public:
  typedef WorkItemIntersectVoxelMap<InnerNode> WorkItem;
  typedef IntersectVMapRunConfig RunConfig;
  typedef AbstractLoadBalancer<branching_factor, level_count, InnerNode, LeafNode, WorkItem, RunConfig> Base;

  IntersectVMap(NTree<branching_factor, level_count, InnerNode, LeafNode>* ntree,
                const VoxelType* voxel_map,
                const gpu_voxels::Vector3ui voxel_map_dim,
                const gpu_voxels::Vector3ui offset = gpu_voxels::Vector3ui(0, 0, 0),
                const uint32_t min_level = 0);
  virtual ~IntersectVMap();

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
   * Holds the vector of voxel types in collision.
   */
  VoxelTypeFlags<vft_size>* m_dev_result_voxelTypeFlags;

  const gpu_voxels::Vector3ui m_offset;
  const VoxelType* m_voxel_map;
  const gpu_voxels::Vector3ui m_voxel_map_dim;
  const uint32_t m_min_level;
  NTree<branching_factor, level_count, InnerNode, LeafNode>* m_ntree;

public:
  /**
   * Holds the number of detected collisions.
   */
  std::size_t m_num_collisions;

  /**
   * Holds the vector of voxel types in collision.
   */
  VoxelTypeFlags<vft_size> m_result_voxelTypeFlags;
};

}
}
}

#endif
