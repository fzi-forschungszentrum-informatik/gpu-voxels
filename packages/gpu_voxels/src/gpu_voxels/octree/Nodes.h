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
#ifndef GPU_VOXELS_OCTREE_NODES_H_INCLUDED
#define GPU_VOXELS_OCTREE_NODES_H_INCLUDED

#include <cuda.h>
#include <assert.h>

// thrust
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/reduce.h>
#include <thrust/sort.h>
#include <thrust/scan.h>

// icl_environment_gpu
//#include <gpu_voxels/voxelmap/Voxel.h>

#include <gpu_voxels/helpers/common_defines.h>

namespace gpu_voxels {
namespace NTree {

/*
 * Type to store the value of EnumNodeStatus
 */
typedef uint8_t NodeStatus;


static const NodeStatus ns_FREE = 1;
static const NodeStatus ns_UNKNOWN = 2;
static const NodeStatus ns_OCCUPIED = 4;
static const NodeStatus ns_PART = 8;
static const NodeStatus ns_LAST_LEVEL = 16;
static const NodeStatus ns_COLLISION = 32;
static const NodeStatus ns_STATIC_MAP = 64;
static const NodeStatus ns_DYNAMIC_MAP = 128;

static const NodeStatus STATUS_OCCUPANCY_MASK = ns_FREE | ns_UNKNOWN | ns_OCCUPIED;
static const NodeStatus STATUS_OCCUPANCY_MASK_INV = ~STATUS_OCCUPANCY_MASK;

/*
 * Computes the new node status based on the status of the child nodes.
 */
__host__   __device__   inline NodeStatus getNewStatus(const NodeStatus child_status_or,
                                                    const NodeStatus child_status_and,
                                                    const uint8_t level_of_node)
{
  NodeStatus new_status = 0;
  NodeStatus tmp_child_status_and = (child_status_and & (ns_FREE | ns_UNKNOWN | ns_OCCUPIED));
  NodeStatus tmp_child_status_or = (child_status_or & (ns_FREE | ns_UNKNOWN | ns_OCCUPIED));

  if (((tmp_child_status_and == ns_FREE) | (tmp_child_status_and == ns_UNKNOWN)
      | (tmp_child_status_and == ns_OCCUPIED)) & ((tmp_child_status_or & ~tmp_child_status_and) == 0))
    new_status = tmp_child_status_and;
  else
    new_status = tmp_child_status_or | ns_PART;
  new_status |= child_status_or & (ns_STATIC_MAP | ns_DYNAMIC_MAP);

  if (level_of_node == 1)
    new_status |= ns_LAST_LEVEL;
  return new_status;
}

__host__   __device__   __forceinline__ enum gpu_voxels::BitVoxelMeaning statusToBitVoxelMeaning(NodeStatus status)
{
  status = (status & (ns_FREE | ns_UNKNOWN | ns_OCCUPIED));
  enum gpu_voxels::BitVoxelMeaning res;

  switch (status)
  {
    case ns_FREE:
    case ns_FREE | ns_UNKNOWN:
      // TODO: specify BitVoxelMeaning for free space
      res = gpu_voxels::eBVM_FREE;
      break;
    case ns_UNKNOWN:
      res = gpu_voxels::eBVM_UNKNOWN;
      break;
    case ns_OCCUPIED:
    case ns_OCCUPIED | ns_FREE:
    case ns_OCCUPIED | ns_FREE | ns_UNKNOWN:
    case ns_OCCUPIED | ns_UNKNOWN:
      // TODO: decide between static and dynamic voxel
      res = gpu_voxels::eBVM_OCCUPIED;
      break;
  }
  return res;
}

__host__   __device__   __forceinline__ enum gpu_voxels::BitVoxelMeaning statusToBitVoxelMeaning(uint8_t* mapping_lookup,
                                                                                 NodeStatus status)
{
  return gpu_voxels::BitVoxelMeaning(mapping_lookup[status & (ns_FREE | ns_UNKNOWN | ns_OCCUPIED | ns_COLLISION)]);
}

/*
 * Type to store the value of EnumNodeFlags
 */
typedef uint8_t NodeFlags;

/*
 * InnerNode flags needed for propagate
 */
    //nf_RESERVED = 15,
static const NodeFlags nf_UPDATE_SUBTREE = 64; // indicates that all nodes in it's subtree have to be updated with this nodes' value
static const NodeFlags nf_NEEDS_UPDATE = 128; // indicates that this node needs to be updated since it was newly created or any node of it's subtree changed

}  // end of ns
}  // end of ns

#endif /* NODES_H_ */
