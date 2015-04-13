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
#ifndef GPU_VOXELS_OCTREE_ROBOTNODES_H_INCLUDED
#define GPU_VOXELS_OCTREE_ROBOTNODES_H_INCLUDED

#include <gpu_voxels/octree/Nodes.h>
#include <gpu_voxels/octree/DataTypes.h>
//#include <gpu_voxels/octree/EnvironmentNodes.h>

namespace gpu_voxels {
namespace NTree {

namespace Environment {
class LeafNode;
class InnerNode;
}

namespace Robot {

/*
 * Data structure to store the occupancy information of different samples or swept volumes in a leaf node of a tree
 */
class LeafNode
{
public:
  enum VoxelType
  {
    FROM_TYPE = 1, TO_TYPE = VOXELLIST_FLAGS_SIZE * 8
  };

  // default constructor needed
  __device__ __host__ __forceinline__ LeafNode()
  {

  }

  __device__ __forceinline__
  void init_d()
  {
    memset(m_voxel_type, 0, sizeof(m_voxel_type));
  }

  __device__ __forceinline__ void initLastLevel_d()
  {
    init_d();
  }

  __host__
  void init_h()
  {
    for (uint32_t i = 0; i < VOXELLIST_FLAGS_SIZE; ++i)
      m_voxel_type[i] = 0;
  }

  __host__
  void init_h(uint32_t level)
  {
    init_h();
  }

  __device__ __host__ __forceinline__
  bool isOccupied()
  {
    return m_voxel_type[0] & 0xFFFFFFEF;
  }

  __device__ __host__ __forceinline__
  void setOccupied(uint32_t voxelType)
  {
    // TODO handle different voxel_types
    assert(FROM_TYPE <= voxelType && voxelType <= TO_TYPE);
    m_voxel_type[voxelType / sizeof(uint32_t)] = m_voxel_type[voxelType / sizeof(uint32_t)]
        | (voxelType % sizeof(uint32_t));
  }

  __device__ __host__ __forceinline__
  void setOccupied()
  {
    m_voxel_type[0] = (m_voxel_type[0] | 0x00000001);
  }

  __device__ __host__ __forceinline__
  void setChildPtr(void* child)
  {
    // dummy
  }

  __device__ __host__ __forceinline__
  void* getChildPtr()
  {
    // function not supported by this type
    assert(0);
    return 0;
  }

  __device__ __host__ __forceinline__
  void setFree(uint32_t voxelType)
  {
    // TODO handle different voxel_types
    assert(FROM_TYPE <= voxelType && voxelType <= TO_TYPE);
    m_voxel_type[voxelType / sizeof(uint32_t)] = m_voxel_type[voxelType / sizeof(uint32_t)]
        & ~(1 << (voxelType % sizeof(uint32_t)));
    m_voxel_type[0] = (m_voxel_type[0] & ~0x00000001);
  }

#ifdef DISABLE_SEPARATE_COMPILTION
  __device__ __host__
  bool isInConflict(Environment::LeafNode env_LeafNode);
#endif

  __device__ __host__ __forceinline__
  bool isInConflict(Robot::LeafNode rob_LeafNode)
  {
    return isOccupied() & rob_LeafNode.isOccupied();
  }

private:
  uint32_t m_voxel_type[VOXELLIST_FLAGS_SIZE];
}
;

class InnerNode
{
private:
  __device__ __host__ __forceinline__
  void init(NodeStatus status)
  {
    m_child_low = 0;
    m_child_high = 0;
    m_status = status;
    alignment = 0;

  }

public:

  // default constructor needed
  __device__ __host__ __forceinline__ InnerNode()
  {

  }

  __device__ __forceinline__
  void init_d()
  {
    init(ns_FREE);
    memset(m_voxel_type, 0xFFFFFFFF, sizeof(m_voxel_type));
  }

  __device__ __forceinline__
  void initLastLevel_d()
  {
    init(NodeStatus(ns_FREE | ns_LAST_LEVEL));
    memset(m_voxel_type, 0xFFFFFFFF, sizeof(m_voxel_type));
  }

  __device__ __host__ __forceinline__
  bool hasStatus(NodeStatus status)
  {
    return (getStatus() & status) > 0;
  }

  __device__ __host__ __forceinline__
  bool isOccupied()
  {
    return hasStatus(NodeStatus (ns_OCCUPIED | ns_PART));
  }

  __device__ __host__ __forceinline__
  void setOccupied()
  {
    // clear all flags except LAST_LEVEL and set PART_OCCUPIED
    setStatus(NodeStatus ((getStatus() & ns_LAST_LEVEL) | ns_PART));
  }

  __device__ __host__ __forceinline__
  NodeStatus getStatus()
  {
    return m_status;
  }

  __device__ __host__ __forceinline__
  void setStatus(NodeStatus status)
  {
    assert(uint32_t(status) <= 0xFF);
    m_status = status;
  }

  __device__ __host__ __forceinline__
  void* getChildPtr()
  {
    return (void*) ((uint64_t(m_child_high) << (8 * sizeof(m_child_low))) | uint64_t(m_child_low));
  }

  __device__ __host__ __forceinline__
  void setChildPtr(void* child)
  {
    assert(
        uint64_t(child) != 0
            && uint64_t(child) < (uint64_t(1) << uint64_t(8 * (sizeof(m_child_high) + sizeof(m_child_low)))));
    m_child_high = (uint8_t) (uint64_t(child) >> (8 * sizeof(m_child_low)));
    m_child_low = (uint32_t) (uint64_t(child) & 0xFFFFFFFF);
  }

#ifdef DISABLE_SEPARATE_COMPILTION
  __device__ __host__
  bool isInConflict(Environment::InnerNode env_InnerNode);
#endif

  __device__ __host__ __forceinline__
  bool isInConflict(Robot::InnerNode rob_InnerNode)
  {
    return isOccupied() & rob_InnerNode.isOccupied();
  }

private:
  uint32_t m_child_low;
  uint8_t m_child_high;
  uint8_t m_status;
  uint16_t alignment;
  uint32_t m_voxel_type[VOXELLIST_FLAGS_SIZE];
};
}

} // end of ns
} // end of ns

#endif /* ROBOTNODES_H_ */
