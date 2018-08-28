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
#ifndef GPU_VOXELS_OCTREE_ENVIRONMENT_NODES_H_INCLUDED
#define GPU_VOXELS_OCTREE_ENVIRONMENT_NODES_H_INCLUDED

#include <gpu_voxels/octree/Nodes.h>
#include <gpu_voxels/octree/DataTypes.h>
#include <ostream>
#include <istream>
//#include <gpu_voxels/octree/RobotNodes.h>

namespace gpu_voxels {
namespace NTree {

// -- forward declaration --
namespace Robot {
class LeafNode;
class InnerNode;
}

namespace Environment {

/**
 * @brief Holds the basic data of a deterministic node.
 */
class BasicData
{
public:
  __host__ __device__
  BasicData()
  {
  }

  __host__ __device__
  BasicData(const NodeStatus status, const NodeFlags flags)
  {
    m_status = status;
    m_flags = flags;
  }

  __device__ __host__ __forceinline__
  bool hasFlags(const NodeFlags flags) const
  {
    return (m_flags & flags) == flags;
  }

  __host__
  friend std::ostream& operator<<(std::ostream& os, const BasicData& dt)
  {
      os << dt.m_status << " " << dt.m_flags;
      return os;
  }

  __host__
  friend std::istream& operator>>(std::istream& in, BasicData& dt)
  {
    in >> dt.m_status;
    in >> dt.m_flags;
    return in;
  }

  NodeStatus m_status;
  NodeFlags m_flags;
};

/**
 * @brief Data represented by a deterministic node of a \c NTree.
 */
class NodeData
{
public:
  typedef Environment::BasicData BasicData;

  __host__ __device__
  NodeData()
  {

  }

  /**
   * @brief NodeData
   * @param voxelID Uniquely identifies any node of the tree in combination with \c level.
   * @param level Level of this node in the tree.
   * @param basic_data
   */
  __host__ __device__
  NodeData(const OctreeVoxelID voxelID, const voxel_count level, const BasicData basic_data)
  {
    m_voxel_id = voxelID;
    m_level = level;
    m_basic_data = basic_data;
  }

  __host__
  friend std::ostream& operator<<(std::ostream& os, const NodeData& dt)
  {
      os << dt.m_voxel_id << " " << dt.m_level << " " << dt.m_basic_data;
      return os;
  }

  __host__
  friend std::istream& operator>>(std::istream& in, NodeData& dt)
  {
    in >> dt.m_voxel_id;
    in >> dt.m_level;
    in >> dt.m_basic_data;
    return in;
  }

  OctreeVoxelID m_voxel_id;
  voxel_count m_level;
  BasicData m_basic_data;
};

/**
 * @brief Super class for all deterministic nodes of a \c NTree.
 */
class Node
{
public:
  typedef Environment::NodeData NodeData;
  /**
   * Type used for ray casting with this kind of nodes.
   */
  typedef struct
  {
    typedef NodeStatus Type;
    Type value;
  } RayCastType; // wrap in struct to be able to overload functions

  __device__ __host__ __forceinline__
  Node() :
      m_status(0)
  {
  }

  __device__  __host__  __forceinline__
  NodeStatus getStatus() const
  {
    return m_status;
  }

  __device__ __host__ __forceinline__
  void setStatus(const NodeStatus status)
  {
    m_status = status;
  }

  /**
   * @brief hasStatus
   * @param status
   * @return Returns \c true if this node has the given status, false otherwise.
   */
  __device__ __host__ __forceinline__
  bool hasStatus(const NodeStatus status) const
  {
    return (getStatus() & status) == status;
  }

  //TODO remove/move out of Node
  __device__ __host__ __forceinline__
  bool isUnknown() const
  {
    return hasStatus(ns_UNKNOWN);
  }

  //TODO remove/move out of Node
  __device__ __host__ __forceinline__
  bool isOccupied() const
  {
    return hasStatus(ns_OCCUPIED);
  }

  //TODO remove/move out of Node
  __device__ __host__ __forceinline__
  bool isFree() const
  {
    return hasStatus(ns_FREE);
  }

#ifndef DISABLE_SEPARATE_COMPILTION
  __device__ __host__
  bool isInConflict(Robot::LeafNode rob_LeafNode);
#endif

  /**
   * @brief Checks whether this node is in conflict with the given node. That's the case if both nodes are occupied.
   * @param node
   * @return \c True if there is a conflict.
   */
  //TODO remove/move out of Node
  __device__ __host__ __forceinline__
  bool isInConflict(Environment::Node node) const
  {
    return isOccupied() & node.isOccupied();
  }

protected:
  NodeStatus m_status;
};

/**
 * @brief Deterministic leaf node of a \c NTree.
 */
class LeafNode: public Node
{
public:
  static const NodeStatus INVALID_STATUS = ns_UNKNOWN | ns_FREE | ns_OCCUPIED;

  // default constructor needed otherwise nvcc bug:
  //UNREACHABLE executed!
  //Stack dump:
  //0.      Running pass 'NVPTX DAG->DAG Pattern Instruction Selection' on function '@_ZN15icl_environment3gpu5NTree17kernel_clearNodesINS1_11Environment9InnerNodeEEEvmPT_j'
  //Aborted (core dumped)
  __device__ __host__ __forceinline__
  LeafNode()
  {
  }

  /**
   * @brief Extracts the data represented by this node.
   * @param voxel_id Uniquely identifies any node of the tree in combination with \c level.
   * @param level Level of this node in the tree.
   * @return
   */
  __device__  __host__  __forceinline__
  NodeData extractData(const OctreeVoxelID voxel_id,
                                                           const voxel_count level) const
  {
    return NodeData(voxel_id, level, NodeData::BasicData(getStatus(), 0));
  }
};

/**
 * @brief Deterministic inner node of a \c NTree.
 */
class InnerNode: public Node
{
public:

  __device__ __host__ __forceinline__
  InnerNode() : m_flags(0), m_child_low(0), m_child_middle(0), m_child_high(0)
  {
  }

  /**
   * @brief getChildPtr
   * @return Pointer to an array of child nodes.
   */
  __device__ __host__ __forceinline__
  void* getChildPtr() const
  {
    return (void*) ((uint64_t(m_child_high) << ((sizeof(m_child_low) + sizeof(m_child_middle)) << 3))
        | (uint64_t(m_child_middle) << (sizeof(m_child_low) << 3))
        |  uint64_t(m_child_low));
//    return (void*) ((uint64_t(m_child_high) << (8 * sizeof(m_child_low))) | uint64_t(m_child_low));
  }

  /**
   * @brief Sets the pointer to an array of child nodes.
   * @param child
   */
  __device__ __host__ __forceinline__
  void setChildPtr(void* const child)
  {
//    assert(uint64_t(child) < (uint64_t(1) << uint64_t(8 * (sizeof(m_child_high) + sizeof(m_child_low)))));
//    m_child_high = (uint16_t) (uint64_t(child) >> (8 * sizeof(m_child_low)));
//    m_child_low = (uint32_t) (uint64_t(child) & 0xFFFFFFFF);
    assert(uint64_t(child) < (uint64_t(1) << uint64_t(8 * (sizeof(m_child_high) + sizeof(m_child_middle) + sizeof(m_child_low)))));
    uint64_t ptr = uint64_t(child);
    m_child_low = uint16_t(ptr);
    ptr =  ptr >> (sizeof(m_child_low) << 3);
    m_child_middle = uint16_t(ptr);
    ptr =  ptr >> (sizeof(m_child_middle) << 3);
    m_child_high = uint16_t(ptr);
  }

#ifndef DISABLE_SEPARATE_COMPILTION
  __device__ __host__
  bool isInConflict(Robot::InnerNode rob_InnerNode);

#endif

  /**
   * @brief Extracts the data represented by this node.
   * @param voxel_id Uniquely identifies any node of the tree in combination with \c level.
   * @param level Level of this node in the tree.
   * @return
   */
  __device__  __host__  __forceinline__
  NodeData extractData(const OctreeVoxelID voxel_id,
                                                           const voxel_count level) const
  {
    return NodeData(voxel_id, level, NodeData::BasicData(m_status, m_flags));
  }

  __device__  __host__  __forceinline__
  NodeFlags getFlags() const
  {
    return m_flags;
  }

  __device__ __host__ __forceinline__
  void setFlags(const NodeFlags flags)
  {
    m_flags = flags;
  }

  /**
   * @brief hasFlags
   * @param flags
   * @return Returns \c true if this node has the given flags, false otherwise.
   */
  __device__ __host__ __forceinline__
  bool hasFlags(const NodeFlags flags) const
  {
    return (getFlags() & flags) == flags;
  }

  __device__ __host__ __forceinline__
  void clearNeedsUpdate()
  {
    setFlags(getFlags() & ~(nf_NEEDS_UPDATE | nf_UPDATE_SUBTREE));
  }

protected:
  NodeFlags m_flags;
  uint16_t m_child_low;
  uint16_t m_child_middle;
  uint16_t m_child_high;

//  uint32_t m_child_low;
//  uint16_t m_child_high;
//  NodeFlags m_flags;
};

} // end of ns

}  // end of ns
}  // end of ns

#endif /* ENVIRONMENT_NODES_H_ */
