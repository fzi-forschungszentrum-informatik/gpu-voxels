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
#ifndef GPU_VOXELS_OCTREE_ENV_NODES_PROBABILISTIC_H_INCLUDED
#define GPU_VOXELS_OCTREE_ENV_NODES_PROBABILISTIC_H_INCLUDED

#include <gpu_voxels/octree/Nodes.h>
#include <gpu_voxels/octree/DataTypes.h>
//#include <gpu_voxels/octree/RobotNodes.h>
#include <gpu_voxels/octree/EnvironmentNodes.h>
#include <ostream>
#include <istream>

namespace gpu_voxels {
namespace NTree {

// -- forward declaration --
namespace Robot {
class LeafNode;
class InnerNode;
}

namespace Environment {


/**
 * @brief Holds the basic data of a probabilistic node.
 */
class BasicDataProb: public BasicData
{
public:
  __host__ __device__
  BasicDataProb() : BasicData()
  {

  }

  __host__ __device__
  BasicDataProb(const NodeStatus status, const NodeFlags flags, const Probability occupancy) :
      BasicData(status, flags)
  {
    m_occupancy = occupancy;
  }

  __host__
  friend std::ostream& operator<<(std::ostream& os, const BasicDataProb& dt)
  {
    os << dt.m_occupancy << " " << BasicData(dt);
    return os;
  }

  __host__
  friend std::istream& operator>>(std::istream& in, BasicDataProb& dt)
  {
    in >> dt.m_occupancy;
    BasicData* tmp = &dt;
    in >> *tmp;
    return in;
  }

  Probability m_occupancy;
};

/**
 * @brief Data represented by a probabilistic node of a \c NTree.
 */
class NodeDataProb
{
public:
  typedef Environment::BasicDataProb BasicData;

  __host__ __device__
  NodeDataProb()
  {

  }

  __host__ __device__
  NodeDataProb(const OctreeVoxelID voxelID, const voxel_count level, const BasicData basic_data)
  {
    m_voxel_id = voxelID;
    m_level = level;
    m_basic_data = basic_data;
  }

  __host__
  friend std::ostream& operator<<(std::ostream& os, const NodeDataProb& dt)
  {
      os << dt.m_voxel_id << " " << dt.m_level << " " << dt.m_basic_data;
      return os;
  }

  __host__
  friend std::istream& operator>>(std::istream& in, NodeDataProb& dt)
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
 * @brief Super class for all probabilistic nodes of a \c NTree. Log-odd representation is used for the occupation probability.
 *
 * Avoids overhead of virtual inheritance in sub classes \c LeafNodeProb and \c InnerNodeProb by not inherit from class \c Node due to diamond shape.
 */
class NodeProb
{
public:  
  typedef Environment::NodeDataProb NodeData;
  /**
  * The probability is used for ray casting whith this kind of nodes.
  */
  typedef struct
  {
    typedef Probability Type;
    Type value;
  } RayCastType;

  __device__ __host__
  NodeProb() :
      m_occupancy(0)
  {
  }

  __device__ __host__ __forceinline__
  Probability getOccupancy() const
  {
    return m_occupancy;
  }

  __device__ __host__ __forceinline__
  void setOccupancy(const Probability occupancy)
  {
    m_occupancy = occupancy;
  }

protected:
  /**
   * @brief log-odd representation of the occupation probability
   */
  Probability m_occupancy;
};

/**
 * @brief Probalilistic leaf node of a \c NTree.
 */
class LeafNodeProb: public LeafNode, public NodeProb
{
public:
  typedef NodeProb::NodeData NodeData;
  typedef NodeProb::RayCastType RayCastType;

  // default constructor needed otherwise nvcc bug:
  //UNREACHABLE executed!
  //Stack dump:
  //0.      Running pass 'NVPTX DAG->DAG Pattern Instruction Selection' on function '@_ZN15icl_environment3gpu5NTree17kernel_clearNodesINS1_11Environment9InnerNodeEEEvmPT_j'
  //Aborted (core dumped)
  __device__ __host__ __forceinline__
  LeafNodeProb()
  {
  }

#include "EnvNodesProbCommon.h"

  __device__ __host__ __forceinline__
  NodeData extractData(const OctreeVoxelID voxel_id, const voxel_count level) const
  {
    return NodeData(voxel_id, level, NodeData::BasicData(m_status, 0, m_occupancy));
  }
};

/**
 * @brief Probalilistic inner node of a \c NTree.
 * Has to implement nearly the same methods as \c LeafNodePro to avoid the overhead for virtual inheritance due to diamond shape.
 */
class InnerNodeProb: public InnerNode, public NodeProb
{
public:
  typedef NodeProb::NodeData NodeData;
  typedef NodeProb::RayCastType RayCastType;

  // default constructor needed
  __device__ __host__ __forceinline__
  InnerNodeProb()
  {
  }

  __device__ __host__ __forceinline__
  NodeData extractData(const OctreeVoxelID voxel_id, const voxel_count level) const
  {
    return NodeData(voxel_id, level, NodeData::BasicData(m_status, m_flags, m_occupancy));
  }

#include "EnvNodesProbCommon.h"

  //TODO remove/move out of Node
  __device__ __host__ __forceinline__
  bool isInConflict(const InnerNodeProb env_InnerNode) const
  {
    return isOccupied() & env_InnerNode.isOccupied();
  }
};

}  // end of ns

} // end of ns
} // end of ns

#endif /* ENVIRONMENT_NODES_H_ */
