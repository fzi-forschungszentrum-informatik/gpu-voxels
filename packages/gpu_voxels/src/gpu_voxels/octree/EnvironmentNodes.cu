// this is for emacs file handling -*- mode: c++; indent-tabs-mode: nil -*-

// -- BEGIN LICENSE BLOCK ----------------------------------------------
// -- END LICENSE BLOCK ------------------------------------------------

//----------------------------------------------------------------------
/*!\file
 *
 * \author  Florian Drews
 * \date    2013-12-11
 *
 */
//----------------------------------------------------------------------/*
#include "EnvironmentNodes.h"
#include "RobotNodes.h"


namespace gpu_voxels {
namespace NTree {
namespace Environment {

#ifndef DISABLE_SEPARATE_COMPILTION
__device__ __host__
bool LeafNode::isInConflict(Robot::LeafNode rob_LeafNode)
{
  return isOccupied() & rob_LeafNode.isOccupied();
}

__device__ __host__
bool InnerNode::isInConflict(Robot::InnerNode rob_InnerNode)
{
  return isOccupied() & rob_InnerNode.isOccupied();
}

#endif

} // end of ns
} // end of ns
} // end of ns
