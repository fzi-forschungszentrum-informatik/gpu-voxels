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
 * \author  Andreas Hermann
 * \date    2015-03-11
 *
 * This defines a general Node of a robot, defined by its pose
 * and scale.
 *
 */
//----------------------------------------------------------------------

#include <gpu_voxels/robot/urdf_robot/node.h>

namespace gpu_voxels {
namespace robot {

node::node() {
  visible = true;
  has_data = false;
}

void node::setVisible(bool visibility)
{
  visible = visibility;
}

void node::setHasData(bool flag)
{
  has_data = flag;
}

bool node::hasData()
{
  return has_data;
}

void node::setPose(const KDL::Frame &_pose)
{
  pose = _pose;
}
void node::setScale(gpu_voxels::Vector3f _scale)
{
  scale = _scale;
}

KDL::Frame node::getPose()
{
  return pose;
}

} // end of NS
} // end of NS
