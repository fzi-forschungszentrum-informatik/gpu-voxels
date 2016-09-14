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
 * This defines a general Node of a robot, defined by its pose.
 * Can also contain a PointCloud, if it is a link.
 *
 */
//----------------------------------------------------------------------
#ifndef GPU_VOXELS_ROBOT_URDF_ROBOT_NODE_H_INCLUDED
#define GPU_VOXELS_ROBOT_URDF_ROBOT_NODE_H_INCLUDED

#include <vector>

#include <gpu_voxels/helpers/cuda_datatypes.h>
#include <gpu_voxels/logging/logging_robot.h>
#include <kdl/frames.hpp>

namespace gpu_voxels {
namespace robot {

class node
{
private:
  bool visible;
  KDL::Frame pose;
  Vector3f scale;
  bool has_data;

public:
  node();

  void setVisible(bool visibility);
  void setHasData(bool flag);
  bool hasData();
  void setPose(const KDL::Frame& _pose);
  void setScale(gpu_voxels::Vector3f _scale);

  KDL::Frame getPose();

};



} // end of NS
} // end of NS

#endif
