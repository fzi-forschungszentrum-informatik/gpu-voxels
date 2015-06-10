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
 * \author  Willow Garage
 * \author  Andreas Hermann
 * \date    2015-03-11
 *
 * Largest parts of this class were copied from ROS rviz and adopted
 * to load pointclouds.
 *
 */
//----------------------------------------------------------------------

/*
 * Copyright (c) 2013, Willow Garage, Inc.
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 *     * Redistributions of source code must retain the above copyright
 *       notice, this list of conditions and the following disclaimer.
 *     * Redistributions in binary form must reproduce the above copyright
 *       notice, this list of conditions and the following disclaimer in the
 *       documentation and/or other materials provided with the distribution.
 *     * Neither the name of the Willow Garage, Inc. nor the names of its
 *       contributors may be used to endorse or promote products derived from
 *       this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 * ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE
 * LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
 * CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
 * SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
 * INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
 * CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
 * ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 * POSSIBILITY OF SUCH DAMAGE.
 */

#ifndef GPU_VOXELS_ROBOT_URDF_ROBOT_ROBOT_JOINT_H_INCLUDED
#define GPU_VOXELS_ROBOT_URDF_ROBOT_ROBOT_JOINT_H_INCLUDED

#include <string>
#include <map>

#include "gpu_voxels/helpers/cuda_datatypes.h"
#include "gpu_voxels/robot/urdf_robot/node.h"
#include <boost/shared_ptr.hpp>
#include <Eigen/Geometry>
#include <kdl/frames.hpp>

namespace urdf {
  class ModelInterface;
  class Link;
  class Joint;
  class Geometry;
  class Pose;
}

namespace gpu_voxels {
namespace robot {

class Robot;

/**
 * \struct RobotJoint
 * \brief Contains any data we need from a joint in the robot.
 */
class RobotJoint
{
public:
  RobotJoint( Robot* robot, const boost::shared_ptr<const urdf::Joint>& joint );
  virtual ~RobotJoint();

  KDL::Frame getPose();
  void setPose(const KDL::Frame &parent_link_pose);

  const std::string& getName() const { return name_; }
  const std::string& getParentLinkName() const { return parent_link_name_; }
  const std::string& getChildLinkName() const { return child_link_name_; }
  RobotJoint* getParentJoint();

  bool poseCalculated() { return pose_calculated_; }
  void resetPoseCalculated() { pose_calculated_ = false; }
  void setJointValue(float joint_value);
  float getJointValue() { return joint_value_; }

protected:
  Robot* robot_;
  std::string name_;
  std::string parent_link_name_;
  std::string child_link_name_;

private:
  bool pose_calculated_;
  float joint_value_;
  KDL::Frame joint_origin_pose_;
  KDL::Frame pose_property_;


};

} // namespace robot
} // namespace gpu_voxels

#endif
