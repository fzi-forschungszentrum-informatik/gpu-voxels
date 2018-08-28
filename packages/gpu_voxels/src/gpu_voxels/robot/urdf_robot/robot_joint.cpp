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
 * Copyright (c) 2008, Willow Garage, Inc.
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

#include "gpu_voxels/robot/urdf_robot/robot_joint.h"
#include "gpu_voxels/robot/urdf_robot/robot_link.h"
#include "gpu_voxels/robot/urdf_robot/robot.h"

#include <urdf_model/model.h>
#include <urdf_model/link.h>
#include <urdf_model/joint.h>

#include "gpu_voxels/logging/logging_robot.h"

namespace gpu_voxels {
namespace robot {

RobotJoint::RobotJoint( Robot* robot, const urdf::JointConstPtr& joint )
  : robot_( robot )
  , name_( joint->name )
  , parent_link_name_( joint->parent_link_name )
  , child_link_name_( joint->child_link_name )
  , pose_calculated_( false )
  , joint_value_(0.0)
{
  pose_property_ = KDL::Frame();

  const urdf::Vector3& pos = joint->parent_to_joint_origin_transform.position;
  const urdf::Rotation& rot = joint->parent_to_joint_origin_transform.rotation;

  KDL::Vector pos2 = KDL::Vector(pos.x, pos.y, pos.z);


  KDL::Rotation rot2 = KDL::Rotation::Quaternion(rot.x, rot.y, rot.z, rot.w);

  joint_origin_pose_ = KDL::Frame(rot2, pos2);
  if(joint->limits)
  {
    lower_joint_limit_ = joint->limits->lower;
    upper_joint_limit_ = joint->limits->upper;
  }else{
    lower_joint_limit_ = 0.0;
    upper_joint_limit_ = 0.0;
  }

  LOGGING_DEBUG_C(RobotLog, RobotJoint,
                  "Creating new Robot Joint named " << name_ << "Min = " << lower_joint_limit_ << " Max = " << upper_joint_limit_ << endl);
}

RobotJoint::~RobotJoint()
{
}

RobotJoint* RobotJoint::getParentJoint()
{
  RobotLink* parent_link = robot_->getLink(parent_link_name_);
  if (!parent_link)
    return NULL;

  const std::string& parent_joint_name = parent_link->getParentJointName();
  if (parent_joint_name.empty())
    return NULL;

  return robot_->getJoint(parent_joint_name);
}

void RobotJoint::setJointValue(float joint_value)
{
  joint_value_ = joint_value;
}

void RobotJoint::setPose(const KDL::Frame &parent_link_pose)
{
  pose_property_ = parent_link_pose;
}

/*!
 * Only returns the origin of the joint.
 * NO rotation is calculated here!
 */
KDL::Frame RobotJoint::getPose()
{
  KDL::Frame ret;

  RobotLink* parent_link;

  if(!pose_calculated_)
  {
    parent_link = robot_->getLink(parent_link_name_);
    if(parent_link)
    {
      pose_property_ = parent_link->getPose();
    }else{
      LOGGING_ERROR_C(RobotLog, RobotJoint,
                      "RobotJoint::getPose() could not find a parent link called [" << parent_link_name_ << "]!" << endl);
      return ret;
    }
    pose_calculated_ = true;
  }
  return pose_property_;
}

} // namespace robot
} // namespace gpu_voxels

