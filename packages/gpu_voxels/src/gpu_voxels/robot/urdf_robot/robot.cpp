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

#include "gpu_voxels/robot/urdf_robot/robot.h"
#include "gpu_voxels/logging/logging_robot.h"

namespace gpu_voxels {
namespace robot {

Robot::Robot()
{
  root_visual_node_    = new node;
  root_collision_node_ = new node;
  root_other_node_     = new node;
}

Robot::~Robot()
{
  clear();
}

void Robot::clear()
{
  M_NameToLink::iterator link_it = links_.begin();
  M_NameToLink::iterator link_end = links_.end();
  for ( ; link_it != link_end; ++link_it )
  {
    RobotLink* link = link_it->second;
    delete link;
  }

  M_NameToJoint::iterator joint_it = joints_.begin();
  M_NameToJoint::iterator joint_end = joints_.end();
  for ( ; joint_it != joint_end; ++joint_it )
  {
    RobotJoint* joint = joint_it->second;
    delete joint;
  }

  links_.clear();
  joints_.clear();

}

void Robot::load( const urdf::ModelInterface &urdf, bool visual, bool collision )
{
  // clear out any data (properties, shapes, etc) from a previously loaded robot.
  clear();

  // the root link is discovered below.  Set to NULL until found.
  root_link_ = NULL;

  // Create properties for each link.
  {
    typedef std::map<std::string, boost::shared_ptr<urdf::Link> > M_NameToUrdfLink;
    M_NameToUrdfLink::const_iterator link_it = urdf.links_.begin();
    M_NameToUrdfLink::const_iterator link_end = urdf.links_.end();
    for( ; link_it != link_end; ++link_it )
    {
      const boost::shared_ptr<const urdf::Link>& urdf_link = link_it->second;
      std::string parent_joint_name;

      if (urdf_link != urdf.getRoot() && urdf_link->parent_joint)
      {
        parent_joint_name = urdf_link->parent_joint->name;
      }

      RobotLink* link = new RobotLink(this, urdf_link, parent_joint_name,
                                   visual, collision, link_pointclouds_);

      if (urdf_link == urdf.getRoot())
      {
        root_link_ = link;
      }

      links_[urdf_link->name] = link;
    }
  }

  // Create properties for each joint.
  {
    typedef std::map<std::string, boost::shared_ptr<urdf::Joint> > M_NameToUrdfJoint;
    M_NameToUrdfJoint::const_iterator joint_it = urdf.joints_.begin();
    M_NameToUrdfJoint::const_iterator joint_end = urdf.joints_.end();
    for( ; joint_it != joint_end; ++joint_it )
    {
      const boost::shared_ptr<const urdf::Joint>& urdf_joint = joint_it->second;
      RobotJoint* joint = new RobotJoint(this, urdf_joint);

      joints_[urdf_joint->name] = joint;
    }
  }

  link_pointclouds_.syncToDevice(); // after all links have been created, we sync them to the GPU

  // finally create a KDL representation of the kinematic tree:
  if (!kdl_parser::treeFromUrdfModel(urdf, tree)){
    LOGGING_ERROR_C(RobotLog, Robot,
                    "Failed to extract kdl tree from xml robot description!" << endl);
    exit(-1);
  }

  LOGGING_INFO_C(RobotLog, Robot,
                 "Constructed KDL tree has " << tree.getNrOfJoints() << " Joints and "
                 << tree.getNrOfSegments() << " segments." << endl);

}

RobotLink* Robot::getLink( const std::string& name )
{
  M_NameToLink::iterator it = links_.find( name );
  if ( it == links_.end() )
  {
    LOGGING_ERROR_C(RobotLog, Robot,
                    "Link " << name << " does not exist!" << endl);
    return NULL;
  }
  return it->second;
}

RobotJoint* Robot::getJoint( const std::string& name )
{
  M_NameToJoint::iterator it = joints_.find( name );
  if ( it == joints_.end() )
  {
    LOGGING_ERROR_C(RobotLog, Robot,
                    "Joint " << name << " does not exist!" << endl);
    return NULL;
  }
  return it->second;
}

const MetaPointCloud& Robot::getRobotClouds()
{
  return link_pointclouds_;
}

void Robot::updatePointcloud(const std::string &link_name, const std::vector<Vector3f> &cloud)
{
  link_pointclouds_.updatePointCloud(link_name, cloud, true);
}

void Robot::setConfiguration(const std::map<std::string, float> &joint_values)
{
  // Reset all calculation flags.
  for (M_NameToLink::const_iterator link=links_.begin(); link != links_.end(); link++)
  {
    link->second->resetPoseCalculated();
  }

  for (M_NameToJoint::const_iterator joint=joints_.begin(); joint != joints_.end(); joint++)
  {
    joint->second->resetPoseCalculated();
  }

  for (std::map<std::string, float>::const_iterator joint=joint_values.begin(); joint != joint_values.end(); joint++)
  {
    RobotJoint* rob_joint = getJoint(joint->first);
    if(rob_joint)
    {
      rob_joint->setJointValue(joint->second);
    }else{
      LOGGING_ERROR_C(RobotLog, Robot,
                      "Joint " << joint->first << " not found and not updated." << endl);
    }
  }
}

void Robot::getConfiguration(std::map<std::string, float> jointmap)
{
  for (M_NameToJoint::const_iterator joint=joints_.begin(); joint != joints_.end(); joint++)
  {
    jointmap[joint->first] = joint->second->getJointValue();
  }
}

void Robot::setPose(const KDL::Frame &pose )
{
  root_visual_node_->setPose( pose );
  root_collision_node_->setPose( pose );
}


void Robot::setScale(const gpu_voxels::Vector3f &scale )
{
  root_visual_node_->setScale( scale );
  root_collision_node_->setScale( scale );
}

KDL::Frame Robot::getPose()
{
  return root_visual_node_->getPose();
}


} // namespace robot
} // namespace gpu_voxels
