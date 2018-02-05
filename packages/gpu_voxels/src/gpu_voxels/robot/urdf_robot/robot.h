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

#ifndef GPU_VOXELS_ROBOT_URDF_ROBOT_ROBOT_H_INCLUDED
#define GPU_VOXELS_ROBOT_URDF_ROBOT_ROBOT_H_INCLUDED

#include "gpu_voxels/robot/robot_interface.h"
#include "gpu_voxels/robot/urdf_robot/robot_link.h"
#include "gpu_voxels/robot/urdf_robot/robot_joint.h"
#include "gpu_voxels/robot/urdf_robot/node.h"
#include "gpu_voxels/helpers/MetaPointCloud.h"
#include <string>
#include <map>
#if __CUDACC_VER_MAJOR__ >= 9
#undef __CUDACC_VER__
#define __CUDACC_VER__ 90000
#endif
#include <Eigen/Geometry>
#include <urdf_model/model.h>
#include <kdl/frames_io.hpp>
#include <kdl_parser/kdl_parser.hpp>
#include <boost/filesystem.hpp>

namespace urdf {
  class ModelInterface;
  class Link;
  class Joint;
}

namespace gpu_voxels {
namespace robot {

typedef std::map< std::string, RobotLink* > M_NameToLink;
typedef std::map< std::string, RobotJoint* > M_NameToJoint;

/**
 * \class Robot
 *
 * A helper class to draw a representation of a robot, as specified by a URDF.  Can display either the visual models of the robot,
 * or the collision models.
 */
class Robot : public RobotInterface
{
public:
  Robot();
  virtual ~Robot();

  /**
   * \brief Loads meshes/primitives from a robot description.  Calls clear() before loading.
   *
   * @param urdf The robot description to read from
   * @param path_to_pointclouds The filesystem path to the pointclouds
   * @param discretization_distance Point distance when sampling geometric URDF primitives
   * @param visual Whether or not to load the visual representation
   * @param collision Whether or not to load the collision representation
   */
  void load( const urdf::ModelInterface &urdf, const boost::filesystem::path &path_to_pointclouds,
             const float discretization_distance, bool visual = true, bool collision = true);

  /**
   * \brief Clears all data loaded from a URDF
   */
  void clear();


  /**
   * @brief getJointNames Reads all joint names
   * @param jointnames Vector of jointnames that will get extended
   */
  void getJointNames(std::vector<std::string> &jointnames);

  /**
   * @brief update Sets the robot configuration
   * @param joint_values Map of jointnames and values
   */
  void setConfiguration(const JointValueMap &joint_values);

  /**
   * @brief getConfiguration Gets the robot configuration
   * @param joint_values Map of jointnames and values.
   * This will get enlarged about missing joint names.
   */
  void getConfiguration(JointValueMap &jointmap);

  /**
   * @brief getLowerJointLimits Gets the robot configuration
   * @param lower_limits Map of jointnames and values.
   * This will get enlarged about missing joint names.
   */
  void getLowerJointLimits(JointValueMap &lower_limits);

  /**
   * @brief getUpperJointLimits Gets the robot configuration
   * @param upper_limits Map of jointnames and values.
   * This will get enlarged about missing joint names.
   */
  void getUpperJointLimits(JointValueMap &upper_limits);

  /**
   * @brief updatePointcloud Changes the geometry of a single link.
   * Useful when grasping an object, changing a tool
   * or interpreting point cloud data from an onboard sensor as a robot link.
   * @param link_name Link to modify
   * @param cloud New geometry
   */
  void updatePointcloud(const std::string &link_name, const std::vector<Vector3f> &cloud);

  /**
   * @brief getTransformedClouds Only a dummy overwritten by robot_to_gpu!
   * @return NULL pointer
   */
  virtual const MetaPointCloud *getTransformedClouds();


  RobotLink* getRootLink() { return root_link_; }
  RobotLink* getLink( const std::string& name );
  RobotJoint* getJoint( const std::string& name );

  const M_NameToLink& getLinks() const { return links_; }
  const M_NameToJoint& getJoints() const { return joints_; }

  const std::string& getName() { return name_; }

  node* getVisualNode() { return root_visual_node_; }
  node* getCollisionNode() { return root_collision_node_; }
  node* getOtherNode() { return root_other_node_; }


  KDL::Frame getPose();
  void setPose(const KDL::Frame &pose);
  void setScale(const gpu_voxels::Vector3f &scale );

  const MetaPointCloud & getRobotClouds();

  const KDL::Tree& getTree() { return tree; }

  const MetaPointCloud* getLinkPointclouds() { return &link_pointclouds_; }

private:

  MetaPointCloud link_pointclouds_;

  KDL::Tree tree;



protected:

  M_NameToLink links_;                      ///< Map of name to link info, stores all loaded links.
  M_NameToJoint joints_;                    ///< Map of name to joint info, stores all loaded joints.
  RobotLink *root_link_;

  node* root_visual_node_;           ///< Node all our visual nodes are children of
  node* root_collision_node_;        ///< Node all our collision nodes are children of
  node* root_other_node_;

  std::string name_;
};

} // namespace robot
} // namespace gpu_voxels

#endif
