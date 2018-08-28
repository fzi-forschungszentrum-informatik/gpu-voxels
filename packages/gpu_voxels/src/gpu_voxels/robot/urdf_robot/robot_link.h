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

#ifndef GPU_VOXELS_ROBOT_URDF_ROBOT_ROBOT_LINK_H_INCLUDED
#define GPU_VOXELS_ROBOT_URDF_ROBOT_ROBOT_LINK_H_INCLUDED

#include <string>
#include <map>
#include "gpu_voxels/robot/urdf_robot/node.h"
#include "gpu_voxels/robot/urdf_robot/robot_joint.h"
#include "gpu_voxels/helpers/MetaPointCloud.h"
#if __CUDACC_VER_MAJOR__ >= 9
#undef __CUDACC_VER__
#define __CUDACC_VER__ 90000
#endif
#include <Eigen/Geometry>
#include <kdl/frames.hpp>
#include <boost/filesystem.hpp>

#include <boost/typeof/typeof.hpp>
#include <boost/utility/declval.hpp>
#include <urdf/model.h>

namespace urdf {
  class ModelInterface;
  class Link;
  typedef BOOST_TYPEOF(boost::declval<urdf::ModelInterface>().links_) VectorOfLinkConstPtr;
  typedef typename VectorOfLinkConstPtr::mapped_type LinkConstPtr;
  class Geometry;
  typedef boost::shared_ptr<const Geometry> GeometryConstPtr;
  class Pose;
}

class Robot;

namespace gpu_voxels {
namespace robot {

/**
 * \struct RobotLink
 * \brief Contains any data we need from a link in the robot.
 */
class RobotLink
{
public:
  RobotLink(Robot* robot,
             const urdf::LinkConstPtr& link,
             const std::string& parent_joint_name,
             bool visual,
             bool collision,
             const float discretization_distance,
             const boost::filesystem::path &path_to_pointclouds,
             MetaPointCloud& link_pointclouds);

  virtual ~RobotLink();

  // access
  const std::string& getName() const { return name_; }
  const std::string& getParentJointName() const { return parent_joint_name_; }
  const std::vector<std::string>& getChildJointNames() const { return child_joint_names_; }

  node* getVisualNode() const { return visual_node_; }
  node* getCollisionNode() const { return collision_node_; }
  Robot* getRobot() const { return robot_; }


  void setPoses( const KDL::Frame& visual_pose, const KDL::Frame& collision_pose);


  KDL::Frame getPose();
  Matrix4f getPoseAsGpuMat4f();

  bool hasGeometry() const;

  void resetPoseCalculated() { pose_calculated_ = false; }
  bool poseCalculated() { return pose_calculated_; }


private:
  void createEntityForGeometryElement( const urdf::LinkConstPtr& link, const urdf::Geometry& geom, const urdf::Pose& origin, node*& entity );

  void createVisual( const urdf::LinkConstPtr& link);
  void createCollision( const urdf::LinkConstPtr& link);

  Robot* robot_;
  std::string name_;
  std::string parent_joint_name_;
  std::vector<std::string> child_joint_names_;

  KDL::Frame pose_property_;
  std::vector<node*> visual_meshes_;    ///< The entities representing the visual mesh of this link (if they exist)
  std::vector<node*> collision_meshes_; ///< The entities representing the collision mesh of this link (if they exist)

  node* visual_node_;              ///< The scene node the visual meshes are attached to
  node* collision_node_;           ///< The scene node the collision meshes are attached to
  node* offset_node;

  float discretization_distance_;
  MetaPointCloud& link_pointclouds_;
  bool pose_calculated_;
  boost::filesystem::path path_to_pointclouds_;
};

} // namespace robot
} // namespace gpu_voxels

#endif
