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

#include <math.h>

#include <urdf_model/model.h>
#include <urdf_model/link.h>

#include "gpu_voxels/logging/logging_robot.h"
#include "gpu_voxels/helpers/PointcloudFileHandler.h"
#include "gpu_voxels/helpers/GeometryGeneration.h"
#include "gpu_voxels/robot/urdf_robot/robot.h"
#include "gpu_voxels/robot/urdf_robot/robot_link.h"
#include "gpu_voxels/robot/urdf_robot/robot_joint.h"

namespace fs=boost::filesystem;

namespace gpu_voxels {
namespace robot {

RobotLink::RobotLink(Robot* robot,
                      const urdf::LinkConstPtr& link,
                      const std::string& parent_joint_name,
                      bool visual,
                      bool collision, const float discretization_distance,
                      const boost::filesystem::path &path_to_pointclouds,
                      MetaPointCloud& link_pointclouds)
: robot_( robot )
, name_( link->name )
, parent_joint_name_( parent_joint_name )
, visual_meshes_()
, collision_meshes_()
, discretization_distance_(discretization_distance)
, link_pointclouds_(link_pointclouds)
, pose_calculated_(false)
, path_to_pointclouds_(path_to_pointclouds)
{
  pose_property_ = KDL::Frame();

  visual_node_ = new node;
  collision_node_ = new node;
  offset_node = new node;

  // we prefer collisions over visuals
  if (collision) createCollision(link);
  else if (visual) createVisual(link);


  // create description and fill in child_joint_names_ vector
  std::stringstream desc;
  if (parent_joint_name_.empty())
  {
    desc << "Root Link " << name_;
  }
  else
  {
    desc << "Link " << name_;
    desc << " with parent joint " << parent_joint_name_;
  }

  if (link->child_joints.empty())
  {
    desc << " has no children.";
  }
  else
  {
    desc
      << " has " 
      << link->child_joints.size();

    if (link->child_joints.size() > 1)
    {
      desc << " child joints: ";
    }
    else
    {
      desc << " child joint: ";
    }

    std::vector<boost::shared_ptr<urdf::Joint> >::const_iterator child_it = link->child_joints.begin();
    std::vector<boost::shared_ptr<urdf::Joint> >::const_iterator child_end = link->child_joints.end();
    for ( ; child_it != child_end ; ++child_it )
    {
      urdf::Joint *child_joint = child_it->get();
      if (child_joint && !child_joint->name.empty())
      {
        child_joint_names_.push_back(child_joint->name);
        desc << child_joint->name << ((child_it+1 == child_end) ? "." : ", ");
      }
    }
  }
  if (hasGeometry())
  {
    if (visual_meshes_.empty())
    {
      desc << "  This link has collision geometry but no visible geometry.";
    }
    else if (collision_meshes_.empty())
    {
      desc << "  This link has visible geometry but no collision geometry.";
    }
  }
  else
  {
    desc << "  This link has NO geometry.";
  }

  LOGGING_DEBUG_C(RobotLog, RobotLink, desc.str() << endl);

}

RobotLink::~RobotLink()
{
  for( size_t i = 0; i < visual_meshes_.size(); i++ )
  {
    delete visual_meshes_[i];
  }
  visual_meshes_.clear();

  for( size_t i = 0; i < collision_meshes_.size(); i++ )
  {
    delete collision_meshes_[i];
  }
  collision_meshes_.clear();

  delete visual_node_;
  delete collision_node_;
  delete offset_node;
}

bool RobotLink::hasGeometry() const
{
  return visual_meshes_.size() + collision_meshes_.size() > 0;
}

void RobotLink::createEntityForGeometryElement(const urdf::LinkConstPtr& link, const urdf::Geometry& geom, const urdf::Pose& origin, node*& entity)
{
  gpu_voxels::Vector3f scale;

  KDL::Vector offset_pos = KDL::Vector(origin.position.x, origin.position.y, origin.position.z);
  KDL::Rotation offset_rot = KDL::Rotation::Quaternion(origin.rotation.x, origin.rotation.y, origin.rotation.z, origin.rotation.w);
  KDL::Frame offset_pose = KDL::Frame(offset_rot, offset_pos);

  PointCloud link_cloud;

  switch (geom.type)
  {
  case urdf::Geometry::SPHERE:
  {
    const urdf::Sphere& sphere = static_cast<const urdf::Sphere&>(geom);
    link_cloud.update(geometry_generation::createSphereOfPoints(Vector3f(0), sphere.radius, discretization_distance_));
    entity->setHasData(true);
    break;
  }
  case urdf::Geometry::BOX:
  {
    const urdf::Box& box = static_cast<const urdf::Box&>(geom);
    Vector3f half_dim(box.dim.x / 2.0, box.dim.y / 2.0, box.dim.z / 2.0);
    link_cloud.update(geometry_generation::createBoxOfPoints((Vector3f(0) - half_dim), (Vector3f(0) + half_dim), discretization_distance_));
    break;
  }
  case urdf::Geometry::CYLINDER:
  {
    const urdf::Cylinder& cylinder = static_cast<const urdf::Cylinder&>(geom);
    link_cloud.update(geometry_generation::createCylinderOfPoints(Vector3f(0), cylinder.radius, cylinder.length, discretization_distance_));
    entity->setHasData(true);
    break;
  }
  case urdf::Geometry::MESH:
  {
    std::vector<Vector3f> tmp_vec3f_cloud;
    const urdf::Mesh& mesh = static_cast<const urdf::Mesh&>(geom);

    if ( mesh.filename.empty() )
      return;

    scale = gpu_voxels::Vector3f(mesh.scale.x, mesh.scale.y, mesh.scale.z);
    
    fs::path p(mesh.filename);
    fs::path pc_file = path_to_pointclouds_ / fs::path(p.stem().string() + std::string(".binvox"));

    LOGGING_DEBUG_C(RobotLog, RobotLink, "Loading pointcloud of link " << pc_file.string() << endl);
    if(!file_handling::PointcloudFileHandler::Instance()->loadPointCloud(
         pc_file.string(), false, tmp_vec3f_cloud, false, Vector3f(0), 1.0))
    {
      LOGGING_ERROR_C(RobotLog, RobotLink,
                      "Could not read file [" << pc_file.string() <<
                      "]. Adding single point instead..." << endl);
      tmp_vec3f_cloud.push_back(Vector3f());
    }

    link_cloud.update(tmp_vec3f_cloud);
    entity->setHasData(true);

    break;
  }
  default:
    LOGGING_WARNING_C(RobotLog, RobotLink,
                      "Link " << link->name <<
                      " has unsupported geometry type " << geom.type << "." << endl);
    break;
  }

  if ( entity )
  {
    double roll, pitch, yaw;
    offset_rot.GetRPY(roll, pitch, yaw);

    Matrix4f trafo(Matrix4f::createFromRotationAndTranslation(Matrix3f::createFromRPY(roll, pitch, yaw),
                                                              Vector3f(origin.position.x, origin.position.y, origin.position.z)));
    link_cloud.transformSelf(&trafo);

    link_pointclouds_.addCloud(link_cloud, false, name_);

    // TODO: does this work as expected? offset_node is a member of RobotLink. createEntityForGeometryElement could be called for multiple collision groups!
    offset_node->setScale(scale);
    offset_node->setPose(offset_pose);
  }
}

void RobotLink::createCollision(const urdf::LinkConstPtr& link)
{
  bool valid_collision_found = false;
#if URDF_MAJOR_VERSION == 0 && URDF_MINOR_VERSION == 2
  std::map<std::string, boost::shared_ptr<std::vector<boost::shared_ptr<urdf::Collision> > > >::const_iterator mi;
  for( mi = link->collision_groups.begin(); mi != link->collision_groups.end(); mi++ )
  {
    if( mi->second )
    {
      std::vector<boost::shared_ptr<urdf::Collision> >::const_iterator vi;
      for( vi = mi->second->begin(); vi != mi->second->end(); vi++ )
      {
        boost::shared_ptr<urdf::Collision> collision = *vi;
        if( collision && collision->geometry )
        {
          node* collision_mesh = new node();
          createEntityForGeometryElement( link, *collision->geometry, collision->origin, collision_mesh );
          if( collision_mesh )
          {
            collision_meshes_.push_back( collision_mesh );
            valid_collision_found = true;
          }
        }
      }
    }
  }
#else
  std::vector<boost::shared_ptr<urdf::Collision> >::const_iterator vi;
  for( vi = link->collision_array.begin(); vi != link->collision_array.end(); vi++ )
  {
    boost::shared_ptr<urdf::Collision> collision = *vi;
    if( collision && collision->geometry )
    {
      node* collision_mesh = new node();
      createEntityForGeometryElement( link, *collision->geometry, collision->origin, collision_mesh );
      if( collision_mesh )
      {
        collision_meshes_.push_back( collision_mesh );
        valid_collision_found = true;
      }
    }
  }
#endif

  if( !valid_collision_found && link->collision && link->collision->geometry )
  {
    node* collision_mesh = new node();
    createEntityForGeometryElement( link, *link->collision->geometry, link->collision->origin, collision_mesh );
    if( collision_mesh )
    {
      collision_meshes_.push_back( collision_mesh );
    }
  }
}

void RobotLink::createVisual(const urdf::LinkConstPtr& link )
{
  bool valid_visual_found = false;
#if URDF_MAJOR_VERSION == 0 && URDF_MINOR_VERSION == 2
  std::map<std::string, boost::shared_ptr<std::vector<boost::shared_ptr<urdf::Visual> > > >::const_iterator mi;
  for( mi = link->visual_groups.begin(); mi != link->visual_groups.end(); mi++ )
  {
    if( mi->second )
    {
      std::vector<boost::shared_ptr<urdf::Visual> >::const_iterator vi;
      for( vi = mi->second->begin(); vi != mi->second->end(); vi++ )
      {
        boost::shared_ptr<urdf::Visual> visual = *vi;
        if( visual && visual->geometry )
        {
          node* visual_mesh = new node();
          createEntityForGeometryElement( link, *visual->geometry, visual->origin, visual_mesh );
          if( visual_mesh )
          {
            visual_meshes_.push_back( visual_mesh );
            valid_visual_found = true;
          }
        }
      }
    }
  }
#else
  std::vector<boost::shared_ptr<urdf::Visual> >::const_iterator vi;
  for( vi = link->visual_array.begin(); vi != link->visual_array.end(); vi++ )
  {
    boost::shared_ptr<urdf::Visual> visual = *vi;
    if( visual && visual->geometry )
    {
      node* visual_mesh = new node();
      createEntityForGeometryElement( link, *visual->geometry, visual->origin, visual_mesh );
      if( visual_mesh )
      {
        visual_meshes_.push_back( visual_mesh );
        valid_visual_found = true;
      }
    }
  }
#endif

  if( !valid_visual_found && link->visual && link->visual->geometry )
  {
    node* visual_mesh = new node();
    createEntityForGeometryElement( link, *link->visual->geometry, link->visual->origin, visual_mesh );
    if( visual_mesh )
    {
      visual_meshes_.push_back( visual_mesh );
    }
  }
}

void RobotLink::setPoses( const KDL::Frame& visual_pose, const KDL::Frame& collision_pose)
{
  visual_node_->setPose( visual_pose );
  collision_node_->setPose( collision_pose );

  pose_property_ = visual_pose;
}


KDL::Frame RobotLink::getPose()
{
  KDL::Frame ret;
  KDL::Frame parent_pose;
  KDL::Frame local_pose;

  RobotJoint* parent_joint;

  if(!pose_calculated_)
  {
    // if this is the root link, the parent pose is the robot pose
    if(this == robot_->getRootLink())
    {
      pose_property_ = robot_->getPose();
    }else{
      parent_joint = robot_->getJoint(parent_joint_name_);
      if(parent_joint)
      {
        parent_pose = parent_joint->getPose();
      }else{
        LOGGING_ERROR_C(RobotLog, RobotLink,
                        "RobotLink::getPose() could not find a parent joint called ["
                        << parent_joint_name_ << "]!" << endl);
        return ret;
      }

      // we now have the parent pose, around which we rotate the current segment:
      std::map<std::string,KDL::TreeElement>::const_iterator local_segment = robot_->getTree().getSegment(name_);

      if(local_segment != robot_->getTree().getSegments().end())
      {
        local_pose = local_segment->second.segment.pose(parent_joint->getJointValue());
        visual_node_->setPose(parent_pose);
      }else{
        LOGGING_ERROR_C(RobotLog, RobotLink,
                        "RobotLink::getPose() could not find local KDL segment named [" << name_ << "]!" << endl);
        return ret;
      }
      pose_property_ = parent_pose * local_pose;

    }
    pose_calculated_ = true;
  }
  return pose_property_;
}

Matrix4f RobotLink::getPoseAsGpuMat4f()
{
  // call getPose() explicitely and do NOT use pose_property_
  // to trigger recursive calculation
  KDL::Frame pose = getPose();

  Matrix4f transformation(
    pose.M.data[0], pose.M.data[1], pose.M.data[2], pose.p.x(),
    pose.M.data[3], pose.M.data[4], pose.M.data[5], pose.p.y(),
    pose.M.data[6], pose.M.data[7], pose.M.data[8], pose.p.z(),
    0.0,            0.0,            0.0,            1.0
  );

  return transformation;
}

} // namespace robot
} // namespace gpu_voxels

