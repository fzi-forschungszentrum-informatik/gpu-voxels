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
 * \author  Sebastian Klemm
 * \date    2012-09-14
 *
 */
//----------------------------------------------------------------------
#ifndef GPU_VOXELS_KINECT_H_INCLUDED
#define GPU_VOXELS_KINECT_H_INCLUDED

#include <pcl/io/pcd_io.h>
#include <pcl/point_types.h>
#include <pcl/io/openni_grabber.h>
#include <pcl/visualization/cloud_viewer.h>
#include <pcl/visualization/image_viewer.h>
#include <pcl/pcl_macros.h>

#include <boost/thread.hpp>
#include <vector>
#include <gpu_voxels/helpers/cuda_datatypes.h>

//#include "NTree.h"
//#include "EnvironmentNodes.h"
#include <gpu_voxels/octree/test/Provider.h>
#include <gpu_voxels/octree/DataTypes.h>
//#include "Sensor.h"
#include <gpu_voxels/octree/test/ArgumentHandling.h>
#include <gpu_voxels/octree/test/SensorData.h>

#include <gpu_voxels/logging/logging_octree.h>

namespace gpu_voxels {
namespace NTree {

class ONIWrapper: public pcl::OpenNIGrabber
{
public:

  boost::shared_ptr<pcl::PointCloud<pcl::PointXYZ> > convertToXYZPointCloud(
      const boost::shared_ptr<openni_wrapper::DepthImage> &depth) const
  {

//    if (depth_map[depth_idx] == 0 || depth_map[depth_idx] == depth->getNoSampleValue()
//        || depth_map[depth_idx] == depth->getShadowValue())
//    {
//      // not valid
//      pt.x = pt.y = pt.z = bad_point;
//      continue;
//    }

    return this->OpenNIGrabber::convertToXYZPointCloud(depth);
  }

  /*
   * Software License Agreement (BSD License)
   *
   * Point Cloud Library (PCL) - www.pointclouds.org
   * Copyright (c) 2010-2011, Willow Garage, Inc.
   * Copyright (c) 2012-, Open Perception, Inc.
   *
   * All rights reserved.
   *
   * Redistribution and use in source and binary forms, with or without
   * modification, are permitted provided that the following conditions
   * are met:
   *
   * * Redistributions of source code must retain the above copyright
   * notice, this list of conditions and the following disclaimer.
   * * Redistributions in binary form must reproduce the above
   * copyright notice, this list of conditions and the following
   * disclaimer in the documentation and/or other materials provided
   * with the distribution.
   * * Neither the name of the copyright holder(s) nor the names of its
   * contributors may be used to endorse or promote products derived
   * from this software without specific prior written permission.
   *
   * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
   * "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
   * LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS
   * FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE
   * COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,
   * INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
   * BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
   * LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
   * CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
   * LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN
   * ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
   * POSSIBILITY OF SUCH DAMAGE.
   *
   */
  void getConvertParameter(uint32_t& width, uint32_t& height, float& constant_x, float& constant_y,
                           float& centerX, float& centerY)
  {
    // https://github.com/PointCloudLibrary/pcl/blob/master/io/src/openni_grabber.cpp
    width = depth_width_;
    height = depth_height_;

    constant_x = 1.0f / device_->getDepthFocalLength(depth_width_);
    constant_y = 1.0f / device_->getDepthFocalLength(depth_width_);
    centerX = ((float) depth_width_ - 1.f) / 2.f;
    centerY = ((float) depth_height_ - 1.f) / 2.f;

    if (pcl_isfinite (depth_focal_length_x_))
      constant_x = 1.0f / static_cast<float>(depth_focal_length_x_);

    if (pcl_isfinite (depth_focal_length_y_))
      constant_y = 1.0f / static_cast<float>(depth_focal_length_y_);

    if (pcl_isfinite (depth_principal_point_x_))
      centerX = static_cast<float>(depth_principal_point_x_);

    if (pcl_isfinite (depth_principal_point_y_))
      centerY = static_cast<float>(depth_principal_point_y_);

//    no sample value 0 no shadow value 0
//    width 640 height 480
//    constant_x 0.001753 constant_y 0.001753
//    centerX 319.500000 centerY 239.500000



    printf("width %u height %u\n", width, height);
    printf("constant_x %f constant_y %f\n", constant_x, constant_y);
    printf("centerX %f centerY %f\n", centerX, centerY);

//    if (depth_map[depth_idx] == 0 ||
//                  depth_map[depth_idx] == depth->getNoSampleValue () ||
//                  depth_map[depth_idx] == depth->getShadowValue ())
//              {
//                // not valid
//                pt.x = pt.y = pt.z = bad_point;
//                continue;
//              }
  }
};

class Kinect: public SensorData
{
public:

//  typedef NTree<Provider::BRANCHING_FACTOR, Provider::LEVEL_COUNT, Environment::InnerNode,
//      Environment::LeafNode> EnvironmentTree;

//! Constructor
  Kinect(Provider::Provider* provider, const Provider::Provider_Parameter* parameter);

//  Kinect(EnvironmentTree* m_ntree, Sensor* sensor, Vector3f voxel_dimension, bool enable_raycasting,
//         bool cut_real_robot);

//! Destructor
  virtual ~Kinect();

  virtual void run();
  virtual void stop();
  virtual bool isRunning();
  virtual void takeImage();

protected:
  pcl::Grabber* m_interface;
  Vector3f* m_data;
//  EnvironmentTree* m_ntree;
//  Sensor* m_sensor;
//  Vector3f m_voxel_dimension;
//  ONIWrapper* oni_wrapper;

  bool m_running;
//  bool m_enable_raycasting;
//  bool m_cut_real_robot;
  volatile uint32_t m_counter;
  bool m_working;
  uint32_t m_frame;
  float m_frame_per_sec;
  const Provider::Provider_Parameter* m_parameter;

  static const uint32_t width = 640;
  static const uint32_t height = 480;
  static const uint32_t capture_size = width * height;
  double m_avg_callback;

  //pcl::visualization::CloudViewer m_viewer;

  virtual void cloud_callback(const pcl::PointCloud<pcl::PointXYZ>::ConstPtr &cloud);

  virtual void depth_callback(const boost::shared_ptr<openni_wrapper::DepthImage> &depth_image);

};


}
}
#endif
