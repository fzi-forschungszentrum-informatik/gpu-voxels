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
 * \date    2012-09-13
 *
 */
//----------------------------------------------------------------------

#include "Kinect.h"

#include <iostream>
#include <math.h>
#include <utility>

namespace gpu_voxels {

Kinect::Kinect(std::string identifier)
  : m_data(640*480)
  , m_running(false)
  , m_identifier(identifier)
{
}

Kinect::~Kinect()
{
  if (m_running)
  {
    stop();
  }
}


void Kinect::cloud_callback(const pcl::PointCloud<pcl::PointXYZ>::ConstPtr &cloud)
{

  for (uint32_t i=0; i<cloud->points.size(); i++)
  {
    m_data[i].x = cloud->points[i].x;
    m_data[i].y = cloud->points[i].y;
    m_data[i].z = cloud->points[i].z;

    //    // cut kinect data to a specific range: (debugging)
    //    const float max_range = 2500;
    //    if (m_data[i].z > max_range)
    //    {
    //      m_data[i].x = NAN;
    //      m_data[i].y = NAN;
    //      m_data[i].z = NAN;
    //    }

    //printf("kinect point: %f, %f, %f\n", m_data[i].x,m_data[i].y, m_data[i].z);
  }

  LOGGING_TRACE_C(Gpu_voxels_helpers, Kinect, "Kinect point cloud callback: point count: " << cloud->points.size() << endl);
}

void Kinect::run()
{
  LOGGING_INFO_C(Gpu_voxels_helpers, Kinect, "Kinect: starting capture interface." << endl);
  m_interface = new pcl::OpenNIGrabber(m_identifier);

  boost::function<void(const pcl::PointCloud<pcl::PointXYZ>::ConstPtr&)> f_cb =
      boost::bind(&Kinect::cloud_callback, this, _1);

  m_interface->registerCallback(f_cb);

  m_interface->start();
  m_running = true;
  LOGGING_INFO_C(Gpu_voxels_helpers, Kinect, "Kinect: capture interface started." << endl);
}

void Kinect::stop()
{
  LOGGING_INFO_C(Gpu_voxels_helpers, Kinect, "Kinect: stopping capture interface." << endl);
  m_interface->stop();
  delete m_interface;
  m_running = false;
  LOGGING_INFO_C(Gpu_voxels_helpers, Kinect, "Kinect: capture interface stopped" << endl);
}

bool Kinect::isRunning()
{
  return m_running;
}

} // end of namespace gpu_voxels
