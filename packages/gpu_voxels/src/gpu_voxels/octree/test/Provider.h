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
 * \author  Florian Drews
 * \date    2014-04-04
 *
 */
//----------------------------------------------------------------------

#ifndef GPU_VOXELS_OCTREE_TEST_PROVIDER_H_INCLUDED
#define GPU_VOXELS_OCTREE_TEST_PROVIDER_H_INCLUDED
#include <gpu_voxels/helpers/CompileIssues.h>

#include <gpu_voxels/octree/test/ArgumentHandling.h>
#include <gpu_voxels/octree/PointCloud.h>
#include <gpu_voxels/octree/Sensor.h>
#include <gpu_voxels/helpers/cuda_datatypes.h>

#include <boost/interprocess/managed_shared_memory.hpp>
#include <boost/interprocess/containers/vector.hpp>
#include <boost/interprocess/allocators/allocator.hpp>
#include <boost/lexical_cast.hpp>
#include <boost/thread.hpp>

namespace gpu_voxels {
namespace NTree {
namespace Provider {

/**
 * Superclass for different map providers like NTree and VoxelMap.
 */
class Provider
{
public:

  Provider();

  virtual ~Provider();

  /**
   * Initialize the Provider with the given parameter.
   */
  virtual void init(Provider_Parameter& parameter);

  /**
   * Let the visualizer render a new frame.
   */
  virtual void visualize() = 0;

  /**
   * Make collision check against Provider set by setCollideWith(..)
   */
  virtual void collide() = 0;

  /**
   * Waits till there is new data for the visualizer
   */
  virtual bool waitForNewData(volatile bool* stop) = 0;

  /**
   * Callback method for new kinect data
   */
  virtual void newSensorData(const DepthData* h_depth_data, const uint32_t width, const uint32_t height) = 0;

  /**
   * Callback method for new kinect data
   */
  virtual void newSensorData(gpu_voxels::Vector3f* h_point_cloud, const uint32_t num_points, const uint32_t width,
                             const uint32_t height) = 0;


  /**
   * Update the sensor pose, to insert new data at the right place in the map.
   */
  virtual void updateSensorPose(float yaw, float pitch, float roll, gpu_voxels::Vector3f position);

  /**
   * Update the sensor pose, to insert new data at the right place in the map.
   */
  virtual void updateSensorPose(float yaw, float pitch, float roll);

  /**
   * Set the Provider to make collision checks with.
   */
  virtual void setCollideWith(Provider* collide_with);

  virtual void setChanged(bool changed);

  virtual bool getChanged();

  virtual void lock();

  virtual void unlock();

protected:
  boost::mutex m_mutex;
  bool m_changed;
  Sensor m_sensor;
  gpu_voxels::Vector3f m_sensor_orientation;
  gpu_voxels::Vector3f m_sensor_position;
  Provider* m_collide_with;
  std::string m_shared_mem_id;
  std::string m_segment_name;
  boost::interprocess::managed_shared_memory m_segment;
  Provider_Parameter* m_parameter;

  static const uint32_t buffer_watch_delay = 10000; // 100 fps
};

}
}
}

#endif
