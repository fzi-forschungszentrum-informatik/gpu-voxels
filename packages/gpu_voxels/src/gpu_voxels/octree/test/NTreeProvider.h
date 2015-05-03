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
 * \date    2014-02-13
 *
 */
//----------------------------------------------------------------------

#ifndef NTREEPROVIDER_H_
#define NTREEPROVIDER_H_

#include <gpu_voxels/octree/test/Provider.h>
#include <gpu_voxels/octree/EnvironmentNodes.h>
#include <gpu_voxels/octree/EnvNodesProbabilistic.h>
#include <gpu_voxels/octree/NTree.h>
#include <gpu_voxels/octree/DataTypes.h>
#include <gpu_voxels/octree/DefaultCollider.h>
#include <float.h>

// ROS
#include <ros/ros.h>
#include <sensor_msgs/PointCloud2.h>
#include <tf/transform_listener.h>

namespace gpu_voxels {
namespace NTree {
namespace Provider {

static const uint32_t BRANCHING_FACTOR = 8;
static const uint32_t LEVEL_COUNT = 15;
static const uint64_t NUM_VOXEL = pow(BRANCHING_FACTOR, LEVEL_COUNT - 1);

class NTreeProvider: public Provider
{
public:
#ifdef PROBABILISTIC_TREE
  typedef Environment::InnerNodeProb InnerNode;
  typedef Environment::LeafNodeProb LeafNode;
#else
  typedef Environment::InnerNode InnerNode;
  typedef Environment::LeafNode LeafNode;
#endif


  NTreeProvider();

  virtual ~NTreeProvider();

  virtual void visualize();

  virtual void init(Provider_Parameter& parameter);

  virtual void newSensorData(const DepthData* h_depth_data, const uint32_t width, const uint32_t height);

  virtual void newSensorData(gpu_voxels::Vector3f* h_point_cloud, const uint32_t num_points, const uint32_t width,
                             const uint32_t height);
  virtual void collide();

  virtual bool waitForNewData(volatile bool* stop);

  // Needed for ROS handling
  void ros_point_cloud(const sensor_msgs::PointCloud2::ConstPtr& msg, const uint32_t type);

  void ros_point_cloud_back(const sensor_msgs::PointCloud2::ConstPtr& msg);

  void ros_point_cloud_front(const sensor_msgs::PointCloud2::ConstPtr& msg);

protected:
  virtual uint32_t generateCubes_wo_locking(Cube** cubes);

  virtual void collide_wo_locking();

protected:
  NTree<BRANCHING_FACTOR, LEVEL_COUNT, InnerNode, LeafNode>* m_ntree;
  uint32_t m_min_level;
  uint32_t* m_shm_superVoxelSize;
  cudaIpcMemHandle_t* m_shm_memHandle;
  uint32_t* m_shm_numCubes;
  bool* m_shm_bufferSwapped;
  int32_t m_fps_rebuild;
  Vector3ui map_data_offset;
  static boost::mutex m_shared_mutex;

  // Needed for ROS handling
  ros::AsyncSpinner* m_spinner;
  ros::NodeHandle* m_node_handle;
  ros::Subscriber* m_subscriber_front;
  ros::Subscriber* m_subscriber_back;
  tf::TransformListener* m_tf_listener;
  thrust::device_vector<Voxel> *d_free_space_voxel;
  thrust::device_vector<Voxel> *d_object_voxel;
  thrust::device_vector<Voxel> *d_free_space_voxel2;
  thrust::device_vector<Voxel> *d_object_voxel2;
  bool m_internal_buffer_1;
  thrust::device_vector<Cube> *m_d_cubes_1;
  thrust::device_vector<Cube> *m_d_cubes_2;
};

}
}
}

#endif /* NTREEPROVIDER_H_ */
