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
 * \author  Andreas Hermann
 * \date    2014-06-08
 *
 * This high level API offers a lot of convenience functionality
 * and additional sanity checks. If you prefer a streamlined high
 * performance API, you can manage the shared pointers to maps and
 * robots by yourself and directly work with them.
 *
 */
//----------------------------------------------------------------------
#ifndef GPU_VOXELS_GPU_VOXELS_H_INCLUDED
#define GPU_VOXELS_GPU_VOXELS_H_INCLUDED

#include <cstdio>
#include <iostream>
#include <map>
#include <boost/shared_ptr.hpp>
#include <stdint.h> // for fixed size datatypes

#include <gpu_voxels/GpuVoxelsMap.h>
#include <gpu_voxels/ManagedMap.h>
#include <gpu_voxels/vis_interface/VisVoxelMap.h>
#include <gpu_voxels/helpers/MetaPointCloud.h>
#include <gpu_voxels/voxelmap/VoxelMap.h>
#include <gpu_voxels/octree/Octree.h>

#include <gpu_voxels/robot/KinematicLink.h>
#include <gpu_voxels/robot/KinematicChain.h>

#include <boost/interprocess/managed_shared_memory.hpp>
#include <boost/interprocess/containers/vector.hpp>
#include <boost/interprocess/allocators/allocator.hpp>
#include <boost/lexical_cast.hpp>

#include <gpu_voxels/logging/logging_gpu_voxels.h>

/**
 * @namespace gpu_voxels
 * Library for GPU based Voxel Collision Detection
 */
namespace gpu_voxels {

typedef boost::shared_ptr<cudaIpcMemHandle_t> CudaIpcMemHandleSharedPtr;
typedef std::map<std::string, ManagedMap > ManagedMaps;
typedef ManagedMaps::iterator ManagedMapsIterator;

typedef boost::shared_ptr<KinematicLink> KinematicLinkSharedPtr;

typedef boost::shared_ptr<KinematicChain> KinematicChainSharedPtr;
typedef std::map<std::string, KinematicChainSharedPtr > ManagedRobots;
typedef ManagedRobots::iterator ManagedRobotsIterator;


class GpuVoxels
{
public:
  /*!
   * \brief gvl Constructor, that defines the general resolution and size of the represented volume.
   * This is relevant for the VoxelMap / VoxelList.
   * The Octree depth will be chosen accordingly.
   *
   * \param dim_x The map's x dimension
   * \param dim_y The map's y dimension
   * \param dim_z The map's z dimension
   * \param voxel_side_length Defines the maximum resolution
   *
   */
  GpuVoxels(const uint32_t dim_x, const uint32_t dim_y, const uint32_t dim_z, const float voxel_side_length);

  ~GpuVoxels();

  /*!
   * \brief addMap Add a new map to GVL.
   * \param map_type Choose between a representation: Octree, Voxelmap, Voxellist are possible
   * \param map_name The name of the map for later identification
   * \return Returns true, if adding was successful, false otherwise
   */
  bool addMap(const MapType map_type, const std::string &map_name);

  /*!
   * \brief delMap Remove a map from GVL.
   * \param map_name Name of the map, that should be deleted.
   * \return Returns true, if deleting was successful, false otherwise
   */
  bool delMap(const std::string &map_name);

  /*!
   * \brief clearMap Deletes ALL data from the map
   * \param map_name Which map to clear
   */
  bool clearMap(const std::string &map_name);

  /*!
   * \brief clearMap Deletes a special voxel type from the map
   * \param map_name Which map to clear
   * \param voxel_type Which type of voxels to clear
   */
  bool clearMap(const std::string &map_name, VoxelType voxel_type);

  /*!
   * \brief getMap Gets a const pointer to the map.
   * \param map_name The name of the queried map.
   * \return Pointer to the queried map. NULL if map was not found.
   */
  GpuVoxelsMapSharedPtr getMap(const std::string &map_name);

  /*!
   * \brief visualizeMap Visualizes the map only if necessary. That's the case if it's enforced by \code force_repaint = true or the visualizer requested it.
   * \param map_name Name of the map, that should be visualized.
   * \param force_repaint True to force a repainting of the map. e.g. needed to visualize changed map data
   * \return Returns true, if there was work to do, false otherwise.
   */
  bool visualizeMap(const std::string &map_name, const bool force_repaint = true);

  /*!
   * \brief addRobot Define a robot with its geometries and kinematic structure
   * \param robot_name Name of the robot, used as handler
   * \param dh_params DH representation of the robots kinematics. Has to be of the same dimensionality as the \code robot_cloude
   * \param paths_to_pointclouds Files on disk that hold the pointcloud representation of the robot geometry
   * \return true, if robot was added, false otherwise
   */
  bool addRobot(const std::string &robot_name, const std::vector<KinematicLink::DHParameters> &dh_params, const std::vector<std::string> &paths_to_pointclouds);

  /*!
   * \brief addRobot Define a robot with its geometries and kinematic structure
   * \param robot_name Name of the robot, used as handler
   * \param dh_params DH representation of the robots kinematics. Has to be of the same dimensionality as the \code robot_cloude
   * \param robot_cloud Metapointcloud representation of the robot links
   * \return true, if robot was added, false otherwise
   */
  bool addRobot(const std::string &robot_name, const std::vector<KinematicLink::DHParameters> &dh_params, const MetaPointCloud &robot_cloud);

  /*!
   * \brief updateRobotPose Changes the robot joint configuration. Call \code insertRobotIntoMap() afterwards!
   * \param robot_name Name of the robot to update
   * \param joint_values Vector of new joint angles
   * \param new_base_pose 6D Base link pose of the robot
   * \return true, if update was successful
   */
  bool updateRobotPose(std::string robot_name, std::vector<float> joint_values, Matrix4f *new_base_pose = NULL);

  /*!
   * \brief insertRobotIntoMap Writes a robot with its current pose into a map
   * \param robot_name Name of the robot to use
   * \param map_name Name of the map to insert the robot
   * \return true, if robot was added, false otherwise
   */
  bool insertRobotIntoMap(std::string robot_name, std::string map_name, const VoxelType voxel_type);

  /*!
  * \brief insertBoxIntoMap Helper function to generate obstacles. This inserts a box object.
  * \param corner_min Coordinates of the lower, left corner in the front.
  * \param corner_max Coordinates of the upper, right corner in the back.
  * \param map_name Name of the map to insert the box
  * \param voxel_type The kind of voxel to insert
  * \param points_per_voxel Point density. This is only relevant to test probabilistic maps.
  */
  bool insertBoxIntoMap(const Vector3f &corner_min, const Vector3f &corner_max, std::string map_name, const VoxelType voxel_type, uint16_t points_per_voxel = 1);

  /*!
   * \brief updateRobotPart Changes the geometry of a single robot link. This is useful when changing a tool,
   * grasping an object of when interpreting sensor data from an onboard sensor as a robot link.
   * Caution: This function requires intensive memory access, if the size of the pointcloud changes!
   * Call \code insertRobotIntoMap() afterwards!
   * \param robot_name Name of the robot beeing modified
   * \param link Index of the link that is modified
   * \param pointcloud New pointcloud of the link
   * \return true, if robot was modified, false otherwise
   */
  bool updateRobotPart(std::string robot_name, size_t link, const std::vector<Vector3f> pointcloud);

  /*!
   * \brief getVisualization Gets a handle to the visualization interface of this map.
   * \return pointer to \code VisProvider of the map with the given name.
   */
  VisProvider* getVisualization(const std::string &map_name);

private:

  ManagedMaps m_managed_maps;
  ManagedRobots m_managed_robots;
  uint32_t m_dim_x;
  uint32_t m_dim_y;
  uint32_t m_dim_z;
  float m_voxel_side_length;
};

} // end of namespace
#endif
