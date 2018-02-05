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
 * \date    2015-03-11
 *
 * This holds the GPU relevant parts of a robot model:
 * Read-only pointclouds of all links and their transformed
 * counterpoarts.
 * This class also triggers the transformations.
 *
 */
//----------------------------------------------------------------------



#ifndef GPU_VOXELS_ROBOT_URDF_ROBOT_ROBOT_TO_GPU_H_INCLUDED
#define GPU_VOXELS_ROBOT_URDF_ROBOT_ROBOT_TO_GPU_H_INCLUDED

#include "gpu_voxels/robot/urdf_robot/robot.h"

namespace gpu_voxels {
namespace robot {


class RobotToGPU : public Robot
{
public:
  RobotToGPU(const float voxel_side_length, const std::string &_path, const bool &use_model_path);
  ~RobotToGPU();

  /**
   * @brief Updates the robot with new joint angles
   * and triggers the transformation kernel.
   * @param jointmap Map of jointnames and values
   */
  void setConfiguration(const JointValueMap &jointmap);

  /**
   * @brief updatePointcloud Updates the internal metapointcloud for the transformed joints
   * as new mem has to be reserved if robots pointcloud changes.
   * @param link_name Link to change
   * @param cloud New pointcloud
   */
  void updatePointcloud(const std::string &link_name, const std::vector<Vector3f> &cloud);

  /**
   * @brief getTransformedClouds
   * @return Pointer to the kinematically transformed clouds.
   */
  const MetaPointCloud *getTransformedClouds();

private:
  MetaPointCloud* m_link_pointclouds_transformed;
  Matrix4f m_transformation;
};

} // namespace robot
} // namespace gpu_voxels

#endif
