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
#include <gpu_voxels/helpers/CudaMath.h>


namespace gpu_voxels {
namespace robot {


class RobotToGPU
{
public:
  RobotToGPU(Robot* robot);
  ~RobotToGPU();

  /**
   * @brief Updates the robot with new joint angles
   * and triggers the transformation kernel.
   * @param jointmap Map of jointnames and values
   */
  void setConfiguration(const std::map<std::string, float> &jointmap);

  void getConfiguration(const std::map<std::string, float> jointmap);

  /**
   * @brief getTransformedClouds
   * @return Pointer to the kinematically transformed clouds.
   */
  const MetaPointCloud *getTransformedClouds();

private:
  Robot* m_robot;
  MetaPointCloud* m_link_pointclouds_transformed;
  CudaMath m_math;

  Matrix4f m_transformation;
  Matrix4f* m_transformation_dev;
  uint32_t m_blocks;
  uint32_t m_threads_per_block;
};

} // namespace robot
} // namespace gpu_voxels

#endif
