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
 * This class combines the GPU and CPU part of the URDF robot model.
 * On construction it parsed an URDF and allocates the robot pointclouds
 * on the GPU.
 * Through runtime, the joints can be updated and the pointclouds
 * will get transformed accordingly.
 *
 */
//----------------------------------------------------------------------

#ifndef GPU_VOXELS_ROBOT_URDF_ROBOT_URDF_ROBOT_H_INCLUDED
#define GPU_VOXELS_ROBOT_URDF_ROBOT_URDF_ROBOT_H_INCLUDED

#include "gpu_voxels/robot/urdf_robot/robot_to_gpu.h"

namespace gpu_voxels {
namespace robot {

/*!
 * \brief The UrdfRobot class
 * Basically wraps the robot_to_gpu.cu into a regular .cpp library
 */
class UrdfRobot : public RobotToGPU
{
public:

  /*!
   * \brief UrdfRobot::UrdfRobot Initializes the robot from a URDF file
   *
   * \param voxel_side_length This is used to voxelize basic URDF geometry elements (0.4 * voxel_side_length)
   * \param path Path to the description URDF file
   * \param use_model_path Use the path specified via
   * GPU_VOXELS_MODEL_PATH environment variable
   */
  UrdfRobot(const float voxel_side_length, const std::string &path, const bool &use_model_path);

  ~UrdfRobot();

};

} // namespace robot
} // namespace gpu_voxels

#endif
