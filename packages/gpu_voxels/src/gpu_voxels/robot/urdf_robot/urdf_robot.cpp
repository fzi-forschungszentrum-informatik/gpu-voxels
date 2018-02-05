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
 *
 */
//----------------------------------------------------------------------

#include "gpu_voxels/robot/urdf_robot/urdf_robot.h"
#include "gpu_voxels/robot/urdf_robot/robot_to_gpu.h"
#include "gpu_voxels/logging/logging_robot.h"
#include "gpu_voxels/helpers/FileReaderInterface.h"

namespace gpu_voxels {
namespace robot {

UrdfRobot::UrdfRobot(const float voxel_side_length, const std::string &path, const bool &use_model_path) :
  RobotToGPU(voxel_side_length, path, use_model_path)
{
}

UrdfRobot::~UrdfRobot()
{
}

} // namespace robot
} // namespace gpu_voxels
