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
#include "gpu_voxels/logging/logging_robot.h"
#include "gpu_voxels/helpers/PointcloudFileHandler.h" // Needed for getGpuVoxelsPath()

namespace gpu_voxels {
namespace robot {

UrdfRobot::UrdfRobot(const std::string &_path, const bool use_model_path)
{
  std::string path;

  // if param is true, prepend the environment variable GPU_VOXELS_MODEL_PATH
  if(use_model_path)
  {
    path = (file_handling::getGpuVoxelsPath() / boost::filesystem::path(_path)).string();
  }else{
    path = _path;
  }
  boost::shared_ptr<urdf::ModelInterface> model_interface_shrd_ptr = urdf::parseURDFFile(path);

  robot = new Robot();
  robot->load(*model_interface_shrd_ptr, true, false);

  rob2gpu = new RobotToGPU(robot);
}

void UrdfRobot::setConfiguration(const JointValueMap &jointmap)
{
  rob2gpu->setConfiguration(jointmap);
}

void UrdfRobot::getConfiguration(JointValueMap &jointmap)
{
  rob2gpu->getConfiguration(jointmap);
}

const MetaPointCloud* UrdfRobot::getTransformedClouds()
{
  return rob2gpu->getTransformedClouds();
}

void UrdfRobot::updatePointcloud(const std::string &link_name, const std::vector<Vector3f> &cloud)
{
  robot->updatePointcloud(link_name, cloud);
}

} // namespace robot
} // namespace gpu_voxels

