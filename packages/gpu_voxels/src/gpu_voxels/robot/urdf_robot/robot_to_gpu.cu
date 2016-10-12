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



#include "gpu_voxels/robot/urdf_robot/robot_to_gpu.h"
#include <gpu_voxels/helpers/cuda_datatypes.h>
#include <gpu_voxels/helpers/PointcloudFileHandler.h>
#include "gpu_voxels/robot/kernels/KinematicOperations.h"
#include "gpu_voxels/robot/urdf_robot/urdf_robot.h"
#include <urdf_model/model.h>
#include <urdf_parser/urdf_parser.h>

namespace gpu_voxels {
namespace robot {


RobotToGPU::RobotToGPU(const std::string &_path, const bool &use_model_path) :
  Robot()
{

  std::string path;

  // if param is true, prepend the environment variable GPU_VOXELS_MODEL_PATH
  path = (getGpuVoxelsPath(use_model_path) / boost::filesystem::path(_path)).string();

  boost::shared_ptr<urdf::ModelInterface> model_interface_shrd_ptr = urdf::parseURDFFile(path);

  Robot::load(*model_interface_shrd_ptr, true, true, use_model_path);

  // allocate a copy of the pointcloud, which will hold the transformed version
  m_link_pointclouds_transformed = new MetaPointCloud(*Robot::getLinkPointclouds());

}

RobotToGPU::~RobotToGPU()
{
}

void RobotToGPU::updatePointcloud(const std::string &link_name, const std::vector<Vector3f> &cloud)
{
  Robot::updatePointcloud(link_name, cloud);
  m_link_pointclouds_transformed->updatePointCloud(link_name, cloud);
}

void RobotToGPU::setConfiguration(const JointValueMap &jointmap)
{
  // first update the joints of the URDF model.
  Robot::setConfiguration(jointmap);


  // iterate over all joints that own a pointcloud.
  for(uint16_t i = 0; i < Robot::getLinkPointclouds()->getNumberOfPointclouds(); i++)
  {
    // get the trafo of the according URDF link
    m_transformation = Robot::getLink( Robot::getLinkPointclouds()->getCloudName(i) )->getPoseAsGpuMat4f();

//    std::cout << "RobotToGPU::update() transform of " << Robot::getLinkPointclouds()->getCloudName(i)
//              << " = " << m_transformation << std::endl;

    Robot::getLinkPointclouds()->transformSubCloud(i, &m_transformation, m_link_pointclouds_transformed);

  }
}

const MetaPointCloud* RobotToGPU::getTransformedClouds()
{
  return m_link_pointclouds_transformed;
}

} // namespace robot
} // namespace gpu_voxels
