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

#include "gpu_voxels/robot/robot_interface.h"
#include "gpu_voxels/robot/urdf_robot/robot.h"
#include "gpu_voxels/robot/urdf_robot/robot_to_gpu.h"
#include <urdf_model/model.h>
#include <urdf_parser/urdf_parser.h>

namespace gpu_voxels {
namespace robot {

class UrdfRobot : public RobotInterface
{
public:

  /**
   * \brief Initializes the robot from a URDF file
   * @param path Path to the description URDF file
   */
  UrdfRobot(const std::string &path, const bool use_model_path);

  /**
   * @brief Updates the joints of the robot
   * @param jointmap Pairs of jointnames and joint values
   * Joints get matched by names, so not all joints have to be
   * specified.
   */
  void setConfiguration(const JointValueMap &jointmap);

  /**
   * @brief getConfiguration Gets the robot configuration
   * @param joint_values Map of jointnames and values.
   * This map will get extended if joints were missing.
   */
  void getConfiguration(JointValueMap &jointmap);

  /**
   * @brief getTransformedClouds
   * @return Pointers to the kinematically transformed clouds.
   */
  const MetaPointCloud *getTransformedClouds();

  /*!
   * \brief updatePointcloud Changes the geometry of a single link.
   * Useful when grasping an object, changing a tool
   * or interpreting point cloud data from an onboard sensor as a robot link.
   * \param link Link to modify
   * \param cloud New geometry
   */
  void updatePointcloud(const std::string &link_name, const std::vector<Vector3f> &cloud);

private:
  Robot* robot; // the URDF description of the robot
  RobotToGPU* rob2gpu; // the GPU part for transformation

};

} // namespace robot
} // namespace gpu_voxels

#endif
