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
 * \date    2015-06-08
 *
 *
 *
 */
//----------------------------------------------------------------------

#ifndef GPU_VOXELS_ROBOT_ROBOT_INTERFACE_H_INCLUDED
#define GPU_VOXELS_ROBOT_ROBOT_INTERFACE_H_INCLUDED

#include <map>
#include <string>
#include "gpu_voxels/helpers/MetaPointCloud.h"

namespace gpu_voxels {
namespace robot {

typedef std::pair<std::string, float> JointValuePair;
typedef std::map<std::string, float> JointValueMap;
typedef JointValueMap::iterator JointValueMapIterator;
typedef JointValueMap::const_iterator JointValueMapConstIterator;


class RobotInterface
{
public:

  /**
   * @brief getJointNames Reads all joint names
   * @param jointnames Vector of jointnames that will get extended
   */
  virtual void getJointNames(std::vector<std::string> &jointnames) = 0;

  /**
   * @brief Updates the joints of the robot
   * @param jointmap Pairs of jointnames and joint values
   * Joints get matched by names, so not all joints have to be
   * specified.
   */
  virtual void setConfiguration(const JointValueMap &jointmap) = 0;

  /**
   * @brief getConfiguration Gets the robot configuration
   * @param joint_values Map of jointnames and values.
   * This map will get extended if joints were missing.
   */
  virtual void getConfiguration(JointValueMap &jointmap) = 0;

  /**
   * @brief getTransformedClouds
   * @return Pointers to the kinematically transformed clouds.
   */
  virtual const MetaPointCloud *getTransformedClouds() = 0;

  /*!
   * \brief updatePointcloud Changes the geometry of a single link.
   * Useful when grasping an object, changing a tool
   * or interpreting point cloud data from an onboard sensor as a robot link.
   * \param link Link to modify
   * \param cloud New geometry
   */
  virtual void updatePointcloud(const std::string &link_name, const std::vector<Vector3f> &cloud) = 0;

  /**
   * @brief getLowerJointLimits Gets the minimum joint values
   * @param lower_limits Map of jointnames and values.
   * This map will get extended if joints were missing.
   */
  virtual void getLowerJointLimits(JointValueMap &lower_limits) = 0;

  /**
   * @brief getUpperJointLimits Gets the maximum joint values
   * @param upper_limits Map of jointnames and values.
   * This map will get extended if joints were missing.
   */
  virtual void getUpperJointLimits(JointValueMap &upper_limits) = 0;
};

} // namespace robot
} // namespace gpu_voxels

#endif /* GPU_VOXELS_ROBOT_ROBOT_INTERFACE_H_INCLUDED */
