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
 * \author  Sebastian Klemm
 * \date    2012-09-13
 *
 */
//----------------------------------------------------------------------
#ifndef GPU_VOXELS_ROBOT_DH_ROBOT_KINEMATIC_CHAIN_H_INCLUDED
#define GPU_VOXELS_ROBOT_DH_ROBOT_KINEMATIC_CHAIN_H_INCLUDED

#include <vector>
#include <string>
#include <map>
#include <utility>
#include <cuda_runtime.h>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>

#include <gpu_voxels/robot/robot_interface.h>
#include <gpu_voxels/helpers/cuda_handling.hpp>
#include <gpu_voxels/helpers/cuda_datatypes.h>
#include <gpu_voxels/helpers/MetaPointCloud.h>
#include <gpu_voxels/robot/dh_robot/KinematicLink.h>

#include <gpu_voxels/logging/logging_robot.h>
#include <sstream>

namespace gpu_voxels {
namespace robot {

class KinematicChain: public RobotInterface
{
public:

  /*!
   * \brief KinematicChain constructor that loads pointclouds from files
   * \param linknames Names of all links
   * \param dh_params DH Parameters of all links
   * \param paths_to_pointclouds Filepaths to pointclouds
   * \param use_model_path Use GPU_MODEL_PATH to search for pointclouds
   *
   * Important: Linknames have to be the same as Pointcloud names
   * (derived from paths_to_pointclouds), if they should be kinematically
   * transformed.
   */
  __host__
  KinematicChain(const std::vector<std::string> &linknames,
                 const std::vector<robot::DHParameters> &dh_params,
                 const std::vector<std::string> &paths_to_pointclouds,
                 const bool use_model_path);

  /*!
   * \brief KinematicChain constructor that takes existing pointcloud
   * \param linknames Names of all links
   * \param dh_params DH Parameters of all links
   * \param pointclouds Existing Meta-Pointcloud of robt
   * (e.g. including attached Sensor Pointcloud)
   *
   * Important: Linknames have to be the same as Pointcloud names
   * (derived from paths_to_pointclouds), if they should be kinematically
   * transformed.
   */
  __host__
  KinematicChain(const std::vector<std::string> &linknames,
                 const std::vector<robot::DHParameters> &dh_params,
                 const MetaPointCloud &pointclouds);

  __host__
  ~KinematicChain();

  /**
   * @brief getJointNames Reads all joint names
   * @param jointnames Vector of jointnames that will get extended
   */
  virtual void getJointNames(std::vector<std::string> &jointnames);

  /*!
   * \brief setConfiguration Sets a robot configuration
   * and triggers pointcloud transformation.
   * \param joint_values Robot joint values. Will get
   * matched by names, so not all joints have to be specified.
   */
  __host__
  virtual void setConfiguration(const JointValueMap &joint_values);

  /*!
   * \brief setConfiguration Reads the current config
   * \param joint_values This map will get extended, if
   * jointnames are missing.
   */
  __host__
  virtual void getConfiguration(JointValueMap &joint_values);


  /**
   * @brief getLowerJointLimits Gets the minimum joint values
   * @param lower_limits Map of jointnames and values.
   * This map will get extended if joints were missing.
   */
  virtual void getLowerJointLimits(JointValueMap &lower_limits);

  /**
   * @brief getUpperJointLimits Gets the maximum joint values
   * @param upper_limits Map of jointnames and values.
   * This map will get extended if joints were missing.
   */
  virtual void getUpperJointLimits(JointValueMap &upper_limits);

  /**
   * @brief getTransformedClouds
   * @return Pointers to the kinematically transformed clouds.
   */
  virtual const MetaPointCloud *getTransformedClouds() { return m_transformed_links_meta_cloud; }

  /*!
   * \brief updatePointcloud Changes the geometry of a single link.
   * Useful when grasping an object, changing a tool
   * or interpreting point cloud data from an onboard sensor as a robot link.
   * \param link Link to modify
   * \param cloud New geometry
   */
  virtual void updatePointcloud(const std::string &link_name, const std::vector<Vector3f> &cloud);



  //! for testing purposes
  //__host__
  //void transformPointAlongChain(Vector3f point);

private:
  void init(const std::vector<std::string> &linknames,
       const std::vector<robot::DHParameters> &dh_params);

  std::vector<std::string> m_linknames;
  MetaPointCloud* m_links_meta_cloud;
  MetaPointCloud* m_transformed_links_meta_cloud;

  /* host stored contents */
  //! pointer to the kinematic links
  std::map<std::string, KinematicLinkSharedPtr> m_links;

  cudaEvent_t m_start;
  cudaEvent_t m_stop;

  Matrix4f m_dh_transformation;
};

} // end of namespace
} // end of namespace
#endif
