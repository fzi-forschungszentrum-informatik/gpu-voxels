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
#ifndef GPU_VOXELS_KINEMATIC_CHAIN_H_INCLUDED
#define GPU_VOXELS_KINEMATIC_CHAIN_H_INCLUDED

#include <vector>
#include <string>
#include <map>
#include <utility>
#include <cuda_runtime.h>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>

#include <gpu_voxels/helpers/cuda_handling.hpp>
#include <gpu_voxels/helpers/cuda_datatypes.h>
#include <gpu_voxels/helpers/MetaPointCloud.h>
#include <gpu_voxels/helpers/CudaMath.h>
#include <gpu_voxels/robot/KinematicLink.h>

#include <gpu_voxels/logging/logging_robot.h>
#include <sstream>

namespace gpu_voxels {

class KinematicChain
{
public:
  /*!
   * \brief KinematicChain
   * \param links Local copy is created
   * \param link_meta_cloud Local copy is created
   * \param basis_transformation Initial pose
   * \param fksolver
   * \param linknames
   * \return
   */
  __host__
  KinematicChain(const std::vector<KinematicLinkSharedPtr> &links, const MetaPointCloud &link_meta_cloud, Matrix4f basis_transformation,
                 std::map<unsigned int, std::string>* linknames = 0);

  /*!
   * destructor.
   */
  __host__
  ~KinematicChain();


  /*! Apply configuration to point clouds. Will
   * update the kinematic chain and the
   * point cloud transformations
   * @param[in] joint_values Robot joint values to set
   */
  __host__
  void setConfiguration(std::vector<float> joint_values);

  //! sets the new base transformation and calls setConfiguration for the joint values
  __host__
  void setConfiguration(Matrix4f basis_transformation, std::vector<float> joint_values);

  //! get size of kinematic chain
  __host__
  uint8_t getKinematicChainSize()
  {
    return m_size;
  }

  const MetaPointCloud* getTransformedLinks() { return m_transoformed_links_meta_cloud; }

  /*!
   * \brief updatePointcloud Changes the geometry of a single link. Useful when grasping an object, changing a tool
   * or interpreting point cloud data from an onboard sensor as a robot link.
   * \param link Link to modify
   * \param cloud New geometry
   */
  void updatePointcloud(uint16_t link, const std::vector<Vector3f> &cloud);

  //! for testing purposes
  __host__
  void transformPointAlongChain(Vector3f point);

  //! same function as for device
  __host__
  void convertDHtoMHost(float theta, float d, float b, float a, float alpha, float q, uint8_t joint_type, Matrix4f& m);
private:

  /*!  Update the joint values on host and device.
   *  Calls a kernel to update DH matrices
   */
  __host__
  void updateJointValues();

  /*!  Uses current joint values of the links within
   *  the kinematic chain to perform a sequence of
   *  links.size() transformations for the point clouds
   *  of each joint.
   */
  __host__
  void update();

  MetaPointCloud* m_links_meta_cloud;
  MetaPointCloud* m_transoformed_links_meta_cloud;

  /* host stored contents */
  uint8_t m_size;
  //! pointer to the kinematic links
  std::vector<KinematicLinkSharedPtr> m_links;

  thrust::host_vector<uint8_t> m_joint_types;
  thrust::host_vector<KinematicLink::DHParameters> m_dh_parameters;

  Matrix4f m_basis_transformation;

  cudaEvent_t m_start;
  cudaEvent_t m_stop;
  CudaMath m_math;
  uint32_t m_blocks;
  uint32_t m_threads_per_block;


  /* device stored contents */
  thrust::device_vector<uint8_t> m_dev_joint_types;
  thrust::device_vector<KinematicLink::DHParameters> m_dev_dh_parameters;
  thrust::device_vector<Matrix4f> m_dev_transformations;
  thrust::device_vector<Matrix4f> m_dev_local_transformations;

  Matrix4f* m_dev_basis_transformation;

  std::map<unsigned int, std::string> m_pointcloud_to_linkname;
};

} // end of namespace
#endif
