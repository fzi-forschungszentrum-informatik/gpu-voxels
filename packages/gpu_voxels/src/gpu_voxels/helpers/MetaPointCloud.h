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
 * \date    2014-06-17
 *
 * This is a management structure to handle arrays of PointClouds
 * on the GPU. Such as RobotLinks or sensor-data.
 *
 */
//----------------------------------------------------------------------
#ifndef GPU_VOXELS_HELPERS_METAPOINTCLOUD_H_INCLUDED
#define GPU_VOXELS_HELPERS_METAPOINTCLOUD_H_INCLUDED

#include <stdint.h> // for fixed size datatypes
#include <vector>
#include <cuda_runtime.h>

#include <gpu_voxels/helpers/cuda_datatypes.h>
#include <gpu_voxels/helpers/cuda_handling.hpp>
#include <gpu_voxels/helpers/PointcloudFileHandler.h>
#include <gpu_voxels/helpers/kernels/MetaPointCloudOperations.h>

#include <gpu_voxels/logging/logging_gpu_voxels_helpers.h>

namespace gpu_voxels {

class MetaPointCloud
{
public:

  MetaPointCloud();
  MetaPointCloud(const std::vector<std::string> &_point_cloud_files, bool use_model_path);
  MetaPointCloud(const std::vector<std::string> &_point_cloud_files,
                 const std::vector<std::string> &_point_cloud_names, bool use_model_path);
  MetaPointCloud(const std::vector<uint32_t> &_point_cloud_sizes);
  MetaPointCloud(const std::vector< std::vector<Vector3f> > &point_clouds);
  MetaPointCloud(const MetaPointCloud &other);
  MetaPointCloud(const MetaPointCloud *other);


  //! Destructor
  ~MetaPointCloud();

  void addCloud(uint32_t cloud_size);
  void addCloud(const std::vector<Vector3f> &cloud, bool sync = false, const std::string &name = "");
  void addClouds(const std::vector<std::string> &_point_cloud_files, bool use_model_path);
  std::string getCloudName(uint16_t i) const;
  const std::map<uint16_t, std::string> getCloudNames() const;
  int16_t getCloudNumber(const std::string& name) const;



  /*!
   * \brief syncToDevice copies a specific map to the GPU
   * \param cloud The cloud that will be synced
   */
  void syncToDevice(uint16_t cloud);

  /*!
   * \brief syncToDevice Syncs all clouds to the device at once.
   */
  void syncToDevice();

  void syncToHost();

  /*!
   * \brief updatePointCloud This updates a specific cloud on the host.
   * Call syncToDevice() after updating all clouds or set sync to true
   * to only sync this current cloud to the GPU.
   * \param cloud Id of the cloud to update
   * \param pointcloud The new cloud. May differ in size.
   * \param sync If set to true, only this modified cloud is synced to the GPU.
   */
  void updatePointCloud(uint16_t cloud, const std::vector<Vector3f> &pointcloud, bool sync = false);

  /*!
   * \brief updatePointCloud This updates a specific cloud on the host.
   * Call syncToDevice() after updating all clouds or set sync to true
   * to only sync this current cloud to the GPU.
   * \param cloud_name Name of the cloud to update
   * \param pointcloud The new cloud. May differ in size.
   * \param sync If set to true, only this modified cloud is synced to the GPU.
   */
  void updatePointCloud(const std::string cloud_name, const std::vector<Vector3f> &pointcloud, bool sync = false);

  /*!
   * \brief updatePointCloud This updates a specific cloud on the host.
   * Call syncToDevice() after updating all clouds or set sync to true
   * to only sync this current cloud to the GPU.
   * \param cloud Id of the cloud to update
   * \param pointcloud The new cloud
   * \param pointcloud_size Size of the new cloud
   * \param sync If set to true, only this modified cloud is synced to the GPU.
   */
  void updatePointCloud(uint16_t cloud, const Vector3f* pointcloud, uint32_t pointcloud_size, bool sync = false);

  /*!
   * \brief getNumberOfPointclouds
   * \return Number of clouds in the MetaPointCloud
   */
  uint16_t getNumberOfPointclouds() const { return m_point_clouds_local->num_clouds; }

  /*!
   * \brief getPointcloudSize Returns the number of elements in one point cloud
   * \param cloud The ID of the cloud.
   * \return Numper of points in one pointcloud.
   */
  uint32_t getPointcloudSize(uint16_t cloud = 0) const;

  /*!
   * \brief getAccumulatedPointcloudSize
   * \return Accumulated size of all point clouds
   */
  uint32_t getAccumulatedPointcloudSize() const;

  /*!
   * \brief getPointcloudSizes
   * \return A vector of the sizes of all point clouds.
   */
  const std::vector<uint32_t>& getPointcloudSizes() const { return m_point_cloud_sizes; }

  /*!
   * \brief getPointCloud
   * \param cloud Which cloud to return.
   * \return A pointer to the host point cloud.
   */
  const Vector3f* getPointCloud(uint16_t cloud) const { return m_point_clouds_local->clouds_base_addresses[cloud]; }

  /*!
   * \brief getDevicePointer
   * \return Returns a writable pointer to the device data
   */
  MetaPointCloudStruct* getDevicePointer() { return m_dev_ptr_to_point_clouds_struct; }

  /*!
   * \brief getDeviceConstPointer
   * \return Returns a const pointer to the device data for RO access
   */
  const MetaPointCloudStruct* getDeviceConstPointer() const { return m_dev_ptr_to_point_clouds_struct; }

  void debugPointCloud() const;

private:

  /*!
   * \brief Init does the allocation of Device and Host memory
   * \param _point_cloud_sizes The point cloud sizes that are required for the malloc
   */
  void init(const std::vector<uint32_t> &_point_cloud_sizes);

  /*!
   * \brief MetaPointCloud::Destruct Private destructor that is also called, when a
   * new cloud is added.
   */
  void destruct();

  std::vector<uint32_t> m_point_cloud_sizes; //basically only needed for copy constructor
  std::map<uint16_t, std::string> m_point_cloud_names;

  uint32_t m_accumulated_pointcloud_size;
  uint16_t m_num_clouds;
  Vector3f* m_accumulated_cloud;

  MetaPointCloudStruct *m_point_clouds_local;
  MetaPointCloudStruct *m_dev_point_clouds_local;

  Vector3f* m_dev_ptr_to_accumulated_cloud;
  MetaPointCloudStruct* m_dev_ptr_to_point_clouds_struct;
  Vector3f** m_dev_ptrs_to_addrs;
  uint32_t *m_dev_ptr_to_cloud_sizes;
  Vector3f** m_dev_ptr_to_clouds_base_addresses;
};

}
#endif
