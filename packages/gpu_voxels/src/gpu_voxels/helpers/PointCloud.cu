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
 * \author  Herbert Pietrzyk
 * \date    2016-05-25
 *
 */
//----------------------------------------------------------------------
#include "PointCloud.h"
#include <gpu_voxels/helpers/kernels/MetaPointCloudOperations.h>
#include <gpu_voxels/logging/logging_gpu_voxels.h>
#include <gpu_voxels/helpers/kernels/HelperOperations.h>
#include <gpu_voxels/helpers/PointcloudFileHandler.h>
#include <sensor_msgs/point_cloud2_iterator.h>

namespace gpu_voxels
{

PointCloud::PointCloud()
{
  m_points_dev = NULL;
  m_points_size = 0;
  HANDLE_CUDA_ERROR(
      cudaMalloc((void** )&m_transformation_dev, sizeof(Matrix4f)));
}

PointCloud::PointCloud(const std::vector<Vector3f> &points)
{
  m_points_size = points.size();

  HANDLE_CUDA_ERROR(
      cudaMalloc((void** ) &m_points_dev, m_points_size * sizeof(Vector3f)));

  HANDLE_CUDA_ERROR(
      cudaMalloc((void** )&m_transformation_dev, sizeof(Matrix4f)));

  HANDLE_CUDA_ERROR(
      cudaMemcpy(m_points_dev, points.data(),
                 sizeof(Vector3f) * m_points_size, cudaMemcpyHostToDevice));
}

PointCloud::PointCloud(const Vector3f *points, uint32_t size)
{
  HANDLE_CUDA_ERROR(
      cudaMalloc((void** ) &m_points_dev, size * sizeof(Vector3f)));

  HANDLE_CUDA_ERROR(
      cudaMalloc((void** )&m_transformation_dev, sizeof(Matrix4f)));

  HANDLE_CUDA_ERROR(
      cudaMemcpy(m_points_dev, points, sizeof(Vector3f) * size, cudaMemcpyHostToDevice));

  m_points_size = size;
}

PointCloud::PointCloud(const PointCloud &other)
{
  m_points_size = other.getPointCloudSize();

  HANDLE_CUDA_ERROR(
      cudaMalloc((void** ) &m_points_dev, m_points_size * sizeof(Vector3f)));

  HANDLE_CUDA_ERROR(
      cudaMalloc((void** )&m_transformation_dev, sizeof(Matrix4f)));

  HANDLE_CUDA_ERROR(
      cudaMemcpy(m_points_dev, other.getConstDevicePointer(),
                 sizeof(Vector3f) * m_points_size, cudaMemcpyDeviceToDevice));
}

PointCloud::PointCloud(const std::string &path_to_file, bool use_model_path)
{
  std::vector<Vector3f> host_point_cloud;

  if(!file_handling::PointcloudFileHandler::Instance()->loadPointCloud(path_to_file, use_model_path, host_point_cloud))
  {
    LOGGING_ERROR_C(Gpu_voxels_helpers, PointCloud,
                    "Could not read file " << path_to_file << icl_core::logging::endl);
    return;
  }

  m_points_size = host_point_cloud.size();

  HANDLE_CUDA_ERROR(
      cudaMalloc((void** ) &m_points_dev, m_points_size * sizeof(Vector3f)));

  HANDLE_CUDA_ERROR(
      cudaMalloc((void** )&m_transformation_dev, sizeof(Matrix4f)));

  HANDLE_CUDA_ERROR(
      cudaMemcpy(m_points_dev, host_point_cloud.data(),
                 sizeof(Vector3f) * m_points_size, cudaMemcpyHostToDevice));
}


PointCloud& PointCloud::operator=(const PointCloud& other)
{
  if (this != &other) // self-assignment check expected
  {
    resize(other.getPointCloudSize());

    HANDLE_CUDA_ERROR(
        cudaMemcpy(m_points_dev, other.getConstDevicePointer(),
                   sizeof(Vector3f) * m_points_size, cudaMemcpyDeviceToDevice));
  }
  return *this;
}


PointCloud::~PointCloud()
{
  if (m_points_dev)
    HANDLE_CUDA_ERROR(cudaFree(m_points_dev));
  if (m_transformation_dev)
    HANDLE_CUDA_ERROR(cudaFree(m_transformation_dev));
}


void PointCloud::resize(uint32_t new_number_of_points)
{
  if(m_points_size != new_number_of_points)
  {
    m_points_size = new_number_of_points;

    if(m_points_dev)
    {
      HANDLE_CUDA_ERROR(cudaFree(m_points_dev));
    }
    HANDLE_CUDA_ERROR(
        cudaMalloc((void** ) &m_points_dev, m_points_size * sizeof(Vector3f)));
  }
}

bool PointCloud::operator==(const PointCloud& other) const
{
  // Things are clear if self comparison:
  if(this == &other)
  {
    LOGGING_DEBUG_C(Gpu_voxels_helpers, PointCloud, "Clouds are the same object." << icl_core::logging::endl);
    return true;
  }
  // Size has to match:
  if(m_points_size != other.m_points_size)
  {
    LOGGING_DEBUG_C(Gpu_voxels_helpers, PointCloud, "Pointcloud size does not match." << icl_core::logging::endl);
    return false;
  }

  // Allocate result memory on Host:
  bool *host_equality_results = new bool[cMAX_NR_OF_BLOCKS];

  bool *dev_equality_results;
  // allocate result memory on device:
  HANDLE_CUDA_ERROR(cudaMalloc((void** )&dev_equality_results, cMAX_NR_OF_BLOCKS * sizeof(bool)));

  // initialze results memory to true:
  HANDLE_CUDA_ERROR(cudaMemset(dev_equality_results, false, cMAX_NR_OF_BLOCKS * sizeof(bool)));

  // do the actual comparison:
  computeLinearLoad(m_points_size, &m_blocks, &m_threads_per_block);
  HANDLE_CUDA_ERROR(cudaDeviceSynchronize());

  kernelCompareMem<<<m_blocks, m_threads_per_block>>>(m_points_dev, other.m_points_dev,
                                                      m_points_size * sizeof(Vector3f), dev_equality_results);
  CHECK_CUDA_ERROR();

  HANDLE_CUDA_ERROR(cudaDeviceSynchronize());
  // copy back the whole array
  HANDLE_CUDA_ERROR(
      cudaMemcpy(host_equality_results, dev_equality_results, cMAX_NR_OF_BLOCKS * sizeof(bool),
                 cudaMemcpyDeviceToHost));

  bool ret = true;

  // check only used portion of array:
  for (uint32_t i = 0; i < m_blocks; i++)
  {
    if(!host_equality_results[i])
    {
      LOGGING_DEBUG_C(Gpu_voxels_helpers, PointCloud, "Clouds data is different!" << icl_core::logging::endl);
      ret = false;
      break;
    }
  }

  // clean up:
  delete host_equality_results;
  HANDLE_CUDA_ERROR(cudaFree(dev_equality_results));
  return ret;
}



void PointCloud::add(const PointCloud *cloud)
{
  this->add(cloud->getPoints(), cloud->getPointCloudSize());
}

void PointCloud::add(const std::vector<Vector3f> &points)
{
  this->add(points.begin().base(), points.size());
}

void PointCloud::add(const Vector3f* points, uint32_t size)
{
  Vector3f* tmp_dev;

  HANDLE_CUDA_ERROR(cudaMalloc((void** ) &tmp_dev, (size + m_points_size) * sizeof(Vector3f)));

  HANDLE_CUDA_ERROR(
      cudaMemcpy(tmp_dev, m_points_dev, sizeof(Vector3f) * m_points_size, cudaMemcpyDeviceToDevice));

  HANDLE_CUDA_ERROR(
      cudaMemcpy(tmp_dev + m_points_size, points, sizeof(Vector3f) * size, cudaMemcpyHostToDevice));

  HANDLE_CUDA_ERROR(cudaFree(m_points_dev));
  m_points_dev = tmp_dev;
  m_points_size = size + m_points_size;
}

void PointCloud::update(const PointCloud *cloud)
{
  this->update(cloud->getPoints(), cloud->getPointCloudSize());
}

void PointCloud::update(const std::vector<Vector3f> &points)
{
  this->update(points.data(), points.size());
}

void PointCloud::update(const Vector3f *points, uint32_t size)
{
  resize(size);

  HANDLE_CUDA_ERROR(
      cudaMemcpy(m_points_dev, points, sizeof(Vector3f) * size, cudaMemcpyHostToDevice));

}

void PointCloud::transformSelf(const Matrix4f *transform)
{
  this->transform(transform, this);
}

void PointCloud::transform(const Matrix4f *transform, PointCloud *transformed_cloud)
{

  HANDLE_CUDA_ERROR(
      cudaMemcpy(m_transformation_dev, transform, sizeof(Matrix4f), cudaMemcpyHostToDevice));

  Vector3f* transformed_dev;
  if(transformed_cloud == this)
  {
    transformed_dev = m_points_dev;
  }
  else
  {
    transformed_cloud->resize(m_points_size);
    transformed_dev = transformed_cloud->getDevicePointer();
  }

  computeLinearLoad(m_points_size,
                    &m_blocks, &m_threads_per_block);
  cudaDeviceSynchronize();
  // transform the cloud via Kernel.
  kernelTransformCloud<<< m_blocks, m_threads_per_block >>>
     (m_transformation_dev,
      m_points_dev,
      transformed_dev,
      m_points_size);
  CHECK_CUDA_ERROR();

  HANDLE_CUDA_ERROR(cudaDeviceSynchronize());
}


void PointCloud::scaleSelf(const Vector3f* scaling)
{
  this->scale(scaling, this);
}

void PointCloud::scale(const Vector3f* scaling, PointCloud* transformed_cloud)
{

  Vector3f* transformed_dev;
  if(transformed_cloud == this)
  {
    transformed_dev = m_points_dev;
  }
  else
  {
    transformed_cloud->resize(m_points_size);
    transformed_dev = transformed_cloud->getDevicePointer();
  }

  computeLinearLoad(m_points_size,
                    &m_blocks, &m_threads_per_block);
  cudaDeviceSynchronize();
  // transform the cloud via Kernel.
  kernelScaleCloud<<< m_blocks, m_threads_per_block >>>
     (*scaling,
      m_points_dev,
      transformed_dev,
      m_points_size);
  CHECK_CUDA_ERROR();

  HANDLE_CUDA_ERROR(cudaDeviceSynchronize());
}

Vector3f* PointCloud::getDevicePointer()
{
  return m_points_dev;
}

const Vector3f* PointCloud::getConstDevicePointer() const
{
  return m_points_dev;
}

Vector3f* PointCloud::getPoints() const
{
  Vector3f* tmp_h = (Vector3f*)malloc(m_points_size * sizeof(Vector3f));
  HANDLE_CUDA_ERROR(
      cudaMemcpy(tmp_h, m_points_dev, m_points_size * sizeof(Vector3f), cudaMemcpyDeviceToHost));

  return tmp_h;
}


size_t PointCloud::getPointCloudSize() const
{
  return m_points_size;
}


void PointCloud::print()
{
  Vector3f* tmp = this->getPoints();

  for(size_t i = 0; i < this->getPointCloudSize(); i++)
  {
    std::cout << "Point " << i << ": (" << tmp[i].x << ", " << tmp[i].y << ", " << tmp[i].z << ")" << std::endl;
  }
  std::cout << std::endl << std::endl;
}

sensor_msgs::PointCloud2 PointCloud::getPointCloud2()
{
  sensor_msgs::PointCloud2 pointcloud_msg;

  Vector3f* tmp = this->getPoints();
  size_t mPointCloudSize = this->getPointCloudSize();

  pointcloud_msg.header.frame_id = "/heifu0/odom";//voxel_point_cloud_frame; // use same reference as in transform_...
  pointcloud_msg.header.stamp = ros::Time::now();
  pointcloud_msg.height = 1;
  pointcloud_msg.width = mPointCloudSize;
  pointcloud_msg.row_step = pointcloud_msg.width * pointcloud_msg.point_step;

  sensor_msgs::PointCloud2Modifier modifier(pointcloud_msg);
  modifier.setPointCloud2FieldsByString(1, "xyz");

  sensor_msgs::PointCloud2Iterator<float>iter_x(pointcloud_msg, "x");
  sensor_msgs::PointCloud2Iterator<float>iter_y(pointcloud_msg, "y");
  sensor_msgs::PointCloud2Iterator<float>iter_z(pointcloud_msg, "z");
  
  for(size_t i=0; i<mPointCloudSize; i++)
  {
      *iter_x = tmp[i].x;
      *iter_y = tmp[i].y;
      *iter_z = tmp[i].z;

      ++iter_x; ++iter_y; ++iter_z;
  }
  return pointcloud_msg;
}

}// end namespace gpu_voxels
