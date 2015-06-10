// this is for emacs file handling -*- mode: c++; indent-tabs-mode: nil -*-

// -- BEGIN LICENSE BLOCK ----------------------------------------------
// -- END LICENSE BLOCK ------------------------------------------------

//----------------------------------------------------------------------
/*!\file
 *
 * \author  Andreas Hermann
 * \date    2014-06-17
 *
 */
//----------------------------------------------------------------------
#include "MetaPointCloud.h"
#include <cstdlib>

namespace gpu_voxels {

void MetaPointCloud::init(const std::vector<uint32_t> &_point_cloud_sizes)
{
  m_point_cloud_sizes = _point_cloud_sizes;
  m_num_clouds = _point_cloud_sizes.size();
  m_accumulated_pointcloud_size = 0;
  m_accumulated_cloud = 0;
  m_dev_point_clouds_local = 0;
  m_dev_ptr_to_point_clouds_struct = 0;
  m_dev_ptr_to_cloud_sizes = 0;
  m_dev_ptr_to_accumulated_cloud = 0;
  m_dev_ptr_to_clouds_base_addresses = 0;
  m_point_clouds_local = 0;

  // allocate point clouds space on host:
  m_point_clouds_local = new MetaPointCloudStruct();
  m_point_clouds_local->num_clouds = m_num_clouds;
  m_point_clouds_local->cloud_sizes = new uint32_t[m_num_clouds];
  m_point_clouds_local->clouds_base_addresses = new Vector3f*[m_num_clouds];

  for (uint16_t i = 0; i < m_num_clouds; i++)
  {
    m_accumulated_pointcloud_size += _point_cloud_sizes[i];
  }

  m_point_clouds_local->accumulated_cloud_size = m_accumulated_pointcloud_size;
  m_accumulated_cloud = new Vector3f[m_accumulated_pointcloud_size];
  memset(m_accumulated_cloud, 0, sizeof(Vector3f)*m_accumulated_pointcloud_size);

  Vector3f* tmp_addr = m_accumulated_cloud;
  for (uint16_t i = 0; i < m_num_clouds; i++)
  {
    m_point_clouds_local->cloud_sizes[i] = _point_cloud_sizes[i];
    m_point_clouds_local->clouds_base_addresses[i] = tmp_addr;
    tmp_addr += _point_cloud_sizes[i];
  }

  // allocate structure on device
  HANDLE_CUDA_ERROR(cudaMalloc((void** )&m_dev_ptr_to_point_clouds_struct, sizeof(MetaPointCloudStruct)));

  // allocate space for array of point clouds sizes on device and save pointers in host local copy:
  HANDLE_CUDA_ERROR(cudaMalloc((void** )&m_dev_ptr_to_cloud_sizes, m_num_clouds * sizeof(uint32_t)));
  // copy the sizes to the device
  HANDLE_CUDA_ERROR(
      cudaMemcpy(m_dev_ptr_to_cloud_sizes, m_point_clouds_local->cloud_sizes, m_num_clouds * sizeof(uint32_t),
                 cudaMemcpyHostToDevice));

  // allocate space for array of point clouds base adresses on device and save pointers in host local copy:
  HANDLE_CUDA_ERROR(
      cudaMalloc((void** )&m_dev_ptr_to_clouds_base_addresses, m_num_clouds * sizeof(Vector3f*)));

  // allocate the accumulated point cloud space
  HANDLE_CUDA_ERROR(
      cudaMalloc((void** )&m_dev_ptr_to_accumulated_cloud, m_accumulated_pointcloud_size * sizeof(Vector3f)));

  // copy the base adresses to the device
  m_dev_ptrs_to_addrs = new Vector3f*[m_num_clouds];
  Vector3f* ptr_iterator = m_dev_ptr_to_accumulated_cloud;
  for (uint16_t i = 0; i < m_num_clouds; i++)
  {
    m_dev_ptrs_to_addrs[i] = ptr_iterator;
    //printf("Addr of cloud %d = %p\n", i , m_dev_ptrs_to_addrs[i]);
    ptr_iterator += _point_cloud_sizes[i];
  }
  HANDLE_CUDA_ERROR(
      cudaMemcpy(m_dev_ptr_to_clouds_base_addresses, m_dev_ptrs_to_addrs, m_num_clouds * sizeof(Vector3f*),
                 cudaMemcpyHostToDevice));

  //printf("Addr of m_dev_ptr_to_clouds_base_addresses: %p\n", m_dev_ptr_to_clouds_base_addresses);

  // copy the structure with the device pointers to the device
  m_dev_point_clouds_local = new MetaPointCloudStruct();
  m_dev_point_clouds_local->num_clouds = m_num_clouds;
  m_dev_point_clouds_local->accumulated_cloud_size = m_accumulated_pointcloud_size;
  m_dev_point_clouds_local->cloud_sizes = m_dev_ptr_to_cloud_sizes;
  m_dev_point_clouds_local->clouds_base_addresses = m_dev_ptr_to_clouds_base_addresses;
  HANDLE_CUDA_ERROR(
      cudaMemcpy(m_dev_ptr_to_point_clouds_struct, m_dev_point_clouds_local, sizeof(MetaPointCloudStruct),
                 cudaMemcpyHostToDevice));

  LOGGING_INFO_C(
      Gpu_voxels_helpers,
      MetaPointCloud,
      "This MetaPointCloud requires: " << (m_accumulated_pointcloud_size * sizeof(Vector3f)) / 1024.0 / 1024.0 << "MB on the GPU and on the Host" << endl);

}


void MetaPointCloud::addClouds(const std::vector<std::string> &_point_cloud_files, bool use_model_path)
{
  std::vector<std::vector<Vector3f> > *point_clouds = new std::vector<std::vector<Vector3f> >(
      _point_cloud_files.size());

  for (size_t i = 0; i < _point_cloud_files.size(); i++)
  {
    if(!file_handling::PointcloudFileHandler::Instance()->loadPointCloud(_point_cloud_files.at(i), use_model_path, point_clouds->at(i)))
    {
      LOGGING_ERROR_C(Gpu_voxels_helpers, MetaPointCloud,
                      "Could not read file " << _point_cloud_files.at(i) << icl_core::logging::endl);
      return;
    }
  }

  std::vector<uint32_t> point_cloud_sizes(point_clouds->size());
  for (size_t i = 0; i < point_clouds->size(); i++)
  {
    point_cloud_sizes.at(i) = point_clouds->at(i).size();
  }
  init(point_cloud_sizes);

  for (size_t i = 0; i < point_clouds->size(); i++)
  {
    updatePointCloud(i, point_clouds->at(i), false);
  }
  syncToDevice();
  delete point_clouds;
}


MetaPointCloud::MetaPointCloud(const std::vector<std::string> &_point_cloud_files, bool use_model_path)
{
  addClouds(_point_cloud_files, use_model_path);
}

MetaPointCloud::MetaPointCloud(const std::vector<std::string> &_point_cloud_files,
               const std::vector<std::string> &_point_cloud_names, bool use_model_path)

{
  addClouds(_point_cloud_files, use_model_path);

  if(_point_cloud_files.size() == _point_cloud_names.size())
  {
    for(size_t i = 0; i < _point_cloud_files.size(); i++)
    {
      m_point_cloud_names[i] = _point_cloud_names[i];
    }
  }else{
    LOGGING_ERROR_C(Gpu_voxels_helpers, MetaPointCloud,
                    "Number of names differs to number of pointcloud files!" << icl_core::logging::endl);
  }
}


MetaPointCloud::MetaPointCloud()
{
  const std::vector<uint32_t> _point_cloud_sizes(0,0);
  init(_point_cloud_sizes);
}

MetaPointCloud::MetaPointCloud(const std::vector<uint32_t> &_point_cloud_sizes)
{
  init(_point_cloud_sizes);
}

MetaPointCloud::MetaPointCloud(const MetaPointCloud &other)
{
  init(other.getPointcloudSizes());
  m_point_cloud_names = other.getCloudNames();
  for (uint16_t i = 0; i < other.getNumberOfPointclouds(); i++)
  {
    updatePointCloud(i, other.getPointCloud(i), other.getPointcloudSize(i), false);
  }
  syncToDevice();
}

MetaPointCloud::MetaPointCloud(const MetaPointCloud *other)
{
  init(other->getPointcloudSizes());
  m_point_cloud_names = other->getCloudNames();
  for (uint16_t i = 0; i < other->getNumberOfPointclouds(); i++)
  {
    updatePointCloud(i, other->getPointCloud(i), other->getPointcloudSize(i), false);
  }
  syncToDevice();
}

MetaPointCloud::MetaPointCloud(const std::vector<std::vector<Vector3f> > &point_clouds)
{
  std::vector<uint32_t> point_cloud_sizes(point_clouds.size());
  for (size_t i = 0; i < point_clouds.size(); i++)
  {
    point_cloud_sizes[i] = point_clouds[i].size();
  }
  init(point_cloud_sizes);

  for (size_t i = 0; i < point_clouds.size(); i++)
  {
    updatePointCloud(i, point_clouds[i], false);
  }
  syncToDevice();
}

void MetaPointCloud::addCloud(uint32_t cloud_size)
{
  std::vector<uint32_t> new_sizes = getPointcloudSizes();
  new_sizes.push_back(cloud_size);

  // backup the current clouds
  uint32_t old_accumulated_size = m_accumulated_pointcloud_size;
  Vector3f *tmp_clouds = new Vector3f[old_accumulated_size];
  memcpy(tmp_clouds, m_accumulated_cloud,
         sizeof(Vector3f) * old_accumulated_size);

  // Destruct current clouds on host and device
  destruct();
  // Allocate new mem
  init(new_sizes);

  // Restore previous data to new mem addresses
  memcpy(m_accumulated_cloud, tmp_clouds,
         sizeof(Vector3f) * old_accumulated_size);

  delete tmp_clouds;
}

void MetaPointCloud::addCloud(const std::vector<Vector3f> &cloud, bool sync, const std::string& name )
{
  // Allocate mem and restore old data
  addCloud(cloud.size());

  // Copy the new cloud to host memory
  memcpy(m_point_clouds_local->clouds_base_addresses[m_num_clouds-1], cloud.data(),
         sizeof(Vector3f) * m_point_clouds_local->cloud_sizes[m_num_clouds-1]);

  if(!name.empty())
  {
    m_point_cloud_names[m_num_clouds-1] = name;
  }

  if (sync)
  {
    syncToDevice(m_num_clouds-1);
  }
}

std::string MetaPointCloud::getCloudName(uint16_t i) const
{
  std::map<uint16_t, std::string>::const_iterator it = m_point_cloud_names.find(i);
  if(it != m_point_cloud_names.end())
  {
    return it->second;
  }
  return std::string();
}

int16_t MetaPointCloud::getCloudNumber(const std::string& name) const
{
  for (std::map<uint16_t, std::string>::const_iterator it=m_point_cloud_names.begin(); it!=m_point_cloud_names.end(); ++it)
  {
    if(name.compare(it->second) == 0)
    {
      return it->first;
    }
  }
  return -1;
}

void MetaPointCloud::destruct()
{
  if (m_dev_ptr_to_point_clouds_struct)
    HANDLE_CUDA_ERROR(cudaFree(m_dev_ptr_to_point_clouds_struct));
  if (m_dev_ptr_to_cloud_sizes)
    HANDLE_CUDA_ERROR(cudaFree(m_dev_ptr_to_cloud_sizes));
  if (m_dev_ptr_to_accumulated_cloud)
    HANDLE_CUDA_ERROR(cudaFree(m_dev_ptr_to_accumulated_cloud));
  if (m_dev_ptr_to_clouds_base_addresses)
    HANDLE_CUDA_ERROR(cudaFree(m_dev_ptr_to_clouds_base_addresses));
  if (m_accumulated_cloud)
    delete (m_accumulated_cloud);
  if (m_dev_point_clouds_local)
    delete (m_dev_point_clouds_local);
  if (m_dev_ptrs_to_addrs)
    delete (m_dev_ptrs_to_addrs);
  if (m_point_clouds_local->cloud_sizes)
    delete (m_point_clouds_local->cloud_sizes);
  if (m_point_clouds_local->clouds_base_addresses)
    delete (m_point_clouds_local->clouds_base_addresses);
  if (m_point_clouds_local)
    delete (m_point_clouds_local);
}

MetaPointCloud::~MetaPointCloud()
{
  destruct();
}

void MetaPointCloud::syncToDevice()
{
  // copy all clouds to the device
  HANDLE_CUDA_ERROR(
      cudaMemcpy(m_dev_ptrs_to_addrs[0], m_point_clouds_local->clouds_base_addresses[0],
      sizeof(Vector3f) * m_accumulated_pointcloud_size, cudaMemcpyHostToDevice));
}

void MetaPointCloud::syncToHost()
{
  // copy all clouds to the host
  HANDLE_CUDA_ERROR(
      cudaMemcpy(m_point_clouds_local->clouds_base_addresses[0], m_dev_ptrs_to_addrs[0],
      sizeof(Vector3f) * m_accumulated_pointcloud_size, cudaMemcpyDeviceToHost));
}

void MetaPointCloud::syncToDevice(uint16_t cloud)
{
  if (cloud < m_num_clouds)
  {
    // copy only the indicated cloud
    HANDLE_CUDA_ERROR(
        cudaMemcpy(m_dev_ptrs_to_addrs[cloud], m_point_clouds_local->clouds_base_addresses[cloud],
                   sizeof(Vector3f) * m_point_clouds_local->cloud_sizes[cloud], cudaMemcpyHostToDevice));
  }
  else
  {
    LOGGING_ERROR_C(Gpu_voxels_helpers, MetaPointCloud,
                    "Cloud " << cloud << "does not exist" << icl_core::logging::endl);
  }
}

void MetaPointCloud::updatePointCloud(uint16_t cloud, const std::vector<Vector3f> &pointcloud, bool sync)
{
  updatePointCloud(cloud, pointcloud.data(), pointcloud.size(), sync);
}

void MetaPointCloud::updatePointCloud(const std::string cloud_name, const std::vector<Vector3f> &pointcloud, bool sync)
{
  updatePointCloud(getCloudNumber(cloud_name), pointcloud.data(), pointcloud.size(), sync);
}

void MetaPointCloud::updatePointCloud(uint16_t cloud, const Vector3f* pointcloud, uint32_t pointcloud_size,
                                      bool sync)
{
  if (pointcloud_size == m_point_clouds_local->cloud_sizes[cloud])
  {
    // Copy the cloud to host memory
    memcpy(m_point_clouds_local->clouds_base_addresses[cloud], pointcloud,
           sizeof(Vector3f) * pointcloud_size);
  }
  else
  {
    LOGGING_WARNING_C(Gpu_voxels_helpers, MetaPointCloud,
                    "Size of pointcloud changed! Rearanging memory" << icl_core::logging::endl);

    std::vector<uint32_t> new_sizes = getPointcloudSizes();
    new_sizes.at(cloud) = pointcloud_size;

    // backup the current clouds
    std::vector<Vector3f*> tmp_clouds;
    for(uint16_t i = 0; i < m_num_clouds; i++)
    {
      if(i != cloud)
      {
        tmp_clouds.push_back(new Vector3f[m_point_cloud_sizes.at(i)]);
        memcpy(tmp_clouds.at(i), m_point_clouds_local->clouds_base_addresses[i],
               sizeof(Vector3f) * m_point_cloud_sizes.at(i));
      }else{
        // skip the modified cloud
      }
    }
    destruct();      // Destruct current clouds on host and device
    init(new_sizes); // Allocate new mem
    // Restore previous data to new mem addresses
    for(uint16_t i = 0; i < m_num_clouds; i++)
    {
      if(i != cloud)
      {
        memcpy(m_point_clouds_local->clouds_base_addresses[i], tmp_clouds[i],
               sizeof(Vector3f) * m_point_cloud_sizes.at(i));
      }else{
        memcpy(m_point_clouds_local->clouds_base_addresses[i], pointcloud,
               sizeof(Vector3f) * m_point_cloud_sizes.at(i));
      }
    }
    for(std::size_t i = 0; i < tmp_clouds.size(); i++)
    {
      delete tmp_clouds.at(i);
    }
  }
  if (sync)
  {
    syncToDevice(cloud);
  }
}

uint32_t MetaPointCloud::getPointcloudSize(uint16_t cloud) const
{
  return m_point_clouds_local->cloud_sizes[cloud];
}

uint32_t MetaPointCloud::getAccumulatedPointcloudSize() const
{
  return m_accumulated_pointcloud_size;
}

const std::map<uint16_t, std::string> MetaPointCloud::getCloudNames() const
{
  return m_point_cloud_names;
}

void MetaPointCloud::debugPointCloud() const
{

  printf("================== hostMetaPointCloud DBG ================== \n");

  printf("hostDebugMetaPointCloud DBG: NumClouds: %d \n", m_num_clouds);

  printf("hostDebugMetaPointCloud DBG: m_dev_ptr_to_clouds_base_addresses: %p \n",
          m_point_clouds_local->clouds_base_addresses);

  for(int i = 0; i < m_point_clouds_local->num_clouds; i++)
  {
      printf("hostDebugMetaPointCloud DBG: '%s' CloudSize[%d]: %d, clouds_base_addresses[%d]: %p \n",
             this->getCloudName(i).c_str(),
             i, m_point_clouds_local->cloud_sizes[i],
             i, m_point_clouds_local->clouds_base_addresses[i]);

      if (m_point_clouds_local->cloud_sizes[i] > 0)
      {
        Vector3f min_xyz = m_point_clouds_local->clouds_base_addresses[i][0];
        Vector3f max_xyz = m_point_clouds_local->clouds_base_addresses[i][0];
        for (uint32_t j = 1; j < m_point_clouds_local->cloud_sizes[i]; j++)
        {
          min_xyz.x = std::min(min_xyz.x, m_point_clouds_local->clouds_base_addresses[i][j].x);
          min_xyz.y = std::min(min_xyz.y, m_point_clouds_local->clouds_base_addresses[i][j].y);
          min_xyz.z = std::min(min_xyz.z, m_point_clouds_local->clouds_base_addresses[i][j].z);

          max_xyz.x = std::max(max_xyz.x, m_point_clouds_local->clouds_base_addresses[i][j].x);
          max_xyz.y = std::max(max_xyz.y, m_point_clouds_local->clouds_base_addresses[i][j].y);
          max_xyz.z = std::max(max_xyz.z, m_point_clouds_local->clouds_base_addresses[i][j].z);
        }

        printf("hostDebugMetaPointCloud DBG: Cloud %d bounds: Min[%f, %f, %f], Max[%f, %f, %f] \n",
               i, min_xyz.x, min_xyz.y, min_xyz.z, max_xyz.x, max_xyz.y, max_xyz.z);
      }
  }

  printf("================== END hostDebugMetaPointCloud DBG ================== \n");

  kernelDebugMetaPointCloud<<< 1,1 >>>(m_dev_ptr_to_point_clouds_struct);
}

} // end of ns
