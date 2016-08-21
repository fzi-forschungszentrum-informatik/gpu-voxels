#ifndef IBEONCOMFILTERGPU_H
#define IBEONCOMFILTERGPU_H

#include <cuda_runtime.h>
//#include <gpu_voxels/GpuVoxels.h>
#include <gpu_voxels/helpers/MetaPointCloud.h>
#include <gpu_voxels/helpers/CudaMath.h>
#include <thrust/host_vector.h>


namespace gpu_voxels {
namespace classification {

class IbeoNcomFilterGPU
{
public:
  IbeoNcomFilterGPU(std::string pcName);

  void transformPointCloud(Matrix4f* transform_matrix, gpu_voxels::MetaPointCloud* mpc, MetaPointCloud *transform_mpc);
  void transformPointCloudTest(Matrix4f* transform_matrix, std::vector<Vector3f>* pointcloud, Vector3f* out_pointcloud_d, int* out_size);
  void transformPointCloudSTD(Matrix4f* transform_matrix, std::vector<Vector3f>* pointcloud);


  Vector3f* getTransformedPointCloud();

  void getNumberOfMaxThreads();

private:
  std::string pointCloudName;

  //Matrix4f transform_matrix;
  Matrix4f* transform_matrix_dev;

  uint32_t m_blocks;
  uint32_t m_threads_per_block;

  Vector3f* transformed;

};

}//end namespace classification
}//end namespace gpu_voxels

#endif // IBEONCOMFILTERGPU_H
