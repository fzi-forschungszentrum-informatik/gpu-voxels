#include "IbeoNcomFilterGPU.h"

//#include <gpu_voxels/helpers/CudaMath.h>
#include <gpu_voxels/helpers/cuda_datatypes.h>
#include <gpu_voxels/robot/kernels/KinematicOperations.h>
#include <thrust/device_vector.h>
#include <gpu_classification/logging/logging_classification.h>


namespace gpu_voxels {
namespace classification {

IbeoNcomFilterGPU::IbeoNcomFilterGPU(std::string pcName)
  :pointCloudName(pcName)
{
  HANDLE_CUDA_ERROR(cudaMalloc((void** )&transform_matrix_dev, sizeof(Matrix4f)));

  //    gvl = new GpuVoxels(200, 200, 200, 0.1);

  //    //create new Map
  //    gvl->addMap(MT_BITVECTOR_OCTREE, "myFilterMap");
}

struct MatrixTransform
{
  Matrix4f trans;

  MatrixTransform(Matrix4f m)
  {
    trans = m;
  }

  __host__ __device__
  Vector3f operator() (Vector3f a)
  {
    return trans * a;
  }
};

void IbeoNcomFilterGPU::transformPointCloud(Matrix4f *transform_matrix, MetaPointCloud *mpc, MetaPointCloud *transform_mpc)
{

  for(uint16_t i = 0; i < mpc->getNumberOfPointclouds(); i++)
  {
    // get the trafo of the according URDF link
    //m_transformation = Robot::getLink( Robot::getLinkPointclouds()->getCloudName(i) )->getPoseAsGpuMat4f();
    //    std::cout << "RobotToGPU::update() transform of " << robot->getLinkPointclouds()->getCloudName(i)
    //              << " = " << transformation << std::endl;

    HANDLE_CUDA_ERROR(cudaMemcpy(transform_matrix_dev, transform_matrix, sizeof(Matrix4f), cudaMemcpyHostToDevice));

    computeLinearLoad(mpc->getPointcloudSize(i),
                      &m_blocks, &m_threads_per_block);
    cudaDeviceSynchronize();
    // transform the cloud via Kernel.

    if(m_blocks != 0 && m_threads_per_block != 0)
    {
      /*kernelKinematicChainTransform<<< m_blocks, m_threads_per_block >>>(i, transform_matrix_dev,
                                                                        mpc->getDeviceConstPointer(),
                                                                        transform_mpc->getDevicePointer());*/
      LOGGING_ERROR(Classification, "kernelKinematicChainTransform() DOES NOT EXIST ANYMORE !!!"<< endl);

    }
    HANDLE_CUDA_ERROR(cudaDeviceSynchronize());
  }

}

void IbeoNcomFilterGPU::transformPointCloudTest(Matrix4f* transform_matrix, std::vector<Vector3f>* pointcloud, Vector3f* out_pointcloud_d, int* out_size)
{
  MatrixTransform trans(*transform_matrix);

  thrust::device_vector<Vector3f> pc(*pointcloud);

  thrust::for_each(pc.begin(),
                   pc.end(),
                   trans);

  out_pointcloud_d = thrust::raw_pointer_cast(&pc[0]);
  *out_size = pc.size();
}

void IbeoNcomFilterGPU::transformPointCloudSTD(Matrix4f* transform_matrix, std::vector<Vector3f>* pointcloud)
{
  MatrixTransform trans(*transform_matrix);

  thrust::device_vector<Vector3f> pc(*pointcloud);
  thrust::transform(pc.begin(),
                    pc.end(),
                    pc.begin(),
                    trans);
  thrust::host_vector<Vector3f> temp(pc);
  for(uint i = 0; i < pointcloud->size(); i++)
  {
    (*pointcloud)[i] = temp[i];
  }

}

Vector3f* IbeoNcomFilterGPU::getTransformedPointCloud()
{
  return transformed;
}

void IbeoNcomFilterGPU::getNumberOfMaxThreads()
{
  int deviceCount, device;
  //int gpuDeviceCount = 0;
  struct cudaDeviceProp properties;
  cudaError_t cudaResultCode = cudaGetDeviceCount(&deviceCount);
  if (cudaResultCode != cudaSuccess)
    deviceCount = 0;
  /* machines with no GPUs can still report one emulation device */
  for (device = 0; device < deviceCount; ++device) {
    cudaGetDeviceProperties(&properties, device);
    if (properties.major != 9999) /* 9999 means emulation only */
      if (device==0)
      {
        printf("multiProcessorCount %d\n",properties.multiProcessorCount);
        printf("maxThreadsPerMultiProcessor %d\n",properties.maxThreadsPerMultiProcessor);
      }
  }
}


} //end namespace classification
} //end namespace gpu_voxels
