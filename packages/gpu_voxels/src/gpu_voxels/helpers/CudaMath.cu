// this is for emacs file handling -*- mode: c++; indent-tabs-mode: nil -*-

// -- BEGIN LICENSE BLOCK ----------------------------------------------
// -- END LICENSE BLOCK ------------------------------------------------

//----------------------------------------------------------------------
/*!\file
 *
 * \author  Sebastian Klemm
 * \date    2012-08-23
 *
 */
//----------------------------------------------------------------------
#include "CudaMath.h"
#include "cuda_handling.hpp"
#include "kernels/PointCloudOperations.h"
#include <cstdio>
#include <iostream>

#include <thrust/device_vector.h>
#include <thrust/transform.h>


namespace gpu_voxels {

CudaMath::CudaMath()
  : m_max_nr_of_devices(10),
    m_max_nr_of_blocks(65535),
    m_max_nr_of_threads_per_block(1024)
{
}

CudaMath::CudaMath(unsigned int max_nr_of_devices, unsigned int max_nr_of_blocks, unsigned int max_nr_of_threads_per_block)
 : m_max_nr_of_devices(max_nr_of_devices),
   m_max_nr_of_blocks(max_nr_of_blocks),
   m_max_nr_of_threads_per_block(max_nr_of_threads_per_block)
{
}

CudaMath::~CudaMath()
{
}

void CudaMath::printMatrix(const Matrix4f& matrix)
{
  printf("  %0.7lf  %0.7lf  %0.7lf  %0.7lf\n",   matrix.a11, matrix.a12, matrix.a13, matrix.a14);
  printf("  %0.7lf  %0.7lf  %0.7lf  %0.7lf\n",   matrix.a21, matrix.a22, matrix.a23, matrix.a24);
  printf("  %0.7lf  %0.7lf  %0.7lf  %0.7lf\n",   matrix.a31, matrix.a32, matrix.a33, matrix.a34);
  printf("  %0.7lf  %0.7lf  %0.7lf  %0.7lf\n\n", matrix.a41, matrix.a42, matrix.a43, matrix.a44);
}

void CudaMath::computeLinearLoad(const uint32_t nr_of_items, uint32_t* blocks, uint32_t* threads_per_block)
{

//  if (nr_of_items <= m_max_nr_of_blocks)
//  {
//    *blocks = nr_of_items;
//    *threads_per_block = 1;
//  }
//  else
//  {
    if (nr_of_items <= m_max_nr_of_blocks * m_max_nr_of_threads_per_block)
    {
      *blocks = (nr_of_items + m_max_nr_of_threads_per_block - 1)
              / m_max_nr_of_threads_per_block;                          // calculation replaces a ceil() function
      *threads_per_block = m_max_nr_of_threads_per_block;
    }
    else
    {
      /* In this case the kernel must perform multiple runs because
       * nr_of_items is larger than the gpu can handle at once.
       * To overcome this limits use standard parallelism offsets
       * as when programming host code (increment by the number of all threads
       * running). Use something like
       *
       *   uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
       *
       *   while (i < nr_of_items)
       *   {
       *     // perform some kernel operations here
       *
       *     // increment by number of all threads that are running
       *     i += blockDim.x * gridDim.x;
       *   }
       */
      *blocks = m_max_nr_of_blocks;
      *threads_per_block = m_max_nr_of_threads_per_block;
    }

}

void CudaMath::Vec3ToMat4(const Vector3f& vec_in, Matrix4f& mat_out)
{
  mat_out.a14 = vec_in.x;
  mat_out.a24 = vec_in.x;
  mat_out.a34 = vec_in.x;
  mat_out.a44 = 1;
}

void CudaMath::Mat3ToMat4(const Matrix3d& mat_in, Matrix4f& mat_out)
{
  mat_out.a11 = mat_in.a11;  mat_out.a12 = mat_in.a12;  mat_out.a13 = mat_in.a13;  mat_out.a14 = 0;
  mat_out.a21 = mat_in.a21;  mat_out.a22 = mat_in.a22;  mat_out.a23 = mat_in.a13;  mat_out.a24 = 0;
  mat_out.a31 = mat_in.a31;  mat_out.a32 = mat_in.a32;  mat_out.a33 = mat_in.a13;  mat_out.a34 = 0;
  mat_out.a41 = 0;           mat_out.a42 = 0;           mat_out.a43 = 0;           mat_out.a44 = 1;
}

void CudaMath::Mat3AndVec3ToMat4(const Matrix3d& mat_in, const Vector3f& vec_in, Matrix4f& mat_out)
{
  mat_out.a11 = mat_in.a11;  mat_out.a12 = mat_in.a12;  mat_out.a13 = mat_in.a13;  mat_out.a14 = vec_in.x;
  mat_out.a21 = mat_in.a21;  mat_out.a22 = mat_in.a22;  mat_out.a23 = mat_in.a23;  mat_out.a24 = vec_in.y;
  mat_out.a31 = mat_in.a31;  mat_out.a32 = mat_in.a32;  mat_out.a33 = mat_in.a33;  mat_out.a34 = vec_in.z;
  mat_out.a41 = 0;           mat_out.a42 = 0;           mat_out.a43 = 0;           mat_out.a44 = 1;
}

void CudaMath::Mat3AndVec3ToMat4(const Matrix3f& mat_in, const Vector3f& vec_in, Matrix4f& mat_out)
{
  mat_out.a11 = mat_in.a11;  mat_out.a12 = mat_in.a12;  mat_out.a13 = mat_in.a13;  mat_out.a14 = vec_in.x;
  mat_out.a21 = mat_in.a21;  mat_out.a22 = mat_in.a22;  mat_out.a23 = mat_in.a23;  mat_out.a24 = vec_in.y;
  mat_out.a31 = mat_in.a31;  mat_out.a32 = mat_in.a32;  mat_out.a33 = mat_in.a33;  mat_out.a34 = vec_in.z;
  mat_out.a41 = 0;           mat_out.a42 = 0;           mat_out.a43 = 0;           mat_out.a44 = 1;
}



void CudaMath::Mat3AndVec4ToMat4(const Matrix3d& mat_in, const Vector4d& vec_in, Matrix4f& mat_out)
{
  mat_out.a11 = mat_in.a11;  mat_out.a12 = mat_in.a12;  mat_out.a13 = mat_in.a13;  mat_out.a14 = vec_in.x;
  mat_out.a21 = mat_in.a21;  mat_out.a22 = mat_in.a22;  mat_out.a23 = mat_in.a23;  mat_out.a24 = vec_in.y;
  mat_out.a31 = mat_in.a31;  mat_out.a32 = mat_in.a32;  mat_out.a33 = mat_in.a33;  mat_out.a34 = vec_in.z;
  mat_out.a41 = 0;           mat_out.a42 = 0;           mat_out.a43 = 0;           mat_out.a44 = vec_in.w;
}



void CudaMath::transpose(const Matrix4f& mat_in, Matrix4f& mat_out)
{
  mat_out.a11 = mat_in.a11;  mat_out.a12 = mat_in.a21;  mat_out.a13 = mat_in.a31;  mat_out.a14 = mat_in.a41;
  mat_out.a21 = mat_in.a12;  mat_out.a22 = mat_in.a22;  mat_out.a23 = mat_in.a32;  mat_out.a24 = mat_in.a42;
  mat_out.a31 = mat_in.a13;  mat_out.a32 = mat_in.a23;  mat_out.a33 = mat_in.a33;  mat_out.a34 = mat_in.a43;
  mat_out.a41 = mat_in.a14;  mat_out.a42 = mat_in.a24;  mat_out.a43 = mat_in.a34;  mat_out.a44 = mat_in.a44;
}

//void CudaMath::invert(const Matrix4f& mat_in, Matrix4f& mat_out)
//{
//  Eigen::Matrix4f mat;
//  mat << mat_in.a11, mat_in.a12, mat_in.a13, mat_in.a14,
//         mat_in.a21, mat_in.a22, mat_in.a23, mat_in.a24,
//         mat_in.a31, mat_in.a32, mat_in.a33, mat_in.a34,
//         mat_in.a41, mat_in.a42, mat_in.a43, mat_in.a44;
//
//  mat = mat.inverse().eval();
//
//  mat_out.a11 = mat(0, 0);  mat_out.a12 = mat(0, 1);  mat_out.a13 = mat(0, 2);  mat_out.a14 = mat(0, 3);
//  mat_out.a21 = mat(1, 0);  mat_out.a22 = mat(1, 1);  mat_out.a23 = mat(1, 2);  mat_out.a24 = mat(1, 3);
//  mat_out.a31 = mat(2, 0);  mat_out.a32 = mat(2, 1);  mat_out.a33 = mat(2, 2);  mat_out.a34 = mat(2, 3);
//  mat_out.a41 = mat(3, 0);  mat_out.a42 = mat(3, 1);  mat_out.a43 = mat(3, 2);  mat_out.a44 = mat(3, 3);
//}

void CudaMath::transform(uint32_t nr_of_transformations, Matrix4f& base,
                         const Matrix4f* relatives, Matrix4f* absolutes, bool invert_base)
{
  assert(invert_base == false); 

  const uint32_t array_size_bytes = nr_of_transformations * sizeof(Matrix4f);

  Matrix4f* device_base;
  Matrix4f* device_relatives;
  Matrix4f* device_absolutes;

  if (invert_base)
  {
    // invert(base, base);
  }

  // copy matrices to device
  HANDLE_CUDA_ERROR(cudaMalloc((void**)&device_base, sizeof(Matrix4f)));
  HANDLE_CUDA_ERROR(cudaMalloc((void**)&device_relatives, array_size_bytes));
  HANDLE_CUDA_ERROR(cudaMalloc((void**)&device_absolutes, array_size_bytes));
  HANDLE_CUDA_ERROR(cudaMemcpy(device_base, &base, sizeof(Matrix4f), cudaMemcpyHostToDevice));
  HANDLE_CUDA_ERROR(cudaMemcpy(device_relatives, relatives, array_size_bytes, cudaMemcpyHostToDevice));
  HANDLE_CUDA_ERROR(cudaMemcpy(device_absolutes, absolutes, array_size_bytes, cudaMemcpyHostToDevice));

  // transform
  uint32_t blocks;
  uint32_t threads_per_block;
  computeLinearLoad(nr_of_transformations, &blocks, &threads_per_block);
  HANDLE_CUDA_ERROR(cudaDeviceSynchronize());
  kernelMultiplyMatrixNbyOne<<< blocks, threads_per_block>>>
      (nr_of_transformations, device_base, device_relatives, device_absolutes);
  // copy results to host
  HANDLE_CUDA_ERROR(cudaMemcpy(absolutes, device_absolutes, array_size_bytes, cudaMemcpyDeviceToHost));
}


float CudaMath::interpolateLinear(float value1, float value2, float ratio)
{
  return (value1 * (1.0 - ratio) + value2 * ratio);
}


double CudaMath::interpolateLinear(double value1, double value2, double ratio)
{
  return (value1 * (1.0 - ratio) + value2 * ratio);
}

std::vector<float> CudaMath::interpolateLinear(const std::vector<float>& joint_state1,
                                               const std::vector<float>& joint_state2, float ratio)
{
  assert(joint_state1.size() == joint_state2.size());

  std::vector<float> result(joint_state1.size());
  for (std::size_t i=0; i<joint_state1.size(); ++i)
  {
    result.at(i) = interpolateLinear(joint_state1.at(i), joint_state2.at(i), ratio);

  }
  return result;
}

std::vector<double> CudaMath::interpolateLinear(const std::vector<double>& joint_state1,
                                                const std::vector<double>& joint_state2, double ratio)
{
  assert(joint_state1.size() == joint_state2.size());

  std::vector<double> result(joint_state1.size());
  for (std::size_t i=0; i<joint_state1.size(); ++i)
  {
    result.at(i) = interpolateLinear(joint_state1.at(i), joint_state2.at(i), ratio);

  }
  return result;
}


JointValueMap CudaMath::interpolateLinear(const JointValueMap& joint_state1,
                                          const JointValueMap& joint_state2, float ratio)
{
  assert(joint_state1.size() == joint_state2.size());

  JointValueMap result(joint_state1);
  for (JointValueMap::const_iterator it=joint_state1.begin();
       it!=joint_state1.end(); ++it)
  {
    result[it->first] = interpolateLinear(joint_state1.at(it->first),
                                          joint_state2.at(it->first), ratio);
  }
  return result;
}


struct TransformPointWithMatrix4f
{
	Matrix4f m_matrix;

  __host__ __device__
  TransformPointWithMatrix4f(Matrix4f matrix) : m_matrix(matrix)
  {
    printf("ERROR: Function TransformPointWithMatrix4f is not yet implemented!");
  }

  __host__ __device__
  Vector3f operator()(Vector3f x)
  {
    return m_matrix * x;
  }
};

void CudaMath::transform(std::vector<Vector3f>& host_point_cloud, Matrix4f matrix)
{
	thrust::device_vector<Vector3f> device_point_cloud(host_point_cloud);
	thrust::transform(device_point_cloud.begin(), device_point_cloud.end(), device_point_cloud.begin(), TransformPointWithMatrix4f(matrix));
    HANDLE_CUDA_ERROR(cudaMemcpy(&host_point_cloud[0], thrust::raw_pointer_cast(device_point_cloud.data()), host_point_cloud.size() * sizeof(Vector3f), cudaMemcpyDeviceToHost));
}


} // end of namespace
