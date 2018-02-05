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
 * \date    2016-05-25
 *
 */
//----------------------------------------------------------------------
#ifndef GPU_VOXELS_VOXELMAP_TESTS_HPP_INCLUDED
#define GPU_VOXELS_VOXELMAP_TESTS_HPP_INCLUDED


#include <gpu_voxels/voxelmap/Tests.h>
#include <gpu_voxels/voxelmap/TemplateVoxelMap.hpp>
#include <gpu_voxels/voxelmap/kernels/VoxelMapTests.hpp>

#include <thrust/device_vector.h>
#include <thrust/device_ptr.h>
#include <thrust/random/linear_congruential_engine.h>
#include <thrust/random/uniform_real_distribution.h>

namespace gpu_voxels {
namespace voxelmap {
namespace test {




struct RandGen
{
  thrust::uniform_real_distribution<float> dist_x;
  thrust::uniform_real_distribution<float> dist_y;
  thrust::uniform_real_distribution<float> dist_z;

  __host__   __device__
  RandGen(Vector3f min, Vector3f max) {
    // create a uniform_real_distribution to produce floats
    dist_x = thrust::uniform_real_distribution<float>(min.x, max.x);
    dist_y = thrust::uniform_real_distribution<float>(min.y, max.y);
    dist_z = thrust::uniform_real_distribution<float>(min.z, max.z);
  }

  __host__   __device__
  Vector3f operator() (const unsigned int n)
  {    
    // create a minstd_rand object to act as our source of randomness
    thrust::minstd_rand rnd;
    rnd.discard(n);
    return Vector3f(dist_x(rnd), dist_y(rnd), dist_z(rnd));
  }
};

template<class Voxel>
void triggerAddressingTest(Vector3ui dimensions, float voxel_side_length,
                           size_t nr_of_tests, bool *success)
{
  thrust::device_vector<Vector3f> dev_testpoint_list(nr_of_tests);

  srand(time(0));
  Voxel* voxelmap_base_adress;
  voxelmap_base_adress = (Voxel*)1234;

  bool* dev_success;
  HANDLE_CUDA_ERROR(cudaMalloc((void** )&dev_success, sizeof(bool)));
  HANDLE_CUDA_ERROR(cudaMemcpy(dev_success, success, sizeof(bool), cudaMemcpyHostToDevice));

  RandGen myRandGen(Vector3f(), dimensions * voxel_side_length);

  thrust::counting_iterator<unsigned int> index_sequence_begin(0);
  thrust::transform(index_sequence_begin, index_sequence_begin + nr_of_tests, dev_testpoint_list.begin(), myRandGen);

  u_int32_t num_blocks;
  u_int32_t threads_per_block;
  computeLinearLoad(nr_of_tests, &num_blocks, &threads_per_block);
  kernelAddressingTest<<< num_blocks, threads_per_block >>> (voxelmap_base_adress, dimensions, voxel_side_length,
                                                             thrust::raw_pointer_cast(dev_testpoint_list.data()),
                                                             nr_of_tests, dev_success);
  CHECK_CUDA_ERROR();
  HANDLE_CUDA_ERROR(cudaDeviceSynchronize());
  HANDLE_CUDA_ERROR(cudaMemcpy(success, dev_success, sizeof(bool), cudaMemcpyDeviceToHost));
}


} // end of namespace
} // end of namespace
} // end of namespace

#endif
