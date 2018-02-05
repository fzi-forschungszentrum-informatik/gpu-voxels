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
 * \author  Florian Drews
 * \date    2013-11-15
 *
 */
//----------------------------------------------------------------------

#ifndef KERNEL_POINTCLOUD_H_
#define KERNEL_POINTCLOUD_H_

#include <cuda_runtime.h>
#include <stdint.h>
#include <gpu_voxels/octree/Voxel.h>
#include <gpu_voxels/octree/Morton.h>
#include <gpu_voxels/octree/Sensor.h>
#include <gpu_voxels/helpers/cuda_datatypes.h>
#include <gpu_voxels/octree/PointCloud.h>
#include <gpu_voxels/octree/DataTypes.h>
#include <float.h>
#include <thrust/device_vector.h>
#include <cmath>

namespace gpu_voxels {
namespace NTree {

/**
 * Converts an array of voxel identified by Cartesian coordinates x y z to voxel identified by their morton code.
 */__global__
void kernel_toMortonCode(uint3* inputVoxel, voxel_count numVoxel, OctreeVoxelID* outputVoxel);

__global__
void kernel_transformKinectPoints(gpu_voxels::Vector3f* point_cloud, OctreeVoxelID num_points, Voxel* voxel,
                                  Sensor* sensor, gpu_voxels::Vector3f voxel_dimension);

__global__ void kernel_transformKinectPoints_simple(gpu_voxels::Vector3f* point_cloud, const voxel_count num_points,
                                                    OctreeVoxelID* voxel, Sensor* sensor,
                                                    const uint32_t resolution);

template<bool COUNT_MODE>
__global__
void kernel_voxelize(OctreeVoxelID* voxelInput, const voxel_count numVoxel, voxel_count* countVoxel,
                     Voxel* voxel_output)
{
  const uint32_t NUM_THREADS = WARP_SIZE;
  assert(NUM_THREADS == blockDim.x); // otherwise an inter-WARP reduce/prefixSum is needed

  const voxel_count chunk_size = ceil(double(numVoxel) / (gridDim.x));
  const voxel_count block_id = blockIdx.x;
  const voxel_count from = chunk_size * block_id;
  const voxel_count to = (OctreeVoxelID) min((unsigned long long int) (from + chunk_size),
                                       (unsigned long long int) numVoxel);
  const uint32_t thread_id = threadIdx.x;

  __shared__ voxel_count shared_voxel_count[NUM_THREADS];
  __shared__ voxel_count shared_write_position;

  voxel_count my_voxel_count = 0;

  if (!COUNT_MODE)
  {
    if (thread_id == 0)
    {
      shared_write_position = countVoxel[block_id];
    }
  }
  __syncthreads();

  for (voxel_count i = from; i < to; i += NUM_THREADS)
  {
    voxel_count my_id = i + thread_id;
    bool is_active = my_id < to;
    OctreeVoxelID my_voxel_id = is_active ? voxelInput[my_id] : INVALID_VOXEL;
    bool is_new_voxel = is_active & ((my_id == 0) || voxelInput[my_id - 1] != voxelInput[my_id]);
    if (COUNT_MODE)
    {
      my_voxel_count += is_new_voxel;
    }
    else
    {
      uint32_t new_voxel_votes = BALLOT(is_new_voxel);
      if (is_new_voxel)
      {
        const uint32_t my_write_pos = shared_write_position
            + __popc(new_voxel_votes << (WARP_SIZE - thread_id));

        // set start index
        voxel_output[my_write_pos].coordinates.x = my_id;

        // set voxel id
        voxel_output[my_write_pos].voxelId = my_voxel_id;

        // set end index
        if (my_id != 0)
          voxel_output[my_write_pos - 1].coordinates.y = my_id - 1;

//        // ## count number of points in my voxel ##
//        uint32_t num_points_in_voxel = __clz(__brev(new_voxel_votes) << (thread_id + 1)) + 1;
//        num_points_in_voxel = min(num_points_in_voxel, WARP_SIZE - thread_id);
//        num_points_in_voxel = min(num_points_in_voxel, to - my_id);
//        uint32_t occupancy = sensor->sensorModel.INITIAL_PROBABILITY * num_points_in_voxel;
//
//        // check for overflow
//        assert(
//            (voxel_output[shared_write_position + my_write_pos].coordinates.x + occupancy)
//                > max(voxel_output[shared_write_position + my_write_pos].coordinates.x, occupancy));
//
//        // exploit coordinates to sum the occupancy
//        atomicAdd(&voxel_output[shared_write_position + my_write_pos].coordinates.x, occupancy);
//
//        // ## sum up previous voxel points ###
//
//
//
//        // ## first thread sets the id ##

      }
      __syncthreads();

      if (thread_id == 0)
        shared_write_position += __popc(new_voxel_votes);
      __syncthreads();
    }
  }
  __syncthreads();

  if (!COUNT_MODE)
  {
    assert(shared_write_position == countVoxel[block_id + 1]);
  }

  if (COUNT_MODE)
  {
    shared_voxel_count[thread_id] = my_voxel_count;
    __syncthreads();

    REDUCE(shared_voxel_count, thread_id, NUM_THREADS, +);

    if (thread_id == 0)
      countVoxel[block_id] = shared_voxel_count[0];
  }
}

__global__
void kernel_voxelize_finalStep(OctreeVoxelID* voxelInput, voxel_count numVoxel, const voxel_count num_output_voxel,
                               Voxel* voxel_output, Sensor* sensor);

__global__
void kernel_countVoxel(Voxel* voxelInput, OctreeVoxelID numVoxel, OctreeVoxelID* countVoxel);

__global__
void kernel_combineEqualVoxel(Voxel* voxelInput, OctreeVoxelID numVoxel, OctreeVoxelID* countVoxel, Voxel* outputVoxel,
                              Sensor* sensor);

__global__
void kernel_toMortonCode(Vector3ui* inputVoxel, voxel_count numVoxel, OctreeVoxelID* outputVoxel);

__global__
void kernel_transformDepthImage(DepthData* depth_image, gpu_voxels::Vector3f* d_point_cloud, Sensor* sensor,
                                const DepthData invalid_measure);

//__global__
//void kernel_preprocessObjectDepthImage(DepthData* d_depth_image, const uint32_t width, const uint32_t height,
//                                       const DepthData noSampleValue, const DepthData shadowValue,
//                                       const DepthData max_sensor_distance);
//
//__global__
//void kernel_preprocessFreeSpaceDepthImage(DepthData* d_depth_image, const uint32_t width,
//                                          const uint32_t height, const DepthData noSampleValue,
//                                          const DepthData shadowValue, const DepthData max_sensor_distance);

__global__
void kernel_preprocessDepthImage(DepthData* d_depth_image, const uint32_t width, const uint32_t height,
                                 const SensorDataProcessing arguments);

__global__
void kernel_toVoxels(const Vector3f *input_points, size_t num_points, Vector3ui* output_voxels, float voxel_side_length);
}
}

#endif /* KERNEL_POINTCLOUD_H_ */
