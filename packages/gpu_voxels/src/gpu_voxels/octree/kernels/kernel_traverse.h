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

#ifndef KERNEL_TRAVERSE_H_
#define KERNEL_TRAVERSE_H_

#include <gpu_voxels/helpers/cuda_handling.h>
#include <gpu_voxels/octree/Morton.h>
#include <gpu_voxels/octree/NTree.h>
#include <gpu_voxels/octree/kernels/kernel_common.h>

// gpu_voxels
#include <gpu_voxels/vis_interface/VisualizerInterface.h>
#include <gpu_voxels/voxelmap/kernels/VoxelMapOperations.h>

namespace gpu_voxels {
namespace NTree {

__device__
void compute_new_status(const NodeStatus status, const bool do_final_computation, NodeStatus* new_status,
                        const uint32_t work_lane_mask, const uint32_t warp_lane, const uint32_t level,
                        int* shared_mem)
{
  //printf("%u input %u | %u\n", threadIdx.x, uint32_t(status), uint32_t(status & ns_OCCUPIED));

  const uint32_t log_free = uint32_t(log2f(ns_FREE));
  const uint32_t log_unknown = uint32_t(log2f(ns_UNKNOWN));
  const uint32_t log_occupied = uint32_t(log2f(ns_OCCUPIED));
  const uint32_t log_static = uint32_t(log2f(ns_STATIC_MAP));
  const uint32_t log_dynamic = uint32_t(log2f(ns_DYNAMIC_MAP));

  // ##### Ballot has a problem if its called right after another #####
  // Leads to wrong _ballot() result values
  // printf() before/after/between solves the problem !?
  uint32_t free_votes = BALLOT(status & ns_FREE);
  uint32_t unknown_votes = BALLOT(status & ns_UNKNOWN);
  uint32_t occupied_votes = BALLOT(status & ns_OCCUPIED);
  uint32_t static_votes = BALLOT(status & ns_STATIC_MAP);
  uint32_t dynamic_votes = BALLOT(status & ns_DYNAMIC_MAP);

  if (do_final_computation)
  {
    //printf("%u ballot %u\n", threadIdx.x, occupied_votes);
    const NodeStatus my_free_votes = (free_votes >> warp_lane) & work_lane_mask;
    const NodeStatus my_unknown_votes = (unknown_votes >> warp_lane) & work_lane_mask;
    const NodeStatus my_occupied_votes = (occupied_votes >> warp_lane) & work_lane_mask;
    const NodeStatus my_static_votes = (static_votes >> warp_lane) & work_lane_mask;
    const NodeStatus my_dynamic_votes = (dynamic_votes >> warp_lane) & work_lane_mask;

    NodeStatus my_status_or = ((my_free_votes > 0) << log_free) | ((my_unknown_votes > 0) << log_unknown)
        | ((my_occupied_votes > 0) << log_occupied) | ((my_static_votes > 0) << log_static)
        | ((my_dynamic_votes > 0) << log_dynamic);

    // my_static_votes and my_dynamic_votes not needed for and-operation
    NodeStatus my_status_and = ((my_free_votes == work_lane_mask) << log_free)
        | ((my_unknown_votes == work_lane_mask) << log_unknown)
        | ((my_occupied_votes == work_lane_mask) << log_occupied);

    *new_status = getNewStatus(my_status_or, my_status_and, level);
  }

//  const int mem_index = threadIdx.x / 8;
//  if (do_final_computation)
//  {
//    shared_mem[mem_index] = 0xFF;
//  }
//  atomicAnd(&shared_mem[mem_index], status);
//  const NodeStatus my_status_and = NodeStatus(shared_mem[mem_index]);
//  if (do_final_computation)
//  {
//    shared_mem[mem_index] = 0;
//  }
//  atomicOr(&shared_mem[mem_index], status);
//  const NodeStatus my_status_or = NodeStatus(shared_mem[mem_index]);
//
//  if (do_final_computation)
//  {
//    *new_status = getNewStatus(my_status_or, my_status_and, level);
//  }
}

struct unary_zero: public thrust::unary_function<uint32_t, bool>
{
  __host__ __device__
  bool operator()(const uint32_t& x)
  {
    return x == 0;
  }
};

template<typename T>
struct isZero: public thrust::unary_function<T, T>
{
  __host__ __device__ T operator()(const T &x) const
  {
    return (x == 0);
  }
};
// end negate


__device__ __forceinline__
void getNextCoordinates(const gpu_voxels::Vector3ui coordinate, uint32_t child, uint8_t level,
                        gpu_voxels::Vector3ui& new_coordinate_min, gpu_voxels::Vector3ui& new_coordinate_max)
{
  const uint32_t coordinate_step = uint32_t(const_cube_side_length[level]);
  gpu_voxels::Vector3ui shift;
  shift.x = (child & 0x01) * coordinate_step;
  shift.y = ((child & 0x02) >> 1) * coordinate_step;
  shift.z = ((child & 0x04) >> 2) * coordinate_step;

  new_coordinate_max = coordinate + gpu_voxels::Vector3ui(coordinate_step) + shift;
  new_coordinate_min = coordinate + shift;
}

}
}

#endif
