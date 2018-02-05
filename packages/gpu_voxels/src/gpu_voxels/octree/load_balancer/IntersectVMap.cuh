
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
 * \date    2014-08-28
 *
 */
//----------------------------------------------------------------------/*
#ifndef GPU_VOXELS_OCTREE_LOAD_BALANCER_INTERSECT_VMAP_CUH_INCLUDED
#define GPU_VOXELS_OCTREE_LOAD_BALANCER_INTERSECT_VMAP_CUH_INCLUDED

#include <gpu_voxels/octree/load_balancer/IntersectVMap.h>
#include <gpu_voxels/octree/load_balancer/AbstractLoadBalancer.cuh>
#include <gpu_voxels/octree/load_balancer/kernel_config/IntersectVMap.cuh>

namespace gpu_voxels {
namespace NTree {
namespace LoadBalancer {

template<std::size_t branching_factor, std::size_t level_count, class InnerNode, class LeafNode,
     int vft_size, bool set_collision_flag, bool compute_voxelTypeFlags, class VoxelType>
IntersectVMap<branching_factor, level_count, InnerNode, LeafNode, vft_size, set_collision_flag, compute_voxelTypeFlags, VoxelType>::IntersectVMap(
        NTree<branching_factor, level_count, InnerNode, LeafNode>* ntree,
        const VoxelType* voxel_map,
        const gpu_voxels::Vector3ui voxel_map_dim,
        const gpu_voxels::Vector3i offset,
        const uint32_t min_level)
    :
    Base(),
    m_dev_num_collisions(NULL),
    m_dev_result_voxelTypeFlags(NULL),
    m_ntree(ntree),
    m_voxel_map(voxel_map),
    m_voxel_map_dim(voxel_map_dim),
    m_offset(offset),
    m_min_level(min_level),
    m_num_collisions(0),
    m_result_voxelTypeFlags()
{

}

template<std::size_t branching_factor, std::size_t level_count, class InnerNode, class LeafNode,
     int vft_size, bool set_collision_flag, bool compute_voxelTypeFlags, class VoxelType>
IntersectVMap<branching_factor, level_count, InnerNode, LeafNode, vft_size, set_collision_flag, compute_voxelTypeFlags, VoxelType>::~IntersectVMap()
{

}

template<std::size_t branching_factor, std::size_t level_count, class InnerNode, class LeafNode,
     int vft_size, bool set_collision_flag, bool compute_voxelTypeFlags, class VoxelType>
bool IntersectVMap<branching_factor, level_count, InnerNode, LeafNode, vft_size, set_collision_flag, compute_voxelTypeFlags, VoxelType>::doPreparations()
{
  bool ret = Base::doPreparations();

  // Alloc device mem
  HANDLE_CUDA_ERROR(cudaMalloc(&m_dev_num_collisions, sizeof(std::size_t)));

  // Push first work item on stack
  InnerNode a;
  HANDLE_CUDA_ERROR(cudaMemcpy(&a, m_ntree->m_root, sizeof(InnerNode), cudaMemcpyDeviceToHost));
  Base::m_init_work_item = WorkItem((InnerNode*) a.getChildPtr(), gpu_voxels::Vector3ui(0), level_count - 2, true, true);

  // Alloc and clear mem for resulting vector of voxel types in collision
  if (compute_voxelTypeFlags)
  {
    m_result_voxelTypeFlags->clear();
    HANDLE_CUDA_ERROR(cudaMalloc((void ** ) &m_dev_result_voxelTypeFlags, sizeof(BitVector<vft_size>)));
    HANDLE_CUDA_ERROR(
        cudaMemcpy(m_dev_result_voxelTypeFlags, m_result_voxelTypeFlags, sizeof(BitVector<vft_size>),
                   cudaMemcpyHostToDevice));
  }

  // Init mem
  HANDLE_CUDA_ERROR(cudaMemset(m_dev_num_collisions, 0, sizeof(std::size_t)));

  return ret;
}

// ---------- Abstract method implementation --------------
template<std::size_t branching_factor, std::size_t level_count, class InnerNode, class LeafNode,
     int vft_size, bool set_collision_flag, bool compute_voxelTypeFlags, class VoxelType>
void IntersectVMap<branching_factor, level_count, InnerNode, LeafNode, vft_size, set_collision_flag, compute_voxelTypeFlags, VoxelType>::doWork()
{
  // Passing of template types needed by the kernel function through the helper struct KernelConfig
  typedef IntersectVMapKernelConfig<
            RunConfig::NUM_TRAVERSAL_THREADS,
            branching_factor,
            InnerNode,
            LeafNode,
            vft_size,
            set_collision_flag,
            compute_voxelTypeFlags,
            VoxelType> KernelConfig;

  // Parameters for the kernel function required for the load balancing concept.
  const typename KernelConfig::AbstractKernelParameters abstract_parameters(
      Base::m_dev_work_stacks1,
      Base::m_dev_work_stacks1_item_count,
      Base::STACK_SIZE_PER_TASK,
      Base::m_dev_tasks_idle_count,
      Base::getIdleCountThreshold());

  // Specific parameters for the kernel function.
  typename KernelConfig::KernelParams kernel_params(abstract_parameters,
    m_dev_num_collisions,
    m_offset,
    m_voxel_map,
    m_voxel_map_dim,
    m_min_level,
    m_dev_result_voxelTypeFlags);


  // Call the templated kernel function. It's behavior is defined by the given KernelConfig.
  size_t dynamic_shared_mem_size = sizeof(typename KernelConfig::SharedMem) + sizeof(typename KernelConfig::SharedVolatileMem);
  kernelLBWorkConcept<KernelConfig><<<Base::NUM_TASKS, RunConfig::NUM_TRAVERSAL_THREADS, dynamic_shared_mem_size>>>(kernel_params);
  HANDLE_CUDA_ERROR(cudaDeviceSynchronize());
}

template<std::size_t branching_factor, std::size_t level_count, class InnerNode, class LeafNode,
     int vft_size, bool set_collision_flag, bool compute_voxelTypeFlags, class VoxelType>
void IntersectVMap<branching_factor, level_count, InnerNode, LeafNode, vft_size, set_collision_flag, compute_voxelTypeFlags, VoxelType>::doPostCalculations()
{
  // Copy results from device to host
  HANDLE_CUDA_ERROR(
      cudaMemcpy(&m_num_collisions, m_dev_num_collisions, sizeof(std::size_t), cudaMemcpyDeviceToHost));
  HANDLE_CUDA_ERROR(
      cudaMemcpy(&m_result_voxelTypeFlags, m_dev_result_voxelTypeFlags, sizeof(BitVector<vft_size>), cudaMemcpyDeviceToHost));
}
// --------------------------------------------------------

template<std::size_t branching_factor, std::size_t level_count, class InnerNode, class LeafNode,
     int vft_size, bool set_collision_flag, bool compute_voxelTypeFlags, class VoxelType>
void IntersectVMap<branching_factor, level_count, InnerNode, LeafNode, vft_size, set_collision_flag, compute_voxelTypeFlags, VoxelType>::doCleanup()
{
  Base::doCleanup();

  // Free allocated device mem
  if (m_dev_num_collisions)
  {
    HANDLE_CUDA_ERROR(cudaFree(m_dev_num_collisions));
    m_dev_num_collisions = NULL;
  }

  if (m_dev_result_voxelTypeFlags)
  {
    HANDLE_CUDA_ERROR(cudaFree(m_dev_result_voxelTypeFlags));
    m_dev_result_voxelTypeFlags = NULL;
  }
}

}
}
}

#endif
