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
 * \date    2014-08-08
 *
 */
//----------------------------------------------------------------------/*
#ifndef GPU_VOXELS_OCTREE_LOAD_BALANCER_EXTRACT_CUH_INCLUDED
#define GPU_VOXELS_OCTREE_LOAD_BALANCER_EXTRACT_CUH_INCLUDED

#include <gpu_voxels/octree/load_balancer/Extract.h>
#include <gpu_voxels/octree/load_balancer/AbstractLoadBalancer.cuh>
#include <gpu_voxels/octree/load_balancer/kernel_config/Extract.cuh>

namespace gpu_voxels {
namespace NTree {
namespace LoadBalancer {

template<std::size_t branching_factor, std::size_t level_count, class InnerNode, class LeafNode, bool clear_collision_flag, bool count_mode>
Extract<branching_factor, level_count, InnerNode, LeafNode, clear_collision_flag, count_mode>::Extract(
        NTree<branching_factor, level_count, InnerNode, LeafNode>* ntree,
        NodeData* dev_node_data,
        const uint32_t node_data_size,
        uint8_t* dev_status_selection,
        const uint32_t min_level) :
    Base(),
    m_ntree(ntree),
    m_dev_node_data(dev_node_data),
    m_node_data_size(node_data_size),
    m_dev_status_selection(dev_status_selection),
    m_min_level(min_level),
    m_num_elements(0),
    m_dev_global_voxel_list_count(NULL)
{

}

template<std::size_t branching_factor, std::size_t level_count, class InnerNode, class LeafNode, bool clear_collision_flag, bool count_mode>
Extract<branching_factor, level_count, InnerNode, LeafNode, clear_collision_flag, count_mode>::~Extract()
{

}

// ---------- Abstract method implementation --------------
template<std::size_t branching_factor, std::size_t level_count, class InnerNode, class LeafNode, bool clear_collision_flag, bool count_mode>
void Extract<branching_factor, level_count, InnerNode, LeafNode, clear_collision_flag, count_mode>::doWork()
{
  // Passing of template types needed by the kernel function through the helper struct KernelConfig
  typedef ExtractNTreeKernelConfig<
      RunConfig::NUM_TRAVERSAL_THREADS,
      branching_factor,
      InnerNode,
      LeafNode,
      NodeData,
      clear_collision_flag,
      count_mode> KernelConfig;

  // Parameters for the kernel function required for the load balancing concept.
  const typename KernelConfig::Base::AbstractKernelParameters abstract_parameters(
      Base::m_dev_work_stacks1,
      Base::m_dev_work_stacks1_item_count,
      Base::STACK_SIZE_PER_TASK,
      Base::m_dev_tasks_idle_count,
      Base::getIdleCountThreshold());

  // Specific parameters for the kernel function.
  typename KernelConfig::KernelParams kernel_params(
    abstract_parameters,
    m_min_level,
    m_dev_node_data,
    m_node_data_size,
    m_dev_global_voxel_list_count);

  // Call the templated kernel function. It's behavior is defined by the given KernelConfig.
  size_t dynamic_shared_mem_size = sizeof(typename KernelConfig::SharedMem) + sizeof(typename KernelConfig::SharedVolatileMem);
  kernelLBWorkConcept<KernelConfig><<<Base::NUM_TASKS, RunConfig::NUM_TRAVERSAL_THREADS, dynamic_shared_mem_size>>>(kernel_params);
  HANDLE_CUDA_ERROR(cudaDeviceSynchronize());
}

template<std::size_t branching_factor, std::size_t level_count, class InnerNode, class LeafNode, bool clear_collision_flag, bool count_mode>
void Extract<branching_factor, level_count, InnerNode, LeafNode, clear_collision_flag, count_mode>::doPostCalculations()
{
    // Copy results from device to host
    HANDLE_CUDA_ERROR(
        cudaMemcpy(&m_num_elements, m_dev_global_voxel_list_count, sizeof(uint32_t), cudaMemcpyDeviceToHost));
}
// --------------------------------------------------------

template<std::size_t branching_factor, std::size_t level_count, class InnerNode, class LeafNode, bool clear_collision_flag, bool count_mode>
bool Extract<branching_factor, level_count, InnerNode, LeafNode, clear_collision_flag, count_mode>::doPreparations()
{
  bool ret = Base::doPreparations();

  // Push first work item on stack
  InnerNode a;
  HANDLE_CUDA_ERROR(cudaMemcpy(&a, m_ntree->m_root, sizeof(InnerNode), cudaMemcpyDeviceToHost));
  if (!a.hasStatus(ns_PART))
    return false;
  Base::m_init_work_item = WorkItem((InnerNode*) a.getChildPtr(), 0, level_count - 2);

  // ##### init status selection lookup table ####
  if (m_dev_status_selection == NULL)
  {
    // extract all
    uint8_t selection[extract_selection_size];
    memset(selection, 1, extract_selection_size * sizeof(uint8_t));
    HANDLE_CUDA_ERROR(
        cudaMemcpyToSymbol(const_extract_selection, selection, extract_selection_size * sizeof(uint8_t), 0,
                           cudaMemcpyHostToDevice));
  }
  else
  {
    // copy selection lookup table to constant memory
    HANDLE_CUDA_ERROR(
        cudaMemcpyToSymbol(const_extract_selection, m_dev_status_selection,
                           extract_selection_size * sizeof(uint8_t), 0, cudaMemcpyDeviceToDevice));
  }
  HANDLE_CUDA_ERROR(cudaDeviceSynchronize());
  // ##############################################

  // Alloc device mem
  HANDLE_CUDA_ERROR(cudaMalloc(&m_dev_global_voxel_list_count, sizeof(uint32_t)));

  // Init mem
  HANDLE_CUDA_ERROR(cudaMemset(m_dev_global_voxel_list_count, 0, sizeof(uint32_t)));

  return ret;
}

template<std::size_t branching_factor, std::size_t level_count, class InnerNode, class LeafNode, bool clear_collision_flag, bool count_mode>
void Extract<branching_factor, level_count, InnerNode, LeafNode, clear_collision_flag, count_mode>::doCleanup()
{
  Base::doCleanup();

  // Free allocated device mem
  if(m_dev_global_voxel_list_count)
  {
    HANDLE_CUDA_ERROR(cudaFree(m_dev_global_voxel_list_count));
    m_dev_global_voxel_list_count = NULL;
  }
}

}
}
}

#endif
