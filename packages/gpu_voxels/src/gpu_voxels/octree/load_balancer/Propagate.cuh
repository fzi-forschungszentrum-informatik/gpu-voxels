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
 * \date    2014-08-31
 *
 */
//----------------------------------------------------------------------/*
#ifndef GPU_VOXELS_OCTREE_LOAD_BALANCER_PROPAGATE_CUH_INCLUDED
#define GPU_VOXELS_OCTREE_LOAD_BALANCER_PROPAGATE_CUH_INCLUDED

#include <gpu_voxels/octree/load_balancer/Propagate.h>
#include <gpu_voxels/octree/load_balancer/AbstractLoadBalancer.cuh>
#include <gpu_voxels/octree/load_balancer/kernel_config/Propagate.cuh>

namespace gpu_voxels {
namespace NTree {
namespace LoadBalancer {

template<std::size_t branching_factor, std::size_t level_count, class InnerNode, class LeafNode>
Propagate<branching_factor, level_count, InnerNode, LeafNode>::Propagate(
        NTree<branching_factor, level_count, InnerNode, LeafNode>* ntree,
        const uint32_t num_tasks) :
    Base(Base::DEFAULT_IDLE_THESHOLD, num_tasks),
    m_ntree(ntree)
{

}

template<std::size_t branching_factor, std::size_t level_count, class InnerNode, class LeafNode>
Propagate<branching_factor, level_count, InnerNode, LeafNode>::~Propagate()
{

}

template<std::size_t branching_factor, std::size_t level_count, class InnerNode, class LeafNode>
bool Propagate<branching_factor, level_count, InnerNode, LeafNode>::doPreparations()
{
  bool ret = Base::doPreparations();

  // Alloc device mem
  // Push first work item on stack
  InnerNode a;
  HANDLE_CUDA_ERROR(cudaMemcpy(&a, m_ntree->m_root, sizeof(InnerNode), cudaMemcpyDeviceToHost));
  if (!a.hasStatus(ns_PART)) // abort if there is only the root node
    return false;
  Base::m_init_work_item = WorkItem((InnerNode*) a.getChildPtr(), m_ntree->m_root, true, level_count - 2, false);

  return ret;
}

// ---------- Abstract method implementation --------------
template<std::size_t branching_factor, std::size_t level_count, class InnerNode, class LeafNode>
void Propagate<branching_factor, level_count, InnerNode, LeafNode>::doWork()
{
  // Passing of template types needed by the kernel function through the helper struct KernelConfig
  typedef PropagateKernelConfig<RunConfig::NUM_TRAVERSAL_THREADS,
      branching_factor,
      InnerNode,
      LeafNode,
      level_count> KernelConfig;

  // Parameters for the kernel function required for the load balancing concept.
  const typename KernelConfig::AbstractKernelParameters abstract_parameters(
        Base::m_dev_work_stacks1,
        Base::m_dev_work_stacks1_item_count,
        Base::STACK_SIZE_PER_TASK,
        Base::m_dev_tasks_idle_count,
        Base::getIdleCountThreshold());

  // Specific parameters for the kernel function.
  typename KernelConfig::KernelParams kernel_params(abstract_parameters);

  // Call the templated kernel function. It's behavior is defined by the given KernelConfig.
  size_t dynamic_shared_mem_size = sizeof(typename KernelConfig::SharedMem) + sizeof(typename KernelConfig::SharedVolatileMem);
  kernelLBWorkConcept<KernelConfig><<<Base::NUM_TASKS, RunConfig::NUM_TRAVERSAL_THREADS, dynamic_shared_mem_size>>>(kernel_params);
  HANDLE_CUDA_ERROR(cudaDeviceSynchronize());
}

template<std::size_t branching_factor, std::size_t level_count, class InnerNode, class LeafNode>
void Propagate<branching_factor, level_count, InnerNode, LeafNode>::doPostCalculations()
{
  // Nothing to do
}
// --------------------------------------------------------

template<std::size_t branching_factor, std::size_t level_count, class InnerNode, class LeafNode>
void Propagate<branching_factor, level_count, InnerNode, LeafNode>::doCleanup()
{
  Base::doCleanup();
}

}
}
}

#endif
