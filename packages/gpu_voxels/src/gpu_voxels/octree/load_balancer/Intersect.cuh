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
 * \date    2014-07-23
 *
 */
//----------------------------------------------------------------------/*
#ifndef GPU_VOXELS_OCTREE_LOAD_BALANCER_INTERSECT_CUH_INCLUDED
#define GPU_VOXELS_OCTREE_LOAD_BALANCER_INTERSECT_CUH_INCLUDED

#include <gpu_voxels/octree/load_balancer/Intersect.h>
#include <gpu_voxels/octree/load_balancer/AbstractLoadBalancer.cuh>
#include <gpu_voxels/octree/load_balancer/kernel_config/Intersect.cuh>

namespace gpu_voxels {
namespace NTree {
namespace LoadBalancer {

template<std::size_t branching_factor, std::size_t level_count, class InnerNode, class LeafNode,
      class InnerNode2, class LeafNode2, class Collider, bool mark_collisions>
Intersect<branching_factor, level_count, InnerNode, LeafNode, InnerNode2, LeafNode2, Collider, mark_collisions>::Intersect(
    NTree<branching_factor, level_count, InnerNode, LeafNode>* ntree_a,
    NTree<branching_factor, level_count, InnerNode2, LeafNode2>* ntree_b,
    const uint32_t min_level, Collider collider)
    :
    Base(),
    m_dev_num_collisions(NULL),
    m_num_collisions(SSIZE_MAX),
    m_collider(collider),
    m_min_level(min_level),
    m_ntree_a(ntree_a),
    m_ntree_b(ntree_b)
{

}

template<std::size_t branching_factor, std::size_t level_count, class InnerNode, class LeafNode,
      class InnerNode2, class LeafNode2, class Collider, bool mark_collisions>
Intersect<branching_factor, level_count, InnerNode, LeafNode, InnerNode2, LeafNode2, Collider, mark_collisions>::~Intersect()
{

}

template<std::size_t branching_factor, std::size_t level_count, class InnerNode, class LeafNode,
      class InnerNode2, class LeafNode2, class Collider, bool mark_collisions>
bool Intersect<branching_factor, level_count, InnerNode, LeafNode, InnerNode2, LeafNode2, Collider, mark_collisions>::doPreparations()
{
  bool ret = Base::doPreparations();

  // Alloc device mem
  HANDLE_CUDA_ERROR(cudaMalloc(&m_dev_num_collisions, sizeof(std::size_t)));

  // Push first work item on stack
  InnerNode a;
  InnerNode2 b;
  HANDLE_CUDA_ERROR(cudaMemcpy(&a, m_ntree_a->m_root, sizeof(InnerNode), cudaMemcpyDeviceToHost));
  HANDLE_CUDA_ERROR(cudaMemcpy(&b, m_ntree_b->m_root, sizeof(InnerNode2), cudaMemcpyDeviceToHost));
  Base::m_init_work_item = WorkItem((InnerNode*) a.getChildPtr(), (InnerNode2*) b.getChildPtr(), level_count - 2, true, true);

  // Init mem
  HANDLE_CUDA_ERROR(cudaMemset(m_dev_num_collisions, 0, sizeof(std::size_t)));

  return ret;
}

// ---------- Abstract method implementation --------------
template<std::size_t branching_factor, std::size_t level_count, class InnerNode, class LeafNode,
      class InnerNode2, class LeafNode2, class Collider, bool mark_collisions>
void Intersect<branching_factor, level_count, InnerNode, LeafNode, InnerNode2, LeafNode2, Collider, mark_collisions>::doWork()
{
  // Passing of template types needed by the kernel function through the helper struct KernelConfig
  typedef IntersectNTreeKernelConfig<
      RunConfig::NUM_TRAVERSAL_THREADS,
      branching_factor,
      InnerNode,
      LeafNode,
      InnerNode2,
      LeafNode2,
      mark_collisions,
      Collider> KernelConfig;

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
    m_dev_num_collisions,
    m_collider);

  // Call the templated kernel function. It's behavior is defined by the given KernelConfig.
  size_t dynamic_shared_mem_size = sizeof(typename KernelConfig::SharedMem) + sizeof(typename KernelConfig::SharedVolatileMem);
  kernelLBWorkConcept<KernelConfig><<<Base::NUM_TASKS, RunConfig::NUM_TRAVERSAL_THREADS, dynamic_shared_mem_size>>>(kernel_params);
  HANDLE_CUDA_ERROR(cudaDeviceSynchronize());
}

template<std::size_t branching_factor, std::size_t level_count, class InnerNode, class LeafNode,
      class InnerNode2, class LeafNode2, class Collider, bool mark_collisions>
void Intersect<branching_factor, level_count, InnerNode, LeafNode, InnerNode2, LeafNode2, Collider, mark_collisions>::doPostCalculations()
{
  // Copy results from device to host
  HANDLE_CUDA_ERROR(
      cudaMemcpy(&m_num_collisions, m_dev_num_collisions, sizeof(std::size_t), cudaMemcpyDeviceToHost));
}
// --------------------------------------------------------

template<std::size_t branching_factor, std::size_t level_count, class InnerNode, class LeafNode,
      class InnerNode2, class LeafNode2, class Collider, bool mark_collisions>
void Intersect<branching_factor, level_count, InnerNode, LeafNode, InnerNode2, LeafNode2, Collider, mark_collisions>::doCleanup()
{
  Base::doCleanup();

  // Free allocated device mem
  if (m_dev_num_collisions)
  {
    HANDLE_CUDA_ERROR(cudaFree(m_dev_num_collisions));
    m_dev_num_collisions = NULL;
  }
}

}
}
}

#endif
