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
#ifndef GPU_VOXELS_OCTREE_LOAD_BALANCER_ABSTRACT_LOAD_BALANCER_CUH_INCLUDED
#define GPU_VOXELS_OCTREE_LOAD_BALANCER_ABSTRACT_LOAD_BALANCER_CUH_INCLUDED

#include <gpu_voxels/octree/load_balancer/AbstractLoadBalancer.h>
#include <gpu_voxels/octree/load_balancer/kernel_config/LoadBalance.cuh>
#include <gpu_voxels/helpers/cuda_handling.h>

namespace gpu_voxels {
namespace NTree {
namespace LoadBalancer {

template<std::size_t branching_factor, std::size_t level_count, class InnerNode, class LeafNode,
    class WorkItem, typename RunConfig>
AbstractLoadBalancer<branching_factor, level_count, InnerNode, LeafNode, WorkItem, RunConfig>::AbstractLoadBalancer(
    const float idle_threshold, const uint32_t num_tasks) :
    IDLE_THESHOLD(idle_threshold),
    NUM_TASKS(num_tasks),
    m_dev_work_stacks1(NULL),
    m_dev_work_stacks2(NULL),
    m_dev_work_stacks1_item_count(NULL),
    m_init_work_item(),
    m_dev_tasks_idle_count(NULL)
{
}

template<std::size_t branching_factor, std::size_t level_count, class InnerNode, class LeafNode,
    class WorkItem, typename RunConfig>
AbstractLoadBalancer<branching_factor, level_count, InnerNode, LeafNode, WorkItem, RunConfig>::~AbstractLoadBalancer()
{
  doCleanup();
}

template<std::size_t branching_factor, std::size_t level_count, class InnerNode, class LeafNode,
    class WorkItem, typename RunConfig>
void AbstractLoadBalancer<branching_factor, level_count, InnerNode, LeafNode, WorkItem, RunConfig>::run()
{
  bool preparations_success = doPreparations();
  if(preparations_success)
  {
      HANDLE_CUDA_ERROR(cudaMemcpy(&m_dev_work_stacks1[0], &m_init_work_item, sizeof(WorkItem), cudaMemcpyHostToDevice));
      uint32_t initial_stack_count = 1;
      HANDLE_CUDA_ERROR(cudaMemcpy(&m_dev_work_stacks1_item_count[0], &initial_stack_count, sizeof(uint32_t), cudaMemcpyHostToDevice));

      HANDLE_CUDA_ERROR(cudaDeviceSynchronize());
      std::size_t total_work_items = 1;
      std::size_t num_balance_tasks = 0;
      std::size_t num_work_tasks = 0;
      while (total_work_items > 0)
      {
        if (!HANDLE_CUDA_ERROR(cudaMemset(m_dev_tasks_idle_count, 0, sizeof(uint32_t)))) break; // avoid error spam
        doWork();
        if (!HANDLE_CUDA_ERROR(cudaDeviceSynchronize())) break; // avoid error spam

        ++num_work_tasks;

        uint32_t idle_count;
        if (!HANDLE_CUDA_ERROR(cudaMemcpy(&idle_count, m_dev_tasks_idle_count, sizeof(uint32_t), cudaMemcpyDeviceToHost)))
        {
          break; // avoid error spam
        }

        // Perform load balancing step if necessary
        if (idle_count >= (NUM_TASKS * IDLE_THESHOLD))
        {
          total_work_items = doBalance();
          ++num_balance_tasks;

          // Swap stack pointers due to load balance from stack1 to stack2
          WorkItem* tmp = m_dev_work_stacks1;
          m_dev_work_stacks1 = m_dev_work_stacks2;
          m_dev_work_stacks2 = tmp;
        }
      }
      doPostCalculations();
  }
  doCleanup();
}

template<std::size_t branching_factor, std::size_t level_count, class InnerNode, class LeafNode,
    class WorkItem, typename RunConfig>
bool AbstractLoadBalancer<branching_factor, level_count, InnerNode, LeafNode, WorkItem, RunConfig>::doPreparations()
{
  // Allocates the work stacks
  HANDLE_CUDA_ERROR(cudaMalloc(&m_dev_work_stacks1, sizeof(WorkItem) * NUM_TASKS * STACK_SIZE_PER_TASK));
  HANDLE_CUDA_ERROR(cudaMalloc(&m_dev_work_stacks2, sizeof(WorkItem) * NUM_TASKS * STACK_SIZE_PER_TASK));
  HANDLE_CUDA_ERROR(cudaMalloc(&m_dev_work_stacks1_item_count, sizeof(uint32_t) * NUM_TASKS));

  // Some other allocations
  HANDLE_CUDA_ERROR(cudaMalloc(&m_dev_tasks_idle_count, sizeof(uint32_t)));

  // Init memory
  HANDLE_CUDA_ERROR(cudaMemset(m_dev_work_stacks1_item_count, 0, sizeof(uint32_t) * NUM_TASKS));
  HANDLE_CUDA_ERROR(cudaMemset(m_dev_tasks_idle_count, 0, sizeof(uint32_t)));

  return true;
}

template<std::size_t branching_factor, std::size_t level_count, class InnerNode, class LeafNode,
    class WorkItem, typename RunConfig>
std::size_t AbstractLoadBalancer<branching_factor, level_count, InnerNode, LeafNode, WorkItem, RunConfig>::doBalance()
{
  uint32_t num_total_work_items = 0;

  balanceWorkStacks<RunConfig::NUM_BALANCE_THREADS, WorkItem, branching_factor, level_count>(
      m_dev_work_stacks1,
      m_dev_work_stacks2,
      m_dev_work_stacks1_item_count,
      NUM_TASKS,
      &num_total_work_items,
      STACK_SIZE_PER_TASK);

  return num_total_work_items;
}

template<std::size_t branching_factor, std::size_t level_count, class InnerNode, class LeafNode,
    class WorkItem, typename RunConfig>
void AbstractLoadBalancer<branching_factor, level_count, InnerNode, LeafNode, WorkItem, RunConfig>::doCleanup()
{
  // Free allocated the work stacks
  if (m_dev_work_stacks1)
  {
    HANDLE_CUDA_ERROR(cudaFree(m_dev_work_stacks1));
    m_dev_work_stacks1 = NULL;
  }
  if (m_dev_work_stacks2)
  {
    HANDLE_CUDA_ERROR(cudaFree(m_dev_work_stacks2));
    m_dev_work_stacks2 = NULL;
  }
  if (m_dev_work_stacks1_item_count)
  {
    HANDLE_CUDA_ERROR(cudaFree(m_dev_work_stacks1_item_count));
    m_dev_work_stacks1_item_count = NULL;
  }

  // Some other frees
  if (m_dev_tasks_idle_count)
  {
    HANDLE_CUDA_ERROR(cudaFree(m_dev_tasks_idle_count));
    m_dev_tasks_idle_count = NULL;
  }
}

template<std::size_t branching_factor, std::size_t level_count, class InnerNode, class LeafNode,
class WorkItem, typename RunConfig>
uint32_t AbstractLoadBalancer<branching_factor, level_count, InnerNode, LeafNode, WorkItem, RunConfig>::getIdleCountThreshold()
{
  return max(1, uint32_t(IDLE_THESHOLD * NUM_TASKS));
}

}
}
}

#endif
