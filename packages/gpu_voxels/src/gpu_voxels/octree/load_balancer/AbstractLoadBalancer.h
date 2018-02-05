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
#ifndef GPU_VOXELS_OCTREE_LOAD_BALANCER_ABSTRACT_LOAD_BALANCER_H_INCLUDED
#define GPU_VOXELS_OCTREE_LOAD_BALANCER_ABSTRACT_LOAD_BALANCER_H_INCLUDED

#include <gpu_voxels/octree/NTree.h>

namespace gpu_voxels {
namespace NTree {
namespace LoadBalancer {

template<int _num_threads>
struct RunConfig
{
public:
  static const int NUM_BALANCE_THREADS = _num_threads;
  static const int NUM_TRAVERSAL_THREADS = _num_threads;
};

/**
 * @brief This abstract template class manages the workflow of the load balancing concept.
 * The general workflow consists of:
 * 1. Preparations like memory allocations (\code doPreparations() \endcode)
 * 2. Do load balanced kernel calls till work is done (\code doPreparations() \endcode)
 * 3. Clean-up like memory frees and collecting of results (\code doCleanup() doPostCalculations() \endcode)
 *
 * @tparam branching_factor Branching factor of the corresponding \code NTree \endcode
 * @tparam level_count Number of levels of the corresponding \code NTree \endcode
 * @tparam InnerNode Inner node type of the corresponding \code NTree \endcode
 * @tparam LeafNode Leaf node type of the corresponding \code NTree \endcode
 * @tparam WorkItem Work item type to use for the stack items
 * @tparam RunConfig Run configuration which defines the number of threads to use.
 */
template<std::size_t branching_factor, std::size_t level_count, class InnerNode, class LeafNode,
    class WorkItem, typename RunConfig>
class AbstractLoadBalancer
{
protected:
#if __cplusplus > 199711L
// initializing "static const float" is not in C++98. C++11 offers "static constexpr float"
  static constexpr float DEFAULT_IDLE_THESHOLD = 2.0f/3.0f;
#else
  static const float DEFAULT_IDLE_THESHOLD = 2.0f/3.0f;
#endif
public:
  /**
   * @brief AbstractLoadBalancer constructor. Default parameters choosen by resulting in good performance of experimental evaluations for GTX Titan.
   * @param idle_threshold Defines when to do a load balancing step by the percentage of idle tasks.
   * @param num_tasks Number of tasks to use.
   */
  AbstractLoadBalancer(const float idle_threshold = DEFAULT_IDLE_THESHOLD, const uint32_t num_tasks = 2688);

  virtual ~AbstractLoadBalancer();

  /**
   * @brief Run the load balancer and do the work.
   */
  virtual void run();

  /**
   * @brief getIdleCountThreshold calculated from \code IDLE_THESHOLD * NUM_TASKS \endcode
   * @return Returns the maximum allowed number of idle tasks for a load balancing step. Minimum is 1.
   */
  uint32_t getIdleCountThreshold();

protected:
  /**
   * @brief Preparations like malloc etc. necessary before doing the load balanced work.
   * @return Returns true on success, false otherwise.
   */
  virtual bool doPreparations();

  /**
   * @brief Do parallel work till there is a load imbalance.
   */
  virtual void doWork() = 0;

  /**
   * @brief Balance the work.
   * @return Returns the total number of work items in all stacks.
   */
  virtual std::size_t doBalance();

  /**
   * @brief Do some final calculations before clean-up after all load balancing work is done.
   */
  virtual void doPostCalculations() = 0;

  /**
   * @brief Clean-up allocated memory of \code doPreparations() \endcode
   */
  virtual void doCleanup();

protected:
  /**
   * @brief Number of stack items per task. Some reasonable value.
   */
  static const std::size_t STACK_SIZE_PER_TASK = 700;

  // -------------------- Host -----------------------------
  /**
   * @brief Threshold of idle tasks when the load balance should occur.
   */
  const float IDLE_THESHOLD;

  /**
   * @brief Number of tasks to use.
   */
  const uint32_t NUM_TASKS;

  /**
   * @brief First work item to start the work with.
   */
  WorkItem m_init_work_item;

  // -------------------- Device ----------------------------
  /**
   * @brief Two arrays of stacks for swapping the work items for load balancing.
   */
  WorkItem* m_dev_work_stacks1;
  WorkItem* m_dev_work_stacks2;

  /**
   * @brief Number of work items in each stack.
   */
  uint32_t* m_dev_work_stacks1_item_count;

  /**
   * @brief Number of tasks that idle.
   */
  uint32_t* m_dev_tasks_idle_count;

};

}
}
}

#endif
