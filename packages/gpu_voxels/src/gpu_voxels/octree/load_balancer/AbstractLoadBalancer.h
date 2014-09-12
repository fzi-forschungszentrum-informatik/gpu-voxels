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
 * Abstract template class for all load balancing problems of the NTree. These are traversal, intersection, data extraction etc.
 */
template<std::size_t branching_factor, std::size_t level_count, class InnerNode, class LeafNode,
    class WorkItem, typename RunConfig>
class AbstractLoadBalancer
{
protected:
  static const float DEFAULT_IDLE_THESHOLD = 2.0f/3.0f;
public:
  AbstractLoadBalancer(const float idle_threshold = DEFAULT_IDLE_THESHOLD, const uint32_t num_tasks = 2688);

  virtual ~AbstractLoadBalancer();

  /**
   * Run the load balancer and do the work.
   */
  virtual void run();

  uint32_t getIdleCountThreshold();

protected:
  /**
   * Preparations like malloc etc. necessary before doing the load balanced work.
   */
  virtual bool doPreparations();

  /**
   * Do parallel work till there is a load imbalance.
   */
  virtual void doWork() = 0;

  /**
   * Balance the work.
   * @return Returns the total number of work items in all stacks.
   */
  virtual std::size_t doBalance();

  /**
   * Do some final calculations before clean-up after all load balancing work is done.
   */
  virtual void doPostCalculations() = 0;

  /**
   * Clean-up allocated memory of \code doPreparations()
   */
  virtual void doCleanup();

protected:

//  /**
//   * Good performing parameters of experimental evaluation for GTX Titan
//   */
//  static const uint32_t NUM_BALANCE_TASKS = 2688;
//  static const uint32_t NUM_BALANCE_THREADS = 128;

  /**
   * Number of stack items per task. Some reasonable value.
   */
  static const std::size_t STACK_SIZE_PER_TASK = 700;

  // -------------------- Host -----------------------------
  /**
   * Threshold of idle tasks when the load balance should occur.
   */
  const float IDLE_THESHOLD;

  const uint32_t NUM_TASKS;

  /**
   * First work item to start the work with.
   */
  WorkItem m_init_work_item;

  // -------------------- Device ----------------------------
  /**
   * Two arrays of stacks for swapping the work items for load balancing.
   */
  WorkItem* m_dev_work_stacks1;
  WorkItem* m_dev_work_stacks2;

  /**
   * Number of work items in each stack.
   */
  uint32_t* m_dev_work_stacks1_item_count;

  /**
   * Number of tasks that idle.
   */
  uint32_t* m_dev_tasks_idle_count;

};

}
}
}

#endif
