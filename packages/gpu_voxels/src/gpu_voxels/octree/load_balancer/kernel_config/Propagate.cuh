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
 * \date    2014-08-29
 *
 */
//----------------------------------------------------------------------/*
#ifndef GPU_VOXELS_OCTREE_LOAD_BALANCED_KERNEL_CONFIG_PROPAGATE_CUH_INCLUDED
#define GPU_VOXELS_OCTREE_LOAD_BALANCED_KERNEL_CONFIG_PROPAGATE_CUH_INCLUDED

#include <cuda_runtime.h>
#include <gpu_voxels/octree/DataTypes.h>
#include <gpu_voxels/octree/kernels/kernel_common.h>
#include <gpu_voxels/octree/NTreeData.h>
#include <gpu_voxels/octree/load_balancer/kernel_config/LoadBalance.cuh>

namespace gpu_voxels {
namespace NTree {
namespace LoadBalancer {

/**
 * @brief This struct defines the shared memory, variables, kernel functions etc. needed to do propagate node changes up and down the \code NTree \endcode with help of the load balancing concept.
 */
template<std::size_t num_threads,
  std::size_t branching_factor,
  typename InnerNode,
  typename LeafNode,
  std::size_t level_count>
struct PropagateKernelConfig: public AbstractKernelConfig<WorkItemPropagate<InnerNode>,
    num_threads, branching_factor>
{
public:
  typedef WorkItemPropagate<InnerNode> WorkItem;
  typedef AbstractKernelConfig<WorkItem, num_threads, branching_factor> Base;

  struct SharedMemConfig : public Base::AbstractSharedMemConfig
  {
  protected:
    static const uint32_t num_warps = num_threads / WARP_SIZE;
  public:
    uint32_t want_to_abort;
    uint32_t warp_prefix_sum[(num_warps + 1) * 2];
    uint16_t level_start_index[level_count];
    uint16_t level_end_index[level_count];
    uint32_t warp_votes[(num_warps + 1) * 2];
  };

  struct SharedVolatileMemConfig: public Base::AbstractSharedVolatileMemConfig
  {
  public:
    uint8_t bottom_up_reduction[num_threads];
  };

  struct VariablesConfig: public Base::AbstractVariablesConfig
  {
  public:
    std::size_t my_num_collisions;
    uint32_t make_progress_count;

    __host__ __device__
    VariablesConfig() :
        my_num_collisions(0),
        make_progress_count(num_threads)
    {
    }
  };

  struct ConstConfig : public Base::AbstractConstConfig
  {
  public:
      const NodeStatus status_mask;
      const NodeStatus top_down_status_mask;
      const NodeStatus bottom_up_status_mask;
      const uint32_t work_lane_mask;
      const uint32_t num_warps;

      __host__ __device__
    ConstConfig(const dim3 p_grid_dim,
                const dim3 p_block_dim,
                const dim3 p_block_ids,
                const dim3 p_thread_ids,
                const uint32_t p_stack_size_per_task) :
        Base::AbstractConstConfig(p_grid_dim,
                            p_block_dim,
                            p_block_ids,
                            p_thread_ids,
                            p_stack_size_per_task,
                            p_stack_size_per_task - num_threads),
        status_mask(ns_FREE | ns_UNKNOWN | ns_OCCUPIED),
        top_down_status_mask(~(ns_PART | ns_LAST_LEVEL)),
        bottom_up_status_mask(~(ns_PART | ns_LAST_LEVEL)),
        work_lane_mask((1 << branching_factor) - 1),
        num_warps(num_threads / WARP_SIZE)
    {

    }
  };

  typedef SharedMemConfig SharedMem;
  typedef SharedVolatileMemConfig SharedVolatileMem;
  typedef VariablesConfig Variables;
  typedef ConstConfig Constants;
  typedef typename Base::AbstractKernelParameters KernelParams;

  __device__
  static void doLoadBalancedWork(SharedMem* const shared_mem, volatile SharedVolatileMem* const shared_volatile_mem,
                                 Variables& variables, const Constants& constants, KernelParams& kernel_params)
  {
    // check the level ordering of the stack work items
#ifndef NDEBUG
    if (constants.thread_id == 0)
    {
      for (int32_t i = 0; i < (variables.num_work_items - 1); ++i)
        assert(shared_mem->work_item_cache[i].level >= shared_mem->work_item_cache[i + 1].level);
    }
#endif
    __syncthreads();

    bool insert_bottom_up_work_item = false;
    bool insert_top_down_work_item = false;
    bool lastLevel = false;
    InnerNode* node;
    bool is_top_down_mode = true;
    bool update_subtree = false;

    __threadfence(); // ensure that all threads of device see the status changes of the parent nodes

    // fetch data, read flags and propagate top-down for InnerNodes
    if (variables.is_active)
    {
      node = &shared_mem->work_item_cache[constants.work_index].node[constants.work_lane];
      assert(node != NULL);
      is_top_down_mode = shared_mem->work_item_cache[constants.work_index].is_top_down;
      update_subtree = shared_mem->work_item_cache[constants.work_index].update_subtree;

      lastLevel = node->hasStatus(ns_LAST_LEVEL);
      insert_top_down_work_item = !lastLevel & is_top_down_mode & node->hasStatus(ns_PART)
          & (node->hasFlags(nf_NEEDS_UPDATE) | update_subtree);
#ifdef PROPAGATE_BOTTOM_UP
      insert_bottom_up_work_item = is_top_down_mode & (constants.work_lane == 0) & !lastLevel;
#endif

      // marker which marks a new level of work items
      assert(shared_mem->work_item_cache[constants.work_index].level < level_count);
      const bool new_level_start = (constants.work_lane == 0)
          && (constants.work_index == 0
              || (shared_mem->work_item_cache[constants.work_index - 1].level
                  != shared_mem->work_item_cache[constants.work_index].level));
      if (new_level_start)
        shared_mem->level_start_index[shared_mem->work_item_cache[constants.work_index].level] =
            constants.thread_id;

      const bool level_end = (constants.work_lane == 0)
          && (((constants.work_index + 1) == variables.num_work_items)
              || shared_mem->work_item_cache[constants.work_index].level
                  != shared_mem->work_item_cache[constants.work_index + 1].level);
      if (level_end)
        shared_mem->level_end_index[shared_mem->work_item_cache[constants.work_index].level] = constants.thread_id
            + branching_factor;

      // handle top-down update
      if (is_top_down_mode)
      {
        // top-down update for expanding compressed nodes
        if (((node->getStatus() & constants.status_mask) == 0))
          topDownUpdate(node, shared_mem->work_item_cache[constants.work_index].parent_node,
                        constants.top_down_status_mask, constants.status_mask);
        else if (update_subtree)
          // top-down update for updating a subtree with a single value
          topDownSubtreeUpdate(node, shared_mem->work_item_cache[constants.work_index].parent_node,
                               constants.top_down_status_mask, constants.status_mask);
      }

      if (!node->hasStatus(ns_PART))
      {
        // node is already updated since there is no bottom-up step needed
        node->clearNeedsUpdate();
      }
    }
    bool make_progress = variables.is_active;

    __threadfence_block(); // ensure that all threads of this block see the top-down status changes

    // handle LeafNodes for top-down and bottom-up step
    if (variables.is_active & lastLevel)
    {
#pragma unroll
      for (uint32_t i = 0; i < branching_factor; ++i)
      {
        // ## top-down ##
        InnerNode* my_node = &shared_mem->work_item_cache[constants.work_index].node[i];
        if (my_node->hasStatus(ns_PART))
        {
          const bool leaf_update_subtree = shared_mem->work_item_cache[constants.work_index].update_subtree
              || my_node->hasFlags(nf_UPDATE_SUBTREE);
          LeafNode* leafNode = &((LeafNode*) my_node->getChildPtr())[constants.work_lane];
          if ((leafNode->getStatus() & LeafNode::INVALID_STATUS) == LeafNode::INVALID_STATUS)
            topDownUpdate(leafNode, my_node, constants.top_down_status_mask, constants.status_mask);
          if (leaf_update_subtree)
            // top-down update for updating a subtree with a single value
            topDownSubtreeUpdate(leafNode, my_node, constants.top_down_status_mask,
                                 constants.status_mask);

          //__threadfence_block();
#ifdef PROPAGATE_BOTTOM_UP
          bottomUpUpdate<branching_factor>(leafNode, my_node, shared_volatile_mem->bottom_up_reduction,
                                           constants.thread_id);

          if (constants.work_lane == 0)
          {
            // reset flags
            my_node->clearNeedsUpdate();
          }
#endif
        }
      }
    }
    __threadfence(); // ensure that all threads of the device see the status change of the bottom-up step

#ifdef PROPAGATE_BOTTOM_UP
    // handle InnerNodes for bottom-up step
    if (variables.is_active & (lastLevel | !is_top_down_mode))
    {
      const uint32_t is_ready_votes = BALLOT(!node->hasFlags(nf_NEEDS_UPDATE));
      const uint8_t my_is_ready_votes = (is_ready_votes
          >> (branching_factor * (constants.warp_lane / branching_factor))) & constants.work_lane_mask;
      if (my_is_ready_votes == constants.work_lane_mask)
      {
        assert(shared_mem->work_item_cache[constants.work_index].level >= 1);

        InnerNode* parent = shared_mem->work_item_cache[constants.work_index].parent_node;
        bottomUpUpdate<branching_factor>(node, parent, shared_volatile_mem->bottom_up_reduction,
                                         constants.thread_id);

        if (constants.work_lane == 0)
        {
          // reset nf_NEEDS_UPDATE
          parent->clearNeedsUpdate();
        }
      }
      else
      {
        // push work item back in queue and handle it some other time
        insert_bottom_up_work_item |= (constants.work_lane == 0);
        make_progress = false;
      }
    }
#endif

    // ### add new item to work queue ###
    // compute offsets needed to keep the level ordering of the stack and so assure the max. memory usage of it
    {
      {
        uint32_t all_insert_votes[2];
        all_insert_votes[0] = BALLOT(insert_bottom_up_work_item);
        all_insert_votes[1] = BALLOT(insert_top_down_work_item);
        if (constants.warp_lane <= 1)
        {
          const uint32_t index = constants.warp_id + constants.warp_lane * (constants.num_warps + 1);
          // make warp local votes available for every warp
          shared_mem->warp_votes[index] = all_insert_votes[constants.warp_lane];
          // count warp votes and make them available for every warp
          shared_mem->warp_prefix_sum[index] = __popc(all_insert_votes[constants.warp_lane]);
        }
      }
      __syncthreads();

      // sequential warp prefix sum
      if (constants.thread_id <= 1)
      {
        const uint32_t index = constants.num_warps + constants.thread_id * (constants.num_warps + 1);
        shared_mem->warp_votes[index] = 0;
        shared_mem->warp_prefix_sum[index] = 0;
        uint32_t my_insert_count = 0;
#pragma unroll
        for (uint32_t w = 0; w < (constants.num_warps + 1); ++w)
        {
          const uint32_t offset = w + constants.thread_id * (constants.num_warps + 1);
          const uint32_t tmp = shared_mem->warp_prefix_sum[offset];
          shared_mem->warp_prefix_sum[offset] = my_insert_count;
          my_insert_count += tmp;
        }
      }
      __syncthreads();

      // insert existing work items of bottom-up step
      if (variables.is_active & insert_bottom_up_work_item)
      {
        const uint32_t my_level_start =
            shared_mem->level_start_index[shared_mem->work_item_cache[constants.work_index].level];
        const uint32_t start_warp = my_level_start / WARP_SIZE;
        const uint32_t local_index_start_warp = my_level_start % WARP_SIZE;
        assert(my_level_start < num_threads);

        // number of items of same kind previous to my position
        uint32_t my_index = shared_mem->warp_prefix_sum[constants.warp_id]
            + __popc(shared_mem->warp_votes[constants.warp_id] << (WARP_SIZE - constants.warp_lane));

        // add number of items of other kind till the level start position
        my_index += shared_mem->warp_prefix_sum[start_warp + constants.num_warps + 1]
            + __popc(
                shared_mem->warp_votes[start_warp + constants.num_warps + 1]
                    << (WARP_SIZE - local_index_start_warp));

        assert(
            my_index
                < (shared_mem->warp_prefix_sum[constants.num_warps]
                    + shared_mem->warp_prefix_sum[2 * constants.num_warps + 1]));
        assert((shared_mem->num_stack_work_items + my_index) < kernel_params.stack_size_per_task);

        WorkItem* work_item = &shared_mem->my_work_stack[shared_mem->num_stack_work_items + my_index];
        *work_item = shared_mem->work_item_cache[constants.work_index];
        work_item->is_top_down = false; // mark as bottom-up work item
      }

      // insert new items of top-down step
      if (variables.is_active & insert_top_down_work_item)
      {
        assert(node != NULL);
        assert(node->getChildPtr() != NULL);

        const uint32_t my_level_end =
            shared_mem->level_end_index[shared_mem->work_item_cache[constants.work_index].level];
        const uint32_t end_warp = my_level_end / WARP_SIZE;
        const uint32_t local_index_end_warp = my_level_end % WARP_SIZE;
        assert(my_level_end <= num_threads);

        // number of items of same kind previous to my position
        uint32_t my_index = shared_mem->warp_prefix_sum[constants.warp_id + constants.num_warps + 1]
            + __popc(
                shared_mem->warp_votes[constants.warp_id + constants.num_warps + 1]
                    << (WARP_SIZE - constants.warp_lane));

        // add number of items of other kind till the NEXT level start position
        my_index += shared_mem->warp_prefix_sum[end_warp]
            + __popc(shared_mem->warp_votes[end_warp] << (WARP_SIZE - local_index_end_warp));

        assert(
            my_index
                < (shared_mem->warp_prefix_sum[constants.num_warps]
                    + shared_mem->warp_prefix_sum[2 * constants.num_warps + 1]));
        assert((shared_mem->num_stack_work_items + my_index) < kernel_params.stack_size_per_task);

        shared_mem->my_work_stack[shared_mem->num_stack_work_items + my_index] = WorkItem(
            (InnerNode*) node->getChildPtr(), node, true,
            shared_mem->work_item_cache[constants.work_index].level - 1,
            update_subtree || node->hasFlags(nf_UPDATE_SUBTREE));
      }
      __syncthreads();

#ifndef NDEBUG
      // ##### Had to remove the following code to compile without errors ####
      // The error message isn't useful at all, since it complains about device functions like ____syncthreads.
      // It's probably a compiler bug, due to too much used registers,
      // since only the following line also leads to the same problem: const bool foo = true;
      // #####################################################################
      // TODO Deal with compiler problem to enable the assertions
//      uint32_t bottom_up_count = __syncthreads_count(insert_bottom_up_work_item);
//      uint32_t top_down_count = __syncthreads_count(insert_top_down_work_item);
//      assert(bottom_up_count == shared_mem->warp_prefix_sum[constants.num_warps]);
//      assert(top_down_count == shared_mem->warp_prefix_sum[2 * constants.num_warps + 1]);
#endif

      if (constants.thread_id == 0)
      {
        // check stack sorting by level
#ifndef NDEBUG
        int32_t num_inserts = shared_mem->warp_prefix_sum[constants.num_warps]
            + shared_mem->warp_prefix_sum[2 * constants.num_warps + 1];
        for (int32_t i = 0; i < (num_inserts - 1); ++i)
          assert(shared_mem->my_work_stack[shared_mem->num_stack_work_items + i].level >= shared_mem->my_work_stack[shared_mem->num_stack_work_items + i + 1].level);
#endif

        shared_mem->num_stack_work_items += shared_mem->warp_prefix_sum[constants.num_warps]
            + shared_mem->warp_prefix_sum[2 * constants.num_warps + 1];
      }
      __syncthreads();
    }
    // ### --> work items of bottom-up step are removed from the queue ###

    variables.make_progress_count = __syncthreads_count(make_progress);
  }

  __device__
  static void doReductionWork(SharedMem* const shared_mem, volatile SharedVolatileMem* const shared_volatile_mem,
                              Variables& variables, const Constants& constants, KernelParams& kernel_params)
  {
    // Nothing to do
  }

  __device__
  static bool abortLoop(SharedMem* const shared_mem, volatile SharedVolatileMem* const shared_volatile_mem,
                              Variables& variables, const Constants& constants, KernelParams& kernel_params)
  {
    return Base::abortLoop(shared_mem, shared_volatile_mem, variables, constants, kernel_params)
            || variables.make_progress_count <= 0;
  }
};

}

}
}

#endif
