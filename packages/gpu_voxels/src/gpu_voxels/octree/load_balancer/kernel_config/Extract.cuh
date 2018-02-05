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
#ifndef GPU_VOXELS_OCTREE_LOAD_BALANCED_KERNEL_CONFIG_EXTRACT_CUH_INCLUDED
#define GPU_VOXELS_OCTREE_LOAD_BALANCED_KERNEL_CONFIG_EXTRACT_CUH_INCLUDED

#include <gpu_voxels/octree/DataTypes.h>
#include <gpu_voxels/octree/kernels/kernel_common.h>
#include <gpu_voxels/octree/Morton.h>
#include <gpu_voxels/octree/NTreeData.h>
#include <gpu_voxels/octree/load_balancer/kernel_config/LoadBalance.cuh>

namespace gpu_voxels {
namespace NTree {
namespace LoadBalancer {

/**
 * @brief This struct defines the shared memory, variables, kernel functions etc. needed to extract data from an \code NTree \endcode with help of the load balancing concept.
 * @tparam clear_collision_flag \code true \endcode to clear the collision flag while extracting data.
 * @tparam count_mode \code true \endcode to only compute the number of nodes that would be extracted from the \code NTree \endcode
 */
template<std::size_t num_threads,
  std::size_t branching_factor,
  typename InnerNode,
  typename LeafNode,
  typename NodeData,
  bool clear_collision_flag,
  bool count_mode>
struct ExtractNTreeKernelConfig: public AbstractKernelConfig<WorkItemIntersect<InnerNode, InnerNode>,
    num_threads, branching_factor>
{
public:
  typedef WorkItemExtract<InnerNode> WorkItem;
  typedef AbstractKernelConfig<WorkItem, num_threads, branching_factor> Base;

   /**
   * @brief Extend the shared memory data with values needed for extracting data from the \code NTree \endcode
   */
  struct SharedMemConfig : public Base::AbstractSharedMemConfig
  {
  public:
      uint32_t votes_new_queue_items[num_threads / WARP_SIZE];
      uint32_t votes_voxel_list[num_threads / WARP_SIZE];
      uint32_t votes_voxel_list_last_level[num_threads / WARP_SIZE];
      uint32_t new_queue_items_count;
      uint32_t voxel_list_count;
      uint32_t voxel_list_last_level_count;
      uint32_t voxel_list_offset;
  };

  /**
  * @brief Extend the constants with values needed for extracting data from the \code NTree \endcode
  */
  struct ConstConfig : public Base::AbstractConstConfig
  {
  public:
    __host__ __device__
    ConstConfig(const dim3 p_grid_dim,
                const dim3 p_block_dim,
                const dim3 p_block_ids,
                const dim3 p_thread_ids,
                const uint32_t p_stack_size_per_task) :
                Base::AbstractConstConfig(
                    p_grid_dim,
                    p_block_dim,
                    p_block_ids,
                    p_thread_ids,
                    p_stack_size_per_task,
                    p_stack_size_per_task + num_threads / branching_factor - num_threads)
    {

    }
  };

  /**
  * @brief Extend the kernel parameters with values needed for extracting data from the \code NTree \endcode
  */
  struct KernelParameters: public Base::AbstractKernelParameters
  {
  public:
    const uint32_t min_level;
    NodeData* node_data;
    const uint32_t node_data_size;
    uint32_t* global_voxel_list_count;
    const MortonCube borders;

    __host__ __device__
    KernelParameters(const typename Base::AbstractKernelParameters& abstract_params,
                     const uint32_t p_min_level,
                     NodeData* p_node_data,
                     const uint32_t p_node_data_size,
                     uint32_t* p_global_voxel_list_count) :
        Base::AbstractKernelParameters(abstract_params),
        min_level(p_min_level),
        node_data(p_node_data),
        node_data_size(p_node_data_size),
        global_voxel_list_count(p_global_voxel_list_count),
        borders()
    {

    }

    __host__ __device__
    KernelParameters(const typename Base::AbstractKernelParameters& abstract_params,
                     const uint32_t p_min_level,
                     NodeData* p_node_data,
                     const uint32_t p_node_data_size,
                     uint32_t* p_global_voxel_list_count,
                     const MortonCube p_borders) :
        Base::AbstractKernelParameters(abstract_params),
        min_level(p_min_level),
        node_data(p_node_data),
        node_data_size(p_node_data_size),
        global_voxel_list_count(p_global_voxel_list_count),
        borders(p_borders)
    {

    }
  };

  /*
   * Redefine/override the data types needed for extracting data from the \code NTree \endcode
   *
   */
  typedef SharedMemConfig SharedMem;
  typedef typename Base::AbstractSharedVolatileMemConfig SharedVolatileMem;
  typedef typename Base::AbstractVariablesConfig Variables;
  typedef ConstConfig Constants;
  typedef KernelParameters KernelParams;

  __device__
  static void doLoadBalancedWork(SharedMem* const shared_mem, volatile SharedVolatileMem* const shared_volatile_mem,
                                 Variables& variables, const Constants& constants, KernelParams& kernel_params)
  {
    bool is_part = false;
    bool is_last_level = false;
    InnerNode* node;
    bool is_min_level = false;
    bool is_selected = false;
    if (variables.is_active)
    {
      node = &shared_mem->work_item_cache[constants.work_index].node[constants.work_lane];
      const uint8_t level = shared_mem->work_item_cache[constants.work_index].level;
      assert(level > 0);
      is_min_level = (level == kernel_params.min_level);
      is_last_level = node->hasStatus(ns_LAST_LEVEL);

      // check whether we are in the bounds of the selected volume
      const bool is_in_boarders = (level < kernel_params.borders.m_level)
          || (getZOrderNodeId<branching_factor>(kernel_params.borders.m_voxel_id, level)
              == constants.work_lane);

      is_part = is_in_boarders & node->hasStatus(ns_PART);
      is_selected = is_in_boarders & (const_extract_selection[node->getStatus()] != 0);
      assert(node != NULL);
    }

    if ((constants.thread_id % WARP_SIZE) == 0)
      shared_mem->votes_voxel_list_last_level[constants.warp_id] = 0;

    // ### handle LeafNodes ###
    if (variables.is_active)
    {
      bool is_leaf_vote = false;
      //#pragma unroll
      for (uint32_t i = 0; i < branching_factor; ++i)
      {
        InnerNode* node = &shared_mem->work_item_cache[constants.work_index].node[i];
        is_leaf_vote = node->hasStatus(ns_PART) & node->hasStatus(ns_LAST_LEVEL)
            & (shared_mem->work_item_cache[constants.work_index].level != kernel_params.min_level);
        if (is_leaf_vote)
        {
          LeafNode* leafNode = &((LeafNode*) node->getChildPtr())[constants.work_lane];
          is_leaf_vote = (const_extract_selection[leafNode->getStatus()] != 0);
        }
        uint32_t votes = BALLOT(is_leaf_vote);
        if ((constants.thread_id % WARP_SIZE) == 0)
          shared_mem->votes_voxel_list_last_level[constants.warp_id] += __popc(votes);
      }
    }

    // compute offsets needed to keep the level ordering of the stack and so assure the max. memory usage of it
    uint32_t votes_last_level = BALLOT(is_last_level);
    uint32_t votes_is_part = BALLOT(is_part);
    uint32_t votes_is_active = BALLOT(variables.is_active);
    uint32_t votes_is_min_level = BALLOT(is_min_level);
    uint32_t votes_is_selected = BALLOT(is_selected);
    uint32_t votes_new_queue_items = ~votes_last_level & votes_is_active & votes_is_part
        & ~votes_is_min_level;
    uint32_t votes_voxel_list = votes_is_selected & ((~votes_is_part & votes_is_active) | votes_is_min_level);

    if ((constants.thread_id % WARP_SIZE) == 0)
    {
      shared_mem->votes_new_queue_items[constants.warp_id] = __popc(votes_new_queue_items);
      shared_mem->votes_voxel_list[constants.warp_id] = __popc(votes_voxel_list);
    }
    __syncthreads();

    // sequential warp prefix sum
    if (constants.thread_id == 0)
    {
      shared_mem->new_queue_items_count = shared_mem->voxel_list_count =
          shared_mem->voxel_list_last_level_count = 0;
#pragma unroll
      for (uint32_t w = 0; w < num_threads / WARP_SIZE; ++w)
      {
        uint32_t tmp = shared_mem->votes_new_queue_items[w];
        shared_mem->votes_new_queue_items[w] = shared_mem->new_queue_items_count;
        shared_mem->new_queue_items_count += tmp;

        tmp = shared_mem->votes_voxel_list[w];
        shared_mem->votes_voxel_list[w] = shared_mem->voxel_list_count;
        shared_mem->voxel_list_count += tmp;

        tmp = shared_mem->votes_voxel_list_last_level[w];
        shared_mem->votes_voxel_list_last_level[w] = shared_mem->voxel_list_last_level_count;
        shared_mem->voxel_list_last_level_count += tmp;
      }
    }
    __syncthreads();

    // ### handle new work queue items ###
    if (variables.is_active & is_part & !is_last_level & !is_min_level)
    {
      const uint32_t warpLocalIndex = __popc(
          votes_new_queue_items << (WARP_SIZE - (constants.thread_id % WARP_SIZE)));
      const uint32_t interWarpIndex = shared_mem->votes_new_queue_items[constants.warp_id];
      WorkItem* stackPointer = &shared_mem->my_work_stack[shared_mem->num_stack_work_items + interWarpIndex
          + warpLocalIndex];
      assert(shared_mem->work_item_cache[constants.work_index].level >= 1);
      *stackPointer = WorkItem(
          (InnerNode*) node->getChildPtr(),
          getZOrderNextLevel<branching_factor>(shared_mem->work_item_cache[constants.work_index].nodeId,
                                               constants.work_lane),
          shared_mem->work_item_cache[constants.work_index].level - 1);
    }
    __syncthreads();

    if (constants.thread_id == 0)
    {
      shared_mem->num_stack_work_items += shared_mem->new_queue_items_count;
      const int32_t tmp = (int32_t) atomicAdd(
          kernel_params.global_voxel_list_count,
          shared_mem->voxel_list_count + shared_mem->voxel_list_last_level_count);
      if (!count_mode)
      {
        // assure no overflow occurs for too small cubes array
        shared_mem->voxel_list_offset = min(
            tmp,
            int32_t(
                kernel_params.node_data_size
                    - (shared_mem->voxel_list_count + shared_mem->voxel_list_last_level_count)));
      }
    }
    __syncthreads();

    if (!count_mode)
    {
      // ### handle InnerNode which have no children ###
      if (variables.is_active & is_selected & (!is_part | is_min_level))
      {
        const uint32_t warpLocalIndex = __popc(
            votes_voxel_list << (WARP_SIZE - (constants.thread_id % WARP_SIZE)));
        const uint32_t interWarpIndex = shared_mem->votes_voxel_list[constants.warp_id];
        NodeData* my_cube = &kernel_params.node_data[shared_mem->voxel_list_offset + interWarpIndex
            + warpLocalIndex];

        assert(shared_mem->work_item_cache[constants.work_index].level >= 1);

        const OctreeVoxelID my_node_id = getZOrderNextLevel<branching_factor>(
            shared_mem->work_item_cache[constants.work_index].nodeId, constants.work_lane);
        *my_cube = node->extractData(
            getZOrderLastLevel<branching_factor>(my_node_id,
                                                 shared_mem->work_item_cache[constants.work_index].level),
            shared_mem->work_item_cache[constants.work_index].level);
      }

      if (clear_collision_flag)
      {
        if (variables.is_active)
          node->setStatus(node->getStatus() & ~ns_COLLISION);
      }

      // ### handle LeafNodes ###
      if (variables.is_active & is_last_level & !is_min_level)
      {
        uint32_t interWarpIndex = shared_mem->votes_voxel_list_last_level[constants.warp_id];

        //#pragma unroll
        for (uint32_t i = 0; i < branching_factor; ++i)
        {
          bool is_selected = false;
          InnerNode* node = &shared_mem->work_item_cache[constants.work_index].node[i];
          LeafNode* leafNode;
          const bool has_leaf = node->hasStatus(ns_PART) & node->hasStatus(ns_LAST_LEVEL);
          if (has_leaf)
          {
            leafNode = &((LeafNode*) node->getChildPtr())[constants.work_lane];
            is_selected = (const_extract_selection[leafNode->getStatus()] != 0);
          }
          uint32_t leaf_votes = BALLOT(is_selected);
          if (is_selected)
          {
            uint32_t warpLocalIndex = __popc(leaf_votes << (WARP_SIZE - (constants.thread_id % WARP_SIZE)));
            NodeData* my_cube = &kernel_params.node_data[shared_mem->voxel_list_offset
                + shared_mem->voxel_list_count + interWarpIndex + warpLocalIndex];

            assert(shared_mem->work_item_cache[constants.work_index].level == 1);

            OctreeVoxelID my_node_id = getZOrderNextLevel<branching_factor>(
                shared_mem->work_item_cache[constants.work_index].nodeId, i);
            my_node_id = getZOrderNextLevel<branching_factor>(my_node_id, constants.work_lane);
            *my_cube = leafNode->extractData(my_node_id, 0);
          }
          interWarpIndex += __popc(leaf_votes);

          if (clear_collision_flag)
          {
            if (has_leaf)
              leafNode->setStatus(leafNode->getStatus() & ~ns_COLLISION);
          }
        }
      }
      __syncthreads();
    }
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
    return Base::abortLoop(shared_mem, shared_volatile_mem, variables, constants, kernel_params);
  }
};

}
}
}
#endif
