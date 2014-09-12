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

  typedef SharedMemConfig SharedMem;
  typedef typename Base::AbstractSharedVolatileMemConfig SharedVolatileMem;
  typedef typename Base::AbstractVariablesConfig Variables;
  typedef ConstConfig Constants;
  typedef KernelParameters KernelParams;

  __device__
  static void doLoadBalancedWork(SharedMem& shared_mem, volatile SharedVolatileMem& shared_volatile_mem,
                                 Variables& variables, const Constants& constants, KernelParams& kernel_params)
  {
    bool is_part = false;
    bool is_last_level = false;
    InnerNode* node;
    bool is_min_level = false;
    bool is_selected = false;
    if (variables.is_active)
    {
      node = &shared_mem.work_item_cache[constants.work_index].node[constants.work_lane];
      const uint8_t level = shared_mem.work_item_cache[constants.work_index].level;
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
      shared_mem.votes_voxel_list_last_level[constants.warp_id] = 0;

    // ### handle LeafNodes ###
    if (variables.is_active)
    {
      bool is_leaf_vote = false;
      //#pragma unroll
      for (uint32_t i = 0; i < branching_factor; ++i)
      {
        InnerNode* node = &shared_mem.work_item_cache[constants.work_index].node[i];
        is_leaf_vote = node->hasStatus(ns_PART) & node->hasStatus(ns_LAST_LEVEL)
            & (shared_mem.work_item_cache[constants.work_index].level != kernel_params.min_level);
        if (is_leaf_vote)
        {
          LeafNode* leafNode = &((LeafNode*) node->getChildPtr())[constants.work_lane];
          //is_leaf_vote = (const_extract_selection[leafNode->getStatus() | leafNode->getStatusFlags()] != 0);
          is_leaf_vote = (const_extract_selection[leafNode->getStatus()] != 0);
        }
        uint32_t votes = __ballot(is_leaf_vote);
        if ((constants.thread_id % WARP_SIZE) == 0)
          shared_mem.votes_voxel_list_last_level[constants.warp_id] += __popc(votes);
      }
    }

    // compute offsets needed to keep the level ordering of the stack and so assure the max. memory usage of it
    uint32_t votes_last_level = __ballot(is_last_level);
    uint32_t votes_is_part = __ballot(is_part);
    uint32_t votes_is_active = __ballot(variables.is_active);
    uint32_t votes_is_min_level = __ballot(is_min_level);
    uint32_t votes_is_selected = __ballot(is_selected);
    uint32_t votes_new_queue_items = ~votes_last_level & votes_is_active & votes_is_part
        & ~votes_is_min_level;
    uint32_t votes_voxel_list = votes_is_selected & ((~votes_is_part & votes_is_active) | votes_is_min_level);
    //    uint32_t shared_mem.votes_voxel_list_last_level = votes_is_selected
    //        & (votes_is_part & votes_last_level & votes_is_active & ~votes_is_min_level);
    if ((constants.thread_id % WARP_SIZE) == 0)
    {
      shared_mem.votes_new_queue_items[constants.warp_id] = __popc(votes_new_queue_items);
      shared_mem.votes_voxel_list[constants.warp_id] = __popc(votes_voxel_list);
      //shared_mem.votes_voxel_list_last_level[warpIndex] = __popc(shared_mem.votes_voxel_list_last_level) * branching_factor;
    }
    __syncthreads();

    // sequential warp prefix sum
    if (constants.thread_id == 0)
    {
      shared_mem.new_queue_items_count = shared_mem.voxel_list_count =
          shared_mem.voxel_list_last_level_count = 0;
#pragma unroll
      for (uint32_t w = 0; w < num_threads / WARP_SIZE; ++w)
      {
        uint32_t tmp = shared_mem.votes_new_queue_items[w];
        shared_mem.votes_new_queue_items[w] = shared_mem.new_queue_items_count;
        shared_mem.new_queue_items_count += tmp;

        tmp = shared_mem.votes_voxel_list[w];
        shared_mem.votes_voxel_list[w] = shared_mem.voxel_list_count;
        shared_mem.voxel_list_count += tmp;

        tmp = shared_mem.votes_voxel_list_last_level[w];
        shared_mem.votes_voxel_list_last_level[w] = shared_mem.voxel_list_last_level_count;
        shared_mem.voxel_list_last_level_count += tmp;
      }
    }
    __syncthreads();

    // ### handle new work queue items ###
    if (variables.is_active & is_part & !is_last_level & !is_min_level)
    {
      const uint32_t warpLocalIndex = __popc(
          votes_new_queue_items << (WARP_SIZE - (constants.thread_id % WARP_SIZE)));
      const uint32_t interWarpIndex = shared_mem.votes_new_queue_items[constants.warp_id];
      WorkItem* stackPointer = &shared_mem.my_work_stack[shared_mem.num_stack_work_items + interWarpIndex
          + warpLocalIndex];
      assert(shared_mem.work_item_cache[constants.work_index].level >= 1);
      *stackPointer = WorkItem(
          (InnerNode*) node->getChildPtr(),
          getZOrderNextLevel<branching_factor>(shared_mem.work_item_cache[constants.work_index].nodeId,
                                               constants.work_lane),
          shared_mem.work_item_cache[constants.work_index].level - 1);
    }
    __syncthreads();

    if (constants.thread_id == 0)
    {
      shared_mem.num_stack_work_items += shared_mem.new_queue_items_count;
      const int32_t tmp = (int32_t) atomicAdd(
          kernel_params.global_voxel_list_count,
          shared_mem.voxel_list_count + shared_mem.voxel_list_last_level_count);
      if (!count_mode)
      {
        // assure no overflow occurs for too small cubes array
        shared_mem.voxel_list_offset = min(
            tmp,
            int32_t(
                kernel_params.node_data_size
                    - (shared_mem.voxel_list_count + shared_mem.voxel_list_last_level_count)));
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
        const uint32_t interWarpIndex = shared_mem.votes_voxel_list[constants.warp_id];
        NodeData* my_cube = &kernel_params.node_data[shared_mem.voxel_list_offset + interWarpIndex
            + warpLocalIndex];

        assert(shared_mem.work_item_cache[constants.work_index].level >= 1);

        const VoxelID my_node_id = getZOrderNextLevel<branching_factor>(
            shared_mem.work_item_cache[constants.work_index].nodeId, constants.work_lane);
        *my_cube = node->extractData(
            getZOrderLastLevel<branching_factor>(my_node_id,
                                                 shared_mem.work_item_cache[constants.work_index].level),
            shared_mem.work_item_cache[constants.work_index].level);

        //      my_cube->m_side_length = getVoxelSideLength<branching_factor>(shared_mem.work_item_cache[constants.work_index].level);
        //      my_cube->m_type = (gpu_voxels::Voxel::VoxelType) node->getStatus();  //statusToVoxelType(node->getStatus());
        //      inv_morton_code60(getZOrderLastLevel<branching_factor>(my_node_id, shared_mem.work_item_cache[constants.work_index].level),
        //                        my_cube->m_position);
      }

      if (clear_collision_flag)
      {
        if (variables.is_active)
          node->setStatus(EnumNodeStatus(node->getStatus() & ~ns_COLLISION));
      }

      // ### handle LeafNodes ###
      if (variables.is_active & is_last_level & !is_min_level)
      {
        uint32_t interWarpIndex = shared_mem.votes_voxel_list_last_level[constants.warp_id];
        //      uint32_t warpLocalIndex = __popc(
        //          shared_mem.votes_voxel_list_last_level << (WARP_SIZE - ((constants.work_index * branching_factor) % WARP_SIZE)))
        //          * branching_factor;

        //#pragma unroll
        for (uint32_t i = 0; i < branching_factor; ++i)
        {
          bool is_selected = false;
          InnerNode* node = &shared_mem.work_item_cache[constants.work_index].node[i];
          LeafNode* leafNode;
          const bool has_leaf = node->hasStatus(ns_PART) & node->hasStatus(ns_LAST_LEVEL);
          if (has_leaf)
          {
            leafNode = &((LeafNode*) node->getChildPtr())[constants.work_lane];
            //is_selected = (const_extract_selection[leafNode->getStatus() | leafNode->getStatusFlags()] != 0);
            is_selected = (const_extract_selection[leafNode->getStatus()] != 0);
          }
          uint32_t leaf_votes = __ballot(is_selected);
          if (is_selected)
          {
            uint32_t warpLocalIndex = __popc(leaf_votes << (WARP_SIZE - (constants.thread_id % WARP_SIZE)));
            NodeData* my_cube = &kernel_params.node_data[shared_mem.voxel_list_offset
                + shared_mem.voxel_list_count + interWarpIndex + warpLocalIndex];

            assert(shared_mem.work_item_cache[constants.work_index].level == 1);

            VoxelID my_node_id = getZOrderNextLevel<branching_factor>(
                shared_mem.work_item_cache[constants.work_index].nodeId, i);
            my_node_id = getZOrderNextLevel<branching_factor>(my_node_id, constants.work_lane);
            *my_cube = leafNode->extractData(my_node_id, 0);

            //          my_cube->m_side_length = 1;
            //          my_cube->m_type = (gpu_voxels::Voxel::VoxelType) leafNode->getStatus(); //statusToVoxelType(leafNode->getStatus());
            //          inv_morton_code60(my_node_id, my_cube->m_position);
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
   static void doReductionWork(SharedMem& shared_mem, volatile SharedVolatileMem& shared_volatile_mem,
                               Variables& variables, const Constants& constants, KernelParams& kernel_params)
  {
    // Nothing to do
  }

  __device__
  static bool abortLoop(SharedMem& shared_mem, volatile SharedVolatileMem& shared_volatile_mem,
                              Variables& variables, const Constants& constants, KernelParams& kernel_params)
  {
    return Base::abortLoop(shared_mem, shared_volatile_mem, variables, constants, kernel_params);
  }
};


//template<std::size_t num_threads, std::size_t branching_factor, typename WorkItem, typename InnerNode,
//    typename LeafNode, typename NodeData, bool clear_collision_flag, bool count_mode>
//__global__ void kernelExtractTreeData(WorkItem* work_stacks, uint32_t* work_stacks_item_count,
//                                      const uint32_t stack_size_per_task, uint32_t* tasks_idle_count,
//                                      uint32_t* idle_count_threshold, const uint32_t min_level,
//                                      NodeData* node_data, const uint32_t node_data_size,
//                                      const MortonCube borders = MortonCube())
//{
//  typedef WorkItemExtract<InnerNode> WorkItem;

//  __shared__ uint32_t shared_num_stack_work_items; // number of work items in stack
//  __shared__ WorkItem shared_work_item_cache[num_threads / branching_factor];
//  __shared__ WorkItem* shared_my_work_stack; // pointer to stack of this task

//  // some index calculations
//  const uint32_t block_id = blockIdx.x;
//  const uint32_t constants.thread_id = threadIdx.x;
//  const uint32_t constants.warp_id = constants.thread_id / WARP_SIZE;
//  const uint32_t warp_lane = constants.thread_id % WARP_SIZE;
//  const uint32_t constants.work_index = constants.thread_id / branching_factor;
//  const uint32_t work_lane = constants.thread_id % branching_factor;

//  // ###########################################################
//  // ################### Specific declarations #################
//  // ###########################################################

//  __shared__ uint32_t votes_new_queue_items[num_threads / WARP_SIZE];
//  __shared__ uint32_t votes_voxel_list[num_threads / WARP_SIZE];
//  __shared__ uint32_t shared_mem.votes_voxel_list_last_level[num_threads / WARP_SIZE];
//  __shared__ uint32_t new_queue_items_count;
//  __shared__ uint32_t voxel_list_count;
//  __shared__ uint32_t voxel_list_last_level_count;
//  __shared__ uint32_t voxel_list_offset;

//  const uint32_t block_offset = gridDim.x * blockIdx.y + blockIdx.x;

//  // ###########################################################
//  // ################ End specific declarations ################
//  // ###########################################################

//  const uint32_t stack_items_threshold = stack_size_per_task + num_threads / branching_factor - num_threads;

//  if (constants.thread_id == 0)
//  {
//    shared_num_stack_work_items = work_stacks_item_count[block_id];
//    shared_my_work_stack = &work_stacks[block_id * stack_size_per_task];
//  }
//  __syncthreads();

//  assert(shared_num_stack_work_items < stack_items_threshold);

//  while (true)
//  {
//    if (handleIdleCounter(shared_num_stack_work_items, stack_items_threshold, tasks_idle_count,
//                          idle_count_threshold, constants.thread_id))
//      break;

//    //uint32_t insert_count_tid0 = 0; // number of new work items only maintained for thread 0
//    uint32_t num_work_items = min((uint32_t) (num_threads / branching_factor), shared_num_stack_work_items);
//    const bool is_active = constants.work_index < num_work_items;

//    // every thread grabs some work
//    blockCopy(shared_work_item_cache, &shared_my_work_stack[shared_num_stack_work_items - num_work_items],
//              num_work_items * sizeof(WorkItem), constants.thread_id, num_threads);
//    __syncthreads();

//    // decrease num work items in stack by the grabbed work
//    if (constants.thread_id == 0)
//      shared_num_stack_work_items -= num_work_items;
//    __syncthreads();

//    // ###################################################
//    // ############ Specific implementation ##############
//    // ###################################################

//    bool is_part = false;
//    bool lastLevel = false;
//    InnerNode* node;
//    const bool is_active = constants.work_index < nActive;
//    bool is_min_level = false;
//    bool is_selected = false;
//    if (is_active)
//    {
//      node = &shared_work_item_cache[constants.work_index].node[work_lane];
//      const uint8_t level = shared_work_item_cache[constants.work_index].level;
//      assert(level > 0);
//      is_min_level = (level == min_level);
//      lastLevel = node->hasStatus(ns_LAST_LEVEL);

//      // check whether we are in the bounds of the selected volume
//      const bool is_in_boarders = (level < borders.m_level)
//          || (getZOrderNodeId<branching_factor>(borders.m_voxel_id, level) == work_lane);

//      is_part = is_in_boarders & node->hasStatus(ns_PART);
//      is_selected = is_in_boarders & (const_extract_selection[node->getStatus()] != 0);
//      assert(node != NULL);
//    }

//    if ((constants.thread_id % WARP_SIZE) == 0)
//      shared_mem.votes_voxel_list_last_level[constants.warp_id] = 0;

//    // ### handle LeafNodes ###
//    if (is_active)
//    {
//      bool is_leaf_vote = false;
////#pragma unroll
//      for (uint32_t i = 0; i < branching_factor; ++i)
//      {
//        InnerNode* node = &shared_work_item_cache[constants.work_index].node[i];
//        is_leaf_vote = node->hasStatus(ns_PART) & node->hasStatus(ns_LAST_LEVEL)
//            & (shared_work_item_cache[constants.work_index].level != min_level);
//        if (is_leaf_vote)
//        {
//          LeafNode* leafNode = &((LeafNode*) node->getChildPtr())[work_lane];
//          //is_leaf_vote = (const_extract_selection[leafNode->getStatus() | leafNode->getStatusFlags()] != 0);
//          is_leaf_vote = (const_extract_selection[leafNode->getStatus()] != 0);
//        }
//        uint32_t votes = __ballot(is_leaf_vote);
//        if ((constants.thread_id % WARP_SIZE) == 0)
//          shared_mem.votes_voxel_list_last_level[constants.warp_id] += __popc(votes);
//      }
//    }

//    // compute offsets needed to keep the level ordering of the stack and so assure the max. memory usage of it
//    uint32_t votes_last_level = __ballot(lastLevel);
//    uint32_t votes_is_part = __ballot(is_part);
//    uint32_t votes_is_active = __ballot(is_active);
//    uint32_t votes_is_min_level = __ballot(is_min_level);
//    uint32_t votes_is_selected = __ballot(is_selected);
//    uint32_t votes_new_queue_items = ~votes_last_level & votes_is_active & votes_is_part
//        & ~votes_is_min_level;
//    uint32_t votes_voxel_list = votes_is_selected & ((~votes_is_part & votes_is_active) | votes_is_min_level);
////    uint32_t shared_mem.votes_voxel_list_last_level = votes_is_selected
////        & (votes_is_part & votes_last_level & votes_is_active & ~votes_is_min_level);
//    if ((constants.thread_id % WARP_SIZE) == 0)
//    {
//      votes_new_queue_items[constants.warp_id] = __popc(votes_new_queue_items);
//      votes_voxel_list[constants.warp_id] = __popc(votes_voxel_list);
//      //shared_mem.votes_voxel_list_last_level[warpIndex] = __popc(shared_mem.votes_voxel_list_last_level) * branching_factor;
//    }
//    __syncthreads();

//    // sequential warp prefix sum
//    if (constants.thread_id == 0)
//    {
//      new_queue_items_count = voxel_list_count = voxel_list_last_level_count = 0;
//#pragma unroll
//      for (uint32_t w = 0; w < num_threads / WARP_SIZE; ++w)
//      {
//        uint32_t tmp = votes_new_queue_items[w];
//        votes_new_queue_items[w] = new_queue_items_count;
//        new_queue_items_count += tmp;

//        tmp = votes_voxel_list[w];
//        votes_voxel_list[w] = voxel_list_count;
//        voxel_list_count += tmp;

//        tmp = shared_mem.votes_voxel_list_last_level[w];
//        shared_mem.votes_voxel_list_last_level[w] = voxel_list_last_level_count;
//        voxel_list_last_level_count += tmp;
//      }
//    }
//    __syncthreads();

//    // ### handle new work queue items ###
//    if (is_active & is_part & !lastLevel & !is_min_level)
//    {
//      const uint32_t warpLocalIndex = __popc(votes_new_queue_items << (WARP_SIZE - (constants.thread_id % WARP_SIZE)));
//      const uint32_t interWarpIndex = votes_new_queue_items[constants.warp_id];
//      WorkItem* stackPointer = &shared_my_work_stack[shared_num_stack_work_items + interWarpIndex
//          + warpLocalIndex];
//      assert(shared_work_item_cache[constants.work_index].level >= 1);
//      *stackPointer = WorkItem(
//          (InnerNode*) node->getChildPtr(),
//          getZOrderNextLevel<branching_factor>(shared_work_item_cache[constants.work_index].nodeId, work_lane),
//          shared_work_item_cache[constants.work_index].level - 1);
//    }
//    __syncthreads();

//    if (constants.thread_id == 0)
//    {
//      shared_num_stack_work_items += new_queue_items_count;
//      const int32_t tmp = (int32_t) atomicAdd(tasks_idle_count,
//                                              voxel_list_count + voxel_list_last_level_count);
//      if (!count_mode)
//      {
//        // assure no overflow occurs for too small cubes array
//        voxel_list_offset = min(
//            tmp, int32_t(node_data_size - (voxel_list_count + voxel_list_last_level_count)));
//      }
//    }
//    __syncthreads();

//    if (!count_mode)
//    {
//      // ### handle InnerNode which have no children ###
//      if (is_active & is_selected & (!is_part | is_min_level))
//      {
//        const uint32_t warpLocalIndex = __popc(votes_voxel_list << (WARP_SIZE - (constants.thread_id % WARP_SIZE)));
//        const uint32_t interWarpIndex = votes_voxel_list[constants.warp_id];
//        NodeData* my_cube = &node_data[voxel_list_offset + interWarpIndex + warpLocalIndex];

//        assert(shared_work_item_cache[constants.work_index].level >= 1);

//        const VoxelID my_node_id = getZOrderNextLevel<branching_factor>(
//            shared_work_item_cache[constants.work_index].nodeId, work_lane);
//        *my_cube = node->extractData(
//            getZOrderLastLevel<branching_factor>(my_node_id, shared_work_item_cache[constants.work_index].level),
//            shared_work_item_cache[constants.work_index].level);

////      my_cube->m_side_length = getVoxelSideLength<branching_factor>(shared_work_item_cache[constants.work_index].level);
////      my_cube->m_type = (gpu_voxels::Voxel::VoxelType) node->getStatus();  //statusToVoxelType(node->getStatus());
////      inv_morton_code60(getZOrderLastLevel<branching_factor>(my_node_id, shared_work_item_cache[constants.work_index].level),
////                        my_cube->m_position);
//      }

//      if (clear_collision_flag)
//      {
//        if (is_active)
//          node->setStatus(EnumNodeStatus(node->getStatus() & ~ns_COLLISION));
//      }

//      // ### handle LeafNodes ###
//      if (is_active & lastLevel & !is_min_level)
//      {
//        uint32_t interWarpIndex = shared_mem.votes_voxel_list_last_level[constants.warp_id];
////      uint32_t warpLocalIndex = __popc(
////          shared_mem.votes_voxel_list_last_level << (WARP_SIZE - ((constants.work_index * branching_factor) % WARP_SIZE)))
////          * branching_factor;

////#pragma unroll
//        for (uint32_t i = 0; i < branching_factor; ++i)
//        {
//          bool is_selected = false;
//          InnerNode* node = &shared_work_item_cache[constants.work_index].node[i];
//          LeafNode* leafNode;
//          const bool has_leaf = node->hasStatus(ns_PART) & node->hasStatus(ns_LAST_LEVEL);
//          if (has_leaf)
//          {
//            leafNode = &((LeafNode*) node->getChildPtr())[work_lane];
//            //is_selected = (const_extract_selection[leafNode->getStatus() | leafNode->getStatusFlags()] != 0);
//            is_selected = (const_extract_selection[leafNode->getStatus()] != 0);
//          }
//          uint32_t leaf_votes = __ballot(is_selected);
//          if (is_selected)
//          {
//            uint32_t warpLocalIndex = __popc(leaf_votes << (WARP_SIZE - (constants.thread_id % WARP_SIZE)));
//            NodeData* my_cube = &node_data[voxel_list_offset + voxel_list_count + interWarpIndex
//                + warpLocalIndex];

//            assert(shared_work_item_cache[constants.work_index].level == 1);

//            VoxelID my_node_id = getZOrderNextLevel<branching_factor>(
//                shared_work_item_cache[constants.work_index].nodeId, i);
//            my_node_id = getZOrderNextLevel<branching_factor>(my_node_id, work_lane);
//            *my_cube = leafNode->extractData(my_node_id, 0);

////          my_cube->m_side_length = 1;
////          my_cube->m_type = (gpu_voxels::Voxel::VoxelType) leafNode->getStatus(); //statusToVoxelType(leafNode->getStatus());
////          inv_morton_code60(my_node_id, my_cube->m_position);
//          }
//          interWarpIndex += __popc(leaf_votes);

//          if (clear_collision_flag)
//          {
//            if (has_leaf)
//              leafNode->setStatus(leafNode->getStatus() & ~ns_COLLISION);
//          }
//        }
//      }
//      __syncthreads();
//    }

//    // ###################################################
//    // ########## End specific implementation ############
//    // ###################################################
//  }

//  if (constants.thread_id == 0)
//    work_stacks_item_count[block_id] = shared_num_stack_work_items;
//}

//// ### todo try to template the kernel functions so there is less code duplication ###
//// Separate code: shared mem alloc in global func ==> call general device func and pass pointers to shared mem ==> call specific/overloaded device function for inner loop work
//// pass parameters as struct which is a template parameter to be able to pass the specific data to the specific device function
//
//// todo rethink scetchup
////  =
////  =
////  =
////  v
//
//template<std::size_t num_threads, std::size_t branching_factor, typename WorkItem, typename InnerNode,
//    typename LeafNode, typename Parameters, typename LBClass>
//__global__ void kernelLoadBalanceExample(Parameters params, LBClass general_lb)
//{
//  __shared__ uint32_t shared_num_stack_work_items; // number of work items in stack
//  __shared__ WorkItem shared_work_item_cache[num_threads / branching_factor];
//  __shared__ WorkItem* shared_my_work_stack; // pointer to stack of this task
//
//  params.shared_num_stack_work_items = shared_num_stack_work_items;
//  params.shared_work_item_cache = shared_work_item_cache;
//  params.shared_my_work_stack = shared_my_work_stack;
//  general_lb.general_work_flow(params, general_lb);
//}
//
//template<std::size_t num_threads, std::size_t branching_factor, typename WorkItem, typename InnerNode,
//    typename LeafNode, typename Parameters, typename LBClass>
//__device__ void general_work_flow(Parameters params, LBClass general_lb)
//{
//  // some index calculations
//    const uint32_t block_id = blockIdx.x;
//    const uint32_t constants.thread_id = threadIdx.x;
//    const uint32_t constants.warp_id = constants.thread_id / WARP_SIZE;
//    const uint32_t warp_lane = constants.thread_id % WARP_SIZE;
//    const uint32_t constants.work_index = constants.thread_id / branching_factor;
//    const uint32_t work_lane = constants.thread_id % branching_factor;
//
//    const uint32_t stack_items_threshold = stack_size_per_task + num_threads / branching_factor - num_threads;
//
//    std::size_t my_num_collisions = 0;
//
//    if (constants.thread_id == 0)
//    {
//      shared_num_stack_work_items = work_stacks_item_count[block_id];
//      shared_my_work_stack = &work_stacks[block_id * stack_size_per_task];
//    }
//    __syncthreads();
//
//    assert(shared_num_stack_work_items < stack_items_threshold);
//
//    while (true)
//    {
//      if (handleIdleCounter(shared_num_stack_work_items, stack_items_threshold, tasks_idle_count,
//                            idle_count_threshold, constants.thread_id))
//        break;
//
//      uint32_t insert_count_tid0 = 0; // number of new work items only maintained for thread 0
//      uint32_t num_work_items = min((uint32_t) (num_threads / branching_factor), shared_num_stack_work_items);
//      const bool is_active = constants.work_index < num_work_items;
//
//      // every thread grabs some work
//      blockCopy(shared_work_item_cache, &shared_my_work_stack[shared_num_stack_work_items - num_work_items],
//                num_work_items * sizeof(WorkItem), constants.thread_id, num_threads);
//      __syncthreads();
//
//      // decrease num work items in stack by the grabbed work
//      if (constants.thread_id == 0)
//        shared_num_stack_work_items -= num_work_items;
//      __syncthreads();
//
//      general_lb.specific_work_impl(params, general_lb);
//
//      if (constants.thread_id == 0)
//        shared_num_stack_work_items += insert_count_tid0;
//      __syncthreads();
//    }
//
//    if (constants.thread_id == 0)
//      work_stacks_item_count[block_id] = shared_num_stack_work_items;
//}

//// This general kernel function can't be templated to avoid code duplication, since the specialization may need new shared memory variables.
//// Shared memory variables can only be declared in __global__ functions and not in a __device__ function.
//template<std::size_t num_threads, std::size_t branching_factor, typename WorkItem, typename InnerNode,
//    typename LeafNode>
//__global__ void kernelLoadBalanceExample(WorkItem* work_stacks, uint32_t* work_stacks_item_count,
//                                         const uint32_t stack_size_per_task, uint32_t* tasks_idle_count,
//                                         const uint32_t idle_count_threshold)
//{
//  __shared__ uint32_t shared_num_stack_work_items; // number of work items in stack
//  __shared__ WorkItem shared_work_item_cache[num_threads / branching_factor];
//  __shared__ WorkItem* shared_my_work_stack; // pointer to stack of this task
//
//  // some index calculations
//  const uint32_t block_id = blockIdx.x;
//  const uint32_t constants.thread_id = threadIdx.x;
//  const uint32_t constants.warp_id = constants.thread_id / WARP_SIZE;
//  const uint32_t warp_lane = constants.thread_id % WARP_SIZE;
//  const uint32_t constants.work_index = constants.thread_id / branching_factor;
//  const uint32_t work_lane = constants.thread_id % branching_factor;
//
//  const uint32_t stack_items_threshold = stack_size_per_task + num_threads / branching_factor - num_threads;
//
//  std::size_t my_num_collisions = 0;
//
//  if (constants.thread_id == 0)
//  {
//    shared_num_stack_work_items = work_stacks_item_count[block_id];
//    shared_my_work_stack = &work_stacks[block_id * stack_size_per_task];
//  }
//  __syncthreads();
//
//  assert(shared_num_stack_work_items < stack_items_threshold);
//
//  while (true)
//  {
//    if (handleIdleCounter(shared_num_stack_work_items, stack_items_threshold, tasks_idle_count,
//                          idle_count_threshold, constants.thread_id))
//      break;
//
//    uint32_t insert_count_tid0 = 0; // number of new work items only maintained for thread 0
//    uint32_t num_work_items = min((uint32_t) (num_threads / branching_factor), shared_num_stack_work_items);
//    const bool is_active = constants.work_index < num_work_items;
//
//    // every thread grabs some work
//    blockCopy(shared_work_item_cache, &shared_my_work_stack[shared_num_stack_work_items - num_work_items],
//              num_work_items * sizeof(WorkItem), constants.thread_id, num_threads);
//    __syncthreads();
//
//    // decrease num work items in stack by the grabbed work
//    if (constants.thread_id == 0)
//      shared_num_stack_work_items -= num_work_items;
//    __syncthreads();
//
//    // ################# Implementation needed #####################
//    // todo process work items
//    // todo push new work items on stack
//    // #############################################################
//
//    if (constants.thread_id == 0)
//      shared_num_stack_work_items += insert_count_tid0;
//    __syncthreads();
//  }
//
//  if (constants.thread_id == 0)
//    work_stacks_item_count[block_id] = shared_num_stack_work_items;
//}

}
}
}
#endif
