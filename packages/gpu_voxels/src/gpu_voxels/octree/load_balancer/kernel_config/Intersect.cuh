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
 * \date    2014-07-24
 *
 */
//----------------------------------------------------------------------/*
#ifndef GPU_VOXELS_OCTREE_LOAD_BALANCED_KERNEL_CONFIG_INTERSECT_CUH_INCLUDED
#define GPU_VOXELS_OCTREE_LOAD_BALANCED_KERNEL_CONFIG_INTERSECT_CUH_INCLUDED

#include <gpu_voxels/octree/DataTypes.h>
#include <gpu_voxels/octree/kernels/kernel_common.h>
#include <gpu_voxels/octree/NTreeData.h>
#include <gpu_voxels/octree/load_balancer/kernel_config/LoadBalance.cuh>

namespace gpu_voxels {
namespace NTree {
namespace LoadBalancer {

/**
 * @brief This struct defines the shared memory, variables, kernel functions etc. needed to do a collision check between two \code NTree \endcode with help of the load balancing concept.
 * @tparam set_collision_flag \code true \endcode to set the collision flag of each node in collision. If it's not clear which of two nodes in collision to set, the node of the first \code NTree \endcode is prefered.
 * @tparam Collider The \code Collider \endcode type defines whether two nodes are in collision or not.
 */
template<std::size_t num_threads,
  std::size_t branching_factor,
  typename InnerNode1,
  typename LeafNode1,
  typename InnerNode2,
  typename LeafNode2,
  bool set_collision_flag,
  typename Collider>
struct IntersectNTreeKernelConfig: public AbstractKernelConfig<WorkItemIntersect<InnerNode1, InnerNode2>,
    num_threads, branching_factor>
{
public:
  typedef WorkItemIntersect<InnerNode1, InnerNode2> WorkItem;
  typedef AbstractKernelConfig<WorkItem, num_threads, branching_factor> Base;

  struct SharedVolatileMemConfig: public Base::AbstractSharedVolatileMemConfig
  {
  public:
    uint32_t block_votes[num_threads / WARP_SIZE];
  };  

  struct VariablesConfig: public Base::AbstractVariablesConfig
  {
  public:
    std::size_t my_num_collisions;

    __host__ __device__
    VariablesConfig() :
        my_num_collisions(0)
    {
    }
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
        Base::AbstractConstConfig(p_grid_dim,
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
    std::size_t* num_collisions;
    Collider collider;

    __host__ __device__
    KernelParameters(const typename Base::AbstractKernelParameters& abstract_params, const uint32_t p_min_level,
                     std::size_t* p_num_collisions, Collider p_collider) :
        Base::AbstractKernelParameters(abstract_params),
        min_level(p_min_level),
        num_collisions(p_num_collisions),
        collider(p_collider)
    {

    }
  };

  typedef typename Base::AbstractSharedMemConfig SharedMem;
  typedef SharedVolatileMemConfig SharedVolatileMem;
  typedef VariablesConfig Variables;
  typedef ConstConfig Constants;
  typedef KernelParameters KernelParams;

  __device__
  static void doLoadBalancedWork(SharedMem* const shared_mem, volatile SharedVolatileMem* const shared_volatile_mem,
                                 Variables& variables, const Constants& constants, KernelParams& kernel_params)
  {
    uint32_t insert_count_tid0 = 0; // number of new work items only maintained for thread 0
    bool insert_work_item = false, is_last_level = false;
    InnerNode1* a_node;
    InnerNode2* b_node;
    bool a_active = false, b_active = false, end_here = false;
    if (variables.is_active)
    {
      a_active = shared_mem->work_item_cache[constants.work_index].a_active;
      a_node = shared_mem->work_item_cache[constants.work_index].a;
      a_node += constants.work_lane * a_active;
      b_active = shared_mem->work_item_cache[constants.work_index].b_active;
      b_node = shared_mem->work_item_cache[constants.work_index].b;
      b_node += constants.work_lane * b_active;
      end_here = (kernel_params.min_level >= shared_mem->work_item_cache[constants.work_index].level);
      is_last_level = (a_node->hasStatus(ns_LAST_LEVEL) | b_node->hasStatus(ns_LAST_LEVEL)) & !end_here;
      insert_work_item = kernel_params.collider.collide(*a_node, *b_node) & !is_last_level
          & (a_node->hasStatus(ns_PART) | b_node->hasStatus(ns_PART)) & !end_here;
    }

    // the only active search is ending here
    if (variables.is_active
        && (kernel_params.collider.collide(*a_node, *b_node)
            & ((!a_node->hasStatus(ns_PART) & !b_node->hasStatus(ns_PART)) | end_here)))
    {
      variables.my_num_collisions +=
          const_voxel_at_level[shared_mem->work_item_cache[constants.work_index].level];
      if (set_collision_flag)
      {
        if (a_active)
          a_node->setStatus(a_node->getStatus() | ns_COLLISION);
        else
          b_node->setStatus(b_node->getStatus() | ns_COLLISION);
      }
    }

    // handle leaf nodes
    if (__syncthreads_or(is_last_level) && kernel_params.min_level == 0)
    {
      InnerNode1* a_temp;
      InnerNode2* b_temp;
      const uint32_t leafs_per_work = branching_factor * branching_factor;
      for (uint32_t i = constants.thread_id; i < variables.num_work_items * leafs_per_work; i += num_threads)
      {
        const uint32_t my_work_item = i / leafs_per_work;
        const uint32_t my_inner_node = (i % leafs_per_work) / branching_factor;
        a_temp = shared_mem->work_item_cache[my_work_item].a;
        const bool a_temp_active = shared_mem->work_item_cache[my_work_item].a_active;
        a_temp += my_inner_node * a_temp_active;
        b_temp = shared_mem->work_item_cache[my_work_item].b;
        const bool b_temp_active = shared_mem->work_item_cache[my_work_item].b_active;
        b_temp += my_inner_node * b_temp_active;
        if ((shared_mem->work_item_cache[my_work_item].level == 1)
            & kernel_params.collider.collide(*a_temp, *b_temp)
            & (a_temp->hasStatus(ns_PART) | b_temp->hasStatus(ns_PART)))
        {
          const uint32_t my_leaf_node = i % branching_factor;
          bool is_in_conflict = false;
          if (a_temp->hasStatus(ns_PART) & b_temp->hasStatus(ns_PART))
          {
            LeafNode1* a = &((LeafNode1*) a_temp->getChildPtr())[my_leaf_node];
            LeafNode2* b = &((LeafNode2*) b_temp->getChildPtr())[my_leaf_node];
            is_in_conflict = kernel_params.collider.collide(*a, *b);
            if (set_collision_flag)
            {
              if (is_in_conflict)
                a->setStatus(a->getStatus() | ns_COLLISION);
            }
          }
          else if (a_temp->hasStatus(ns_PART) & !b_temp->hasStatus(ns_PART))
          {
            LeafNode1* a = &((LeafNode1*) a_temp->getChildPtr())[my_leaf_node];
            is_in_conflict = kernel_params.collider.collide(*b_temp, *a);
            if (set_collision_flag)
            {
              if (is_in_conflict)
                a->setStatus(a->getStatus() | ns_COLLISION);
            }
          }
          else if (!a_temp->hasStatus(ns_PART) & b_temp->hasStatus(ns_PART))
          {
            LeafNode2* b = &((LeafNode2*) b_temp->getChildPtr())[my_leaf_node];
            is_in_conflict = kernel_params.collider.collide(*a_temp, *b);
            if (set_collision_flag)
            {
              if (is_in_conflict)
                b->setStatus(b->getStatus() | ns_COLLISION);
            }
          }
          variables.my_num_collisions += is_in_conflict;
        }
      }
    }

    uint32_t all_votes = thread_prefix<num_threads / WARP_SIZE>(shared_volatile_mem->block_votes,
                                                                constants.thread_id,
                                                                insert_count_tid0,
                                                                insert_work_item);

    // add new work items to stack
    if (variables.is_active & insert_work_item)
    {
      const uint32_t warp_local_index = __popc(all_votes << (WARP_SIZE - constants.warp_lane));
      const uint32_t inter_warp_index = shared_volatile_mem->block_votes[constants.warp_id];
      WorkItem* stackPointer = &shared_mem->my_work_stack[shared_mem->num_stack_work_items + inter_warp_index + warp_local_index];
      a_active &= a_node->hasStatus(ns_PART);
      InnerNode1* a_children = a_active ? ((InnerNode1*) a_node->getChildPtr()) : a_node;
      b_active &= b_node->hasStatus(ns_PART);
      InnerNode2* b_children = b_active ? ((InnerNode2*) b_node->getChildPtr()) : b_node;
      *stackPointer = WorkItem(a_children, b_children,
                               shared_mem->work_item_cache[constants.work_index].level - 1, a_active,
                               b_active);
    }
    __syncthreads();

    if (constants.thread_id == 0)
      shared_mem->num_stack_work_items += insert_count_tid0;
    __syncthreads();
  }

  __device__
  static void doReductionWork(SharedMem* const shared_mem, volatile SharedVolatileMem* const shared_volatile_mem,
                              Variables& variables, const Constants& constants, KernelParams& kernel_params)
  {
    // todo is a shared memory reduction more efficient in this case?
    if (variables.my_num_collisions != 0)
      atomicAdd((unsigned long long int*) kernel_params.num_collisions,
                (unsigned long long int) variables.my_num_collisions);
  }
};

}
}
}

#endif
