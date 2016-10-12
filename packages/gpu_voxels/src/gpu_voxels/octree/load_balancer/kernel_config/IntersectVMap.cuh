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
 * \date    2014-08-28
 *
 */
//----------------------------------------------------------------------/*
#ifndef GPU_VOXELS_OCTREE_LOAD_BALANCED_KERNEL_CONFIG_INTERSECT_VMAP_CUH_INCLUDED
#define GPU_VOXELS_OCTREE_LOAD_BALANCED_KERNEL_CONFIG_INTERSECT_VMAP_CUH_INCLUDED

#include <gpu_voxels/octree/DataTypes.h>
#include <gpu_voxels/octree/kernels/kernel_common.h>
#include <gpu_voxels/octree/NTreeData.h>
#include <gpu_voxels/octree/load_balancer/kernel_config/LoadBalance.cuh>
#include <gpu_voxels/helpers/BitVector.h>
#include <gpu_voxels/voxelmap/kernels/VoxelMapOperations.h>

namespace gpu_voxels {
namespace NTree {
namespace LoadBalancer {

__device__ __forceinline__
void getNextCoordinates(const gpu_voxels::Vector3ui coordinate, uint32_t child, uint8_t level,
                        gpu_voxels::Vector3ui& new_coordinate_min, gpu_voxels::Vector3ui& new_coordinate_max)
{
  const uint32_t coordinate_step = uint32_t(const_cube_side_length[level]);
  gpu_voxels::Vector3ui shift;
  shift.x = (child & 0x01) * coordinate_step;
  shift.y = ((child & 0x02) >> 1) * coordinate_step;
  shift.z = ((child & 0x04) >> 2) * coordinate_step;

  new_coordinate_max = coordinate + gpu_voxels::Vector3ui(coordinate_step) + shift;
  new_coordinate_min = coordinate + shift;
}

__device__
__forceinline__
void check_border(const gpu_voxels::Vector3ui& ntree_min, const gpu_voxels::Vector3ui& ntree_max,
                  const gpu_voxels::Vector3ui& voxelmap_min, const gpu_voxels::Vector3ui& voxelmap_max, bool& is_inside,
                  bool& is_at_border)
{
  // separating axis theorem (OBBTree: A hierarchical structure for rapid interference detection)
  is_at_border = ntree_min < voxelmap_max && voxelmap_min < ntree_max;
//  is_at_border = !(ntree_min.x >= voxelmap_max.x || voxelmap_min.x >= ntree_max.x
//      || ntree_min.y >= voxelmap_max.y || voxelmap_min.y >= ntree_max.y || ntree_min.z >= voxelmap_max.z
//      || voxelmap_min.z >= ntree_max.z);

  is_inside = voxelmap_min <= ntree_min && ntree_max <= voxelmap_max;
}

/**
 * @brief This struct defines the shared memory, variables, kernel functions etc. needed to do a collision check between an \code NTree \endcode and a \code VoxelMap \endcode with help of the load balancing concept.
 * @tparam vft_size Size parameter to use for \code BitVector \endcode template. Defines the size in Byte of the voxel-meaning bit-vector.
 * @tparam set_collision_flag \code true \endcode to set the collision flag if necessary
 * @tparam compute_voxelTypeFlags \code true \endcode to compute the voxel meaning flags. Each bit of this vector indicates whether a voxel of the corresponsing meaning caused a collision.
 * @tparam VoxelType The meaning of a voxel of the corresponsing \code VoxelMap \endcode
 */
template<std::size_t num_threads,
  std::size_t branching_factor,
  typename InnerNode,
  typename LeafNode,
  int vtf_size,
  bool set_collision_flag,
  bool compute_voxelTypeFlags,
  class VoxelType>
struct IntersectVMapKernelConfig: public AbstractKernelConfig<WorkItemIntersectVoxelMap<InnerNode>,
    num_threads, branching_factor>
{
public:
  typedef WorkItemIntersectVoxelMap<InnerNode> WorkItem;
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
    BitVector<vtf_size> my_flags;

    __host__ __device__
    VariablesConfig() :
        my_num_collisions(0)
    {
        if(compute_voxelTypeFlags)
            my_flags.clear();;
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
                            p_stack_size_per_task + num_threads / branching_factor - num_threads)
    {

    }
  };

  struct KernelParameters: public Base::AbstractKernelParameters
  {
  public:
    std::size_t* num_collisions;
    const gpu_voxels::Vector3i offset;
    const VoxelType* voxel_map;
    const gpu_voxels::Vector3ui voxel_map_dim;
    const uint32_t min_level;
    BitVector<vtf_size>* result_voxelTypeFlags;

    __host__ __device__
    KernelParameters(const typename Base::AbstractKernelParameters& abstract_params,
                     std::size_t* p_num_collisions,
                     const gpu_voxels::Vector3i p_offset,
                     const VoxelType* p_voxel_map,
                     const gpu_voxels::Vector3ui p_voxel_map_dim,
                     const uint32_t p_min_level,
                     BitVector<vtf_size>* p_result_voxelTypeFlags) :
        Base::AbstractKernelParameters(abstract_params),
        num_collisions(p_num_collisions),
        offset(p_offset),
        voxel_map(p_voxel_map),
        voxel_map_dim(p_voxel_map_dim),
        min_level(p_min_level),
        result_voxelTypeFlags(p_result_voxelTypeFlags)
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
      InnerNode* node;
      bool node_active = false, chk_border = false;
      uint8_t level;
      gpu_voxels::Vector3ui coordinates;
      if (variables.is_active)
      {
        node_active = shared_mem->work_item_cache[constants.work_index].active;
        node = shared_mem->work_item_cache[constants.work_index].node;
        node += constants.work_lane * node_active;
        chk_border = shared_mem->work_item_cache[constants.work_index].check_border;
        level = shared_mem->work_item_cache[constants.work_index].level;

        is_last_level = (level == 1);
        insert_work_item = node->isOccupied() & !is_last_level & (node->hasStatus(ns_PART) | !node_active);
        if (insert_work_item)
        {
          // compute coordinates of this inner node
          gpu_voxels::Vector3ui min, max;
          getNextCoordinates(shared_mem->work_item_cache[constants.work_index].coordinates, constants.work_lane, level, min, max);
          coordinates = min;

          if (chk_border)
          {
            // check whether this inner node is inside the VoxelMap
            bool is_inside, is_at_border;
            check_border(min, max, kernel_params.offset, kernel_params.offset + kernel_params.voxel_map_dim, is_inside, is_at_border);
            insert_work_item = is_inside | is_at_border;
            chk_border = !is_inside;
          }
        }
      }

      // handle leaf nodes
      if (__syncthreads_or(is_last_level))
      {
        InnerNode* temp_node;
        const uint32_t leafs_per_work_item = branching_factor * branching_factor;
        for (uint32_t i = constants.thread_id; i < variables.num_work_items * leafs_per_work_item; i += num_threads)
        {
          const uint32_t my_work_item = i / leafs_per_work_item;
          const uint32_t my_inner_node = (i % leafs_per_work_item) / branching_factor;
          temp_node = shared_mem->work_item_cache[my_work_item].node;
          const bool temp_active = shared_mem->work_item_cache[my_work_item].active;
          temp_node += my_inner_node * temp_active;
          bool chk_border = shared_mem->work_item_cache[my_work_item].check_border;
          const uint8_t level = shared_mem->work_item_cache[my_work_item].level;

          if ((level == 1) && temp_node->isOccupied()
              && (temp_node->hasStatus(ns_PART) || !shared_mem->work_item_cache[my_work_item].active))
          {
            gpu_voxels::Vector3ui min, max;
            getNextCoordinates(shared_mem->work_item_cache[my_work_item].coordinates, my_inner_node, level, min, max);
            gpu_voxels::Vector3ui coordinates = min;
            bool is_inside = true, is_at_border = true;
            if (chk_border)
            {
              // check whether this inner node is inside the VoxelMap
              check_border(min, max, kernel_params.offset, kernel_params.offset + kernel_params.voxel_map_dim, is_inside, is_at_border);
              chk_border = !is_inside;
            }
            if (is_inside | is_at_border)
            {
              const uint32_t my_leaf_node = i % branching_factor;
              getNextCoordinates(coordinates, my_leaf_node, 0, min, max);
              coordinates = min;
              if (chk_border)
              {
                // check whether this inner node is inside the VoxelMap
                check_border(min, max, kernel_params.offset, kernel_params.offset + kernel_params.voxel_map_dim, is_inside, is_at_border);
              }
              if (is_inside | is_at_border)
              {
                const VoxelType* my_voxel = (VoxelType*) gpu_voxels::voxelmap::getVoxelPtr(kernel_params.voxel_map, &kernel_params.voxel_map_dim,
                                                                          coordinates.x - kernel_params.offset.x,
                                                                          coordinates.y - kernel_params.offset.y,
                                                                          coordinates.z - kernel_params.offset.z);
                LeafNode* leaf = NULL;
                bool is_in_conflict = true;
                bool has_leaf = temp_node->hasStatus(ns_PART) && 0 >= kernel_params.min_level;
                if (has_leaf)
                {
                  leaf = &((LeafNode*) temp_node->getChildPtr())[my_leaf_node];
                  is_in_conflict = leaf->isOccupied();
                }

                if (is_in_conflict && isVoxelOccupied(my_voxel))
                {
                  if (set_collision_flag)
                  {
                    if (has_leaf)
                      leaf->setStatus(leaf->getStatus() | ns_COLLISION);
                    else
                      temp_node->setStatus(temp_node->getStatus() | ns_COLLISION);
                  }
                  if (compute_voxelTypeFlags)
                  {
                    variables.my_flags |= my_voxel->voxel_meaning;
                  }
                  ++variables.my_num_collisions;
                }
              }
            }
          }
        }
      }

      uint32_t allVotes = thread_prefix<num_threads / WARP_SIZE>(shared_volatile_mem->block_votes, constants.thread_id, insert_count_tid0,
                                                              insert_work_item);

      // add new work items to stack
      if (variables.is_active & insert_work_item)
      {
        const uint32_t warpLocalIndex = __popc(allVotes << (WARP_SIZE - constants.warp_lane));
        const uint32_t interWarpIndex = shared_volatile_mem->block_votes[constants.warp_id];
        WorkItem* stackPointer = &shared_mem->my_work_stack[shared_mem->num_stack_work_items + interWarpIndex + warpLocalIndex];

        node_active = node->hasStatus(ns_PART) && kernel_params.min_level <= (level - 1);
        InnerNode* children = node_active ? ((InnerNode*) node->getChildPtr()) : node;
        *stackPointer = WorkItem(children, coordinates, level - 1, chk_border, node_active);
      }
      __syncthreads();

      if (constants.thread_id == 0)
        shared_mem->num_stack_work_items += insert_count_tid0;
      __syncthreads();
  }

  __device__
  static void doReductionWork(SharedMem* const shared_mem, volatile SharedVolatileMem* const shared_volatile_mem,
                              Variables& variables, Constants& constants, KernelParams& kernel_params)
  {      
    if (variables.my_num_collisions != 0)
        atomicAdd(kernel_params.num_collisions, variables.my_num_collisions);

    if (compute_voxelTypeFlags)
    {
        if (!variables.my_flags.isZero())
          BitVector<vtf_size>::reduceAtomic(variables.my_flags, *kernel_params.result_voxelTypeFlags);
    }

    // todo is a shared memory reduction more efficient in this case?
    if (variables.my_num_collisions != 0)
      atomicAdd((unsigned long long int*) kernel_params.num_collisions,
                (unsigned long long int) variables.my_num_collisions);
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
