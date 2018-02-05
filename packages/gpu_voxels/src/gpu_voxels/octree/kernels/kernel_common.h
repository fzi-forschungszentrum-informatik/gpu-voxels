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
 * \date    2014-04-17
 *
 */
//----------------------------------------------------------------------
#ifndef KERNEL_COMMON_H_
#define KERNEL_COMMON_H_

#include <gpu_voxels/octree/EnvironmentNodes.h>
#include <gpu_voxels/octree/EnvNodesProbabilistic.h>
#include <gpu_voxels/octree/DataTypes.h>
#include <gpu_voxels/octree/Nodes.h>

#include <gpu_voxels/voxel/BitVoxel.h>
#include <gpu_voxels/voxel/ProbabilisticVoxel.h>

#if CUDA_VERSION < 9000
#define CUB_NS_PREFIX namespace thrust { namespace system { namespace cuda { namespace detail {
#define CUB_NS_POSTFIX                  }                  }                }                  }
#define cub cub_
#include <thrust/system/cuda/detail/cub/util_type.cuh>
#undef cub
#undef CUB_NS_PREFIX
#undef CUB_NS_POSTFIX
namespace cub = thrust::system::cuda::detail::cub_;
#else // Cuda 9 or higher
#define THRUST_CUB_NS_PREFIX namespace thrust {   namespace cuda_cub {
#define THRUST_CUB_NS_POSTFIX }  }
#include <thrust/system/cuda/detail/cub/util_type.cuh>
#undef CUB_NS_PREFIX
#undef CUB_NS_POSTFIX
namespace cub = thrust::cuda_cub::cub;
#endif

// __ballot has been replaced by __ballot_sync in Cuda9
#if(__CUDACC_VER_MAJOR__ >= 9)
#define FULL_MASK 0xffffffff
#define BALLOT(PREDICATE) __ballot_sync(FULL_MASK, PREDICATE)
#define ANYWARP(PREDICATE) __any_sync(FULL_MASK, PREDICATE)
#else
#define BALLOT(PREDICATE) __ballot(PREDICATE)
#define ANYWARP(PREDICATE) __any(PREDICATE)
#endif

#include <cuda_runtime.h>

namespace gpu_voxels {
namespace NTree {

#define extract_selection_size 256

// look-up table in constant memory for extracting only needed data of the NTree with kernel_extractTreeData().
__constant__ uint8_t const_extract_selection[extract_selection_size];
#define const_voxel_at_level_size 20
// number of elements at each tree level
__constant__ OctreeVoxelID const_voxel_at_level[const_voxel_at_level_size]; // max 20 level = 60 bit
__constant__ uint32_t const_cube_side_length[const_voxel_at_level_size]; // max 20 level = 60 bit

//#define MORTON_LOOKUP_SIZE 8
//// look-up table for bit-splitting of morton-code computation
//__constant__ uint16_t const_morton_splitting_lookup[MORTON_LOOKUP_SIZE];
//
//__device__ __forceinline__
//static uint16_t* getMortonLookupPtr(const uint32_t x, const uint32_t y, const uint32_t z)
//{
//return const_morton_lookup + x + third_root(BRANCHING_FACTOR) * y + third_root(BRANCHING_FACTOR) * y * z;
//}

__device__ __forceinline__
void blockCopy(void* dst, void* src, const std::size_t size, const uint32_t idx, const uint32_t nThreads)
{
#pragma unroll
  for (uint32_t i = idx; i < (size >> 2); i += nThreads)
    ((uint32_t*) dst)[i] = ((uint32_t*) src)[i];

  if (idx < (size & 0x03))
    ((char*) dst)[size - idx - 1] = ((char*) src)[size - idx - 1];
}

template<int NUM_WARPS>
__device__ __forceinline__
uint32_t thread_prefix(volatile uint32_t* shr_sum, const uint32_t tid, uint32_t& sum,
                                                  const bool pred)
{
  assert(tid < (32 * 32));
  uint32_t warp_votes = BALLOT(pred); // warp vote
  if (tid % WARP_SIZE == tid / WARP_SIZE)
    shr_sum[tid / WARP_SIZE] = __popc(warp_votes); // population count
  if(NUM_WARPS > 1)
    __syncthreads();

  // exclusive sequential prefix sum
  if (tid == 0)
  {
    sum = 0;
#pragma unroll
    for (int i = 0; i < NUM_WARPS; ++i)
    {
      uint32_t tmp = shr_sum[i];
      shr_sum[i] = sum;
      sum += tmp;
    }
  }
  if(NUM_WARPS > 1)
    __syncthreads();

  return warp_votes;
//
//  int index = shr_sum[tid / WARP_SIZE];
//  return index + __popc(warp_votes << (WARP_SIZE - (tid % WARP_SIZE)));
}

__device__ __forceinline__
uint32_t getThreadPrefix(volatile uint32_t* shared_warp_prefix, const uint32_t warp_votes,
                                                    const uint32_t tid)
{
  return shared_warp_prefix[tid / WARP_SIZE] + __popc(warp_votes << (WARP_SIZE - (tid % WARP_SIZE)));
}

__device__ __forceinline__
uint32_t getThreadPrefix_Inclusive(volatile uint32_t* shared_warp_prefix, const uint32_t warp_votes,
                                                    const uint32_t tid)
{
  return shared_warp_prefix[tid / WARP_SIZE] + __popc(warp_votes << (WARP_SIZE - 1 - (tid % WARP_SIZE)));
}

template<int NUM_WARPS>
__device__ __forceinline__
bool any_thread(const bool pred)
{
  if(NUM_WARPS > 1)
    return __syncthreads_or(pred);
  else
    return ANYWARP(pred);
}

template<std::size_t branching_factor, typename T1, typename T2>
__device__ __forceinline__ T1 warp_reduction(const T1 value, volatile T1* const shared_memory,
                                             const uint32_t thread_id, T2 reduction_op)
{
  const uint32_t index = thread_id % branching_factor;
  const uint32_t first_index = (thread_id / branching_factor) * branching_factor;

  shared_memory[thread_id] = value;

  if (index < 4)
    shared_memory[thread_id] = reduction_op(shared_memory[thread_id], shared_memory[thread_id + 4]);
  if (index < 2)
    shared_memory[thread_id] = reduction_op(shared_memory[thread_id], shared_memory[thread_id + 2]);
  if (index < 1)
    shared_memory[thread_id] = reduction_op(shared_memory[thread_id], shared_memory[thread_id + 1]);

  return shared_memory[first_index];
}

__device__ __forceinline__
bool isVoxelOccupied(const gpu_voxels::ProbabilisticVoxel* voxel)
{
  return voxel->occupancy() >= 50;;
}

template<std::size_t length>
__device__ __forceinline__
bool isVoxelOccupied(const gpu_voxels::BitVoxel<length>* voxel)
{
  return !voxel->bitVector().isZero();;
}

__host__ __device__ __forceinline__
void clearNode(Environment::InnerNode* n)
{
  n->setStatus(ns_STATIC_MAP);
  n->setFlags(nf_NEEDS_UPDATE);
}

__host__ __device__
__forceinline__ void clearNodeLastLevel(Environment::InnerNode* n)
{
  n->setStatus(ns_STATIC_MAP | ns_LAST_LEVEL);
  n->setFlags(nf_NEEDS_UPDATE);
}

__host__ __device__
__forceinline__ void clearNode(Environment::LeafNode* n)
{
  n->setStatus(Environment::LeafNode::INVALID_STATUS | ns_STATIC_MAP);
}

__host__ __device__
__forceinline__ void clearNodeLastLevel(Environment::LeafNode* n)
{
  clearNode(n);
}

__host__ __device__
__forceinline__ void setOccupied(Environment::InnerNode* n, void* childPtr)
{
  n->setStatus(n->getStatus() | ns_PART);
  n->setChildPtr(childPtr);
}

__host__ __device__
__forceinline__ void setOccupied(Environment::LeafNode* n, void* childPtr)
{
  n->setStatus((n->getStatus() & ~Environment::LeafNode::INVALID_STATUS) | ns_OCCUPIED);
}

__host__ __device__
__forceinline__ void insertNode(Environment::InnerNode* n)
{
  n->setStatus(ns_DYNAMIC_MAP);
  n->setFlags(nf_NEEDS_UPDATE);
}

__host__ __device__
__forceinline__ void insertNodeLastLevel(Environment::InnerNode* n)
{
  n->setStatus(ns_DYNAMIC_MAP | ns_LAST_LEVEL);
  n->setFlags(nf_NEEDS_UPDATE);
}

__host__ __device__
__forceinline__ void insertNode(Environment::LeafNode* n)
{
  n->setStatus(Environment::LeafNode::INVALID_STATUS | ns_DYNAMIC_MAP);
}

__host__ __device__
__forceinline__ void insertNodeLastLevel(Environment::LeafNode* n)
{
  insertNode(n);
}

template<typename T1, typename T2>
__host__ __device__
__forceinline__ void _topDownUpdate(T1* node, T2* parent_node, const NodeStatus top_down_status_mask,
                                    const NodeStatus status_mask)
{
  const NodeStatus node_status = (node->getStatus() & ~top_down_status_mask)
      | (parent_node->getStatus() & top_down_status_mask);
  assert(
      ((node_status & status_mask) == ns_UNKNOWN) | ((node_status & status_mask) == ns_FREE)
          | ((node_status & status_mask) == ns_OCCUPIED));
  node->setStatus(node_status);
}

__host__ __device__
__forceinline__ void topDownUpdate(Environment::InnerNode* node, Environment::InnerNode* parent_node,
                                   const NodeStatus top_down_status_mask, const NodeStatus status_mask)
{
  _topDownUpdate(node, parent_node, top_down_status_mask, status_mask);
}

__host__ __device__ __forceinline__
void topDownUpdate(Environment::LeafNode* node, Environment::InnerNode* parent_node,
                   const NodeStatus top_down_status_mask, const NodeStatus status_mask)
{
  _topDownUpdate(node, parent_node, top_down_status_mask, status_mask);
}

__host__ __device__
__forceinline__ void topDownSubtreeUpdate(Environment::InnerNode* node, Environment::InnerNode* parent_node,
                                          const NodeStatus top_down_status_mask, const NodeStatus status_mask)
{
  // nothing to do since the insert method removed already all children of this node and also doesn't set the ns_UPDATE_SUBTREE flag
}

__host__ __device__ __forceinline__
void topDownSubtreeUpdate(Environment::LeafNode* node, Environment::InnerNode* parent_node,
                          const NodeStatus top_down_status_mask, const NodeStatus status_mask)
{
  // nothing to do since the insert method removed already all children of this node and also doesn't set the ns_UPDATE_SUBTREE flag
}

struct BitAnd_op
{
  __device__ __forceinline__ NodeStatus operator()(const NodeStatus a, const NodeStatus b)
  {
    return a & b;
  }
};

struct BitOr_op
{
  __device__ __forceinline__ NodeStatus operator()(const NodeStatus a, const NodeStatus b)
  {
    return a | b;
  }
};

// Needed for checking the NTree sequentially
__host__ __device__
bool isValidParentStatus(void* nodes, const uint32_t node_count, const uint8_t level,
                                    Environment::InnerNode* parent)
{
  const uint8_t status_mask = ~(ns_PART | ns_COLLISION | ns_LAST_LEVEL);
  uint8_t childs_status_flags = 0x00;
  for (uint32_t c = 0; c < node_count; ++c)
  {
    if (level == 0)
    {
      childs_status_flags |= ((Environment::LeafNode*) nodes)[c].getStatus();
    }
    else
      childs_status_flags |= ((Environment::InnerNode*) nodes)[c].getStatus();
  }
  childs_status_flags = (childs_status_flags & status_mask) | ns_PART;
  if (level == 0)
    childs_status_flags |= ns_LAST_LEVEL;
  return parent->getStatus() == childs_status_flags;
}

template<std::size_t branching_factor, typename T1, typename T2>
__device__
__forceinline__ void _bottomUpUpdate(T1* node, T2* parent_node, volatile uint8_t* shared_mem, const uint32_t thread_id,
                                     const uint8_t parent_level)
{
//#define USE_BALLOT

#ifdef USE_BALLOT
//  // #### Code is fast but cuda has problems with multiple __ballot() after each other ####
//  // no solution found for this problem
  assert(branching_factor == 8);
  //printf("%u input %u | %u\n", threadIdx.x, uint32_t(status), uint32_t(status & ns_OCCUPIED));

  const uint32_t work_lane_mask = (1 << branching_factor) - 1;
  const uint32_t log_free = cub::Log2<ns_FREE>::VALUE;
  const uint32_t log_unknown = cub::Log2<ns_UNKNOWN>::VALUE;
  const uint32_t log_occupied = cub::Log2<ns_OCCUPIED>::VALUE;
  const uint32_t log_static = cub::Log2<ns_STATIC_MAP>::VALUE;
  const uint32_t log_dynamic = cub::Log2<ns_DYNAMIC_MAP>::VALUE;
  const NodeStatus status = node->getStatus();

  // ##### Ballot has a problem if its called right after another #####
  // Leads to wrong _ballot() result values
  // printf() before/after/between solves the problem !?
  // works after some refactoring with no substantial change!!
  const uint32_t free_votes = BALLOT(status & ns_FREE);
  const uint32_t unknown_votes = BALLOT(status & ns_UNKNOWN);
  const uint32_t occupied_votes = BALLOT(status & ns_OCCUPIED);
  const uint32_t static_votes = BALLOT(status & ns_STATIC_MAP);
  const uint32_t dynamic_votes = BALLOT(status & ns_DYNAMIC_MAP);

  if ((thread_id % branching_factor) == 0)
  {
    //printf("%u ballot %u\n", threadIdx.x, occupied_votes);
    const uint32_t warp_lane = thread_id % WARP_SIZE;
    const NodeStatus my_free_votes = (free_votes >> warp_lane) & work_lane_mask;
    const NodeStatus my_unknown_votes = (unknown_votes >> warp_lane) & work_lane_mask;
    const NodeStatus my_occupied_votes = (occupied_votes >> warp_lane) & work_lane_mask;
    const NodeStatus my_static_votes = (static_votes >> warp_lane) & work_lane_mask;
    const NodeStatus my_dynamic_votes = (dynamic_votes >> warp_lane) & work_lane_mask;

    const NodeStatus my_status_or = ((my_free_votes > 0) << log_free)
    | ((my_unknown_votes > 0) << log_unknown) | ((my_occupied_votes > 0) << log_occupied)
    | ((my_static_votes > 0) << log_static) | ((my_dynamic_votes > 0) << log_dynamic);

    // my_static_votes and my_dynamic_votes not needed for and-operation
    const NodeStatus my_status_and = ((my_free_votes == work_lane_mask) << log_free)
    | ((my_unknown_votes == work_lane_mask) << log_unknown)
    | ((my_occupied_votes == work_lane_mask) << log_occupied);

    const NodeStatus new_status = getNewStatus(my_status_or, my_status_and, parent_level);
    parent_node->setStatus(new_status);
  }
#else
  const NodeStatus status = node->getStatus();

  volatile NodeStatus* my_shared_mem = (volatile NodeStatus*) shared_mem;
  const NodeStatus my_status_and = warp_reduction<branching_factor>(status, my_shared_mem, thread_id,
                                                                    BitAnd_op());
  const NodeStatus my_status_or = warp_reduction<branching_factor>(status, my_shared_mem, thread_id,
                                                                   BitOr_op());
  if ((thread_id % branching_factor) == 0)
  {
    const NodeStatus new_status = getNewStatus(my_status_or, my_status_and, parent_level);
    parent_node->setStatus(new_status);
  }
#endif

  //  const int mem_index = threadIdx.x / 8;
  //  const bool do_final_computation = (thread_id % branching_factor) == 0;
  //  const NodeStatus status = node->getStatus();
  //  if (do_final_computation)
  //  {
  //    shared_mem[mem_index] = 0xFF;
  //  }
  //  atomicAnd(&shared_mem[mem_index], status);
  //  const NodeStatus my_status_and = NodeStatus(shared_mem[mem_index]);
  //  if (do_final_computation)
  //  {
  //    shared_mem[mem_index] = 0;
  //  }
  //  atomicOr(&shared_mem[mem_index], status);
  //  const NodeStatus my_status_or = NodeStatus(shared_mem[mem_index]);
  //
  //  if (do_final_computation)
  //  {
  //    const NodeStatus new_status = getNewStatus(my_status_or, my_status_and, parent_level);
  //    parent_node->setStatus(new_status);
  //  }
#undef USE_BALLOT
}

template<std::size_t branching_factor>
__device__
__forceinline__ void bottomUpUpdate(Environment::LeafNode* node, Environment::InnerNode* parent_node,
                                    volatile uint8_t* shared_mem, const uint32_t thread_id)
{
  _bottomUpUpdate<branching_factor>(node, parent_node, shared_mem, thread_id, 1);
}

template<std::size_t branching_factor>
__device__
__forceinline__ void bottomUpUpdate(Environment::InnerNode* node, Environment::InnerNode* parent_node,
                                    volatile uint8_t* shared_mem, const uint32_t thread_id)
{
  _bottomUpUpdate<branching_factor>(node, parent_node, shared_mem, thread_id, 99);
}

__host__ __device__
__forceinline__ void setNode(Environment::LeafNode* node,
                             const Environment::LeafNode::NodeData::BasicData set_basic_data,
                             const Environment::LeafNode::NodeData::BasicData reset_basic_data)
{
  node->setStatus((node->getStatus() & ~reset_basic_data.m_status) | set_basic_data.m_status);
}

__host__ __device__
__forceinline__ void updateNode(Environment::LeafNode* node,
                                const Environment::LeafNode::NodeData::BasicData set_basic_data,
                                const Environment::LeafNode::NodeData::BasicData reset_basic_data)
{
  setNode(node, set_basic_data, reset_basic_data);
}

__host__ __device__
__forceinline__ void _setNode(Environment::InnerNode* node, void* const childPtr,
                              const Environment::InnerNode::NodeData::BasicData set_basic_data,
                              const Environment::InnerNode::NodeData::BasicData reset_basic_data)
{
  NodeStatus s = node->getStatus();
  if (childPtr == NULL)
    s = (s & ~(reset_basic_data.m_status)) | set_basic_data.m_status;
  else
  {
    // TODO FREE TREE handle this case since a part of the tree is unlinked!
    node->setChildPtr(childPtr);
    s = s | ns_PART;
  }
  node->setStatus(s);
}

__host__ __device__
__forceinline__ void setNode(Environment::InnerNode* node, void* const childPtr,
                             const Environment::InnerNode::NodeData::BasicData set_basic_data,
                             Environment::InnerNode::NodeData::BasicData reset_basic_data)
{
  //reset_basic_data.m_status |= ns_PART;
  _setNode(node, childPtr, set_basic_data, reset_basic_data);
}

__host__ __device__
__forceinline__ void updateNode(Environment::InnerNode* node, void* const childPtr,
                                const Environment::InnerNode::NodeData::BasicData set_basic_data,
                                const Environment::InnerNode::NodeData::BasicData reset_basic_data)
{
  setNode(node, childPtr, set_basic_data, reset_basic_data);
}

__host__ __device__ __forceinline__
void getRayCastInit(Environment::InnerNode::RayCastType* const init)
{
  init->value = 0;
}

__host__ __device__ __forceinline__
void getRayCastInit(Environment::NodeProb::RayCastType* const init)
{
  init->value = INITIAL_PROBABILITY;
}

__device__ __forceinline__
void handleRayHit(Environment::InnerNode::RayCastType* a, int32_t x, int32_t y, int32_t z)
{
  a->value = ns_FREE;
}

__device__ __forceinline__
bool isValidValue(const Environment::NodeProb::RayCastType& a)
{
  Environment::NodeProb::RayCastType init;
  getRayCastInit(&init);
  return a.value != init.value;
}

__device__ __forceinline__
bool isValidValue(const Environment::InnerNode::RayCastType& a)
{
  Environment::InnerNode::RayCastType init;
  getRayCastInit(&init);
  return a.value != init.value;
}

template<typename InnerNode>
struct Comp_is_valid
{
  __host__ __device__ __forceinline__
  bool operator()(const typename InnerNode::RayCastType& x)
  {
    return isValidValue(x);
  }
};

__device__ __forceinline__
bool isPackingPossible(const Environment::InnerNode::RayCastType max,
                       const Environment::InnerNode::RayCastType min)
{
  typedef Environment::InnerNode::RayCastType::Type T;
  Environment::InnerNode::RayCastType init;
  getRayCastInit(&init);
  return max.value == min.value && min.value != init.value;
}

__device__ __forceinline__
bool maxMinReduction(const Environment::InnerNode::RayCastType my_value,
                     volatile Environment::InnerNode::RayCastType* shared_memory, const uint32_t thread_id,
                     const uint32_t num_threads, const uint32_t branching_factor, const bool is_active,
                     Environment::InnerNode::RayCastType* max_value,
                     Environment::InnerNode::RayCastType* min_value)
{
  const uint32_t warp_lane = thread_id % WARP_SIZE;
  volatile uint16_t* my_shared_mem = (volatile uint16_t*) shared_memory;

  const bool is_valid = is_active && isValidValue(my_value);
  const uint32_t warp_votes = BALLOT(is_valid);

  if (branching_factor > WARP_SIZE)
  {
    // block-local computation
    if (warp_lane == 0)
      my_shared_mem[thread_id / WARP_SIZE] = __popc(warp_votes);

    const bool all_invalid = __syncthreads_and(!is_valid);
    if (all_invalid)
      return true; // nothing to do

    PARTIAL_REDUCE(my_shared_mem, thread_id, num_threads, branching_factor / WARP_SIZE, +);

    max_value->value = min_value->value = (my_shared_mem[thread_id / branching_factor] == branching_factor);
  }
  else
  {
    // warp-local computation
    const uint32_t work_id = warp_lane / branching_factor;
    const uint32_t work_mask = (1 << branching_factor) - 1;
    const uint32_t my_work_votes = (warp_votes >> (work_id * branching_factor)) & work_mask;
    max_value->value = min_value->value = (my_work_votes == work_mask);

    const bool all_invalid = __syncthreads_and(!is_valid);
    if (all_invalid)
      return true; // nothing to do
  }
  return false;
}

__device__ __forceinline__
void packData(Environment::InnerNode::NodeData::BasicData* ptr, const uint32_t pos,
              const Environment::InnerNode::RayCastType max, const Environment::InnerNode::RayCastType min,
              const Environment::InnerNode::RayCastType value)
{
  // nothing to do
  // OctreeVoxelID is enough to know whether the voxel is free or not
  ptr[pos] = Environment::InnerNode::NodeData::BasicData(value.value, 0);
}

__device__ __forceinline__
void packData(Environment::InnerNode::RayCastType* ptr, const Environment::InnerNode::RayCastType max,
              const Environment::InnerNode::RayCastType min, const Environment::InnerNode::RayCastType value)
{
  *ptr = value;
}

__host__ __device__
void getFreeValue(Environment::InnerNode::RayCastType* ptr)
{
  ptr->value = ns_FREE;
}

__host__ __device__
Environment::LeafNode getLeafNode(Environment::InnerNode* ptr)
{
  Environment::LeafNode tmp;
  tmp.setStatus(ptr->getStatus());
  return tmp;
}


// #####################################################
// ############### EnvNodesProbabilistic ###############
// #####################################################

__host__ __device__ __forceinline__
void clearNode(Environment::InnerNodeProb* n)
{
  clearNode(static_cast<Environment::InnerNode*>(n));
  n->setOccupancy(INITIAL_PROBABILITY);
}

__host__ __device__
__forceinline__ void clearNodeLastLevel(Environment::InnerNodeProb* n)
{
  clearNodeLastLevel(static_cast<Environment::InnerNode*>(n));
  n->setOccupancy(INITIAL_PROBABILITY);
}

__host__ __device__
__forceinline__ void clearNode(Environment::LeafNodeProb* n)
{
  clearNode(static_cast<Environment::LeafNode*>(n));
  n->setOccupancy(INITIAL_PROBABILITY);
}

__host__ __device__
__forceinline__ void clearNodeLastLevel(Environment::LeafNodeProb* n)
{
  clearNode(n);
}

__host__ __device__
__forceinline__ void setOccupied(Environment::InnerNodeProb* n, void* childPtr)
{
  setOccupied(static_cast<Environment::InnerNode*>(n), childPtr);
  //n->setOccupancy(MAX_PROBABILITY);
}

__host__ __device__
__forceinline__ void setOccupied(Environment::LeafNodeProb* n, void* childPtr)
{
  n->setStatus((n->getStatus() & ~Environment::LeafNode::INVALID_STATUS));
  //setOccupied(static_cast<Environment::LeafNode*>(n), childPtr);
  n->setOccupancy(MAX_PROBABILITY);
}

__host__ __device__
__forceinline__ void insertNode(Environment::InnerNodeProb* n)
{
  insertNode(static_cast<Environment::InnerNode*>(n));
  n->setOccupancy(INITIAL_PROBABILITY);
}

__host__ __device__
__forceinline__ void insertNodeLastLevel(Environment::InnerNodeProb* n)
{
  insertNodeLastLevel(static_cast<Environment::InnerNode*>(n));
  n->setOccupancy(INITIAL_PROBABILITY);
}

__host__ __device__
__forceinline__ void insertNode(Environment::LeafNodeProb* n)
{
  insertNode(static_cast<Environment::LeafNode*>(n));
  n->setOccupancy(INITIAL_PROBABILITY);
}

__host__ __device__
__forceinline__ void insertNodeLastLevel(Environment::LeafNodeProb* n)
{
  insertNode(n);
}

__host__ __device__ __forceinline__
Probability updateOccupancy(const Probability old_value, const Probability new_value)
{
  // watch out for overflow: cast to int32_t
  return min(max(int32_t(int32_t(old_value) + int32_t(new_value)), int32_t(MIN_PROBABILITY)), int32_t(MAX_PROBABILITY));
}

//template<typename Node>
//__host__ __device__ __forceinline__
//bool _isOccupied(Node* const node)
//{
//  return node->getOccupancy() >= THRESHOLD_OCCUPANCY;
//}
//
//__host__ __device__ __forceinline__
//bool isOccupied(Environment::InnerNodeProb* const node)
//{
//  return _isOccupied(node);
//}
//
//__host__ __device__ __forceinline__
//bool isOccupied(Environment::LeafNodeProb* const node)
//{
//  return _isOccupied(node);
//}

__host__ __device__ __forceinline__
bool isUnknown(Probability const prob)
{
  return prob == UNKNOWN_PROBABILITY;
}

template<typename Node>
__host__ __device__
__forceinline__ void _updateStatus(Node* const node)
{
  // set node occupied/free according to occupancy
  NodeStatus s = ns_UNKNOWN;
  // ## ignore flags and always set to unknown to mark this node as valid ##
//  const Probability p = node->getOccupancy();
//  if (!isUnknown(p))
//    s = (p >= THRESHOLD_OCCUPANCY) ? ns_OCCUPIED : ns_FREE;
  node->setStatus((node->getStatus() & ~(ns_OCCUPIED | ns_FREE | ns_UNKNOWN)) | s);
}

__host__ __device__
__forceinline__ void topDownUpdate(Environment::InnerNodeProb* node, Environment::InnerNodeProb* parent_node,
                                   const NodeStatus top_down_status_mask, const NodeStatus status_mask)
{
//  topDownUpdate(static_cast<Environment::InnerNode*>(node), static_cast<Environment::InnerNode*>(parent_node),
//                top_down_status_mask, status_mask);
  node->setOccupancy(parent_node->getOccupancy());
  _updateStatus(node);
}

__host__ __device__
__forceinline__ void topDownUpdate(Environment::LeafNodeProb* node, Environment::InnerNodeProb* parent_node,
                                   const NodeStatus top_down_status_mask, const NodeStatus status_mask)
{
//  topDownUpdate(static_cast<Environment::LeafNode*>(node), static_cast<Environment::InnerNode*>(parent_node),
//                top_down_status_mask, status_mask);
  node->setOccupancy(parent_node->getOccupancy());
  _updateStatus(node);
}

/**
 * Update instead of set the occupancy of the subtree.
 */__host__ __device__
__forceinline__ void topDownSubtreeUpdate(Environment::InnerNodeProb* node,
                                          Environment::InnerNodeProb* parent_node,
                                          const NodeStatus top_down_status_mask, const NodeStatus status_mask)
{
   const Probability occupancy_parent = parent_node->getOccupancy();
   Probability p = occupancy_parent; // override existing probability, since it gets restore with the bottom-up step
   if(!node->hasStatus(ns_PART))
     p = updateOccupancy(node->getOccupancy(), occupancy_parent);
   node->setOccupancy(p);
//   node->setOccupancy(parent_node->getOccupancy());
}

/**
 * Update instead of set the occupancy of the subtree.
 */__host__ __device__
__forceinline__ void topDownSubtreeUpdate(Environment::LeafNodeProb* node,
                                          Environment::InnerNodeProb* parent_node,
                                          const NodeStatus top_down_status_mask, const NodeStatus status_mask)
{
   //node->setOccupancy(parent_node->getOccupancy());
   node->setOccupancy(updateOccupancy(node->getOccupancy(), parent_node->getOccupancy()));
}

struct _Occupancy_aggregate_op
{
  __device__ __forceinline__ Probability operator()(const Probability a, const Probability b)
  {
    // attention: prefers unknown space over free space
    return max(int(a), int(b));
  }
};

// Needed for checking the NTree sequentially
__host__ __device__
bool isValidParentStatus(void* nodes, const uint32_t node_count, const uint8_t level,
                         Environment::InnerNodeProb* parent)
{
  //printf("level %u\n", level);
  Probability p_max = UNKNOWN_PROBABILITY;
  Probability p_min = MAX_PROBABILITY;
  NodeStatus check_mask =  ns_LAST_LEVEL | ns_PART;
  NodeStatus array[8];
  bool all_non_part = true;
  for (uint32_t c = 0; c < node_count; ++c)
  {
    Probability tmp;
    NodeStatus st;
    if (level == 0){
      tmp = ((Environment::LeafNodeProb*) nodes)[c].getOccupancy();
      st = ((Environment::LeafNodeProb*) nodes)[c].getStatus();
      all_non_part &= true;
    }
    else{
      tmp = ((Environment::InnerNodeProb*) nodes)[c].getOccupancy();
      all_non_part &= !((Environment::InnerNodeProb*) nodes)[c].hasStatus(ns_PART);
      st = ((Environment::InnerNodeProb*) nodes)[c].getStatus();
    }
    array[c] = st;

    p_max = (Probability)max((int)tmp, (int)p_max);
    p_min = (Probability)min((int)tmp, (int)p_min);
  }

  NodeStatus s = 0;
  if (!(p_max == p_min && all_non_part))
    s |= ns_PART;
  if (level == 0)
    s |= ns_LAST_LEVEL;

  bool res = parent->getOccupancy() == p_max && (parent->getStatus() & check_mask) == s;
  if (!res){
    //printf("FAIL\n");
    printf("level %i p_max %i p_min %i parent %i parent status %u status %u\n", level, p_max, p_min, parent->getOccupancy(),
           parent->getStatus(), s);
    for(int i = 0; i < int(node_count); ++i)
      printf("node %i status %u\n", i, array[i]);
    printf("parent %p, first child %p\n", parent, nodes);
  }
  return res;
}

template<std::size_t branching_factor, typename T1, typename T2>
__device__ __forceinline__
void _bottomUpUpdateProb(T1* node, T2* parent_node, volatile uint8_t* shared_mem, const uint32_t thread_id)
{
// ### WARP local computation ###
  assert(branching_factor == 8);
  const uint32_t node_index = thread_id % branching_factor;
  const uint32_t parent_node_index = thread_id / branching_factor;
  const uint32_t parent_node_id = parent_node_index * branching_factor;
  const uint32_t parent_warp_index = parent_node_id % WARP_SIZE;
  volatile Probability* my_shared_mem = (volatile Probability*) shared_mem;
  const Probability my_occupancy = node->getOccupancy();

//  if(uint64_t(parent_node) == 0xb006bfc80){
//    printf("child occ %i status %u\n", my_occupancy, node->getStatus());
//  }

  const Probability max_occupancy = warp_reduction<branching_factor>(my_occupancy, my_shared_mem, thread_id,
                                                                     _Occupancy_aggregate_op());

//  my_shared_mem[thread_id] = my_occupancy;
//
//// max reduction
//  if (node_index < 4)
//    my_shared_mem[thread_id] = (Probability) max(my_shared_mem[thread_id], my_shared_mem[thread_id + 4]);
//  if (node_index < 2)
//    my_shared_mem[thread_id] = (Probability) max(my_shared_mem[thread_id], my_shared_mem[thread_id + 2]);
//  if (node_index < 1)
//    my_shared_mem[thread_id] = (Probability) max(my_shared_mem[thread_id], my_shared_mem[thread_id + 1]);

//  const Probability max_occupancy = my_shared_mem[parent_node_id];
  const uint32_t all_votes = BALLOT((max_occupancy == my_occupancy) & !(node->hasStatus(ns_PART)));

  if (node_index == 0)
  {
// do the final work
    const uint32_t work_lane_mask = (1 << branching_factor) - 1;
    const bool same_votes = ((all_votes >> parent_warp_index) & work_lane_mask) == work_lane_mask;

    const NodeStatus part_flag = same_votes ? 0 : ns_PART;
//    if(same_votes) printf("COMPACT\n");

//    if(uint64_t(parent_node) == 0xb006bfc80){
//       printf("votes %u tid %u\n", all_votes, thread_id);
//       printf("max_occupancy %i parent status %u\n", max_occupancy, parent_node->getStatus());
//    }

    parent_node->setOccupancy(max_occupancy);
    parent_node->setStatus((parent_node->getStatus() & ~ns_PART) | part_flag);
    _updateStatus(parent_node);
  }

//  if (node_index == 0)
//    shared_mem[parent_node_index] = MIN_PROBABILITY;
//  atomicMax(&shared_mem[parent_node_index], node->getOccupancy());
//  // TODO use/implement non-atomic max function
//
//  // compute min to check whether all child have the same value
//  Probability max = Probability(shared_mem[parent_node_index]);
//  if (node_index == 0)
//    shared_mem[parent_node_index] = max;
//  atomicMin(&shared_mem[parent_node_index], node->getOccupancy());
//
//  if (node_index == 0)
//  {
//    parent_node->setOccupancy((Probability) shared_mem[parent_node_index]);
//
//    // set/reset part flag and therefore cut the children or not
//    NodeStatus s = (max != Probability(shared_mem[parent_node_index])) ? ns_PART : 0;
//    parent_node->setStatus((parent_node->getStatus() & ~ns_PART) | s);
//  }
}

template<std::size_t branching_factor>
__device__
__forceinline__ void bottomUpUpdate(Environment::LeafNodeProb* node, Environment::InnerNodeProb* parent_node,
                                    volatile uint8_t* shared_mem, const uint32_t thread_id)
{
//  _updateStatus(node);
//  bottomUpUpdate<branching_factor>(static_cast<Environment::LeafNode*>(node),
//                                   static_cast<Environment::InnerNode*>(parent_node), shared_mem, thread_id);
  _bottomUpUpdateProb<branching_factor, Environment::LeafNodeProb, Environment::InnerNodeProb>(node,
                                                                                               parent_node,
                                                                                               shared_mem,
                                                                                               thread_id);
 // _updateStatus(node);
}

template<std::size_t branching_factor>
__device__
__forceinline__ void bottomUpUpdate(Environment::InnerNodeProb* node, Environment::InnerNodeProb* parent_node,
                                    volatile uint8_t* shared_mem, const uint32_t thread_id)
{
//  _updateStatus(node);
//  bottomUpUpdate<branching_factor>(static_cast<Environment::InnerNode*>(node),
//                                   static_cast<Environment::InnerNode*>(parent_node), shared_mem, thread_id);
  _bottomUpUpdateProb<branching_factor, Environment::InnerNodeProb, Environment::InnerNodeProb>(node,
                                                                                                parent_node,
                                                                                                shared_mem,
                                                                                                thread_id);
 // _updateStatus(node);
}

__host__ __device__
__forceinline__ void updateNode(Environment::LeafNodeProb* node,
                                const Environment::NodeProb::NodeData::BasicData set_basic_data,
                                const Environment::NodeProb::NodeData::BasicData reset_basic_data)
{
  //updateNode(static_cast<Environment::LeafNode*>(node), set_basic_data, reset_basic_data);

  Probability p = node->getOccupancy();
  if (isUnknown(p))
    p = 0;
  node->setOccupancy(updateOccupancy(p, set_basic_data.m_occupancy));
  _updateStatus(node);
}

__host__ __device__
__forceinline__ void setNode(Environment::LeafNodeProb* node,
                             const Environment::NodeProb::NodeData::BasicData set_basic_data,
                             const Environment::NodeProb::NodeData::BasicData reset_basic_data)
{
  //setNode(static_cast<Environment::LeafNode*>(node), set_basic_data, reset_basic_data);

  node->setOccupancy(set_basic_data.m_occupancy);
  _updateStatus(node);
}

__host__ __device__
__forceinline__ void setNode(Environment::InnerNodeProb* node, void* const childPtr,
                             const Environment::NodeProb::NodeData::BasicData set_basic_data,
                             const Environment::NodeProb::NodeData::BasicData reset_basic_data)
{
  setNode(static_cast<Environment::InnerNode*>(node), childPtr, set_basic_data, reset_basic_data);

  if (childPtr == NULL)
  {
    node->setOccupancy(set_basic_data.m_occupancy);
    _updateStatus(node);
  }
}

__host__ __device__ __forceinline__
void updateNode(Environment::InnerNodeProb* node, void* const childPtr,
                const Environment::NodeProb::NodeData::BasicData set_basic_data,
                const Environment::NodeProb::NodeData::BasicData reset_basic_data)
{
  // check whether the node to update contains children which have to be updated with this value too
  // and set the corresponding flag if necessary
  if (childPtr == NULL && !reset_basic_data.hasFlags(nf_UPDATE_SUBTREE) && node->hasStatus(ns_PART))
  {
    //printf("set nf_UPDATE_SUBTREE\n");
    node->setFlags(node->getFlags() | nf_UPDATE_SUBTREE);
  }

  // call set method without resetting the ns_PART flag
  //_setNode(static_cast<Environment::InnerNode*>(node), childPtr, set_basic_data, reset_basic_data);
  setNode(static_cast<Environment::InnerNode*>(node), childPtr, set_basic_data, reset_basic_data);

  if (childPtr == NULL)
  {
    node->setOccupancy(set_basic_data.m_occupancy);
    _updateStatus(node);
  }
}

__device__ __forceinline__
void handleRayHit(Environment::NodeProb::RayCastType* a, int32_t x, int32_t y, int32_t z)
{
  // Concurrent writes to the same cell/voxel can lead to incorrect results.
  // Locking with an atomic operation is too expensive for the runtime and needed memory.
  // A conflict shouldn't happen that often and the resulting data is quite good.
  Environment::NodeProb::RayCastType::Type current = a->value;
  // TODO: Apply advanced sensor model, which considers the distance to the sensor
  // watch out for overflow: cast to int32_t
  a->value = int32_t(current) + max(int32_t(FREE_UPDATE_PROBABILITY), int32_t(MIN_PROBABILITY) - int32_t(current));
}

__device__ __forceinline__
bool isPackingPossible(const Environment::NodeProb::RayCastType max,
                       const Environment::NodeProb::RayCastType min)
{
  return (max.value - min.value) == 0;
}

struct _Max_op
{
  __device__ __forceinline__ Probability operator()(const Probability a, const Probability b)
  {
    return max(int(a), int(b));
  }
};

struct _Min_op
{
  __device__ __forceinline__ Probability operator()(const Probability a, const Probability b)
  {
    return min(int(a), int(b));
  }
};

__device__ __forceinline__
bool maxMinReduction(const Environment::NodeProb::RayCastType my_value,
                     volatile Environment::NodeProb::RayCastType* shared_memory, const uint32_t thread_id,
                     const uint32_t num_threads, const uint32_t branching_factor, const bool is_active,
                     Environment::NodeProb::RayCastType* max_value,
                     Environment::NodeProb::RayCastType* min_value)
{
  volatile Environment::NodeProb::RayCastType::Type* const my_shared_memory =
      (Environment::NodeProb::RayCastType::Type*) shared_memory;
  const uint32_t reduction_index = (thread_id / branching_factor) * branching_factor;
  // max reduce
  PARTIAL_REDUCE(my_shared_memory, thread_id, num_threads, branching_factor, _Max_op());
  if (branching_factor > WARP_SIZE)
    __syncthreads();

  max_value->value = my_shared_memory[reduction_index];
  if (branching_factor > WARP_SIZE)
    __syncthreads();

//  // causes some trouble!
//  const bool all_invalid = __syncthreads_and(!is_active || !isValidValue(*max_value));
//  if (all_invalid)
//    return true; // nothing to do here

  // fill shared mem with thread data
  my_shared_memory[thread_id] = my_value.value;
  if (branching_factor > WARP_SIZE)
    __syncthreads();

  // min reduce
  PARTIAL_REDUCE(my_shared_memory, thread_id, num_threads, branching_factor, _Min_op());
  if (branching_factor > WARP_SIZE)
    __syncthreads();

  min_value->value = my_shared_memory[reduction_index];
  return false;
}

__device__ __forceinline__
void packData(Environment::NodeProb::NodeData::BasicData* ptr, const uint32_t pos,
              const Environment::NodeProb::RayCastType max, const Environment::NodeProb::RayCastType min,
              const Environment::NodeProb::RayCastType value)
{
  //ptr[pos].m_occupancy = max.value - (max.value - min.value) / 2;
  ptr[pos] = Environment::NodeProb::NodeData::BasicData(0, 0, value.value);
}

__device__ __forceinline__
void packData(Environment::NodeProb::RayCastType* ptr, const Environment::NodeProb::RayCastType max,
              const Environment::NodeProb::RayCastType min, const Environment::NodeProb::RayCastType value)
{
  *ptr = value; //max.value - (max.value - min.value) / 2;
}

__host__ __device__
void getFreeValue(Environment::NodeProb::RayCastType* ptr)
{
  ptr->value = MIN_PROBABILITY;
}

__host__ __device__
Environment::LeafNodeProb getLeafNode(Environment::InnerNodeProb* ptr)
{
  Environment::LeafNodeProb tmp;
  tmp.setStatus(ptr->getStatus());
  tmp.setOccupancy(ptr->getOccupancy());
  return tmp;
}

}
}

#endif /* KERNEL_COMMON_H_ */
