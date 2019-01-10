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
 * \date    2013-11-11
 *
 */
//----------------------------------------------------------------------

#ifndef KERNEL_OCTREE_H_
#define KERNEL_OCTREE_H_

#include <assert.h>
#include <stdio.h>
#include <cuda_runtime.h>

#include <gpu_voxels/octree/DataTypes.h>
#include <gpu_voxels/octree/Voxel.h>
#include <gpu_voxels/octree/Morton.h>
#include <gpu_voxels/octree/NTree.h>
#include <gpu_voxels/octree/kernels/kernel_common.h>
//#include "octree/Octree.h"
#include <gpu_voxels/octree/NTreeData.h>

#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/reduce.h>
#include <thrust/iterator/constant_iterator.h>

// gpu_voxels
#include <gpu_voxels/helpers/cuda_datatypes.h>
#include <gpu_voxels/voxelmap/kernels/VoxelMapOperations.h>

#include "shared_voxel.cuh"

namespace gpu_voxels {
namespace NTree {

/*
 * Counts the number of needed parent nodes to store this level. This is done by checking the prefix of their zorder IDs.
 */
template<std::size_t branching_factor, std::size_t level_count, typename InnerNode, typename LeafNode>
__global__
void kernel_countNodes(OctreeVoxelID* const voxel, const OctreeVoxelID numVoxel, const uint32_t level, OctreeVoxelID* const sum)
{
  const OctreeVoxelID chunk_size = ceil(double(numVoxel) / (gridDim.x * blockDim.x));
  const OctreeVoxelID id = (blockIdx.x * blockDim.x + threadIdx.x);
  const OctreeVoxelID from = chunk_size * id;
  const OctreeVoxelID to = (OctreeVoxelID) min((unsigned long long int) (from + chunk_size),
                                   (unsigned long long int) numVoxel);
  OctreeVoxelID neededNodes = 0;

  // handle idle threads
  if (from < numVoxel)
  {
    // left most thread takes care of nodes split across chunks
    OctreeVoxelID lastPrefix = getZOrderPrefix<branching_factor>(voxel[from], level);
    neededNodes = (from == 0 || getZOrderPrefix<branching_factor>(voxel[from - 1], level) != lastPrefix);

    for (OctreeVoxelID i = from + 1; i < to; ++i)
    {
      neededNodes += (getZOrderPrefix<branching_factor>(voxel[i], level) != lastPrefix);
      lastPrefix = getZOrderPrefix<branching_factor>(voxel[i], level);
    }
  }
  sum[id] = neededNodes;
}

/*
 * Kernel to set the newly allocated nodes to its default value (status to unknown)
 */
template<typename T, bool lastLevel>
__global__
void kernel_clearNodes(const voxel_count size, T* const nodes)
{
  // init nodes
  for (uint32_t i = blockIdx.x * blockDim.x + threadIdx.x; i < size; i += gridDim.x * blockDim.x)
  {
    nodes[i] = T();
    if (lastLevel)
      clearNodeLastLevel(&nodes[i]);
    else
      clearNode(&nodes[i]);
  }
}

/**
 * Kernel to set the occupied nodes, save the nodeIDs of the nodes for the next level
 * and set the child pointers for level > 0
 */
template<typename T1, typename T2, std::size_t branching_factor>
__global__
void kernel_setNodes(OctreeVoxelID* const voxel, const OctreeVoxelID numVoxel, const uint32_t level, OctreeVoxelID* const sum, T1* const nodes,
                     OctreeVoxelID* const nodeIds, T2* const childNodes)
{
  const uint32_t chunk_size = ceil(double(numVoxel) / (gridDim.x * blockDim.x));
  const OctreeVoxelID id = (blockIdx.x * blockDim.x + threadIdx.x);
  const OctreeVoxelID from = chunk_size * id;
  const OctreeVoxelID to = (OctreeVoxelID) min((unsigned long long int) (from + chunk_size),
                                   (unsigned long long int) numVoxel);

// handle idle threads
  if (from < numVoxel)
  {
    // left most thread takes care of nodes split across chunks
    OctreeVoxelID lastPrefix = getZOrderPrefix<branching_factor>(voxel[from], level);
    OctreeVoxelID node = (id == 0) ? 0 : sum[id - 1] - 1;
    node += (id == 0 || getZOrderPrefix<branching_factor>(voxel[from - 1], level) == lastPrefix) ? 0 : 1;

    for (uint32_t i = from; i < to; ++i)
    {
      node += (getZOrderPrefix<branching_factor>(voxel[i], level) != lastPrefix);
      const OctreeVoxelID index = branching_factor * node + getZOrderNodeId<branching_factor>(voxel[i], level);

      setOccupied(&nodes[index], childNodes + i * branching_factor);

      lastPrefix = getZOrderPrefix<branching_factor>(voxel[i], level);
      nodeIds[node] = voxel[i];
    }

#ifdef DEBUG_MODE
//    printf("A ID:%i %016llX\n", from, &nodes[from]);
//    printf("D ID:%i %016llX\n", from, *(voxel_int*) ((void*) &nodes[from]));
#endif
  }
}

template<std::size_t branching_factor, std::size_t level_count, typename InnerNode, typename LeafNode>
__global__
void kernel_print(InnerNode* root, InnerNode* stack1, InnerNode* stack2)
{
  int32_t stack1Top = -1;
  int32_t stack2Top = -1;
  stack1[++stack1Top] = *root;
  printf("L%i: ", level_count - 1);
  printf(" root[][%016llX]\n", root);
//printf(" root[][%016llX]\n", (*octree.root).data);

  for (int32_t level = level_count - 2; level >= 0; --level)
  {
    printf("L%i ####################### \n", level);
    while (stack1Top >= 0)
    {
      InnerNode node = stack1[stack1Top--];
      if (node.hasStatus(ns_PART))
      {
        assert(
            (level == 0 && node.hasStatus(ns_LAST_LEVEL)) || (level != 0 && !node.hasStatus(ns_LAST_LEVEL)));
        if (node.hasStatus(ns_LAST_LEVEL))
        {
          LeafNode* child = (LeafNode*) node.getChildPtr();
          printf("Address: %016llX ", child);
          for (uint32_t c = 0; c < branching_factor; ++c)
            printf(" L[%c][%#x] ", child[c].isOccupied() ? 'X' : 'U', c);
          printf("\n");
        }
        else
        {
          InnerNode* child = (InnerNode*) node.getChildPtr();
          printf("Address: %016llX ", child);
          for (uint32_t c = 0; c < branching_factor; ++c)
          {
            stack2[++stack2Top] = child[c];
            printf(" I[%#x][%#x] ", child[c].getStatus(), c);
          }
          printf("\n");
        }
      }
    }
    stack1Top = stack2Top;
    InnerNode* tmp = stack1;
    stack1 = stack2;
    stack2 = tmp;
    stack2Top = -1;
  }
}

template<typename T1, typename T2, typename T3>
struct MyTripple
{
public:
  T1 m_a;
  T2 m_b;
  T3 m_c;

  __host__ __device__ MyTripple()
  {

  }

  __host__ __device__ MyTripple(T1 t1, T2 t2, T3 t3)
  {
    m_a = t1;
    m_b = t2;
    m_c = t3;
  }
};

__device__
static void getStatusString(char* status, uint8_t nodeStatus)
{
  for (uint32_t i = 0; i < 8; ++i)
    status[i] = ' ';
  status[8] = '\0';
  if ((nodeStatus & ns_FREE) > 0)
    status[0] = 'F';
  if ((nodeStatus & ns_UNKNOWN) > 0)
    status[1] = 'U';
  if ((nodeStatus & ns_OCCUPIED) > 0)
    status[2] = 'X';
  if ((nodeStatus & ns_PART) > 0)
    status[3] = 'P';
  if ((nodeStatus & ns_LAST_LEVEL) > 0)
    status[4] = 'L';
  if ((nodeStatus & ns_COLLISION) > 0)
    status[5] = 'C';
  if ((nodeStatus & ns_STATIC_MAP) > 0)
    status[6] = 'S';
  if ((nodeStatus & ns_DYNAMIC_MAP) > 0)
    status[7] = 'D';
}

template<std::size_t branching_factor, std::size_t level_count, typename InnerNode, typename LeafNode>
__global__
void kernel_print2(InnerNode* root, MyTripple<InnerNode*, OctreeVoxelID, bool>* stack1,
                   MyTripple<InnerNode*, OctreeVoxelID, bool>* stack2)
{
  int32_t stack1Top = -1;
  int32_t stack2Top = -1;
  stack1[++stack1Top] = MyTripple<InnerNode*, OctreeVoxelID, bool>(root, 0, false);
//printf("Level %i ####################### \n", level_count - 1);
//printf("root [%lu - %lu] @ %016llX\n", 0, (voxel_id) (pow(branching_factor, level_count) - 1), octree.root);
//printf(" root[][%016llX]\n", (*octree.root).data);

  for (int32_t level = level_count - 1; level >= 0; --level)
  {
    // reverse
    for (uint32_t j = 0; j < ((stack1Top + 1) / 2); ++j)
    {
      MyTripple<InnerNode*, OctreeVoxelID, bool> tmp = stack1[j];
      stack1[j] = stack1[stack1Top - j];
      stack1[stack1Top - j] = tmp;
    }

    printf("\nLevel %i ####################### \n", level);
    while (stack1Top >= 0)
    {
      MyTripple<InnerNode*, OctreeVoxelID, bool> current = stack1[stack1Top--];
      InnerNode* node = current.m_a;
      if (current.m_c)
      {
        // is child node
        char status[8];
        LeafNode* tmp_leaf = (LeafNode*) current.m_a;
        uint8_t s = tmp_leaf->getStatus();
        getStatusString(status, s);
        printf("[%lu][%s][@ ] -- ", current.m_b, status);  // tmp_leaf);
      }
      else
      {
        char status[8];
        getStatusString(status, node->getStatus());

        printf("[%lu - %lu][%s][@ ] -- ", current.m_b,
               current.m_b + (OctreeVoxelID) (powf(branching_factor, level) - 1), (char*) status);  // node);

        if (node->hasStatus(ns_PART))
        {
          InnerNode* child = (InnerNode*) node->getChildPtr();
          for (uint32_t c = 0; c < branching_factor; ++c)
          {
            if (node->hasStatus(ns_LAST_LEVEL))
            {
              stack2[++stack2Top] = MyTripple<InnerNode*, OctreeVoxelID, bool>(
                  (InnerNode*) &((LeafNode*) (child))[c],
                  (OctreeVoxelID) (current.m_b + (c << ((level - 1) * uint32_t(log2(float(branching_factor)))))),
                  true);
            }
            else
              stack2[++stack2Top] = MyTripple<InnerNode*, OctreeVoxelID, bool>(
                  &child[c],
                  (OctreeVoxelID) (current.m_b + (c << ((level - 1) * uint32_t(log2(float(branching_factor)))))),
                  false);
          }
        }
      }
    }
    stack1Top = stack2Top;
    MyTripple<InnerNode*, OctreeVoxelID, bool>* tmp = stack1;
    stack1 = stack2;
    stack2 = tmp;
    stack2Top = -1;
  }
  printf("\n");
}

template<std::size_t branching_factor, std::size_t level_count, typename InnerNode, typename LeafNode>
__global__ void kernel_find(InnerNode* root, Vector3ui* voxel, OctreeVoxelID numVoxel, void** resultNode,
                            enum NodeType* resultNodeType)
{
  const OctreeVoxelID chunk_size = ceil(double(numVoxel) / (gridDim.x * blockDim.x));
  const OctreeVoxelID id = (blockIdx.x * blockDim.x + threadIdx.x);
  const OctreeVoxelID from = chunk_size * id;
  const OctreeVoxelID to = (OctreeVoxelID) min((unsigned long long int) (from + chunk_size),
                                   (unsigned long long int) numVoxel);
  for (OctreeVoxelID i = from; i < to; ++i)
  {
    InnerNode* node = root;
    const OctreeVoxelID nodeID = morton_code60(voxel[i].x, voxel[i].y, voxel[i].z);
    for (uint32_t level = level_count - 2; level > 0 && node->hasStatus(ns_PART); --level)
      node = &((InnerNode*) node->getChildPtr())[getZOrderNodeId<branching_factor>(nodeID, level)];

    if (node->hasStatus(ns_PART))
    {
      resultNode[i] = &((LeafNode*) node->getChildPtr())[getZOrderNodeId<branching_factor>(nodeID, 0)];
      resultNodeType[i] = isLeafNode;
    }
    else
    {
      resultNode[i] = node;
      resultNodeType[i] = isInnerNode;
    }
  }
}


/*!
 * Iterate over an array of Voxels (given as 3D Coordinates), generate their morton code,
 * and look up the corresponding voxel in the octree. ==> Check for collision.
 * This kernel can not handle voxelmeanings from the colliding list!
 */
template<std::size_t branching_factor, std::size_t level_count, typename InnerNode, typename LeafNode>
__global__ void kernel_intersect(InnerNode* root, Vector3ui* voxel, OctreeVoxelID numVoxel,
                                 voxel_count* num_collisions)
{
  __shared__ voxel_count shared_num_collisions[NUM_THREADS_PER_BLOCK];
  const OctreeVoxelID id = (blockIdx.x * blockDim.x + threadIdx.x);

  voxel_count my_num_collisions = 0;
  for (OctreeVoxelID i = id; i < numVoxel; i += gridDim.x * blockDim.x)
  {
    InnerNode* node = root;
    const OctreeVoxelID nodeID = morton_code60(voxel[i].x, voxel[i].y, voxel[i].z);
    for (uint32_t level = level_count - 2; level > 0 && node->hasStatus(ns_PART); --level)
      node = &((InnerNode*) node->getChildPtr())[getZOrderNodeId<branching_factor>(nodeID, level)];

    my_num_collisions +=
        (node->hasStatus(ns_PART)) ? ((LeafNode*) node->getChildPtr())[getZOrderNodeId<branching_factor>(
                                         nodeID, 0)].isOccupied() :
                                     node->isOccupied();
  }
  shared_num_collisions[threadIdx.x] = my_num_collisions;
  __syncthreads();

  REDUCE(shared_num_collisions, threadIdx.x, blockDim.x, +)

  if (threadIdx.x == 0)
    num_collisions[blockIdx.x] = shared_num_collisions[0];
  __syncthreads();
}


// WATCH OUT: This macro will read and write the following variables:
// int level, tmp_level;
// InnerNode* node, tmp_node;
// voxel_count* shared_num_collisions;
// UPDATE_FLAGS must be a complete function

#define TRAVES_MACRO(NODEID, UPDATE_FLAGS, LEVEL) \
    bool collision, collision_w_unknown, isLeaf; \
    LeafNode* leaf = NULL; \
    level = tmp_level; \
    node = tmp_node; \
    for (; level > LEVEL && node->hasStatus(ns_PART); --level) \
      node = &((InnerNode*) node->getChildPtr())[getZOrderNodeId<branching_factor>(NODEID, level)]; \
      if(level == 0 && node->hasStatus(ns_PART)) \
      { \
        leaf = &((LeafNode*) node->getChildPtr())[getZOrderNodeId<branching_factor>(NODEID, 0)]; \
        collision = leaf->isOccupied(); \
        isLeaf = true; \
        if(compute_collsWithUnknown) \
          collision_w_unknown = leaf->isUnknown(); \
      } \
      else \
      { \
        if(level == LEVEL && node->hasStatus(ns_PART)) \
          node = &((InnerNode*) node->getChildPtr())[getZOrderNodeId<branching_factor>(NODEID, level)]; \
        collision = node->isOccupied(); \
        isLeaf = false; \
        if(compute_collsWithUnknown) \
          collision_w_unknown = node->isUnknown(); \
      } \
      if (collision) \
      { \
        ++shared_num_collisions[threadIdx.x]; \
        if(set_collision_flag) \
        { \
          if(isLeaf) \
            leaf->setStatus(leaf->getStatus() | ns_COLLISION); \
          else \
            node->setStatus(node->getStatus() | ns_COLLISION); \
        } \
        if (compute_voxelTypeFlags) \
            UPDATE_FLAGS; \
      } \
      if (collision_w_unknown) \
      { \
        ++shared_num_collisions_w_unknown[threadIdx.x]; \
      }

template<class Voxel>
__device__
void reduceVoxels(Voxel& flags, const int thread_id, const int num_threads, Voxel* shared_mem)
{
  shared_mem[thread_id] = flags;
  __syncthreads();

  for (int r = num_threads / 2; r != 0; r /= 2)
  {
    if (thread_id < r)
      shared_mem[thread_id] = Voxel::reduce(shared_mem[thread_id], shared_mem[thread_id + r]);
    __syncthreads();
  }
  if (thread_id == 0)
    flags = shared_mem[0];
  __syncthreads();
}

/*!
 * Iterate over an array of Voxels (given as Morton Codes),
 * and look up the corresponding voxels in the octree. ==> Check for collision.
 * Per iteration two Voxels from the array are checked at once.
 *
 * WARNING: THIS KERNEL IS NOT COMPLETE (it ignores the voxel flags)!
 */
template<std::size_t branching_factor, std::size_t level_count, typename InnerNode, typename LeafNode,
    bool set_collision_flag, typename VoxelFlags, bool compute_voxelTypeFlags, bool compute_collsWithUnknown>
__global__
void kernel_intersect(InnerNode* root, OctreeVoxelID* voxel, VoxelFlags* voxelFlags, voxel_count num_voxel,
                      voxel_count* num_collisions, VoxelFlags* result_voxelFlags)
{
  __shared__ voxel_count shared_num_collisions[NUM_THREADS_PER_BLOCK];
  __shared__ voxel_count shared_num_collisions_w_unknown[NUM_THREADS_PER_BLOCK];
  const uint32_t STEPS = 2;
  VoxelFlags my_voxel_flags;

  shared_num_collisions[threadIdx.x] = 0;

  if(compute_collsWithUnknown)
  {
    shared_num_collisions_w_unknown[threadIdx.x] = 0;
  }

  for (voxel_count i = STEPS * (blockIdx.x * blockDim.x + threadIdx.x); i < num_voxel;
      i += STEPS * (gridDim.x * blockDim.x))
  {
    const OctreeVoxelID nodeID0 = voxel[i];
    const OctreeVoxelID nodeID1 = (i == (num_voxel - 1)) ? nodeID0 : voxel[i + 1];
    //const voxel_id nodeID7 = voxel[i + 7];
    InnerNode* node = root;
    int common_level = getCommonLevel<branching_factor>(nodeID0, nodeID1);
    int level = level_count - 2;
    for (; level > common_level && node->hasStatus(ns_PART); --level)
      node = &((InnerNode*) node->getChildPtr())[getZOrderNodeId<branching_factor>(nodeID0, level)];

    int tmp_level = level;
    InnerNode* tmp_node = node;

    TRAVES_MACRO(nodeID0, my_voxel_flags |= voxelFlags[i], 0)

    if ((i != (num_voxel - 1)))
    {
      TRAVES_MACRO(nodeID1, my_voxel_flags |= voxelFlags[i], 0)
    }

//    TRAVES_MACRO(nodeID7)
//
//    const voxel_id nodeID1 = voxel[i + 1];
//    TRAVES_MACRO(nodeID1)
//
//    const voxel_id nodeID2 = voxel[i + 2];
//    TRAVES_MACRO(nodeID2)
//
//    const voxel_id nodeID3 = voxel[i + 3];
//    TRAVES_MACRO(nodeID3)
//
//    const voxel_id nodeID4 = voxel[i + 4];
//    TRAVES_MACRO(nodeID4)
//
//    const voxel_id nodeID5 = voxel[i + 5];
//    TRAVES_MACRO(nodeID5)
//
//    const voxel_id nodeID6 = voxel[i + 6];
//    TRAVES_MACRO(nodeID6)
  }
  __syncthreads();

  // The collision counter per thread is incremented by TRAVES_MACRO.
  // So this summs up the collisions per Block
  REDUCE(shared_num_collisions, threadIdx.x, blockDim.x, +)


  // TODO: Fix this! Until then, the computation of flags is disabled.
  // Basically this should reduce the Flags per Block via OR-Operation
  // But: The reduction function overwrites data in the shared_num_collisions with VoxelFlags?!
  if (compute_voxelTypeFlags)
  {
    printf("kernel_Octree.h: compute_voxelTypeFlags IS NOT IMPLEMENTED!");
    assert(false);
    //VoxelFlags::reduce(my_voxel_flags, threadIdx.x, blockDim.x, shared_num_collisions);
  }

  if (threadIdx.x == 0)
  {
    num_collisions[blockIdx.x] = shared_num_collisions[0];
    if (compute_voxelTypeFlags)
      result_voxelFlags[blockIdx.x] = my_voxel_flags;
  }
}

/*!
 * Iterate over a complete Voxelmap, generate the Morton code of each occupied voxel and look it up
 * in the Octree. Then check for collisions and colliding meanings.
 */
template<std::size_t branching_factor, std::size_t level_count, typename InnerNode, typename LeafNode,
    bool set_collision_flag, bool compute_voxelTypeFlags, bool compute_collsWithUnknown, typename VoxelType>
__global__ void kernel_intersect_VoxelMap(InnerNode* root, VoxelType* voxels, uint32_t voxelmap_size,
                                          gpu_voxels::Vector3ui dimensions, voxel_count* num_collisions, voxel_count* num_collisions_w_unknown,
                                          VoxelType* d_result_voxels, const uint32_t min_level,
                                          const Vector3i voxelmap_offset = Vector3i(0))
{
  __shared__ voxel_count shared_num_collisions[NUM_THREADS_PER_BLOCK];
  __shared__ voxel_count shared_num_collisions_w_unknown[NUM_THREADS_PER_BLOCK];
  SharedVoxel<VoxelType> shared;
  VoxelType* shared_voxels = shared.getPointer();
  VoxelType my_voxel_flags;

  if(compute_collsWithUnknown)
  {
    shared_num_collisions_w_unknown[threadIdx.x] = 0;
  }

  shared_num_collisions[threadIdx.x] = 0;
  for (voxel_count i = blockIdx.x * blockDim.x + threadIdx.x; i < voxelmap_size; i += gridDim.x * blockDim.x)
  {
    const VoxelType* my_voxel = &voxels[i];
    if (isVoxelOccupied(my_voxel))
    {
      const OctreeVoxelID nodeID = morton_code60(voxelmap::mapToVoxels(voxels, dimensions, my_voxel) + voxelmap_offset);
      InnerNode* node = root;
      int level = level_count - 2;
      int tmp_level = level;
      InnerNode* tmp_node = node;

      TRAVES_MACRO(nodeID, my_voxel_flags = VoxelType::reduce(my_voxel_flags, *my_voxel), min_level)
    }
  }
  __syncthreads();

  REDUCE(shared_num_collisions, threadIdx.x, blockDim.x, +)

  if (compute_voxelTypeFlags)
    reduceVoxels(my_voxel_flags, threadIdx.x, blockDim.x, shared_voxels);

  if(compute_collsWithUnknown)
    REDUCE(shared_num_collisions_w_unknown, threadIdx.x, blockDim.x, +)

  if (threadIdx.x == 0)
  {
    num_collisions[blockIdx.x] = shared_num_collisions[0];
    if (compute_voxelTypeFlags)
      d_result_voxels[blockIdx.x] = my_voxel_flags;

    if(compute_collsWithUnknown)
      num_collisions_w_unknown[blockIdx.x] = shared_num_collisions_w_unknown[0];
  }
}

/*!
 * Iterate over an array of Voxels (given as 3D Coordinates and (Bit)Voxels),
 * generate their morton code, and look up the corresponding voxel in the octree.
 * ==> Check for collision.
 * This kernel can compute the meanings of the collidng voxels from the list.
 */
template<std::size_t branching_factor, std::size_t level_count, typename InnerNode, typename LeafNode,
    bool set_collision_flag, bool compute_voxelTypeFlags, bool compute_collsWithUnknown, typename VoxelType>
__global__ void kernel_intersect_VoxelList(InnerNode* root, const Vector3ui* voxel_coords, VoxelType* voxels, uint32_t voxellist_size,
                                          voxel_count* num_collisions, voxel_count* num_collisions_w_unknown,
                                          BitVectorVoxel* d_result_voxels, const uint32_t min_level,
                                          const Vector3i voxelmap_offset = Vector3i(0))
{
  __shared__ voxel_count shared_num_collisions[NUM_THREADS_PER_BLOCK];
  __shared__ voxel_count shared_num_collisions_w_unknown[NUM_THREADS_PER_BLOCK];

  if(compute_collsWithUnknown)
  {
    shared_num_collisions_w_unknown[threadIdx.x] = 0;
  }

  SharedVoxel<BitVectorVoxel> shared;
  BitVectorVoxel* shared_voxels = shared.getPointer();
  BitVectorVoxel my_voxel_flags;

  shared_num_collisions[threadIdx.x] = 0;
  for (voxel_count i = blockIdx.x * blockDim.x + threadIdx.x; i < voxellist_size; i += gridDim.x * blockDim.x)
  {
    const VoxelType* my_voxel = &voxels[i];
    if (isVoxelOccupied(my_voxel))
    {
      const OctreeVoxelID nodeID = morton_code60(voxel_coords[i] + voxelmap_offset);
      InnerNode* node = root;
      int level = level_count - 2;
      int tmp_level = level;
      InnerNode* tmp_node = node;

      TRAVES_MACRO(nodeID, my_voxel_flags = BitVectorVoxel::reduce(my_voxel_flags, *(my_voxel)), min_level)
    }
  }
  __syncthreads();

  REDUCE(shared_num_collisions, threadIdx.x, blockDim.x, +)

  if (compute_voxelTypeFlags)
    reduceVoxels(my_voxel_flags, threadIdx.x, blockDim.x, shared_voxels);

  if(compute_collsWithUnknown)
    REDUCE(shared_num_collisions_w_unknown, threadIdx.x, blockDim.x, +)


  if (threadIdx.x == 0)
  {
    num_collisions[blockIdx.x] = shared_num_collisions[0];
    if (compute_voxelTypeFlags)
      d_result_voxels[blockIdx.x] = my_voxel_flags;

    if(compute_collsWithUnknown)
      num_collisions_w_unknown[blockIdx.x] = shared_num_collisions_w_unknown[0];
  }
}


/*!
 * Iterate over an array of Voxels (given as Morton Codes and (Bit)Voxels),
 * and look up the corresponding voxel in the octree.
 * ==> Check for collision.
 * This kernel can compute the meanings of the collidng voxels from the list.
 */
template<std::size_t branching_factor, std::size_t level_count, typename InnerNode, typename LeafNode,
    bool set_collision_flag, bool compute_voxelTypeFlags, bool compute_collsWithUnknown, typename VoxelType>
__global__ void kernel_intersect_MortonVoxelList(InnerNode* root, const OctreeVoxelID* voxel_ids, VoxelType* voxels, uint32_t voxellist_size,
                                                 voxel_count* num_collisions, voxel_count* num_collisions_w_unknown,
                                                 BitVectorVoxel* d_result_voxels, const uint32_t min_level)
{
  __shared__ voxel_count shared_num_collisions[NUM_THREADS_PER_BLOCK];
  __shared__ voxel_count shared_num_collisions_w_unknown[NUM_THREADS_PER_BLOCK];

  SharedVoxel<BitVectorVoxel> shared;
  BitVectorVoxel* shared_voxels = shared.getPointer();
  BitVectorVoxel my_voxel_flags;

  if(compute_collsWithUnknown)
  {
    shared_num_collisions_w_unknown[threadIdx.x] = 0;
  }

  shared_num_collisions[threadIdx.x] = 0;
  for (voxel_count i = blockIdx.x * blockDim.x + threadIdx.x; i < voxellist_size; i += gridDim.x * blockDim.x)
  {
    const VoxelType* my_voxel = &voxels[i];
    if (isVoxelOccupied(my_voxel))
    {
      const OctreeVoxelID nodeID = voxel_ids[i];
      InnerNode* node = root;
      int level = level_count - 2;
      int tmp_level = level;
      InnerNode* tmp_node = node;

      TRAVES_MACRO(nodeID, my_voxel_flags = BitVectorVoxel::reduce(my_voxel_flags, *(my_voxel)), min_level)
    }
  }
  __syncthreads();

  REDUCE(shared_num_collisions, threadIdx.x, blockDim.x, +)

  if (compute_voxelTypeFlags)
    reduceVoxels(my_voxel_flags, threadIdx.x, blockDim.x, shared_voxels);

  if(compute_collsWithUnknown)
    REDUCE(shared_num_collisions_w_unknown, threadIdx.x, blockDim.x, +)

  if (threadIdx.x == 0)
  {
    num_collisions[blockIdx.x] = shared_num_collisions[0];
    if (compute_voxelTypeFlags)
      d_result_voxels[blockIdx.x] = my_voxel_flags;

    if(compute_collsWithUnknown)
      num_collisions_w_unknown[blockIdx.x] = shared_num_collisions_w_unknown[0];
  }
}


/*!
 * Iterate over an array of 3D coordinates and look up the corresponding voxels in the octree.
 * Returns an array of leaf nodes.
 */
template<std::size_t branching_factor, std::size_t level_count, typename InnerNode, typename LeafNode>
__global__ void kernel_find(InnerNode* const root, Vector3ui* const voxel, const voxel_count numVoxel,
                            FindResult<LeafNode>* resultNode)
{

//printf("to: %lu\n", to);
  for (voxel_count i = blockIdx.x * blockDim.x + threadIdx.x; i < numVoxel; i += gridDim.x * blockDim.x)
  {
    const OctreeVoxelID nodeID = morton_code60(voxel[i].x, voxel[i].y, voxel[i].z);
    InnerNode* node = root;
    uint32_t level = level_count - 2;
    for (; level > 0 && node->hasStatus(ns_PART); --level)
    {
      //printf("Address: %016llX\n ", &((InnerNode*) node.getChildPtr())[getZOrderNodeId(nodeID, level)]);
      node = &((InnerNode*) node->getChildPtr())[getZOrderNodeId<branching_factor>(nodeID, level)];
    }

    if (node->hasStatus(ns_PART))
    {
      LeafNode* ptr = &((LeafNode*) node->getChildPtr())[getZOrderNodeId<branching_factor>(nodeID, 0)];
      resultNode[i] = FindResult<LeafNode>(ptr, level, *ptr);

      //resultNode[i] = ((LeafNode*) node.getChildPtr())[getZOrderNodeId<branching_factor>(nodeID, 0)];

      //printf("Address: %016llX\n ", &((LeafNode*) node.getChildPtr())[getZOrderNodeId(nodeID, 0)]);
      //printf("VOXEL %lu Occ. %s\n", nodeID, resultNode[i].isOccupied() ? "TRUE" : "FALSE");
    }
    else
    {
      resultNode[i] = FindResult<LeafNode>(node, level + 1, getLeafNode(node));

//      resultNode[i] = LeafNode();
//      uint8_t status = node.getStatus() & (ns_FREE | ns_UNKNOWN | ns_OCCUPIED);
//      assert(status == ns_FREE || status == ns_OCCUPIED || status == ns_UNKNOWN);
//      resultNode[i].setStatus(status);

//      if (status == ns_FREE)
//        resultNode[i].setFree();
//      else if (status == ns_OCCUPIED)
//        resultNode[i].setOccupied();
//      else
//        resultNode[i].setUnknown();
    }
  }
}

template<std::size_t branching_factor, std::size_t level_count, typename InnerNode, typename LeafNode>
__global__
static void kernel_intersect_shared(InnerNode* robot_root, InnerNode* environment_root,
                                    OctreeVoxelID side_length_in_voxel, uint32_t num_level, OctreeVoxelID* numConflicts,
                                    const OctreeVoxelID splitLevel)
{
  extern __shared__ thrust::pair<InnerNode, InnerNode> stack[];

// octrees have to model the same space
  const OctreeVoxelID numVoxel = side_length_in_voxel * side_length_in_voxel * side_length_in_voxel;
  const uint32_t numLevel = num_level;

// TODO better splitting of work; in worst case only 1/branching_factor * #threads are busy
  const OctreeVoxelID id = blockIdx.x * blockDim.x + threadIdx.x;
  const OctreeVoxelID from = id << (splitLevel * (uint32_t) log2(float(branching_factor)));
//printf("llog = %lu\n", llog);
//printf("splitLevel = %lu\n", splitLevel);
//printf("from = %lu\n", from);

// size of 2 * numLevels
  const uint32_t R_TOP = threadIdx.x * splitLevel * branching_factor;
//const uint32_t R_TOP = id * splitLevel * branching_factor;
//const uint32_t E_TOP = (id * 2 + 1) * numLevel * branching_factor;
  uint32_t r_top = R_TOP;
//uint32_t e_top = E_TOP;

  OctreeVoxelID myNumConflicts = 0;

  InnerNode r_node = *robot_root;
  InnerNode e_node = *environment_root;

  if (from >= numVoxel)
  {
    numConflicts[id] = myNumConflicts;
    return;
  }

// search for "root" of subtree
  OctreeVoxelID level = numLevel - 2;
  for (; level >= splitLevel && r_node.isInConflict(e_node); --level)
  {
    unsigned char child = getZOrderNodeId<branching_factor>(from, level);
    r_node = ((InnerNode*) r_node.getChildPtr())[child];
    e_node = ((InnerNode*) e_node.getChildPtr())[child];
  }
//  if (!r_node->isInConflict(*e_node))
//  {
//    numConflicts[id] = myNumConflicts;
//    return;
//  }

  stack[r_top++] = thrust::make_pair<InnerNode, InnerNode>(r_node, e_node);

  while (r_top != R_TOP)
  {
    thrust::pair<InnerNode, InnerNode> tmp = stack[--r_top];
    r_node = tmp.first;
    e_node = tmp.second;

    if (r_node.hasStatus(ns_LAST_LEVEL) && e_node.hasStatus(ns_LAST_LEVEL))
    {
      if (r_node.isInConflict(e_node))
      {
        // check LeafNodes
        LeafNode* r_leaf = ((LeafNode*) r_node.getChildPtr());
        LeafNode* e_leaf = ((LeafNode*) e_node.getChildPtr());
        //++myNumConflicts;

        //        LeafNode r0 = r_leaf[0];
        //        LeafNode r1 = r_leaf[1];
        //        LeafNode r2 = r_leaf[2];
        //        LeafNode r3 = r_leaf[3];
        //        LeafNode r4 = r_leaf[4];
        //        LeafNode r5 = r_leaf[5];
        //        LeafNode r6 = r_leaf[6];
        //        LeafNode r7 = r_leaf[7];
        //
        //        LeafNode e0 = e_leaf[0];
        //        LeafNode e1 = e_leaf[1];
        //        LeafNode e2 = e_leaf[2];
        //        LeafNode e3 = e_leaf[3];
        //        LeafNode e4 = e_leaf[4];
        //        LeafNode e5 = e_leaf[5];
        //        LeafNode e6 = e_leaf[6];
        //        LeafNode e7 = e_leaf[7];
        //
        //        myNumConflicts += r0.isConflict(e0) ? 1 : 0;
        //        myNumConflicts += r1.isConflict(e1) ? 1 : 0;
        //        myNumConflicts += r2.isConflict(e2) ? 1 : 0;
        //        myNumConflicts += r3.isConflict(e3) ? 1 : 0;
        //        myNumConflicts += r4.isConflict(e4) ? 1 : 0;
        //        myNumConflicts += r5.isConflict(e5) ? 1 : 0;
        //        myNumConflicts += r6.isConflict(e6) ? 1 : 0;
        //        myNumConflicts += r7.isConflict(e7) ? 1 : 0;

#pragma unroll
        for (uint32_t i = 0; i < branching_factor; ++i)
          myNumConflicts += r_leaf[i].isConflict(e_leaf[i]);

//          myNumConflicts += r_leaf[0].isConflict(e_leaf[0]) + r_leaf[1].isConflict(e_leaf[1])
//              + r_leaf[2].isConflict(e_leaf[2]) + r_leaf[3].isConflict(e_leaf[3])
//              + r_leaf[4].isConflict(e_leaf[4]) + r_leaf[5].isConflict(e_leaf[5])
//              + r_leaf[6].isConflict(e_leaf[6]) + r_leaf[7].isConflict(e_leaf[7]);
        //#pragma unroll
        //        for (uint32_t c = 0; c < branching_factor; ++c)
        //        {
        //          myNumConflicts += r_leaf[c].isConflict(e_leaf[c]) ? 1 : 0;
        //        }
      }
    }
    else if (!r_node.hasStatus(ns_LAST_LEVEL) && !e_node.hasStatus(ns_LAST_LEVEL))
    {
      if (r_node.isInConflict(e_node))
      {
        InnerNode* r_inner = ((InnerNode*) r_node.getChildPtr());
        InnerNode* e_inner = ((InnerNode*) e_node.getChildPtr());

#pragma unroll
        for (uint32_t i = 0; i < branching_factor; ++i)
          stack[r_top++] = thrust::make_pair<InnerNode, InnerNode>(r_inner[i], e_inner[i]);

//        stack[r_top++] = thrust::make_pair<InnerNode, InnerNode>(r_inner[0], e_inner[0]);
//        stack[r_top++] = thrust::make_pair<InnerNode, InnerNode>(r_inner[1], e_inner[1]);
//        stack[r_top++] = thrust::make_pair<InnerNode, InnerNode>(r_inner[2], e_inner[2]);
//        stack[r_top++] = thrust::make_pair<InnerNode, InnerNode>(r_inner[3], e_inner[3]);
//        stack[r_top++] = thrust::make_pair<InnerNode, InnerNode>(r_inner[4], e_inner[4]);
//        stack[r_top++] = thrust::make_pair<InnerNode, InnerNode>(r_inner[5], e_inner[5]);
//        stack[r_top++] = thrust::make_pair<InnerNode, InnerNode>(r_inner[6], e_inner[6]);
//        stack[r_top++] = thrust::make_pair<InnerNode, InnerNode>(r_inner[7], e_inner[7]);

        //#pragma unroll
        //        for (uint32_t c = 0; c < branching_factor; ++c)
        //        {
        //          stack[r_top++] = thrust::make_pair<InnerNode*, InnerNode*>(&((InnerNode*) r_node->getChildPtr())[c],
        //                                                                     &((InnerNode*) e_node->getChildPtr())[c]);
        //        }
      }
    }
    else
    {
      // ERROR
      myNumConflicts = INVALID_VOXEL;
      break;
    }
  }

  numConflicts[id] = myNumConflicts;
}

template<std::size_t branching_factor, std::size_t level_count, typename a_InnerNode, typename a_LeafNode,
    typename b_InnerNode, typename b_LeafNode>
__global__
static void kernel_intersect_wo_stack_coalesced(a_InnerNode* nTreeA_root, b_InnerNode* nTreeB_root,
                                                OctreeVoxelID* numConflicts, const uint32_t splitLevel)
{
  extern __shared__ thrust::pair<a_LeafNode*, b_LeafNode*> work[];

// octrees have to model the same space
  const OctreeVoxelID numVoxel = (OctreeVoxelID) powf(double(branching_factor), double(level_count - 1));

  const uint32_t id = blockIdx.x * blockDim.x + threadIdx.x;
  const OctreeVoxelID from = OctreeVoxelID(id) << (splitLevel * (uint32_t) log2(float(branching_factor)));

  if (from >= numVoxel)
  {
    numConflicts[id] = 0;
    return;
  }

  a_InnerNode* a_node = nTreeA_root;
  b_InnerNode* b_node = nTreeB_root;

// search for "root" of subtree
  uint32_t level = level_count - 2;
  for (; level >= splitLevel & a_node->isInConflict(*b_node); --level)
  {
    unsigned char child = getZOrderNodeId<branching_factor>(from, level);
    a_node = &((a_InnerNode*) a_node->getChildPtr())[child];
    b_node = &((b_InnerNode*) b_node->getChildPtr())[child];
  }

  a_LeafNode* a_leaf = 0;
  b_LeafNode* b_leaf = 0;

  if (a_node->isInConflict(*b_node))
  {
    a_leaf = ((a_LeafNode*) a_node->getChildPtr());
    b_leaf = ((b_LeafNode*) b_node->getChildPtr());
  }

// coalesced global memory access
// each InnerNode of last level with it's branching_factor LeafNodes is processed by branching_factor threads in parallel -> every thread gets one LeafNode
// -> coalesced memory access
// pointer are exchanged by shared memory
  work[threadIdx.x] = thrust::make_pair<a_LeafNode*, b_LeafNode*>(a_leaf, b_leaf);
  __syncthreads();

  OctreeVoxelID myNumConflicts = 0;
#pragma unroll
  for (uint32_t i = 0; i < branching_factor; ++i)
    myNumConflicts += work[(threadIdx.x / branching_factor) * branching_factor + i].first[threadIdx.x
        % branching_factor].isInConflict(
        work[(threadIdx.x / branching_factor) * branching_factor + i].second[threadIdx.x % branching_factor]);
  numConflicts[id] = myNumConflicts;
}

template<std::size_t branching_factor, std::size_t level_count, typename InnerNode, typename LeafNode>
__global__
static void kernel_intersect_wo_stack(InnerNode* robot_root, InnerNode* environment_root,
                                      OctreeVoxelID side_length_in_voxel, uint32_t num_level, OctreeVoxelID* numConflicts,
                                      const uint32_t splitLevel)
{
// stack is only not needed for splitLevel == 1
  assert(splitLevel == 1);

// octrees have to model the same space
  const OctreeVoxelID numVoxel = side_length_in_voxel * side_length_in_voxel * side_length_in_voxel;
  const uint32_t numLevel = num_level;

  const uint32_t id = blockIdx.x * blockDim.x + threadIdx.x;
  const OctreeVoxelID from = OctreeVoxelID(id) << (splitLevel * (uint32_t) log2(float(branching_factor)));

  if (from >= numVoxel)
  {
    numConflicts[id] = 0;
    return;
  }

  InnerNode* r_node = robot_root;
  InnerNode* e_node = environment_root;

// search for "root" of subtree
  uint32_t level = numLevel - 2;
  for (; level >= splitLevel & r_node->isInConflict(*e_node); --level)
  {
    unsigned char child = getZOrderNodeId<branching_factor>(from, level);
    r_node = &((InnerNode*) r_node->getChildPtr())[child];
    e_node = &((InnerNode*) e_node->getChildPtr())[child];
  }

  if (!r_node->isInConflict(*e_node))
  {
    numConflicts[id] = 0;
    return;
  }

  LeafNode* r_leaf = ((LeafNode*) r_node->getChildPtr());
  LeafNode* e_leaf = ((LeafNode*) e_node->getChildPtr());

#pragma unroll
  for (uint32_t i = 0; i < branching_factor; ++i)
    numConflicts[id] += r_leaf[i].isConflict(e_leaf[i]);

//  numConflicts[id] += r_leaf[0].isConflict(e_leaf[0]) + r_leaf[1].isConflict(e_leaf[1])
//      + r_leaf[2].isConflict(e_leaf[2]) + r_leaf[3].isConflict(e_leaf[3]) + r_leaf[4].isConflict(e_leaf[4])
//      + r_leaf[5].isConflict(e_leaf[5]) + r_leaf[6].isConflict(e_leaf[6]) + r_leaf[7].isConflict(e_leaf[7]);
}

typedef OctreeVoxelID numChild;

template<typename T1, typename T2, typename T3>
struct Triple
{
  T1 x1;
  T2 x2;
  T3 x3;

  __host__ __device__
  Triple()
  {
  }

  __host__ __device__
  Triple(T1 x1, T2 x2, T3 x3)
  {
    this->x1 = x1;
    this->x2 = x2;
    this->x3 = x3;
  }
};

template<std::size_t branching_factor, std::size_t level_count, typename InnerNode, typename LeafNode>
__global__
static void kernel_intersect_smallStack(InnerNode* robot_root, InnerNode* environment_root,
                                        OctreeVoxelID side_length_in_voxel, uint32_t num_level,
                                        OctreeVoxelID* numConflicts,
                                        Triple<InnerNode*, InnerNode*, numChild>* stack,
                                        const uint32_t splitLevel)
{
// octrees have to model the same space
  const OctreeVoxelID numVoxel = side_length_in_voxel * side_length_in_voxel * side_length_in_voxel;
  const uint32_t numLevel = num_level;

  const uint32_t id = blockIdx.x * blockDim.x + threadIdx.x;
  const OctreeVoxelID from = OctreeVoxelID(id) << (splitLevel * (uint32_t) log2(float(branching_factor)));

// size of 2 * numLevels
  const int32_t R_TOP = id * splitLevel;
  int32_t r_top = R_TOP;

  OctreeVoxelID myNumConflicts = 0;

  if (from >= numVoxel)
  {
    numConflicts[id] = 0;
    return;
  }

  InnerNode* r_node = robot_root;
  InnerNode* e_node = environment_root;

// search for "root" of subtree
  uint32_t level = numLevel - 2;
  for (; level >= splitLevel & r_node->isInConflict(*e_node); --level)
  {
    unsigned char child = getZOrderNodeId<branching_factor>(from, level);
    r_node = &((InnerNode*) r_node->getChildPtr())[child];
    e_node = &((InnerNode*) e_node->getChildPtr())[child];
  }

  if (!r_node->isInConflict(*e_node))
  {
    numConflicts[id] = 0;
    return;
  }

  if (splitLevel == 1)
  {
    LeafNode* r_leaf = ((LeafNode*) r_node->getChildPtr());
    LeafNode* e_leaf = ((LeafNode*) e_node->getChildPtr());
#pragma unroll
    for (uint32_t i = 0; i < branching_factor; ++i)
      myNumConflicts += r_leaf[i].isConflict(e_leaf[i]);
    numConflicts[id] = myNumConflicts;
    return;
  }

  stack[r_top] = Triple<InnerNode*, InnerNode*, numChild>((InnerNode*) r_node->getChildPtr(),
                                                          (InnerNode*) e_node->getChildPtr(), 0);
  while (r_top >= R_TOP)
  {
    Triple<InnerNode*, InnerNode*, numChild> tmp = stack[r_top];
    r_node = &tmp.x1[tmp.x3];
    e_node = &tmp.x2[tmp.x3];

    if (stack[r_top].x3 + 1 != branching_factor)
      stack[r_top].x3 += 1;
    else
      --r_top;

    if (r_node->isInConflict(*e_node))
    {
      // check LeafNodes
      LeafNode* r_leaf = ((LeafNode*) r_node->getChildPtr());
      LeafNode* e_leaf = ((LeafNode*) e_node->getChildPtr());

      if (r_node->hasStatus(ns_LAST_LEVEL) & e_node->hasStatus(ns_LAST_LEVEL))
      {
#pragma unroll
        for (uint32_t i = 0; i < branching_factor; ++i)
          myNumConflicts += r_leaf[i].isConflict(e_leaf[i]);

//        myNumConflicts += r_leaf[0].isConflict(e_leaf[0]) + r_leaf[1].isConflict(e_leaf[1])
//            + r_leaf[2].isConflict(e_leaf[2]) + r_leaf[3].isConflict(e_leaf[3])
//            + r_leaf[4].isConflict(e_leaf[4]) + r_leaf[5].isConflict(e_leaf[5])
//            + r_leaf[6].isConflict(e_leaf[6]) + r_leaf[7].isConflict(e_leaf[7]);
      }
      else if (!r_node->hasStatus(ns_LAST_LEVEL) & !e_node->hasStatus(ns_LAST_LEVEL))
      {
        stack[++r_top] = Triple<InnerNode*, InnerNode*, numChild>((InnerNode*) (void*) r_leaf,
                                                                  (InnerNode*) (void*) e_leaf, 0);
      }
      else
      {
        // ERROR
        myNumConflicts = INVALID_VOXEL;
        printf("ERROR\n");
        break;
      }
    }
  }

  numConflicts[id] = myNumConflicts;
}

/*
 * voxel: 128*1024*1024
 * unoptimized: 350 ms
 * unroll + common stack: 210 ms
 * inline isConflict(): 190 ms
 * LeafNode struct equivalent to voxel of VoxelMap: 260 ms ??? uses less memory and less operations
 *
 * without checking LeafNodes: 70 ms
 *
 *
 *
 */
template<std::size_t branching_factor, std::size_t level_count, typename a_InnerNode, typename a_LeafNode,
    typename b_InnerNode, typename b_LeafNode>
__global__
static void kernel_intersect(a_InnerNode* nTreeA_root, b_InnerNode* nTreeB_root, OctreeVoxelID* numConflicts,
                             thrust::pair<a_InnerNode*, b_InnerNode*>* stack, const OctreeVoxelID splitLevel)
{
// octrees have to model the same space
  const OctreeVoxelID numVoxel = (OctreeVoxelID) powf(double(branching_factor), double(level_count - 1));
//nTree_a.sideLengthInVoxel * nTree_a.sideLengthInVoxel * nTree_a.sideLengthInVoxel;

// TODO better splitting of work; in worst case only 1/branching_factor * #threads are busy
  const OctreeVoxelID id = blockIdx.x * blockDim.x + threadIdx.x;
  const OctreeVoxelID from = id << (splitLevel * (uint32_t) log2(float(branching_factor)));
//printf("llog = %lu\n", llog);
//printf("splitLevel = %lu\n", splitLevel);
//printf("from = %lu\n", from);

// size of 2 * numLevels
  const uint32_t R_TOP = id * splitLevel * branching_factor;
//const uint32_t E_TOP = (id * 2 + 1) * numLevel * branching_factor;
  uint32_t r_top = R_TOP;
//uint32_t e_top = E_TOP;

  OctreeVoxelID myNumConflicts = 0;

  a_InnerNode* a_node = nTreeA_root;
  b_InnerNode* b_node = nTreeB_root;

  if (from >= numVoxel)
  {
    numConflicts[id] = myNumConflicts;
    return;
  }

// search for "root" of subtree
  OctreeVoxelID level = level_count - 2;
  for (; level >= splitLevel && a_node->isInConflict(*b_node); --level)
  {
    unsigned char child = getZOrderNodeId<branching_factor>(from, level);
    a_node = &((a_InnerNode*) a_node->getChildPtr())[child];
    b_node = &((b_InnerNode*) b_node->getChildPtr())[child];
  }
//  if (!r_node->isInConflict(*e_node))
//  {
//    numConflicts[id] = myNumConflicts;
//    return;
//  }

  stack[r_top++] = thrust::make_pair<a_InnerNode*, b_InnerNode*>(a_node, b_node);

  while (r_top != R_TOP)
  {
    thrust::pair<a_InnerNode*, b_InnerNode*> tmp = stack[--r_top];
    a_node = tmp.first;
    b_node = tmp.second;

    if (a_node->isInConflict(*b_node))
    {
      if (a_node->hasStatus(ns_LAST_LEVEL) & b_node->hasStatus(ns_LAST_LEVEL))
      {
        // check LeafNodes
        a_LeafNode* a_leaf = ((a_LeafNode*) a_node->getChildPtr());
        b_LeafNode* b_leaf = ((b_LeafNode*) b_node->getChildPtr());
        //++myNumConflicts;

        //        LeafNode r0 = r_leaf[0];
        //        LeafNode r1 = r_leaf[1];
        //        LeafNode r2 = r_leaf[2];
        //        LeafNode r3 = r_leaf[3];
        //        LeafNode r4 = r_leaf[4];
        //        LeafNode r5 = r_leaf[5];
        //        LeafNode r6 = r_leaf[6];
        //        LeafNode r7 = r_leaf[7];
        //
        //        LeafNode e0 = e_leaf[0];
        //        LeafNode e1 = e_leaf[1];
        //        LeafNode e2 = e_leaf[2];
        //        LeafNode e3 = e_leaf[3];
        //        LeafNode e4 = e_leaf[4];
        //        LeafNode e5 = e_leaf[5];
        //        LeafNode e6 = e_leaf[6];
        //        LeafNode e7 = e_leaf[7];
        //
        //        myNumConflicts += r0.isConflict(e0) ? 1 : 0;
        //        myNumConflicts += r1.isConflict(e1) ? 1 : 0;
        //        myNumConflicts += r2.isConflict(e2) ? 1 : 0;
        //        myNumConflicts += r3.isConflict(e3) ? 1 : 0;
        //        myNumConflicts += r4.isConflict(e4) ? 1 : 0;
        //        myNumConflicts += r5.isConflict(e5) ? 1 : 0;
        //        myNumConflicts += r6.isConflict(e6) ? 1 : 0;
        //        myNumConflicts += r7.isConflict(e7) ? 1 : 0;
        //++myNumConflicts;

#pragma unroll
        for (uint32_t i = 0; i < branching_factor; ++i)
          myNumConflicts += b_leaf[i].isInConflict(a_leaf[i]);

//        myNumConflicts += r_leaf[0].isConflict(e_leaf[0]) + r_leaf[1].isConflict(e_leaf[1])
//            + r_leaf[2].isConflict(e_leaf[2]) + r_leaf[3].isConflict(e_leaf[3])
//            + r_leaf[4].isConflict(e_leaf[4]) + r_leaf[5].isConflict(e_leaf[5])
//            + r_leaf[6].isConflict(e_leaf[6]) + r_leaf[7].isConflict(e_leaf[7]);

        //        myNumConflicts += r_leaf[0].isConflict(e_leaf[0]) ? 1 : 0;
        //        myNumConflicts += r_leaf[1].isConflict(e_leaf[1]) ? 1 : 0;
        //        myNumConflicts += r_leaf[2].isConflict(e_leaf[2]) ? 1 : 0;
        //        myNumConflicts += r_leaf[3].isConflict(e_leaf[3]) ? 1 : 0;
        //        myNumConflicts += r_leaf[4].isConflict(e_leaf[4]) ? 1 : 0;
        //        myNumConflicts += r_leaf[5].isConflict(e_leaf[5]) ? 1 : 0;
        //        myNumConflicts += r_leaf[6].isConflict(e_leaf[6]) ? 1 : 0;
        //        myNumConflicts += r_leaf[7].isConflict(e_leaf[7]) ? 1 : 0;
        //#pragma unroll
        //        for (uint32_t c = 0; c < branching_factor; ++c)
        //        {
        //          myNumConflicts += r_leaf[c].isConflict(e_leaf[c]) ? 1 : 0;
        //        }
      }
      else if (!a_node->hasStatus(ns_LAST_LEVEL) & !b_node->hasStatus(ns_LAST_LEVEL))
      {
        a_InnerNode* a_inner = ((a_InnerNode*) a_node->getChildPtr());
        b_InnerNode* b_inner = ((b_InnerNode*) b_node->getChildPtr());

#pragma unroll
        for (uint32_t i = 0; i < branching_factor; ++i)
          stack[r_top++] = thrust::make_pair<a_InnerNode*, b_InnerNode*>(&a_inner[i], &b_inner[i]);

//        stack[r_top++] = thrust::make_pair<InnerNode*, InnerNode*>(&r_inner[0], &e_inner[0]);
//        stack[r_top++] = thrust::make_pair<InnerNode*, InnerNode*>(&r_inner[1], &e_inner[1]);
//        stack[r_top++] = thrust::make_pair<InnerNode*, InnerNode*>(&r_inner[2], &e_inner[2]);
//        stack[r_top++] = thrust::make_pair<InnerNode*, InnerNode*>(&r_inner[3], &e_inner[3]);
//        stack[r_top++] = thrust::make_pair<InnerNode*, InnerNode*>(&r_inner[4], &e_inner[4]);
//        stack[r_top++] = thrust::make_pair<InnerNode*, InnerNode*>(&r_inner[5], &e_inner[5]);
//        stack[r_top++] = thrust::make_pair<InnerNode*, InnerNode*>(&r_inner[6], &e_inner[6]);
//        stack[r_top++] = thrust::make_pair<InnerNode*, InnerNode*>(&r_inner[7], &e_inner[7]);

        //#pragma unroll
        //        for (uint32_t c = 0; c < branching_factor; ++c)
        //        {
        //          stack[r_top++] = thrust::make_pair<InnerNode*, InnerNode*>(&((InnerNode*) r_node->getChildPtr())[c],
        //                                                                     &((InnerNode*) e_node->getChildPtr())[c]);
        //        }
      }
      else
      {
        // ERROR
        myNumConflicts = INVALID_VOXEL;
        break;
      }
    }
  }

  numConflicts[id] = myNumConflicts;
}

template<std::size_t branching_factor, std::size_t level_count, typename InnerNode, typename LeafNode,
    bool SET_UPDATE_FLAG>
__global__
static void kernel_insert_countNeededNodes(InnerNode* const root,
                                           OctreeVoxelID* const d_voxel,
                                           const voxel_count numVoxel,
                                           voxel_count* const d_neededNodesPerLevel,
                                           void** d_traversalNodes,
                                           uint32_t* const d_traversalLevels,
                                           const int32_t target_level)
{
#define UPDATE_STATUS_EX

  __shared__ uint32_t shared_levelCount[level_count];

  const uint32_t num_threads = blockDim.x;
  const uint32_t task_id = blockIdx.x;
  const uint32_t thread_id = threadIdx.x;
  const voxel_count chunk_size = ceil(double(numVoxel) / gridDim.x);
  const voxel_count from = chunk_size * task_id;
  const voxel_count to = min(from + chunk_size, numVoxel);

// init level count array
#pragma unroll
  for (uint32_t i = thread_id; i < level_count; i += num_threads)
    shared_levelCount[i] = 0;
  __syncthreads();

  for (voxel_count v = from; v < to; v += num_threads)
  {
    const voxel_count index = v + thread_id;
    const bool isActive = (index < to);
    int32_t level = level_count - 2;
    OctreeVoxelID voxelID = INVALID_VOXEL;
    OctreeVoxelID voxelIDNeighbor = INVALID_VOXEL;
    if (isActive)
    {
      voxelID = d_voxel[index];
      //printf("VOXEL[%u]: %lu\n", index, voxelID);
      voxelIDNeighbor = (index == 0) ? ~voxelID : d_voxel[index - 1];
      InnerNode* node = root;

#ifdef UPDATE_STATUS_EX

      // second more advanced solution
      // only one thread sets the flag of each node

//        level = level_count - 1;
//        const uint32_t to_level = max(target_level, 1); // needed to handle LeafNodes separately
//
//        for (; (level > to_level) & node->hasStatus(ns_PART); --level)
//          node = &((InnerNode*)node->getChildPtr())[getZOrderNodeId<branching_factor>(voxelID, level - 1)];
//
//        if (target_level == 0 && node->hasStatus(ns_PART))
//        {
//          node =
//              (InnerNode*) &((LeafNode*) node->getChildPtr())[getZOrderNodeId<branching_factor>(voxelID, 0)];
//          level = 0;
//        }

      const uint32_t to_level = max(target_level, 1);        // needed to handle LeafNodes separately
      const uint32_t common_level = max(getCommonLevel<branching_factor>(voxelID, voxelIDNeighbor),
                                        to_level + 1) - 1;
      level = level_count - 1;

      // traverse common path of tree
      for (; (level > common_level) && node->hasStatus(ns_PART); --level)
        node = &((InnerNode*) node->getChildPtr())[getZOrderNodeId<branching_factor>(voxelID, level - 1)];

      // traverse path on my own
      for (; (level > to_level) && node->hasStatus(ns_PART); --level)
      {
        if (SET_UPDATE_FLAG)
          node->setFlags(NodeFlags(nf_NEEDS_UPDATE));

        node = &((InnerNode*) node->getChildPtr())[getZOrderNodeId<branching_factor>(voxelID, level - 1)];
      }
      if (SET_UPDATE_FLAG)
        node->setFlags(nf_NEEDS_UPDATE);

      if (target_level == 0 && node->hasStatus(ns_PART))
      {
        node = (InnerNode*) &((LeafNode*) node->getChildPtr())[getZOrderNodeId<branching_factor>(voxelID, 0)];
        level = 0;
      }
#else
      // traverse tree
      for (; (level > target_level) & node->hasStatus(ns_PART); --level)
      {
        if (SET_UPDATE_FLAG)
        {
          // first simple solution
          // every thread sets the flag -> many threads set the flag og the same node -> slow???
          node->setFlags(NodeFlags(nf_NEEDS_UPDATE));
        }
        node = &((InnerNode*) node->getChildPtr())[getZOrderNodeId<branching_factor>(voxelID, level)];
      }

      if (SET_UPDATE_FLAG)
      {
        node->setFlags(NodeFlags(nf_NEEDS_UPDATE));
      }

      if (node->hasStatus(ns_PART))
      {
        if (level == 0)
        node =
        (InnerNode*) &((LeafNode*) node->getChildPtr())[getZOrderNodeId<branching_factor>(voxelID, 0)];
        else
        {
          node = &((InnerNode*) node->getChildPtr())[getZOrderNodeId<branching_factor>(voxelID, level)];
        }
        --level;
      }

      // fix current level number
      ++level;
#endif

      // store result of traversal for next pass
      d_traversalNodes[index] = (void*) node;
      d_traversalLevels[index] = level;
    }
    //__syncthreads();

    // node isn't available in tree
    if (isActive & (level != target_level))
    {
      // TODO try to use shuffle() to check left neighbor or shared memory

      // only count nodes not covered by left neighbor voxel
      // level from where both leafs have the same path to the root
      const int commonLevel =
          (index == 0) ? (level + 1) :
                         min((voxel_count) (level + 1),
                             (voxel_count) getCommonLevel<branching_factor>(voxelID, voxelIDNeighbor));
      //printf("commonLevel: %i\n", commonLevel);
      assert(commonLevel - 2 < int(level_count));
      for (int32_t i = commonLevel - 2; i >= target_level; --i)
        atomicAdd(&shared_levelCount[i], branching_factor);
    }
    //__syncthreads();
  }
  __syncthreads();

// copy shared level count data into global memory
#pragma unroll
  for (uint32_t i = thread_id; i < level_count; i += num_threads)
    d_neededNodesPerLevel[gridDim.x * i + task_id] = shared_levelCount[i];
}

template<std::size_t branching_factor, std::size_t level_count, typename T, bool isLastLevel>
__global__
static void kernel_insert_initNeededNodes(T* newNodes, voxel_count numNodes)
{
  const uint32_t num_threads = blockDim.x * gridDim.x;
  const uint32_t thread_id = blockIdx.x * blockDim.x + threadIdx.x;

  for (voxel_count i = thread_id; i < numNodes; i += num_threads)
  {
    newNodes[i] = T();
    if (isLastLevel)
      insertNodeLastLevel(&newNodes[i]);
    else
      insertNode(&newNodes[i]);
  }
}

template<typename InnerNode, bool SET_STATUS, bool SET_UPDATE_FLAG>
__device__ __forceinline__
void insert_setInnerNodeStatus(InnerNode* innerNode, const NodeStatus node_status, const uint32_t level,
                               const uint32_t target_level)
{
  NodeStatus s = innerNode->getStatus();
  if (level == target_level)
  {
    if (SET_STATUS)
      s = (s & ~(ns_FREE | ns_OCCUPIED | ns_UNKNOWN | ns_PART)) | node_status;
  }
  else
  {
    s |= ns_PART;
    if (SET_UPDATE_FLAG)
      innerNode->setFlags(nf_NEEDS_UPDATE);
  }
  innerNode->setStatus(s);
}

template<std::size_t branching_factor, std::size_t level_count, typename InnerNode, typename LeafNode,
std::size_t num_threads, typename Iterator1, typename Iterator2, typename BasicData, bool SET_UPDATE_FLAG>
__global__
static void kernel_insert_setNodes(InnerNode* const root,
                                   OctreeVoxelID* const voxel,
                                   Iterator1 d_set_basic_data,
                                   Iterator2 d_reset_basic_data,
                                   const voxel_count numVoxel,
                                   voxel_count* const prefixSum,
                                   LeafNode* const leafNodes,
                                   InnerNode* const innerNodes,
                                   void** const traversalNodes,
                                   uint32_t* const traversalLevels,
                                   const uint32_t target_level)
{
//#define FIX_STATUS_OF_NODE_AT_SPLIT_UP

  assert(num_threads == blockDim.x);

  const uint32_t num_warps = num_threads / WARP_SIZE;
  const uint32_t num_tasks = gridDim.x;
  const uint32_t task_id = blockIdx.x;
  const uint32_t thread_id = threadIdx.x;
  const voxel_count chunk_size = ceil(double(numVoxel) / gridDim.x);
  const voxel_count from = chunk_size * task_id;
  const voxel_count to = min(from + chunk_size, numVoxel);
  const voxel_count numLeafNodes = prefixSum[num_tasks];

  volatile __shared__ uint32_t shared_levelCount[level_count];
  volatile __shared__ uint32_t shared_prefix_sum[num_warps];

// copy level count array into shared memory
// start prefix sum at level 1
#pragma unroll
  for (uint32_t i = thread_id; i < level_count; i += num_threads)
  {
    uint32_t tmp = prefixSum[i * num_tasks + task_id];
    assert(i == 0 || tmp >= numLeafNodes);
    tmp -= (i == 0) ? 0 : numLeafNodes;
    shared_levelCount[i] = tmp;
  }
  __syncthreads();

  for (voxel_count v = from; v < to; v += num_threads)
  {
//    if (thread_id == 0)
//      printf("\nRun: %i\n", (v - from) / num_threads);

    void* node = NULL;
    const voxel_count index = v + thread_id;
    OctreeVoxelID voxelIDNeighbor = INVALID_VOXEL;
    bool isActive = (index < to);
    int32_t level = level_count - 2;
    void* child = NULL;
    OctreeVoxelID my_voxel_id;
    BasicData my_set_basic_data, my_reset_basic_data;

    uint32_t commonLevel = 0;
    if (isActive)
    {
      my_voxel_id = voxel[index];
      my_set_basic_data = d_set_basic_data[index];
      my_reset_basic_data = d_reset_basic_data[index];
      voxelIDNeighbor = (index == 0) ? INVALID_VOXEL : voxel[index - 1];
      node = traversalNodes[index];
      level = traversalLevels[index];

      // TODO try to use shuffle() to check left neighbor or shared mem
      commonLevel =
          (index == 0) ? (level + 1) :
                         min((voxel_count) (level + 1),
                             (voxel_count) getCommonLevel<branching_factor>(my_voxel_id, voxelIDNeighbor));

      assert(level >= target_level);

      // ##### Handle already in tree available level 0 nodes #####
      if (level == 0)
      {
        updateNode((LeafNode*) node, my_set_basic_data, my_reset_basic_data);
        isActive = false;
//        printf("UPDATE LEAF NODE! %u\n", myVoxel.voxelId);
      }
    }

//    if (thread_id == 0)
//      printf("Available leaf nodes set!\n");

    // ##### set leaf nodes of newly initialized memory #####
    if (target_level == 0)
    {
      const bool isLeftMost = isActive & (commonLevel > 1);
      uint32_t total_sum;
      const uint32_t offset = shared_levelCount[0]; // relies on the __syncthreads() of thread_prefix()
      const uint32_t allVotes = thread_prefix<num_warps>(shared_prefix_sum, thread_id,
                                                           total_sum, isLeftMost);
      if (isActive)
      {
        const uint32_t myIndex = getThreadPrefix_Inclusive(shared_prefix_sum, allVotes, thread_id);
        assert(myIndex > 0 || offset >= branching_factor);
        child = (void*) &leafNodes[(offset + myIndex * branching_factor) - branching_factor];
        LeafNode* const l = &((LeafNode*) child)[getZOrderNodeId<branching_factor>(my_voxel_id, 0)];

        setNode(l, my_set_basic_data, my_reset_basic_data);

        //      printf(
//          "SET LEAF NODE %lu! Address: %u HEX: %p\n",
//          myVoxel.voxelId,
//          shared_levelCount[0] + myIndex * branching_factor - branching_factor
//              + getZOrderNodeId<branching_factor>(myVoxel.voxelId, 0),
//          child);
      }
      //__syncthreads();

      if (thread_id == 0)
        shared_levelCount[0] += total_sum * branching_factor;
      __syncthreads();
    }

//    if (thread_id == 0)
//      printf("New leaf nodes set!\n");

    // ##### set inner nodes of newly initialized memory bottom-up #####
    bool hasWork = false;
    for (uint32_t l = max(1, target_level); any_thread<num_warps>(hasWork = (isActive & (l < commonLevel) & (l < level)));
        ++l)
    {
      const bool isLeftMost = hasWork & (commonLevel > (l + 1));
      uint32_t total_sum;
      const uint32_t offset = shared_levelCount[l]; // relies on the __syncthreads() of thread_prefix()
      const uint32_t allVotes = thread_prefix<num_warps>(shared_prefix_sum, thread_id,
                                                           total_sum, isLeftMost);
      if (hasWork)
      {
        const uint32_t myIndex = getThreadPrefix_Inclusive(shared_prefix_sum, allVotes, thread_id);
        assert(myIndex > 0 || offset >= branching_factor);
        InnerNode* const iNode = &innerNodes[(offset + myIndex * branching_factor) - branching_factor];
        InnerNode* const tmp = &iNode[getZOrderNodeId<branching_factor>(my_voxel_id, l)];

        setNode(tmp, child, my_set_basic_data, my_reset_basic_data);
//
//        if (child != NULL)
//          tmp->setChildPtr(child);
//        else
//        {
//          assert(l == target_level);
//          // TODO FREE TREE handle this case since a part of the tree is unlinked!
//        }
//        insert_setInnerNodeStatus<InnerNode, SET_STATUS, SET_UPDATE_FLAG>(tmp, my_status, l, target_level);

        child = (void*) iNode;
      }
      //__syncthreads();

      // increment offset for next chunk of work
      if (thread_id == 0)
        shared_levelCount[l] += total_sum * branching_factor;
      __syncthreads();
    }

//    if (thread_id == 0)
//      printf("Set inner nodes bottom up finished!\n");

    // set status and child of inner nodes which received a new child
    if (isActive && (level < commonLevel))
    {
      assert(node != NULL);

      updateNode((InnerNode*) node, child, my_set_basic_data, my_reset_basic_data);

//      printf("set connection node %lu to %p\n", myVoxel.voxelId, child);
    }

//    if (thread_id == 0)
//      printf("Run finished!\n");

  }

//  if (thread_id == 0)
//    printf("setNodes() finished!\n");
}

struct BitMapProperties
{
  enum Status
  {
    UNKNOWN = 0, FREE = 1
  };

  uint32_t* coordinate_x;
  uint32_t* coordinate_y;
  uint32_t* coordinate_z;
  uint32_t corrdinates_size;
  uint32_t min_x, min_y, min_z;
  uint32_t max_x, max_y, max_z;
  uint32_t size_x, size_y, size_z;

  uint32_t* d_ptr;
  uint32_t size;

//  cudaPitchedPtr devPitchedPtr;
//
  template<typename T, std::size_t branching_factor>
  __host__ __device__
  __forceinline__ T* computePtr(uint32_t x, uint32_t y, uint32_t z, uint32_t* bit_index)
  {
    if (min_x <= x && x <= max_x && min_y <= y && y <= max_y && min_z <= z && z <= max_z)
    {
      x -= min_x;
      y -= min_y;
      z -= min_z;

      const uint32_t branching_factor_third_root = (uint32_t) powf(branching_factor, 1.0 / 3);
      const uint32_t log_branching_factor_third_root = uint32_t(log2f(branching_factor_third_root));
      const uint32_t num_x_in_element = branching_factor_third_root * sizeof(T) * 8
          / branching_factor_third_root / branching_factor_third_root / branching_factor_third_root;
      const uint32_t num_elements_in_x = size_x / num_x_in_element;
      const uint32_t num_elements_in_y = size_y / branching_factor_third_root;

      uint32_t x_bit_index = x & (1 << (log_branching_factor_third_root - 1));
      uint32_t y_bit_index = y & (1 << (log_branching_factor_third_root - 1));
      uint32_t z_bit_index = z & (1 << (log_branching_factor_third_root - 1));
      *bit_index = morton_code(x_bit_index, y_bit_index, z_bit_index);
      *bit_index += branching_factor * ((x % num_x_in_element) / branching_factor_third_root);
      assert(*bit_index < 32);

      return d_ptr + x / num_x_in_element + y / branching_factor_third_root * num_elements_in_x
          + z / branching_factor_third_root * num_x_in_element * num_elements_in_y;
    }
    else
      return NULL;
  }
//
//  template<typename T>
//  __host__ __device__
//  __forceinline__ T* computeOffsetPtr(uint32_t offset_x, uint32_t offset_y, uint32_t offset_z)
//  {
//    if (offset_x <= size_x && offset_y <= size_y && offset_z <= size_z)
//      return (((T*) devPitchedPtr.ptr) + offset_z * devPitchedPtr.pitch * size_y
//          + offset_y * devPitchedPtr.pitch + offset_x);
//    else
//      return NULL;
//  }
};

template<std::size_t branching_factor, typename InnerNode>
__global__
static void kernel_rayInsert(const gpu_voxels::Vector3ui sensor_origin, voxel_count* free_space_count,
                             MapProperties<typename InnerNode::RayCastType, branching_factor> map_properties)
{
#define USE_MORTON
#define SWAP_FROM_TO

  const uint32_t id = blockIdx.x * blockDim.x + threadIdx.x;
  const uint32_t task_id = blockIdx.x;
  const uint32_t thread_id = threadIdx.x;
  gpu_voxels::Vector3ui from = gpu_voxels::Vector3ui(sensor_origin.x - map_properties.min_x,
                                       sensor_origin.y - map_properties.min_y,
                                       sensor_origin.z - map_properties.min_z);
  uint32_t my_voxel_count = 0;
  const uint32_t num_voxel = map_properties.coordinates_size;

  for (voxel_count i = id; i < num_voxel; i += gridDim.x * blockDim.x)
  {
    gpu_voxels::Vector3ui to = gpu_voxels::Vector3ui(map_properties.coordinate_x[i] - map_properties.min_x,
                                       map_properties.coordinate_y[i] - map_properties.min_y,
                                       map_properties.coordinate_z[i] - map_properties.min_z);
#ifdef SWAP_FROM_TO
    {
    const gpu_voxels::Vector3ui t = from;
    from = to;
    to = t;
    }
#endif


    // differences
    int32_t difference_x = __sad(to.x, from.x, 0);        //(to.x > from.x) ? to.x - from.x : from.x - to.x;
    int32_t x_increment = (to.x > from.x) ? 1 : -1;

    int32_t difference_y = __sad(to.y, from.y, 0); //(to.y > from.y) ? to.y - from.y : from.y - to.y;
    int32_t y_increment = (to.y > from.y) ? 1 : -1;

    int32_t difference_z = __sad(to.z, from.z, 0); //(to.z > from.z) ? to.z - from.z : from.z - to.z;
    int32_t z_increment = (to.z > from.z) ? 1 : -1;

    // start values
    int32_t x = from.x;
    int32_t y = from.y;
    int32_t z = from.z;

    // number of cells to visit
    int32_t n = difference_x + difference_y + difference_z + 1;

    // error between x- and y- difference
    int32_t error_xy = difference_x - difference_y;

    // error between x- and z- difference
    int32_t error_xz = difference_x - difference_z;

    // error between y- and z- difference
    int32_t error_yz = difference_y - difference_z;

    // double differences to avoid float values
    difference_x *= 2;
    difference_y *= 2;
    difference_z *= 2;

    for (; n > 0; n--)
    {
#ifdef USE_MORTON
      typename InnerNode::RayCastType* val = map_properties.computeMortonOffsetPtr(x, y, z);

//      typename InnerNode::RayCastType* val = &map_properties.d_ptr[x + y * map_properties.size_x
//          + z * map_properties.size_x * map_properties.size_y];

#else
      typename InnerNode::RayCastType* val = map_properties.computeOffsetPtr(x, y, z);
#endif
      handleRayHit(val, x, y, z);
      ++my_voxel_count;

// ----

      if ((error_xy > 0) & (error_xz > 0))
      {
        // walk in x direction until error_xy or error_xz is below 0
        x += x_increment;
        error_xy -= difference_y;
        error_xz -= difference_z;
      }
      else
      {
        if (error_yz > 0)
        {
          // walk in y direction
          y += y_increment;
          error_xy += difference_x;
          error_yz -= difference_z;
        }
        else
        {
          // walk in z direction
          z += z_increment;
          error_xz += difference_x;
          error_yz += difference_y;
        }
      }
    }
  }

  free_space_count[task_id * blockDim.x + thread_id] = my_voxel_count;
#undef USE_MORTON
#undef SWAP_FROM_TO
}

template<std::size_t branching_factor, typename InnerNode>
__device__ __forceinline__
gpu_voxels::Vector3ui getFirstVoxel(
    uint32_t cube_id, MapProperties<typename InnerNode::RayCastType, branching_factor>& byte_map_properties)
{
  const uint32_t branching_factor_third_root = (uint32_t) powf(branching_factor, 1.0 / 3);
  const uint32_t sx = byte_map_properties.size_x / branching_factor_third_root;
  const uint32_t sy = byte_map_properties.size_y / branching_factor_third_root;
  const uint32_t plane_x_y = cube_id % (sx * sy);

  gpu_voxels::Vector3ui v(0, 0, 0);
  v.x = (plane_x_y % sx) * branching_factor_third_root;
  v.y = (plane_x_y / sx) * branching_factor_third_root;
  v.z = (cube_id / (sx * sy)) * branching_factor_third_root;

  assert(
      v.x < byte_map_properties.size_x && v.y < byte_map_properties.size_y
          && v.z < byte_map_properties.size_z);

  return v;
}

template<std::size_t branching_factor, bool COUNT_MODE>
__global__
static void kernel_packByteMap_MemEfficient_Coa2(voxel_count* num_this_level, voxel_count* num_next_level,
                                                 MapProperties<uint8_t, branching_factor> byte_map_properties,
                                                 voxel_count* index_this_level = NULL,
                                                 voxel_count* index_next_level = NULL,
                                                 OctreeVoxelID* voxel_id_this_level = NULL,
                                                 OctreeVoxelID* voxel_id_next_level = NULL)
{

  assert(branching_factor == 8);

// each warp handles an x-row and then collects/interchanges the results with the other warps to compute the final result

  const uint32_t NUM_WARPS = 4;
  const uint32_t NUM_THREADS = WARP_SIZE * NUM_WARPS;
  assert(NUM_THREADS == blockDim.x);

  const uint32_t work_items_at_once = (NUM_THREADS * 4) / branching_factor;
  const uint32_t branching_factor_third_root = (uint32_t) powf(branching_factor, 1.0 / 3);

  __shared__ gpu_voxels::Vector3ui shared_cubes[NUM_WARPS];
  __shared__ uint32_t* shared_ptr[NUM_WARPS];
  __shared__ voxel_count shared_num_this_level[NUM_THREADS];
  __shared__ voxel_count shared_num_next_level;
  __shared__ uint32_t shared_votes[2];

  const uint32_t block_id = blockIdx.x;
  const uint32_t num_threads = blockDim.x;
  const uint32_t thread_id = threadIdx.x;
  const uint32_t warp_id = thread_id / WARP_SIZE;
  const uint32_t lane_id = thread_id % WARP_SIZE;

  assert(
      ((byte_map_properties.size_x * byte_map_properties.size_y * byte_map_properties.size_z)
          % branching_factor) == 0);

  const uint32_t num_work_items = (byte_map_properties.size_x * byte_map_properties.size_y
      * byte_map_properties.size_z) / branching_factor;

  assert(((num_work_items) % work_items_at_once) == 0);

  const uint32_t free_mask[4] =
  { 1, 1 << 8, 1 << 16, 1 << 24 };

  const uint32_t is_free_mask = free_mask[0] | free_mask[1];

// init shared memory counter
  shared_num_this_level[thread_id] = 0;
  if (thread_id == 0)
    shared_num_next_level = 0;

  __syncthreads();

  for (uint32_t i = block_id * work_items_at_once; i < num_work_items; i += work_items_at_once * gridDim.x)
  {
#pragma unroll
    for (uint32_t j = thread_id; j < 2; j += num_threads)
      shared_votes[j] = UINT_MAX;

    //shared_warp_vote = UINT_MAX;
    for (uint32_t k = thread_id; k < NUM_WARPS; k += num_threads)
    {
      gpu_voxels::Vector3ui tmp = getFirstVoxel<branching_factor>(i, byte_map_properties);
      tmp.y += k % 2;
      tmp.z += k / 2;
      shared_ptr[k] = (uint32_t*) byte_map_properties.computeOffsetPtr(tmp.x, tmp.y, tmp.z);
      shared_cubes[k] = tmp;

      assert((tmp.x + work_items_at_once * branching_factor_third_root) <= byte_map_properties.size_x);
    }
    __syncthreads();

    uint32_t is_occupied = shared_ptr[warp_id][lane_id];

#pragma unroll
    for (uint32_t i = 0; i < 2; ++i)
    {
      if (COUNT_MODE)
      {
        if (i == 0)
          shared_num_this_level[thread_id] += __popc(is_occupied);
      }

      bool occupied = (((is_occupied >> (i * 16)) & is_free_mask) == is_free_mask);
      uint32_t all_votes = BALLOT(occupied);

      if (lane_id == 0)
        atomicAnd(&shared_votes[i], all_votes);
    }
    __syncthreads();

    // ### handle this level ###
    if (!COUNT_MODE)
    {
#pragma unroll
      for (uint32_t i = 0; i < 2; ++i)
      {
        if ((shared_votes[i] & (1 << lane_id)) == 0)
        {
#pragma unroll
          for (uint32_t k = 0; k < 2; ++k)
          {
            if (((free_mask[i * 2 + k] & is_occupied) != 0) & (voxel_id_this_level != NULL))
            {
              voxel_id_this_level[atomicAdd(index_this_level, 1)] = morton_code60(
                  byte_map_properties.min_x + shared_cubes[warp_id].x + lane_id * 4 + i * 2 + k,
                  byte_map_properties.min_y + shared_cubes[warp_id].y,
                  byte_map_properties.min_z + shared_cubes[warp_id].z);
            }
          }
        }
      }

      // ### handle next level ###
      for (uint32_t i = warp_id; i < 2; i += NUM_WARPS)
      {
        if ((shared_votes[i] & (1 << lane_id)) != 0)
        {
          voxel_id_next_level[atomicAdd(index_next_level, 1)] = morton_code60(
              byte_map_properties.min_x + shared_cubes[0].x + lane_id * 4 + i * 2,
              byte_map_properties.min_y + shared_cubes[0].y, byte_map_properties.min_z + shared_cubes[0].z);
        }
      }
    }

    if (COUNT_MODE)
    {
#pragma unroll
      for (uint32_t i = thread_id; i < 2; i += num_threads)
        atomicAdd(&shared_num_next_level, (uint32_t) __popc(shared_votes[i]));
    }

    __syncthreads();
  }
  __syncthreads();

  if (COUNT_MODE)
  {
    REDUCE(shared_num_this_level, thread_id, NUM_THREADS, +)
//REDUCE(shared_num_next_level, thread_id, work_items_at_once, +)

    if (thread_id == 0)
    {
      num_this_level[block_id] = shared_num_this_level[0] - shared_num_next_level * branching_factor;
      num_next_level[block_id] = shared_num_next_level;
    }
  }
}

template<uint32_t num_threads, std::size_t branching_factor, bool COUNT_MODE, bool MAP_ONLY_MODE,
    bool PACKING, typename InnerNode>
__global__
static void kernel_packMortonL0Map(
    voxel_count* num_this_level,
    voxel_count* num_next_level,
    MapProperties<typename InnerNode::RayCastType, branching_factor> map_properties,
    voxel_count* index_this_level = NULL,
    voxel_count* index_next_level = NULL,
    OctreeVoxelID* voxel_id_this_level = NULL,
    OctreeVoxelID* voxel_id_next_level = NULL,
    typename InnerNode::NodeData::BasicData* basic_data_this_level = NULL,
    typename InnerNode::NodeData::BasicData* basic_data_next_level = NULL,
    MapProperties<typename InnerNode::RayCastType, branching_factor> d_next_level_map =
        MapProperties<typename InnerNode::RayCastType, branching_factor>())
{
  typedef typename InnerNode::RayCastType RayCastType;
  typedef typename RayCastType::Type T;
  const uint32_t num_blocks = gridDim.x;
  const uint32_t block_id = blockIdx.x;
  const uint32_t thread_id = threadIdx.x;
  const uint32_t reduction_index = (thread_id / branching_factor) * branching_factor;
  const uint32_t num_warps = num_threads / WARP_SIZE;
  T* const map_ptr = (T*) map_properties.d_ptr;

  assert(num_threads == blockDim.x);
  assert((num_threads % branching_factor) == 0);

  volatile __shared__ T shared_data[num_threads * sizeof(uint32_t) / sizeof(T)];
  volatile __shared__ uint32_t shared_this_level_prefix[num_warps];
  volatile __shared__ uint32_t shared_next_level_prefix[num_warps];
  __shared__ uint32_t shared_this_level_index;
  __shared__ uint32_t shared_next_level_index;

  volatile uint32_t* const shared_count_reduce = (uint32_t*) shared_data;

  uint32_t my_this_level_count = 0, my_next_level_count = 0;

  for (uint32_t i = block_id * num_threads; i < map_properties.size_v; i += num_blocks * num_threads)
  {
    __syncthreads(); // sync for the use of continue

    const uint32_t work_size = min(num_threads, uint32_t(map_properties.size_v - i));
    const bool is_active = thread_id < work_size;

    // load data into shared mem
    if (is_active)
      shared_data[thread_id] = map_ptr[i + thread_id];
    // T* ptr =  map_ptr + i;
    // (work_size * sizeof(T))
//    blockCopy(shared_data, map_ptr + i, work_size * sizeof(T), thread_id, num_threads);
    __syncthreads();

    RayCastType my_value;
    if (is_active)
      my_value.value = shared_data[thread_id];
    const bool is_invalid = !is_active || !isValidValue(my_value);
    const bool all_invalid = __syncthreads_and(is_invalid);
    if (all_invalid)
      continue; // nothing to do here

    RayCastType max_value, min_value;
    const bool abort = maxMinReduction(my_value, (volatile typename InnerNode::RayCastType*) shared_data,
                                       thread_id, num_threads, branching_factor, is_active, &max_value,
                                       &min_value);
    if (abort)
      continue;

    const bool pack = PACKING ? isPackingPossible(max_value, min_value) : false;
    const bool has_item_this_level = !is_invalid && !pack;
    const bool has_item_next_level = !is_invalid && pack && reduction_index == thread_id;

    if (!COUNT_MODE)
    {
      // ########## compute index in output array for each thread and write data to it ##########
      // handle this level
      uint32_t this_level_sum = 0;
      uint32_t this_level_votes = thread_prefix<num_warps>((uint32_t*) shared_this_level_prefix, thread_id,
                                                           this_level_sum, has_item_this_level);
      if (thread_id == 0 && this_level_sum != 0)
        shared_this_level_index = atomicAdd(index_this_level, this_level_sum);
      __syncthreads();
      if (has_item_this_level)
      {
        const uint32_t pos = shared_this_level_index
            + getThreadPrefix((uint32_t*) shared_this_level_prefix, this_level_votes, thread_id);

//        uint32_t index = i + thread_id;
//        uint32_t plane_x_y = map_properties.size_x * map_properties.size_y;
//        gpu_voxels::Vector3ui blub((index % plane_x_y) % map_properties.size_x,
//                            (index % plane_x_y) / map_properties.size_x, index / plane_x_y);
//        voxel_id_this_level[pos] = morton_code60(
//            blub + gpu_voxels::Vector3ui(map_properties.min_x, map_properties.min_y, map_properties.min_z));

        voxel_id_this_level[pos] = map_properties.computeVoxelID_morton(i + thread_id);
        packData(basic_data_this_level, pos, max_value, min_value, my_value);
      }

      if (MAP_ONLY_MODE)
      {
        if (has_item_next_level)
        {
          RayCastType* const ptr = map_properties.computeNextMapPtr(i + thread_id, d_next_level_map);
          packData(ptr, max_value, min_value, my_value);
        }
      }
      else if (PACKING)
      {
        // handle next level
        uint32_t next_level_sum = 0;
        uint32_t next_level_votes = thread_prefix<num_warps>((uint32_t*) shared_next_level_prefix, thread_id,
                                                             next_level_sum, has_item_next_level);
        if (thread_id == 0 && next_level_sum != 0)
          shared_next_level_index = atomicAdd(index_next_level, next_level_sum);
        __syncthreads();
        if (has_item_next_level)
        {
          const uint32_t pos = shared_next_level_index
              + getThreadPrefix((uint32_t*) shared_next_level_prefix, next_level_votes, thread_id);
          voxel_id_next_level[pos] = map_properties.computeVoxelID_morton(i + thread_id);
          packData(basic_data_next_level, pos, max_value, min_value, my_value);
        }
      }
      // ######################################################################################
    }
    else
    {
      if (has_item_this_level)
        ++my_this_level_count;
      if (has_item_next_level)
        ++my_next_level_count;
    }
  }

  if (COUNT_MODE)
  {
    // count reduce
    __syncthreads();
    shared_count_reduce[thread_id] = my_this_level_count;
    __syncthreads();
    REDUCE(shared_count_reduce, thread_id, num_threads, +)
    if (thread_id == 0)
      num_this_level[block_id] = shared_count_reduce[0];
    __syncthreads();

    shared_count_reduce[thread_id] = my_next_level_count;
    __syncthreads();
    REDUCE(shared_count_reduce, thread_id, num_threads, +)
    if (thread_id == 0)
      num_next_level[block_id] = shared_count_reduce[0];
  }
}

/*
 * Method for computing the required space without duplicates first.
 * Removes the duplicates by moving the elements to a new location in a second call.
 */
template<bool COUNT_MODE>
__global__
static void kernel_removeDuplicates(OctreeVoxelID* free_space, voxel_count num_voxel,
                                    OctreeVoxelID* free_space_wo_duplicates, voxel_count* voxel_count_wo_duplicates)
{
  assert(blockDim.x <= WARP_SIZE); // otherwise have to implement inter warp prefix sum

  __shared__ uint32_t shared_voxel_count[WARP_SIZE];
  __shared__ uint32_t shared_offset;

  const uint32_t num_threads = blockDim.x;
  const uint32_t task_id = blockIdx.x;
  const uint32_t thread_id = threadIdx.x;
  const voxel_count chunk_size = ceil(double(num_voxel) / gridDim.x);
  const voxel_count from = chunk_size * task_id;
  const voxel_count to = (voxel_count) min((unsigned long long int) (from + chunk_size),
                                           (unsigned long long int) num_voxel);

  if (COUNT_MODE)
    shared_voxel_count[thread_id] = 0;
  else
    shared_offset = voxel_count_wo_duplicates[task_id];

  for (voxel_count i = from; i < to; i += num_threads)
  {
    voxel_count my_id = i + thread_id;
    OctreeVoxelID my_element = free_space[my_id];
    bool is_no_duplicate = (my_id < to) && ((my_id == 0) || (my_element != free_space[my_id - 1]))
        && (my_element != INVALID_VOXEL);

// first element in queue of duplicates counts
    if (COUNT_MODE)
      shared_voxel_count[thread_id] += is_no_duplicate;
    else
    {
// vote to compute offset within the block
      uint32_t votes = BALLOT(is_no_duplicate);
      if (is_no_duplicate)
      {
        uint32_t my_index = __popc(votes << (32 - thread_id));
        free_space_wo_duplicates[shared_offset + my_index] = my_element;
      }
      __syncthreads();

      if (thread_id == 0)
        shared_offset += __popc(votes);
      __syncthreads();
    }
  }

// copy into global memory
  if (COUNT_MODE)
  {
    __syncthreads();
    REDUCE(shared_voxel_count, thread_id, num_threads, +);
    if (thread_id == 0)
      voxel_count_wo_duplicates[task_id] = shared_voxel_count[0];
  }
}

/*
 * Computes the number for the next level, by packing the voxel by their parent
 * and the number for the current level, which can't be packed together.
 */
template<std::size_t branching_factor, bool COUNT_MODE>
__global__
static void kernel_packVoxel(OctreeVoxelID* free_space, voxel_count num_voxel, voxel_count* voxel_count_this_level,
                             voxel_count* voxel_count_next_level, uint32_t level,
                             OctreeVoxelID* free_space_this_level, OctreeVoxelID* free_space_next_level)
{
  assert(blockDim.x == WARP_SIZE); // otherwise have to implement inter warp prefix sum

  __shared__ voxel_count shared_count_this_level;
  __shared__ voxel_count shared_count_next_level;
  __shared__ voxel_count shared_num_childs_split_parent;
  __shared__ voxel_count shared_votes_this_level;
  __shared__ bool shared_ends_at_split;
  __shared__ bool shared_is_this_level;
  __shared__ voxel_count shared_num_with_same_parent;

  const uint32_t num_threads = blockDim.x;
  const uint32_t task_id = blockIdx.x;
  const uint32_t thread_id = threadIdx.x;
  const voxel_count chunk_size = ceil(double(num_voxel) / gridDim.x);
  const voxel_count from = chunk_size * task_id;
  const voxel_count to = (voxel_count) min((unsigned long long int) (from + chunk_size),
                                           (unsigned long long int) num_voxel);

  if (thread_id == 0)
  {
    if (COUNT_MODE)
    {
      shared_count_this_level = 0;
      shared_count_next_level = 0;
      shared_num_childs_split_parent = 0;
    }
    else
    {
      shared_count_this_level = voxel_count_this_level[task_id];
      shared_count_next_level = voxel_count_next_level[task_id];
      shared_num_childs_split_parent = 0;
      shared_votes_this_level = 0;
      shared_num_with_same_parent = 0;
    }
  }
  __syncthreads();

  for (voxel_count i = from; i < to; i += num_threads)
  {
    if (thread_id == 0)
    {
      shared_num_childs_split_parent = 0;
      if (!COUNT_MODE)
      {
        shared_votes_this_level = 0;
      }
    }
    __syncthreads();

    voxel_count my_id = i + thread_id;
    uint32_t votes = 0;
    bool has_new_parent = false;
    OctreeVoxelID my_voxel_id = INVALID_VOXEL;
    bool is_active = (my_id < to);

    if (is_active)
    {
      my_voxel_id = free_space[my_id];
      has_new_parent = (my_id == 0)
          || (getZOrderPrefix<branching_factor>(my_voxel_id, level)
              != getZOrderPrefix<branching_factor>(free_space[my_id - 1], level));
//      has_new_parent = (thread_id == 0)
//          || (getZOrderPrefix<branching_factor>(my_voxel_id, level)
//              != getZOrderPrefix<branching_factor>(free_space[my_id - 1], level));
    }
    votes = __brev(BALLOT(has_new_parent));

    bool is_this_level = false;
    uint32_t num_childs_with_same_parent = 0;
    uint32_t num_childs_split_parent = 0;
    if (has_new_parent)
    {
      num_childs_with_same_parent = min(
          min(1 + __clz(votes << (thread_id + 1)), (uint32_t) (WARP_SIZE - thread_id)),
          (uint32_t) (to - my_id));
      assert(num_childs_with_same_parent <= branching_factor && num_childs_with_same_parent != 0);

// check whether the voxel with the same parent is split
// so we need to read the next block of data but only process the split voxels
      if ((num_childs_with_same_parent != branching_factor)
          & ((my_id + num_childs_with_same_parent) != num_voxel)
          & (((my_id + num_childs_with_same_parent) == to)
              | ((thread_id + num_childs_with_same_parent) == WARP_SIZE)))
      {
        // can only happen to one thread per WARP
        shared_num_childs_split_parent = num_childs_split_parent = num_childs_with_same_parent;
      }
      else
      {
        is_this_level = (num_childs_with_same_parent != branching_factor);
      }

      if (num_childs_split_parent == 0)
      {
        if (COUNT_MODE)
        {
          if (is_this_level)
          {
            atomicAdd(&shared_count_this_level, num_childs_with_same_parent);
            //printf("1 this level count id %lu count %u \n", my_voxel_id, num_childs_with_same_parent);
          }
          else
          {
            //printf("1 next level start %lu\n", my_voxel_id);
            atomicAdd(&shared_count_next_level, 1);
          }
        }
        else
        {
          // generate look up bit vectors to be able to determine each thread's work (this level, next level,
          // nothing since it's a left-over from previous block)
          if (is_this_level)
            atomicAdd(&shared_votes_this_level, ((1 << num_childs_with_same_parent) - 1) << thread_id);
        }
      }
    }
    __syncthreads();

    if (!COUNT_MODE)
    {
// ##### handle next level #####
      bool next_level_vote = (num_childs_split_parent == 0) & (is_active) & has_new_parent & !is_this_level;
      uint32_t next_level_votes = BALLOT(next_level_vote);
      if (next_level_vote)
      {
        uint32_t next_level_index = __popc(next_level_votes << (32 - thread_id));
        free_space_next_level[shared_count_next_level + next_level_index] = my_voxel_id;
        //printf("1 write next level voxel %lu\n", my_voxel_id);
      }
      __syncthreads();

      if (thread_id == 0)
        shared_count_next_level += __popc(next_level_votes);
      __syncthreads();

// ##### handle this level #####
      if (((shared_votes_this_level & (1 << thread_id)) > 0) & (free_space_this_level != NULL))
      {
        uint32_t this_level_index = __popc(shared_votes_this_level << (32 - thread_id));
        free_space_this_level[shared_count_this_level + this_level_index] = my_voxel_id;
        //printf("1 write this level voxel %lu\n", my_voxel_id);
      }
      __syncthreads();

      if (thread_id == 0)
        shared_count_this_level += __popc(shared_votes_this_level);
      __syncthreads();
    }

// ##### process next block to handle split parent voxel #####
    if (shared_num_childs_split_parent != 0)
    {
// count voxel with same parent
      my_id += min(min(num_threads, chunk_size), to - i); // TODO move only "num_threads - num_childs_split_parent" far, so thread 0 reads the first voxel

      has_new_parent = false;
      if ((thread_id < (branching_factor - shared_num_childs_split_parent + 1)) & (my_id < num_voxel))
        has_new_parent = (getZOrderPrefix<branching_factor>(free_space[my_id], level)
            != getZOrderPrefix<branching_factor>(free_space[my_id - 1], level));
      __syncthreads();

      votes = __brev(BALLOT(has_new_parent));
      if (thread_id == 0)
      {
        shared_ends_at_split = false;
        shared_is_this_level = false;
        shared_num_with_same_parent = shared_num_childs_split_parent;
        if (has_new_parent)
          shared_ends_at_split = true;
        else
        {
          shared_num_with_same_parent += min(
              min(min(1 + __clz(votes << (thread_id + 1)), (uint32_t) (WARP_SIZE - thread_id)),
                  (uint32_t) (num_voxel - my_id)),
              (uint32_t) (branching_factor - shared_num_childs_split_parent));
          assert(shared_num_with_same_parent <= branching_factor);

          shared_is_this_level = (shared_num_with_same_parent != branching_factor);
        }
      }
      __syncthreads();

      if (COUNT_MODE)
      {
        if (thread_id == 0)
        {
          if (shared_ends_at_split | shared_is_this_level)
          {
            // only handle previous parent node, since it ends at the split border
            shared_count_this_level += shared_num_with_same_parent;
            //printf("2 this level count id %lu count %u \n", my_voxel_id, shared_num_with_same_parent);
          }
          else if (!shared_is_this_level)
          {
            //printf("2 next level start %lu\n", my_voxel_id);
            ++shared_count_next_level;
          }
          else
          {
            // Should never happen
            assert(false);
          }
        }
      }
      else
      {
        if (shared_ends_at_split | shared_is_this_level)
        {
          // ##### handle this level #####
          if ((thread_id < shared_num_with_same_parent) & (free_space_this_level != NULL))
          {
            free_space_this_level[shared_count_this_level + thread_id] = free_space[my_id
                - shared_num_childs_split_parent];
//            printf("2 write this level voxel id %lu count %u tid %u\n",
//                   free_space[my_id - shared_num_childs_split_parent], shared_num_with_same_parent,
//                   thread_id);
          }
          __syncthreads();

          if (thread_id == 0)
            shared_count_this_level += shared_num_with_same_parent;
          __syncthreads();
        }
        else if (!shared_is_this_level)
        {
          if (thread_id == 0)
          {
            free_space_next_level[shared_count_next_level++] = free_space[my_id
                - shared_num_childs_split_parent];
//            printf("2 write next level voxel %lu\n", my_voxel_id);
          }
          __syncthreads();
        }
        __syncthreads();
      }
    }
    __syncthreads();
  }

// copy into global memory
  if (COUNT_MODE)
  {
    if (thread_id == 0)
    {
      voxel_count_this_level[task_id] = shared_count_this_level;
      voxel_count_next_level[task_id] = shared_count_next_level;
    }
    __syncthreads();
  }
}

/*
 * Splits a vector of Voxel into separate vectors.
 */
template<bool need_voxel_id, bool need_occupancy, bool need_coordinates, bool need_separate_coordinates>
__global__
static void kernel_split_voxel_vector(Voxel* voxel, voxel_count num_voxel, OctreeVoxelID* voxel_id,
                                      Probability* occupancy, gpu_voxels::Vector3ui* coordinates, uint32_t* x,
                                      uint32_t* y, uint32_t* z)
{
  for (voxel_count i = blockIdx.x * blockDim.x + threadIdx.x; i < num_voxel; i += blockDim.x * gridDim.x)
  {
    if (need_voxel_id)
      voxel_id[i] = voxel[i].voxelId;
    if (need_occupancy)
      occupancy[i] = voxel[i].getOccupancy();
    if (need_coordinates)
      coordinates[i] = voxel[i].coordinates;
    if (need_separate_coordinates)
    {
      x[i] = voxel[i].coordinates.x;
      y[i] = voxel[i].coordinates.y;
      z[i] = voxel[i].coordinates.z;
    }
  }
}

__global__
static void kernel_checkBlub(OctreeVoxelID* voxel_id1, voxel_count num_voxel, OctreeVoxelID* voxel_id2)
{
  for (uint32_t i = 0; i < num_voxel; ++i)
  {
    if (voxel_id1[i] != voxel_id2[i])
    {
      printf("pos %u voxelList %lu vs. %lu\n", i, voxel_id1[i], voxel_id2[i]);
      break;
    }
  }
}

template<std::size_t branching_factor, std::size_t level_count, typename InnerNode, typename LeafNode>
__device__ __forceinline__
static void traverse(InnerNode* root, OctreeVoxelID voxel, uint32_t target_level, void** out_inner_node,
                     uint32_t* out_level, bool* has_part_status)
{
  InnerNode* node = root;
  int32_t level = level_count - 2;
  for (; (level > target_level) & node->hasStatus(ns_PART); --level)
    node = &((InnerNode*) node->getChildPtr())[getZOrderNodeId<branching_factor>(voxel, level)];

  if (node->hasStatus(ns_PART) & (level == target_level))
  {
    if (level == 0)
      *out_inner_node =
          (void*) &((LeafNode*) (node->getChildPtr()))[getZOrderNodeId<branching_factor>(voxel, level)];
    else
      *out_inner_node =
          (void*) &((InnerNode*) (node->getChildPtr()))[getZOrderNodeId<branching_factor>(voxel, level)];
    --level;
  }
  else
    *out_inner_node = (void*) node;
  ++level;
  *out_level = level;
  *has_part_status = (level != 0) && (((InnerNode*) *out_inner_node)->hasStatus(ns_PART));
}

template<std::size_t branching_factor, std::size_t level_count, typename InnerNode, typename LeafNode>
__global__
static void kernel_checkTree(InnerNode* root, uint8_t* error)
{
  __shared__ InnerNode* shared_stack[level_count];

  *error = 0;

  OctreeVoxelID last_voxel = (OctreeVoxelID) powf(branching_factor, level_count - 1);
  for (OctreeVoxelID i = 0; i < last_voxel && !*error;)
  {
    shared_stack[level_count - 1] = root;
    InnerNode* node = root;
    int32_t level = level_count - 2;
    for (; (level > 0) && node->hasStatus(ns_PART); --level)
    {
      node = &((InnerNode*) node->getChildPtr())[getZOrderNodeId<branching_factor>(i, level)];
      shared_stack[level] = node;
    }
    ++level;
    if (!(shared_stack[level]->hasStatus(ns_PART)))
      ++level;

    // check path to root
    for (uint32_t l = level; l < level_count; ++l)
    {
      bool result = isValidParentStatus(shared_stack[l]->getChildPtr(), branching_factor, l - 1,
                                        shared_stack[l]);
      if (!result)
      {
        printf("ERROR InnerNode on level %u for voxel with id %lu has wrong status flags\n", l, i);
//          printf(
//              "ERROR InnerNode on level %u for voxel with id %lu has wrong status flags: %u but should be %u\n",
//              l, i, uint32_t(parent_status), uint32_t(childs_status_flags));
        *error = 1;
        break;
      }
    }

    i += uint32_t(powf(branching_factor, max(level - 1, 1)));
//printf("i %u\n", i);
  }
}

__global__
static void kernel_splitCoordinates(gpu_voxels::Vector3ui* coordinates, voxel_count num_voxel, uint32_t* x,
                                      uint32_t* y, uint32_t* z)
{
  for (voxel_count i = blockIdx.x * blockDim.x + threadIdx.x; i < num_voxel; i += blockDim.x * gridDim.x)
  {
    x[i] = coordinates[i].x;
    y[i] = coordinates[i].y;
    z[i] = coordinates[i].z;
  }
}

}
}

#endif /* KERNEL_OCTREE_H_ */
