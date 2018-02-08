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
 * \date    2014-12-10
 *
 */
//----------------------------------------------------------------------/*
#ifndef GPU_VOXELS_OCTREE_NTREE_HPP_INCLUDED
#define GPU_VOXELS_OCTREE_NTREE_HPP_INCLUDED

#include <stdio.h>
#include <assert.h>
#include <limits.h>
#include <float.h>
#include <math.h>
#include <algorithm>
#include <ostream>
#include <istream>

// CUDA
#include <cuda_runtime.h>
#include <cuda_profiler_api.h>
#include <driver_types.h>

#include <gpu_voxels/helpers/cuda_handling.h>

// Thrust
#include <thrust/sort.h>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/reduce.h>
#include <thrust/scan.h>
#include <thrust/unique.h>
#include <thrust/set_operations.h>
#include <thrust/device_ptr.h>
#include <thrust/count.h>
#include <thrust/iterator/constant_iterator.h>
#include <thrust/fill.h>
#include <thrust/extrema.h>

#if CUDA_VERSION < 9000
#define CUB_NS_PREFIX namespace thrust { namespace system { namespace cuda { namespace detail {
#define CUB_NS_POSTFIX                  }                  }                }                  }
#define cub cub_
#include <thrust/system/cuda/detail/cub/device/device_radix_sort.cuh>
#undef cub
#undef CUB_NS_PREFIX
#undef CUB_NS_POSTFIX
namespace cub = thrust::system::cuda::detail::cub_;
#else // Cuda 9 or higher
#define THRUST_CUB_NS_PREFIX namespace thrust {   namespace cuda_cub {
#define THRUST_CUB_NS_POSTFIX }  }
#include <thrust/system/cuda/detail/cub/device/device_radix_sort.cuh>
#undef CUB_NS_PREFIX
#undef CUB_NS_POSTFIX
namespace cub = thrust::cuda_cub::cub;
#endif

// Internal dependencies
#include <gpu_voxels/octree/load_balancer/LoadBalancer.cuh>
#include <gpu_voxels/octree/kernels/kernel_common.h>
#include <gpu_voxels/octree/kernels/kernel_traverse.h>
#include <gpu_voxels/octree/kernels/kernel_Octree.h>
#include <gpu_voxels/octree/kernels/kernel_PointCloud.h>
#include <gpu_voxels/octree/DataTypes.h>
#include <gpu_voxels/octree/PointCloud.h>
#include <gpu_voxels/octree/Morton.h>
#include <gpu_voxels/octree/Octree.h>
#include <gpu_voxels/voxelmap/TemplateVoxelMap.hpp>

#include <icl_core_performance_monitor/PerformanceMonitor.h>

#include <gpu_voxels/helpers/cuda_datatypes.h>
#include <gpu_voxels/helpers/cuda_handling.h>
#include <gpu_voxels/helpers/MetaPointCloud.h>

#include <gpu_voxels/logging/logging_octree.h>

//#define DEBUG_MODE
//#define DEBUG_MODE_EX

//#define COALESCED // only works for linearVoxel (all voxel occupied)
//#define WITHOUT_STACK
#define DEFAULT
//#define SHARED_STACK
//#define SMALL_STACK

/*
 * octree
 * occupied voxel: 128*1024*1024
 * map size: 512³=128*1024*1024
 * numThreadsPerBlock = 32*8
 * numBlocks = 8192*8
 *
 * COALESCED:           15 ms (6 ms for comparing leaf nodes -> >170 GB/s)
 * WITHOUT_STACK:       27 ms
 * SMALL_STACK:         32 ms
 * DEFAULT:             50 ms
 * SHARED_STACK:        52 ms
 *
 *
 *
 */

namespace gpu_voxels {
namespace NTree {

/** This is a construct to avoid warnings about an unused variable
 * Later on templates are defined from a const bool, which isn't used anywhere else.
 * These bools are passed to this unused() class.
 */
template<class T> void unused(T&)
{ }

/*
 * #################### Some Helpers ######################
 */

template<typename T>
bool checkSorting(T* data, uint32_t num_items)
{
#ifdef CHECK_SORTING
  LOGGING_DEBUG(OctreeLog, "Check sorting..." << endl);
  // data has to be sorted
  for (uint32_t i = 0; i < num_items - 1; ++i)
  {
    T item_a, item_b;
    cudaMemcpy((void*) &item_a, (void*) &data[i], sizeof(T), cudaMemcpyDeviceToHost);
    cudaMemcpy((void*) &item_b, (void*) &data[i + 1], sizeof(T), cudaMemcpyDeviceToHost);
    if (!(item_a < item_b))
    {
      LOGGING_DEBUG(OctreeLog, "index " << i << " " << item_a << " < " << item_b << endl);
    }
    assert(item_a < item_b);
  }

//#else
//  printf("Skip check sorting!\n");

#endif
  return true;
}

inline
void initRoot(Environment::InnerNode& n)
{
  n.setStatus(n.getStatus() | ns_UNKNOWN);
}

inline
void initRoot(Environment::InnerNodeProb& n)
{
  n.setOccupancy(UNKNOWN_PROBABILITY);
}

//void getOccupancyResetData(Environment::InnerNode::NodeData::BasicData& basic_data)
//{
//  basic_data = Environment::InnerNode::NodeData::BasicData(STATUS_OCCUPANCY_MASK, 0);
//}
//
//void getOccupancyResetData(Environment::NodeProb::NodeData::BasicData& basic_data)
//{
//  basic_data = Environment::NodeProb::NodeData::BasicData(STATUS_OCCUPANCY_MASK, 0, 0);
//}

void getFreeData(Environment::InnerNode::NodeData::BasicData& basic_data)
{
  basic_data = Environment::InnerNode::NodeData::BasicData(ns_FREE, 0);
}

void getFreeData(Environment::NodeProb::NodeData::BasicData& basic_data)
{
  basic_data = Environment::NodeProb::NodeData::BasicData(0, 0, MIN_PROBABILITY);
}

void getOccupiedData(Environment::InnerNode::NodeData::BasicData& basic_data)
{
  basic_data = Environment::InnerNode::NodeData::BasicData(ns_OCCUPIED, 0);
}

void getOccupiedData(Environment::NodeProb::NodeData::BasicData& basic_data)
{
  basic_data = Environment::NodeProb::NodeData::BasicData(0, 0, MAX_PROBABILITY);
}

void getRebuildResetData(Environment::InnerNode::NodeData::BasicData& basic_data)
{
  basic_data = Environment::InnerNode::NodeData::BasicData(0xFF, 0xFF);
}

void getRebuildResetData(Environment::NodeProb::NodeData::BasicData& basic_data)
{
  basic_data = Environment::NodeProb::NodeData::BasicData(0xFF, 0xFF, MIN_PROBABILITY);
}

struct Trafo_Voxel_to_BasicData
{
  __host__ __device__ __forceinline__
  Environment::InnerNode::NodeData::BasicData operator()(const Voxel x)
  {
    Environment::InnerNode::NodeData::BasicData b(0, 0);
    NodeStatus s = 0;
    if (x.getOccupancy() >= THRESHOLD_OCCUPANCY)
      s = ns_OCCUPIED;
    b.m_status = s;
    return b;
  }
};

struct Trafo_Voxel_to_BasicDataProb
{
  __host__ __device__ __forceinline__
  Environment::NodeProb::NodeData::BasicData operator()(const Voxel x)
  {
    Environment::NodeProb::NodeData::BasicData b(0, 0, 0);
    b.m_occupancy = x.getOccupancy();
    return b;
  }
};

void getBasicData(thrust::device_vector<Voxel>& voxel,
                  thrust::device_vector<Environment::InnerNode::NodeData::BasicData>& basic_data)
{
  basic_data.resize(voxel.size());
  thrust::transform(voxel.begin(), voxel.end(), basic_data.begin(), Trafo_Voxel_to_BasicData());
  HANDLE_CUDA_ERROR(cudaDeviceSynchronize());
}

void getBasicData(thrust::device_vector<Voxel>& voxel,
                  thrust::device_vector<Environment::NodeProb::NodeData::BasicData>& basic_data)
{
  basic_data.resize(voxel.size());
  thrust::transform(voxel.begin(), voxel.end(), basic_data.begin(), Trafo_Voxel_to_BasicDataProb());
  HANDLE_CUDA_ERROR(cudaDeviceSynchronize());
}

/**
 * Cut sub-tree for inserted voxel
 */
void getHardInsertResetData(Environment::InnerNode::NodeData::BasicData& basic_data)
{
  basic_data = Environment::InnerNode::NodeData::BasicData(STATUS_OCCUPANCY_MASK | ns_PART, 0);
}

/**
 * Still cut the sub-tree for the inersted voxel in deterministic case
 */
void getSoftInsertResetData(Environment::InnerNode::NodeData::BasicData& basic_data)
{
  basic_data = Environment::InnerNode::NodeData::BasicData(STATUS_OCCUPANCY_MASK | ns_PART, 0);
}

/**
 * Reset data for free_bounding_box
 * Don't set the nf_UPDATE_SUBTREE in this case!
 */
void getFreeBoxResetData(Environment::InnerNode::NodeData::BasicData& basic_data)
{
  basic_data = Environment::InnerNode::NodeData::BasicData(STATUS_OCCUPANCY_MASK | ns_PART,
                                                           nf_UPDATE_SUBTREE);
}

/**
 * Cut sub-tree for inserted voxel
 */
void getHardInsertResetData(Environment::NodeProb::NodeData::BasicData& basic_data)
{
  basic_data = Environment::NodeProb::NodeData::BasicData(STATUS_OCCUPANCY_MASK | ns_PART, 0, 0);
}

/**
 * Don't cut sub-tree for inserted voxel
 * Needed to insert super-voxel for a large free space and therefore update the sub-tree voxel with propagate
 */
void getSoftInsertResetData(Environment::NodeProb::NodeData::BasicData& basic_data)
{
  basic_data = Environment::NodeProb::NodeData::BasicData(STATUS_OCCUPANCY_MASK, 0, 0);
}

/**
 * Reset data for free_bounding_box
 * Don't set the nf_UPDATE_SUBTREE in this case!
 */
void getFreeBoxResetData(Environment::NodeProb::NodeData::BasicData& basic_data)
{
  basic_data = Environment::NodeProb::NodeData::BasicData(STATUS_OCCUPANCY_MASK, nf_UPDATE_SUBTREE, 0);
}

/*
 * ########################################################
 */

template<std::size_t branching_factor, std::size_t level_count, typename InnerNode, typename LeafNode>
NTree<branching_factor, level_count, InnerNode, LeafNode>::NTree(uint32_t numBlocks,
                                                                 uint32_t numThreadsPerBlock,
                                                                 uint32_t resolution)
{
  this->numBlocks = numBlocks;
  this->numThreadsPerBlock = numThreadsPerBlock;
  this->allocInnerNodes = 0;
  this->allocLeafNodes = 0;
  this->m_resolution = resolution;
  this->m_center = gpu_voxels::Vector3ui(pow(pow(branching_factor, 1.0 / 3), level_count - 2));
  this->m_extract_buffer_size = INITIAL_EXTRACT_BUFFER_SIZE;
  this->m_rebuild_buffer_size = INITIAL_REBUILD_BUFFER_SIZE;
  this->m_max_memory_usage = 200 * cMBYTE2BYTE; // 200 MB
  this->m_rebuild_counter = 0;
  this->m_has_data = false;

  InnerNode* r = new InnerNode();
  initRoot(*r);
  r->setStatus(r->getStatus() | ns_STATIC_MAP | ns_DYNAMIC_MAP);
  HANDLE_CUDA_ERROR(cudaMalloc(&m_root, sizeof(InnerNode)));
  HANDLE_CUDA_ERROR(cudaDeviceSynchronize());
  HANDLE_CUDA_ERROR(cudaMemcpy(m_root, r, sizeof(InnerNode), cudaMemcpyHostToDevice));
  HANDLE_CUDA_ERROR(cudaDeviceSynchronize());
  m_allocation_list.push_back(m_root);
  delete r;

  // create default status to BitVoxelMeaning mapping
  const int mapping_size = 256;
  uint8_t mapping[mapping_size];
  memset(mapping, 0, mapping_size * sizeof(uint8_t));

  mapping[ns_FREE] = gpu_voxels::eBVM_FREE;
  mapping[ns_FREE | ns_UNKNOWN] = gpu_voxels::eBVM_FREE;
  //mapping[ns_UNKNOWN] = gpu_voxels::eBVM_UNDEFINED;
  mapping[ns_UNKNOWN] = gpu_voxels::eBVM_UNKNOWN;
  //mapping[ns_FREE | ns_UNKNOWN] = gpu_voxels::eBVM_UNDEFINED;
  mapping[ns_OCCUPIED] = gpu_voxels::eBVM_OCCUPIED;
  mapping[ns_OCCUPIED | ns_FREE] = gpu_voxels::eBVM_OCCUPIED;
  mapping[ns_OCCUPIED | ns_FREE | ns_UNKNOWN] = gpu_voxels::eBVM_OCCUPIED;
  mapping[ns_OCCUPIED | ns_UNKNOWN] = gpu_voxels::eBVM_OCCUPIED;
  for (int i = 0; i < mapping_size; ++i)
    if ((i & ns_COLLISION) == ns_COLLISION)
      mapping[i] = gpu_voxels::eBVM_COLLISION;
  // mapping[i] = gpu_voxels::eBVM_COLLISION;

  HANDLE_CUDA_ERROR(cudaMalloc(&m_status_mapping, mapping_size * sizeof(uint8_t)));
  HANDLE_CUDA_ERROR(cudaDeviceSynchronize());
  HANDLE_CUDA_ERROR(
      cudaMemcpy(m_status_mapping, mapping, mapping_size * sizeof(uint8_t), cudaMemcpyHostToDevice));
  HANDLE_CUDA_ERROR(cudaDeviceSynchronize());

  // create default extract data status selection
  uint8_t selection[extract_selection_size];
  memset(selection, 1, extract_selection_size * sizeof(uint8_t));

  HANDLE_CUDA_ERROR(cudaMalloc(&m_extract_status_selection, extract_selection_size * sizeof(uint8_t)));
  HANDLE_CUDA_ERROR(
      cudaMemcpy(m_extract_status_selection, selection, extract_selection_size * sizeof(uint8_t),
                 cudaMemcpyHostToDevice));
  HANDLE_CUDA_ERROR(cudaDeviceSynchronize());

  init_const_memory();
}

template<std::size_t branching_factor, std::size_t level_count, typename InnerNode, typename LeafNode>
NTree<branching_factor, level_count, InnerNode, LeafNode>::~NTree()
{
  for (uint32_t i = 0; i < m_allocation_list.size(); ++i)
    HANDLE_CUDA_ERROR(cudaFree(m_allocation_list[i]));
  m_allocation_list.clear();

  HANDLE_CUDA_ERROR(cudaFree(m_status_mapping));
  HANDLE_CUDA_ERROR(cudaFree(m_extract_status_selection));
}

template<std::size_t branching_factor, std::size_t level_count, typename InnerNode, typename LeafNode>
void NTree<branching_factor, level_count, InnerNode, LeafNode>::build(
    thrust::host_vector<Vector3ui>& h_points, const bool free_bounding_box)
{
  thrust::device_vector<Vector3ui> d_points = h_points;
  build(d_points, free_bounding_box);
}

template<std::size_t branching_factor, std::size_t level_count, typename InnerNode, typename LeafNode>
void NTree<branching_factor, level_count, InnerNode, LeafNode>::toVoxelCoordinates(
    thrust::host_vector<Vector3f>& h_points, thrust::device_vector<Vector3ui>& d_voxels)
{
  size_t num_points = h_points.size();
  d_voxels.resize(num_points);
  thrust::device_vector<Vector3f> d_points = h_points;
  kernel_toVoxels<<<numBlocks, numThreadsPerBlock>>>(D_PTR(d_points), num_points, D_PTR(d_voxels), float(m_resolution / 1000.0f));
  CHECK_CUDA_ERROR();
  HANDLE_CUDA_ERROR(cudaDeviceSynchronize());
}

/*
 * Due to the fact that of the morton transformation and the sorting, the max speedup is 2 for the current approach,
 * since morton and sorting takes half the time of this method
 */
template<std::size_t branching_factor, std::size_t level_count, typename InnerNode, typename LeafNode>
void NTree<branching_factor, level_count, InnerNode, LeafNode>::build(
    thrust::device_vector<Vector3ui>& d_points, const bool free_bounding_box)
{
#define SORT_WITH_CUB false
  const std::string prefix = __FUNCTION__;
  const std::string temp_timer = prefix + "_temp";
  const std::string temp2_timer = prefix + "_temp2";
  PERF_MON_START(prefix);

//  this->sideLengthInVoxel = sideLengthInVoxel;
//  this->voxelSideLength = voxelSideLength;
//  this->level_count = ceil(
//      log(float(sideLengthInVoxel * sideLengthInVoxel * sideLengthInVoxel)) / log(float(branching_factor)))
//      + 1;
  this->allocInnerNodes = 0;
  this->allocLeafNodes = 0;

  OctreeVoxelID num_points = d_points.size();
  OctreeVoxelID total_num_voxel = num_points;

  // computation of number of blocks and threads due to experimental founding
  uint32_t num_blocks = 4096;
  const float blocks_1 = 32; // first reference point
  const float points_1 = 3000000;
  const float blocks_2 = 512; // second reference point
  const float points_2 = 13000000;
  uint32_t num_threads_per_block = linearApprox(blocks_1, points_1, blocks_2, points_2, num_points, WARP_SIZE, MAX_NUMBER_OF_THREADS);

  // #################################################
  //                       Step 0
  // #################################################
  // transform points into morton code
  // throughput ~ 3.8 GB/s
  PERF_MON_START(temp_timer);
  thrust::device_vector<OctreeVoxelID> d_voxels(num_points);
  kernel_toMortonCode<<<num_blocks, num_threads_per_block>>>(D_PTR(d_points), num_points, D_PTR(d_voxels));
  CHECK_CUDA_ERROR();
  HANDLE_CUDA_ERROR(cudaDeviceSynchronize());
  if(!free_bounding_box)
  {
    // free points as data isn't needed any more
    d_points.clear();
    d_points.shrink_to_fit();
  }

  //// Host-side computation -> slowdown by a factor of about 10
  //  thrust::device_vector<voxel_id> voxel;
  //  thrust::host_vector<voxel_id> h_voxel(h_points.size());
  //  for (uint32_t i = 0; i < h_points.size(); ++i)
  //    h_voxel[i] = morton_code60(h_points[i].x, h_points[i].y, h_points[i].z);
  //  voxel = h_voxel;
  //  HANDLE_CUDA_ERROR(cudaDeviceSynchronize());
  PERF_MON_PRINT_AND_RESET_INFO_P(temp_timer, "ToMorton", prefix);
  PERF_MON_ADD_STATIC_DATA_P("Voxel", num_points, "build");

// #################################################
//                       Step 1
// #################################################
// Sort input voxel
// implements radix sort for primitive types and default comparator
// complexity O(N/P)
#ifdef _IC_DEBUG_
  timespec time = getCPUTime();
#endif
// Throughput of max 690 MKey/s (Key is 8 Byte) thats only about 5.1 GB/s
// Radixsort is implemented in thrust see http://code.google.com/p/back40computing/wiki/RadixSorting#Performance
// There are also performance charts which are conform to my measured performance
// The limit seams to be the memory bandwidth
// Sorting needs at least 2 MKey to reach roughly it's full performance
// Have a look at http://code.google.com/p/back40computing/wiki/RadixSorting#News
// There is something said about a better performing version for small input sizes < 1.2 MKey
// Performing the sorting on the CPU for small inputs is way faster than on GPU (have a look at my own benchmarks)!

  if(SORT_WITH_CUB)
  {
     thrust::device_vector<OctreeVoxelID> voxel_tmp(num_points);
     OctreeVoxelID *d_key_buf = D_PTR(d_voxels);
     OctreeVoxelID *d_key_alt_buf = D_PTR(voxel_tmp);
     cub::DoubleBuffer<OctreeVoxelID> d_keys(d_key_buf, d_key_alt_buf);
     // Determine temporary device storage requirements
     void *d_temp_storage = NULL;
     size_t temp_storage_bytes = 0;
     cub::DeviceRadixSort::SortKeys(d_temp_storage, temp_storage_bytes, d_keys, num_points);
     // Allocate temporary storage
     HANDLE_CUDA_ERROR(cudaMalloc(&d_temp_storage, temp_storage_bytes));
     // Run sorting operation
     cub::DeviceRadixSort::SortKeys(d_temp_storage, temp_storage_bytes, d_keys, num_points);
     HANDLE_CUDA_ERROR(cudaFree(d_temp_storage));
     if(d_keys.Current() != d_key_buf)
       voxel_tmp.swap(d_voxels);
   }
   else
     thrust::sort(d_voxels.begin(), d_voxels.end());
  HANDLE_CUDA_ERROR(cudaDeviceSynchronize()); // sync just like for plain kernel calls
#ifndef LOAD_BALANCING_PROPAGATE
  thrust::device_vector<OctreeVoxelID> voxel_copy = d_voxels;
#endif
  LOGGING_DEBUG(OctreeLog, "thrust::sort(): " << timeDiff(time, getCPUTime()) << " ms" << endl);
#ifdef DEBUG_MODE_EX
  for (uint32_t i = 0; i < num_points; ++i)
  {
    LOGGING_DEBUG(OctreeDebugEXLog, "sorted[" << i << "]: " << d_voxels[i] << << endl);
  }
#endif

  PERF_MON_PRINT_AND_RESET_INFO_P(temp_timer, "Sort", prefix);
  PERF_MON_START(temp2_timer);

  OctreeVoxelID biggest_value = d_voxels.back();
  if(biggest_value >= (OctreeVoxelID) pow(branching_factor, level_count - 1))
  {
    LOGGING_ERROR(OctreeLog, "Point (morton code: " << biggest_value << ") of input data is out of range for the NTree!" << endl);
    assert(false);
  }

  void* childNodes = 0;

#ifdef DEBUG_MODE
  LOGGING_DEBUG(OctreeDebugLog, "allocating parentNodes with size " << num_points << endl);
#endif

// holds the zOrder IDs of the next level, since InnerNode doesn't store these.
// necessary to determine which InnerNodes have the same parent InnerNode
  thrust::device_vector<OctreeVoxelID> nodeIds(num_points);

#ifdef DEBUG_MODE
  LOGGING_DEBUG(OctreeDebugLog, "allocating nodeCount..." << endl);
#endif

  thrust::device_vector<OctreeVoxelID> nodeCount(numBlocks * num_threads_per_block);

#ifdef DEBUG_MODE
  LOGGING_DEBUG(OctreeDebugLog, "loop start" << endl);
#endif

  for (uint32_t level = 0; level < level_count; ++level)
  {
#ifdef DEBUG_MODE
    LOGGING_DEBUG(OctreeDebugLog, << "level: " << level << endl);
#endif

    // #################################################
    //                     Step 2
    // #################################################
    // count needed nodes; compute prefix sum
    kernel_countNodes<branching_factor, level_count, InnerNode, LeafNode> <<<numBlocks, num_threads_per_block>>>(
        D_PTR(d_voxels), num_points,
        level, D_PTR(nodeCount));
        CHECK_CUDA_ERROR();
        HANDLE_CUDA_ERROR(cudaDeviceSynchronize());

        uint32_t lastThread = ceil(double(num_points) / ceil(double(num_points) / (numBlocks * num_threads_per_block)))
        - 1;

#ifdef DEBUG_MODE
        LOGGING_DEBUG(OctreeDebugLog, endl);
        LOGGING_DEBUG(OctreeDebugLog, "numVoxel: " << num_points << endl);
        LOGGING_DEBUG(OctreeDebugLog, "lastThread: " << lastThread << endl);

//        std::cout << std::endl;
//        //    for (uint32_t i = 0; i < nodeCount.size(); ++i)
//        //    std::cout << "count[" << i << "]: " << nodeCount[i] << std::endl;
//
//        std::cout << "numVoxel: " << num_points << std::endl;
//        std::cout << "lastThread: " << lastThread << std::endl;
#endif

        thrust::inclusive_scan(nodeCount.begin(), nodeCount.begin() + lastThread + 1, nodeCount.begin());
        HANDLE_CUDA_ERROR(cudaDeviceSynchronize()); // sync just like for plain kernel calls

#ifdef DEBUG_MODE
        LOGGING_DEBUG(OctreeDebugLog, "voxel counted" <<  endl);
        LOGGING_DEBUG(OctreeDebugLog, endl);
        //std::cout << "voxel counted" << std::endl;
        //std::cout << std::endl;
        //    for (uint32_t i = 0; i <= lastThread; ++i)
        //    std::cout << "scan[" << i << "]: " << nodeCount[i] << std::endl;
#endif

        // #################################################
        //                     Step 3
        // #################################################
        // Allocate nodes, set nodes, set child pointers
        OctreeVoxelID numNodes = nodeCount[lastThread];
        void* nodes = 0;
        HANDLE_CUDA_ERROR(
            cudaMalloc(&nodes,
                branching_factor * numNodes * ((level == 0) ? sizeof(LeafNode) : sizeof(InnerNode))));
        m_allocation_list.push_back(nodes);

#ifdef DEBUG_MODE
        LOGGING_DEBUG(OctreeDebugLog,"numNodes: " << numNodes <<  endl);
#endif

        if (level == 0)
        {
          this->allocLeafNodes += branching_factor * numNodes;
          kernel_clearNodes<LeafNode, false> <<<numBlocks, num_threads_per_block>>>(branching_factor * numNodes, (LeafNode*) nodes);
          CHECK_CUDA_ERROR();
          HANDLE_CUDA_ERROR(cudaDeviceSynchronize());
#ifdef DEBUG_MODE
        LOGGING_DEBUG(OctreeDebugLog, "clearNodes done L0" <<  endl);
#endif
        kernel_setNodes<LeafNode, InnerNode, branching_factor> <<<numBlocks, num_threads_per_block>>>(
            D_PTR(d_voxels), num_points, level, D_PTR(nodeCount),
            (LeafNode*) nodes, D_PTR(nodeIds), (InnerNode*) childNodes);
        CHECK_CUDA_ERROR();
        HANDLE_CUDA_ERROR(cudaDeviceSynchronize());
#ifdef DEBUG_MODE
        LOGGING_DEBUG(OctreeDebugLog, "setNodes done L0" <<  endl);
#endif
      }
      else
      {
        this->allocInnerNodes += branching_factor * numNodes;
#ifdef DEBUG_MODE
        LOGGING_DEBUG(OctreeDebugLog, "clearNodes done" <<  endl);
#endif
        if (level == 1)
        {
          kernel_clearNodes<InnerNode, true> <<<numBlocks, num_threads_per_block>>>(branching_factor * numNodes, (InnerNode*) nodes );
          CHECK_CUDA_ERROR();
          HANDLE_CUDA_ERROR(cudaDeviceSynchronize());
          kernel_setNodes<InnerNode, LeafNode, branching_factor> <<<numBlocks, num_threads_per_block>>>(
              D_PTR(d_voxels), num_points, level, D_PTR(nodeCount),
              (InnerNode*) nodes, D_PTR(nodeIds), (LeafNode*) childNodes);
          HANDLE_CUDA_ERROR(cudaDeviceSynchronize());
        }
        else
        {
          kernel_clearNodes<InnerNode, false> <<<numBlocks, num_threads_per_block>>>(branching_factor * numNodes, (InnerNode*) nodes);
          CHECK_CUDA_ERROR();
          HANDLE_CUDA_ERROR(cudaDeviceSynchronize());
          kernel_setNodes<InnerNode, InnerNode,branching_factor> <<<numBlocks, num_threads_per_block>>>(
              D_PTR(d_voxels), num_points, level, D_PTR(nodeCount),
              (InnerNode*) nodes, D_PTR(nodeIds), (InnerNode*) childNodes);
          HANDLE_CUDA_ERROR(cudaDeviceSynchronize());
        }
#ifdef DEBUG_MODE
        LOGGING_DEBUG(OctreeDebugLog, "setNodes done" <<  endl);
#endif
      }

      d_voxels.swap(nodeIds);
      childNodes = nodes;
      num_points = numNodes;
      PERF_MON_PRINT_AND_RESET_INFO_P(temp_timer, "Build_L" + to_string(level, "%02d"), prefix);
    }
  m_root = (InnerNode*) childNodes;
  InnerNode k;
  HANDLE_CUDA_ERROR(cudaMemcpy(&k, m_root, sizeof(InnerNode), cudaMemcpyDeviceToHost));
  initRoot(k);
  k.setStatus(k.getStatus() | ns_STATIC_MAP);
  HANDLE_CUDA_ERROR(cudaMemcpy(m_root, &k, sizeof(InnerNode), cudaMemcpyHostToDevice));
  HANDLE_CUDA_ERROR(cudaDeviceSynchronize());
  PERF_MON_PRINT_INFO_P(temp2_timer, "Build_L_ALL", prefix);

  if (free_bounding_box)
    this->free_bounding_box(d_points);

  PERF_MON_START(temp_timer);

#ifdef LOAD_BALANCING_PROPAGATE
  propagate(total_num_voxel);
#else
  propagate_bottom_up(D_PTR(voxel_copy), total_num_voxel, 0);
#endif

  m_has_data = true; // indicate that the NTree holds some data

  PERF_MON_PRINT_INFO_P(temp_timer, "Propagate", prefix);

  PERF_MON_ADD_STATIC_DATA_P("InnerNodes", allocInnerNodes, "build");
  PERF_MON_ADD_STATIC_DATA_P("LeafNodes", allocLeafNodes, "build");
  PERF_MON_ADD_STATIC_DATA_P("Mem", getMemUsage(), "build");

  PERF_MON_PRINT_INFO_P(prefix, "", prefix);
#undef SORT_WITH_CUB
}

template<std::size_t branching_factor, std::size_t level_count, typename InnerNode, typename LeafNode>
void NTree<branching_factor, level_count, InnerNode, LeafNode>::print()
{
  thrust::device_vector<InnerNode> stack1(1000000);
  thrust::device_vector<InnerNode> stack2(1000000);
  kernel_print<branching_factor, level_count, InnerNode, LeafNode> <<<1, 1>>>(m_root, D_PTR(stack1), D_PTR(stack2));
  CHECK_CUDA_ERROR();
  HANDLE_CUDA_ERROR(cudaDeviceSynchronize());
}

template<std::size_t branching_factor, std::size_t level_count, typename InnerNode, typename LeafNode>
void NTree<branching_factor, level_count, InnerNode, LeafNode>::print2()
{
  thrust::device_vector<MyTripple<InnerNode*, OctreeVoxelID, bool> > stack1(10000000);
  thrust::device_vector<MyTripple<InnerNode*, OctreeVoxelID, bool> > stack2(10000000);
  kernel_print2<branching_factor, level_count, InnerNode, LeafNode> <<<1, 1>>>(m_root, D_PTR(stack1), D_PTR(stack2));
  CHECK_CUDA_ERROR();
  HANDLE_CUDA_ERROR(cudaDeviceSynchronize());
}

template<std::size_t branching_factor, std::size_t level_count, typename InnerNode, typename LeafNode>
void NTree<branching_factor, level_count, InnerNode, LeafNode>::find(
    thrust::device_vector<Vector3ui> voxel, void** resultNode,
    thrust::device_vector<enum NodeType> resultNodeType)
{
  assert(voxel.size() == resultNodeType.size());
  kernel_find<branching_factor, level_count, InnerNode, LeafNode> <<<numBlocks, numThreadsPerBlock>>>(
      m_root, D_PTR(voxel), voxel.size(),
      resultNode, D_PTR(resultNodeType));
  CHECK_CUDA_ERROR();
  HANDLE_CUDA_ERROR(cudaDeviceSynchronize());
}

template<std::size_t branching_factor, std::size_t level_count, typename InnerNode, typename LeafNode>
voxel_count NTree<branching_factor, level_count, InnerNode, LeafNode>::intersect(
    thrust::host_vector<Vector3ui>& h_voxel)
{
//// Is about 30 ms whereas the current implementation is ~18 ms
//  thrust::device_vector<Vector3ui> d_points = h_voxel;
//  thrust::device_vector<voxel_id> voxel(h_voxel.size());
//  kernel_toMortonCode <<<numBlocks, numThreadsPerBlock>>> (D_PTR(d_points), h_voxel.size(), D_PTR(voxel));
//  CHECK_CUDA_ERROR();
//  HANDLE_CUDA_ERROR(cudaDeviceSynchronize());
//  d_points.clear();
//  d_points.shrink_to_fit();
//  thrust::device_vector<voxel_count> d_num_collisions(numBlocks);
//  thrust::sort(voxel.begin(), voxel.end());
//  HANDLE_CUDA_ERROR(cudaDeviceSynchronize());
//  kernel_intersect<<<numBlocks, numThreadsPerBlock>>>(m_root, D_PTR(voxel), h_voxel.size(), D_PTR(d_num_collisions));
//  CHECK_CUDA_ERROR();
//  HANDLE_CUDA_ERROR(cudaDeviceSynchronize());
//  voxel_count collisions = thrust::reduce(d_num_collisions.begin(), d_num_collisions.end());
//  return collisions;

  timespec t = getCPUTime();
  thrust::device_vector<Vector3ui> d_voxel = h_voxel;
  thrust::device_vector<voxel_count> d_num_collisions(numBlocks);
  HANDLE_CUDA_ERROR(cudaDeviceSynchronize());
  LOGGING_INFO(OctreeLog, "malloc and copy: " << timeDiff(t, getCPUTime()) << " ms" <<  endl);
  t = getCPUTime();
  kernel_intersect<branching_factor, level_count, InnerNode, LeafNode> <<<numBlocks, numThreadsPerBlock>>>(
      m_root, D_PTR(d_voxel), h_voxel.size(),D_PTR(d_num_collisions));
  CHECK_CUDA_ERROR();
  HANDLE_CUDA_ERROR(cudaDeviceSynchronize());
  LOGGING_INFO(OctreeLog, "kernel_intersect(): " << timeDiff(t, getCPUTime()) << " ms" <<  endl);
  t = getCPUTime();
  voxel_count collisions = thrust::reduce(d_num_collisions.begin(), d_num_collisions.end());
  LOGGING_INFO(OctreeLog, "thrust::reduce(): " << timeDiff(t, getCPUTime()) << " ms" <<  endl);
  return collisions;
}


//template<std::size_t branching_factor, std::size_t level_count, typename InnerNode, typename LeafNode>
//template<int VTF_SIZE, bool set_collision_flag, bool compute_voxelTypeFlags, typename VoxelType,
//    bool use_execution_context>
//voxel_count NTree<branching_factor, level_count, InnerNode, LeafNode>::intersect_sparse(
//    gpu_voxels::voxelmap::VoxelMap& voxel_map, BitVector<VTF_SIZE>* h_result_voxelTypeFlags, const uint32_t min_level, gpu_voxels::Vector3i offset)
//{
////  assert(
////      (use_VoxelFlags && result_voxelTypeFlags != NULL) || (!use_VoxelFlags && result_voxelTypeFlags == NULL));
//
//  const std::string prefix = __FUNCTION__;
//  PERF_MON_START(prefix);
//
//  thrust::device_vector<voxel_count> d_num_collisions(numBlocks);
//  thrust::device_vector<BitVector<VTF_SIZE> > d_voxelTypeFlags(numBlocks);
//
//  kernel_intersect_VoxelMap<
//  branching_factor,
//  level_count,
//  InnerNode,
//  LeafNode,
//  set_collision_flag,
//  BitVector<VTF_SIZE>,
//  compute_voxelTypeFlags,
//  VoxelType>
//  <<<numBlocks, numThreadsPerBlock>>>
//      (m_root,
//      voxel_map.getDeviceDataPtr(),
//      voxel_map.getVoxelMapSize(),
//      voxel_map.getDimensions(),
//      D_PTR(d_num_collisions),
//      D_PTR(d_voxelTypeFlags),
//      min_level,
//      offset);
//   CHECK_CUDA_ERROR();
//
//  HANDLE_CUDA_ERROR(cudaDeviceSynchronize());
//
//  thrust::host_vector<voxel_count> h_num_collisions = d_num_collisions;
//  voxel_count collisions = thrust::reduce(h_num_collisions.begin(), h_num_collisions.end());
//  if (compute_voxelTypeFlags)
//  {
//    thrust::host_vector<BitVector<VTF_SIZE> > h_voxelTypeFlags = d_voxelTypeFlags;
//    BitVector<VTF_SIZE> init;
//    init.clear();
//    *h_result_voxelTypeFlags = thrust::reduce(h_voxelTypeFlags.begin(), h_voxelTypeFlags.end(), init,
//                                              typename BitVector<VTF_SIZE>::reduce_op());
//  }
//
//  PERF_MON_PRINT_INFO_P(prefix, "", prefix);
//  PERF_MON_ADD_DATA_NONTIME_P("NumCollisions", collisions, prefix);
//
//  return collisions;
//}

template<std::size_t branching_factor, std::size_t level_count, typename InnerNode, typename LeafNode>
template<bool set_collision_flag, bool compute_voxelTypeFlags, bool compute_collsWithUnknown, typename VoxelType>
voxel_count NTree<branching_factor, level_count, InnerNode, LeafNode>::intersect_sparse(
    gpu_voxels::voxelmap::TemplateVoxelMap<VoxelType>& voxel_map, VoxelType* h_result_voxel, const uint32_t min_level,
    gpu_voxels::Vector3i offset, voxel_count* num_colls_with_unknown_cells)
{
//  assert(
//      (use_VoxelFlags && result_voxelTypeFlags != NULL) || (!use_VoxelFlags && result_voxelTypeFlags == NULL));

  const std::string prefix = __FUNCTION__;
  PERF_MON_START(prefix);

  thrust::device_vector<voxel_count> d_num_collisions(numBlocks);
  thrust::device_vector<voxel_count> d_num_collisions_w_unknown(numBlocks);
  thrust::device_vector<VoxelType> d_voxelTypeFlags(numBlocks);

  kernel_intersect_VoxelMap<
  branching_factor,
  level_count,
  InnerNode,
  LeafNode,
  set_collision_flag,
  compute_voxelTypeFlags,
  compute_collsWithUnknown,
  VoxelType>
  <<<numBlocks, numThreadsPerBlock, numThreadsPerBlock*sizeof(VoxelType)>>>
      (m_root,
      voxel_map.getDeviceDataPtr(),
      voxel_map.getVoxelMapSize(),
      voxel_map.getDimensions(),
      D_PTR(d_num_collisions),
      D_PTR(d_num_collisions_w_unknown),
      D_PTR(d_voxelTypeFlags),
      min_level,
      offset);
  CHECK_CUDA_ERROR();

  HANDLE_CUDA_ERROR(cudaDeviceSynchronize());

  thrust::host_vector<voxel_count> h_num_collisions = d_num_collisions;
  voxel_count collisions = thrust::reduce(h_num_collisions.begin(), h_num_collisions.end());

  if(compute_collsWithUnknown)
  {
    thrust::host_vector<voxel_count> h_num_collisions_w_unknown = d_num_collisions_w_unknown;
    *num_colls_with_unknown_cells = thrust::reduce(h_num_collisions_w_unknown.begin(), h_num_collisions_w_unknown.end());
  }

  if (compute_voxelTypeFlags)
  {
    thrust::host_vector<VoxelType> h_voxelTypeFlags = d_voxelTypeFlags;
    VoxelType init;
    *h_result_voxel = thrust::reduce(h_voxelTypeFlags.begin(), h_voxelTypeFlags.end(), init,
                                              typename VoxelType::reduce_op());
  }

  PERF_MON_PRINT_INFO_P(prefix, "", prefix);
  PERF_MON_ADD_DATA_NONTIME_P("NumCollisions", collisions, prefix);

  return collisions;
}

template<std::size_t branching_factor, std::size_t level_count, typename InnerNode, typename LeafNode>
template<bool set_collision_flag, bool compute_voxelTypeFlags, bool compute_collsWithUnknown, typename VoxelType>
voxel_count NTree<branching_factor, level_count, InnerNode, LeafNode>::intersect_sparse(
    gpu_voxels::voxellist::TemplateVoxelList<VoxelType, MapVoxelID>& voxel_list, BitVectorVoxel* h_result_voxel,
    const uint32_t min_level, gpu_voxels::Vector3i offset, voxel_count* num_colls_with_unknown_cells)
{
//  assert(
//      (use_VoxelFlags && result_voxelTypeFlags != NULL) || (!use_VoxelFlags && result_voxelTypeFlags == NULL));

  const std::string prefix = __FUNCTION__;
  PERF_MON_START(prefix);

  thrust::device_vector<voxel_count> d_num_collisions(numBlocks);
  thrust::device_vector<voxel_count> d_num_collisions_w_unknown(numBlocks);
  thrust::device_vector<BitVectorVoxel> d_voxelTypeFlags(numBlocks);

  kernel_intersect_VoxelList<
  branching_factor,
  level_count,
  InnerNode,
  LeafNode,
  set_collision_flag,
  compute_voxelTypeFlags,
  compute_collsWithUnknown,
  VoxelType>
  <<<numBlocks, numThreadsPerBlock, numThreadsPerBlock*sizeof(VoxelType)>>>
      (m_root,
      voxel_list.getDeviceCoordPtr(),
      voxel_list.getDeviceDataPtr(),
      voxel_list.getDimensions().x,
      D_PTR(d_num_collisions),
      D_PTR(d_num_collisions_w_unknown),
      D_PTR(d_voxelTypeFlags),
      min_level,
      offset);
  CHECK_CUDA_ERROR();

  HANDLE_CUDA_ERROR(cudaDeviceSynchronize());

  thrust::host_vector<voxel_count> h_num_collisions = d_num_collisions;
  voxel_count collisions = thrust::reduce(h_num_collisions.begin(), h_num_collisions.end());

  if(compute_collsWithUnknown)
  {
    thrust::host_vector<voxel_count> h_num_collisions_w_unknown = d_num_collisions_w_unknown;
    *num_colls_with_unknown_cells = thrust::reduce(h_num_collisions_w_unknown.begin(), h_num_collisions_w_unknown.end());
  }

  if (compute_voxelTypeFlags)
  {
    thrust::host_vector<BitVectorVoxel> h_voxelTypeFlags = d_voxelTypeFlags;
    BitVectorVoxel init;
    *h_result_voxel = thrust::reduce(h_voxelTypeFlags.begin(), h_voxelTypeFlags.end(), init,
                                              typename BitVectorVoxel::reduce_op());
  }

  PERF_MON_PRINT_INFO_P(prefix, "", prefix);
  PERF_MON_ADD_DATA_NONTIME_P("NumCollisions", collisions, prefix);

  return collisions;
}


// same as above, but calling a kernel that does not transform the voxel coords to morton codes first but
// compares them directly
template<std::size_t branching_factor, std::size_t level_count, typename InnerNode, typename LeafNode>
template<bool set_collision_flag, bool compute_voxelTypeFlags, bool compute_collsWithUnknown, typename VoxelType>
voxel_count NTree<branching_factor, level_count, InnerNode, LeafNode>::intersect_morton(
    gpu_voxels::voxellist::TemplateVoxelList<VoxelType, OctreeVoxelID>& voxel_list, BitVectorVoxel* h_result_voxel,
    const uint32_t min_level, voxel_count* num_colls_with_unknown_cells)
{
//  assert(
//      (use_VoxelFlags && result_voxelTypeFlags != NULL) || (!use_VoxelFlags && result_voxelTypeFlags == NULL));

  const std::string prefix = __FUNCTION__;
  PERF_MON_START(prefix);

  thrust::device_vector<voxel_count> d_num_collisions(numBlocks);
  thrust::device_vector<voxel_count> d_num_collisions_w_unknown(numBlocks);
  thrust::device_vector<BitVectorVoxel> d_voxelTypeFlags(numBlocks);

  kernel_intersect_MortonVoxelList<
  branching_factor,
  level_count,
  InnerNode,
  LeafNode,
  set_collision_flag,
  compute_voxelTypeFlags,
  compute_collsWithUnknown,
  VoxelType>
  <<<numBlocks, numThreadsPerBlock, numThreadsPerBlock*sizeof(VoxelType)>>>
      (m_root,
      voxel_list.getDeviceIdPtr(),
      voxel_list.getDeviceDataPtr(),
      voxel_list.getDimensions().x,
      D_PTR(d_num_collisions),
      D_PTR(d_num_collisions_w_unknown),
      D_PTR(d_voxelTypeFlags),
      min_level);
  CHECK_CUDA_ERROR();

  HANDLE_CUDA_ERROR(cudaDeviceSynchronize());

  thrust::host_vector<voxel_count> h_num_collisions = d_num_collisions;
  voxel_count collisions = thrust::reduce(h_num_collisions.begin(), h_num_collisions.end());

  if(compute_collsWithUnknown)
  {
    thrust::host_vector<voxel_count> h_num_collisions_w_unknown = d_num_collisions_w_unknown;
    *num_colls_with_unknown_cells = thrust::reduce(h_num_collisions_w_unknown.begin(), h_num_collisions_w_unknown.end());
  }

  if (compute_voxelTypeFlags)
  {
    thrust::host_vector<BitVectorVoxel> h_voxelTypeFlags = d_voxelTypeFlags;
    BitVectorVoxel init;
    *h_result_voxel = thrust::reduce(h_voxelTypeFlags.begin(), h_voxelTypeFlags.end(), init,
                                              typename BitVectorVoxel::reduce_op());
  }

  PERF_MON_PRINT_INFO_P(prefix, "", prefix);
  PERF_MON_ADD_DATA_NONTIME_P("NumCollisions", collisions, prefix);

  return collisions;
}


template<std::size_t branching_factor, std::size_t level_count, typename InnerNode, typename LeafNode>
template<int vft_size, bool set_collision_flag, bool compute_voxelTypeFlags, typename VoxelType>
voxel_count NTree<branching_factor, level_count, InnerNode, LeafNode>::intersect_load_balance(voxelmap::ProbVoxelMap &voxel_map, gpu_voxels::Vector3i offset, const uint32_t min_level,
    BitVector<vft_size>* h_result_voxelTypeFlags)
{
  const std::string prefix = "VoxelMap::" + std::string(__FUNCTION__);
  PERF_MON_START(prefix);

  typedef LoadBalancer::IntersectVMap<
      branching_factor,
      level_count,
      InnerNode,
      LeafNode,
      vft_size,
      set_collision_flag,
      compute_voxelTypeFlags,
      VoxelType> MyLoadBalancer;

  MyLoadBalancer load_balancer(
      this,
      (VoxelType*) voxel_map.getVoidDeviceDataPtr(),
      voxel_map.getDimensions(),
      offset,
      min_level);

  load_balancer.run();

  PERF_MON_PRINT_INFO_P(prefix, "", prefix);
  PERF_MON_ADD_DATA_NONTIME_P("NumCollisions", load_balancer.m_num_collisions, prefix);
//  PERF_MON_ADD_DATA_NONTIME_P("BalanceOverhead", balance_overhead, prefix);
//  PERF_MON_ADD_DATA_NONTIME_P("BalanceTasks", balance_tasks, prefix);

  return load_balancer.m_num_collisions;
}

template<std::size_t branching_factor, std::size_t level_count, typename InnerNode, typename LeafNode>
void NTree<branching_factor, level_count, InnerNode, LeafNode>::find(
    thrust::host_vector<Vector3ui> & h_voxel, thrust::host_vector<FindResult<LeafNode> > & resultNode)
{
  assert(h_voxel.size() == resultNode.size());
  thrust::device_vector<Vector3ui> voxel(h_voxel);
  thrust::device_vector<FindResult<LeafNode> > result(resultNode.size());
  kernel_find<branching_factor, level_count, InnerNode, LeafNode> <<<numBlocks, numThreadsPerBlock>>>(
      m_root, D_PTR(voxel), voxel.size(), D_PTR(result));
  CHECK_CUDA_ERROR();
  HANDLE_CUDA_ERROR(cudaDeviceSynchronize());
  resultNode = result;
}

template<std::size_t branching_factor, std::size_t level_count, typename InnerNode, typename LeafNode>
template<typename o_InnerNode, typename o_LeafNode>
voxel_count NTree<branching_factor, level_count, InnerNode, LeafNode>::intersect(
    NTree<branching_factor, level_count, o_InnerNode, o_LeafNode>* other)
{
  timespec time = getCPUTime();
  const uint32_t numBlocks = this->numBlocks;
  const uint32_t numThreadsPerBlock = this->numThreadsPerBlock;

  thrust::device_vector<OctreeVoxelID> numConflicts(numBlocks * numThreadsPerBlock);

//const uint32_t level_count = robot->level_count;
  LOGGING_INFO(OctreeLog, "level_count:  " << level_count <<  endl);
  const OctreeVoxelID llog = (OctreeVoxelID) (log(float(numBlocks * numThreadsPerBlock)) / log(float(branching_factor)));
  const uint32_t splitLevel = level_count - 1
      - min((unsigned long long) llog, (unsigned long long) (level_count - 2));
  LOGGING_INFO(OctreeLog, "llog: " << llog << " splitLevel " << splitLevel <<  endl);

#ifdef SHARED_STACK
  uint32_t sMem = numThreadsPerBlock * splitLevel * branching_factor * sizeof(thrust::pair<InnerNode, InnerNode>);
  LOGGING_DEBUG(OctreeLog, "sMem size: " << sMem <<  endl);
#endif

#ifdef SMALL_STACK
  thrust::device_vector<Triple<InnerNode*, InnerNode*, numChild> > stack(
      numBlocks * numThreadsPerBlock * splitLevel);
#endif

#ifdef DEFAULT
  thrust::device_vector<thrust::pair<InnerNode*, o_InnerNode*> > stack(
      numBlocks * numThreadsPerBlock * splitLevel * branching_factor);
#endif

  LOGGING_INFO(OctreeLog, "Alloc: " << timeDiff(time, getCPUTime()) << " ms" <<  endl);

  time = getCPUTime();
#ifdef COALESCED
  uint32_t sMem = numThreadsPerBlock * sizeof(thrust::pair<LeafNode*, o_LeafNode*>);
  kernel_intersect_wo_stack_coalesced<branching_factor, level_count, InnerNode, LeafNode, o_InnerNode,
  o_LeafNode> <<<numBlocks, numThreadsPerBlock, sMem>>>(m_root, *other,D_PTR(numConflicts),
      splitLevel);
  CHECK_CUDA_ERROR();
  HANDLE_CUDA_ERROR(cudaDeviceSynchronize());
#endif

#ifdef WITHOUT_STACK
  kernel_intersect_wo_stack<<<numBlocks, numThreadsPerBlock>>>(*robot, *environment,
      D_PTR(numConflicts), splitLevel);
  CHECK_CUDA_ERROR();
  HANDLE_CUDA_ERROR(cudaDeviceSynchronize());
#endif

#ifdef SMALL_STACK
  kernel_intersect_smallStack<<<numBlocks, numThreadsPerBlock>>>(*robot, *environment,
      D_PTR(numConflicts),
      D_PTR(stack), splitLevel);
  CHECK_CUDA_ERROR();
  HANDLE_CUDA_ERROR(cudaDeviceSynchronize());
#endif

#ifdef SHARED_STACK
  kernel_intersect_shared<<<numBlocks, numThreadsPerBlock,sMem>>>(*robot, *environment,
      D_PTR(numConflicts) , splitLevel);
  CHECK_CUDA_ERROR();
  HANDLE_CUDA_ERROR(cudaDeviceSynchronize());
#endif

#ifdef DEFAULT
  kernel_intersect<branching_factor, level_count, InnerNode, LeafNode, o_InnerNode, o_LeafNode> <<<
      numBlocks, numThreadsPerBlock>>>(m_root, other->m_root, D_PTR(numConflicts),
  D_PTR(stack), splitLevel);
  CHECK_CUDA_ERROR();
  HANDLE_CUDA_ERROR(cudaDeviceSynchronize());
#endif

  LOGGING_INFO(OctreeLog, "kernel_intersect: " << timeDiff(time, getCPUTime()) << " ms" <<  endl);

  time = getCPUTime();
  OctreeVoxelID res = thrust::reduce(numConflicts.begin(), numConflicts.end());
  LOGGING_INFO(OctreeLog, "thrust::reduce: " << timeDiff(time, getCPUTime()) << " ms" <<  endl);

  return res;
}

template<std::size_t branching_factor, std::size_t level_count, typename InnerNode, typename LeafNode>
template<typename o_InnerNode, typename o_LeafNode, typename Collider>
OctreeVoxelID NTree<branching_factor, level_count, InnerNode, LeafNode>::intersect_load_balance(
    NTree<branching_factor, level_count, o_InnerNode, o_LeafNode>* other, const uint32_t min_level, Collider collider,
    bool mark_collisions, double* balance_overhead, int* num_balance_tasks)
{
  const std::string prefix = __FUNCTION__;
  PERF_MON_START(prefix);

  std::size_t num_collisions = 0;
  // mark_collisions is not a template parameter, to be able to omit the template parameters for using this function (see GvlNTree.hpp)
  if(mark_collisions)
  {
    typedef LoadBalancer::Intersect<
         branching_factor,
         level_count,
         InnerNode,
         LeafNode,
         o_InnerNode,
         o_LeafNode,
         Collider,
         true> MyLoadBalancer;

    MyLoadBalancer load_balancer(
         this,
         other,
         min_level,
         collider);
     load_balancer.run();
     num_collisions = load_balancer.m_num_collisions;
  }
  else
  {
    typedef LoadBalancer::Intersect<
         branching_factor,
         level_count,
         InnerNode,
         LeafNode,
         o_InnerNode,
         o_LeafNode,
         Collider,
         false> MyLoadBalancer;

     MyLoadBalancer load_balancer(
         this,
         other,
         min_level,
         collider);
     load_balancer.run();
     num_collisions = load_balancer.m_num_collisions;
  }

  PERF_MON_PRINT_INFO_P(prefix, "", prefix);
//  PERF_MON_ADD_DATA_NONTIME_P("BalanceOverhead", *balance_overhead, prefix);
//  PERF_MON_ADD_DATA_NONTIME_P("BalanceTasks", *num_balance_tasks, prefix);
  PERF_MON_ADD_DATA_NONTIME_P("NumCollisions", num_collisions, prefix);

#ifdef INTERSECT_MESSAGES
  LOGGING_INFO(OctreeInsertLog, "used min level: " << min_level <<  endl);
#endif
  return num_collisions;
}

//template<std::size_t branching_factor, std::size_t level_count, typename InnerNode, typename LeafNode>
//void NTree<branching_factor, level_count, InnerNode, LeafNode>::computeFreeSpaceViaRayCast_VoxelList(
//    thrust::device_vector<Voxel>& d_occupied_voxel, gpu_voxels::Vector3ui sensor_origin,
//    thrust::host_vector<thrust::pair<OctreeVoxelID*, voxel_count> >& h_packed_levels)
//{
//  const uint32_t numThreadsPerBlock = 32;
//
//// ##### alloc mem #####
//  timespec time = getCPUTime();
//  const uint32_t kinect_points = d_occupied_voxel.size();
//  const uint32_t max_depth_in_voxel = 2000;
//
//// TODO correct calculation of max_size_free_space based on max_depth_in_voxel!
//  const uint32_t max_size_free_space = ((kinect_points / numThreadsPerBlock) + 1) * numThreadsPerBlock
//      * max_depth_in_voxel;
//  printf("max_size_free_space: %u\n", max_size_free_space);
//  size_t size = max_size_free_space * sizeof(OctreeVoxelID);
//  thrust::device_vector<OctreeVoxelID> d_free_space(max_size_free_space, INVALID_VOXEL);
//  thrust::device_vector<uint32_t> d_voxel_count(numBlocks * numThreadsPerBlock);
//  HANDLE_CUDA_ERROR(cudaDeviceSynchronize());
////d_voxel_count.back() = 0;
//  printf("Alloc %f MB for free space: %f ms\n", double(size) * cBYTE2MBYTE, timeDiff(time, getCPUTime()));
//  time = getCPUTime();
//
//// ##### init free space #####
////  HANDLE_CUDA_ERROR(
////      cudaMemset((void*) D_PTR(d_free_space), (int) INVALID_VOXEL, max_size_free_space * sizeof(voxel_id)));
////  HANDLE_CUDA_ERROR(cudaDeviceSynchronize());
////  double diff = timeDiff(time, getCPUTime());
////  printf("Init free space with %f GB/s: %f ms\n", double(size) / 1024.0 / 1024.0 / 1024.0 / (diff / 1000.0),
////         diff);
////  time = getCPUTime();
//
//// ##### ray cast #####
//  kernel_rayInsert<false, false, branching_factor> <<<numBlocks, numThreadsPerBlock>>>(D_PTR(d_occupied_voxel),kinect_points,
//      sensor_origin, D_PTR(d_free_space), D_PTR(d_voxel_count), max_depth_in_voxel * numThreadsPerBlock);
//  CHECK_CUDA_ERROR();
//  HANDLE_CUDA_ERROR(cudaDeviceSynchronize());
//  printf("Ray cast: %f ms\n", timeDiff(time, getCPUTime()));
//
//// ##### sort #####
//  time = getCPUTime();
//  thrust::sort(d_free_space.begin(), d_free_space.end());
//  HANDLE_CUDA_ERROR(cudaDeviceSynchronize());
//  printf("thrust::sort(): %f ms\n", timeDiff(time, getCPUTime()));
//  time = getCPUTime();
//  uint32_t num_free_voxel = thrust::reduce(d_voxel_count.begin(), d_voxel_count.end());
//  HANDLE_CUDA_ERROR(cudaDeviceSynchronize());
//  d_voxel_count.clear();
//  d_voxel_count.shrink_to_fit();
//  printf("thrust::reduce(): %f ms\n", timeDiff(time, getCPUTime()));
//  printf("num_free_voxel: %u\n", num_free_voxel);
//
//// alternative: http://stackoverflow.com/questions/12463693/how-to-remove-zero-values-from-an-array-in-parallel
//// ##### remove duplicates ######
//  time = getCPUTime();
//  voxel_count num_free_wo_duplicates = 0;
//  uint32_t remove_invalid_val = 0;
//  if (((OctreeVoxelID) d_free_space.back()) == INVALID_VOXEL)
//    remove_invalid_val = 1;
//  d_free_space.erase(thrust::unique(d_free_space.begin(), d_free_space.end()) - remove_invalid_val,
//                     d_free_space.end());
//  num_free_wo_duplicates = d_free_space.size();
//  thrust::device_vector<OctreeVoxelID> d_free_space_wo_duplicates(1);
//  HANDLE_CUDA_ERROR(cudaDeviceSynchronize());
//  d_free_space_wo_duplicates.swap(d_free_space);
//  d_free_space_wo_duplicates.shrink_to_fit();
//
//// num_free_wo_duplicates = d_free_space_wo_duplicates.size();
//  assert(((OctreeVoxelID)d_free_space.back())!= INVALID_VOXEL);
//
////assert(checkSorting(D_PTR(d_free_space_wo_duplicates), num_free_wo_duplicates));
//
////  thrust::device_vector<voxel_count> d_num_voxel_wo_duplicates(numBlocks + 1); // TODO: do this on host side -> faster
////  d_num_voxel_wo_duplicates.back() = 0;
////  kernel_removeDuplicates<true> <<< numBlocks, numThreadsPerBlock >>>(
////      D_PTR(d_free_space), max_size_free_space, NULL, D_PTR(d_num_voxel_wo_duplicates));
////  CHECK_CUDA_ERROR();
////  HANDLE_CUDA_ERROR(cudaDeviceSynchronize());
////  thrust::exclusive_scan(d_num_voxel_wo_duplicates.begin(), d_num_voxel_wo_duplicates.end(),
////                         d_num_voxel_wo_duplicates.begin());
////  HANDLE_CUDA_ERROR(cudaDeviceSynchronize());
////
////  voxel_count num_free_wo_duplicates = (voxel_count) d_num_voxel_wo_duplicates.back();
////  thrust::device_vector<voxel_id> d_free_space_wo_duplicates(num_free_wo_duplicates);
////  kernel_removeDuplicates<false> <<< numBlocks, numThreadsPerBlock >>>(D_PTR(d_free_space), max_size_free_space,
////      D_PTR(d_free_space_wo_duplicates), D_PTR(d_num_voxel_wo_duplicates));
////  CHECK_CUDA_ERROR();
////  HANDLE_CUDA_ERROR(cudaDeviceSynchronize());
////  d_free_space.clear();
////  d_free_space.shrink_to_fit();
////  d_num_voxel_wo_duplicates.clear();
////  d_num_voxel_wo_duplicates.shrink_to_fit();
//
//  printf("remove duplicates: %f ms\n", timeDiff(time, getCPUTime()));
//  printf("Points without duplicates : %u\n", num_free_wo_duplicates);
//
//// print voxel wo duplicates
////printf("\nfree_space_wo_duplicates:\n");
////  thrust::host_vector<voxel_id> h_free_space_wo_duplicates = d_free_space_wo_duplicates;
////  for (uint32_t i = 0; i < h_free_space_wo_duplicates.size(); ++i)
////    printf("%u: %lu\n", i, (voxel_id) h_free_space_wo_duplicates[i]);
////  for (uint32_t i = 0; i < h_free_space_wo_duplicates.size() - 1; ++i)
////    assert((voxel_id )h_free_space_wo_duplicates[i] < (voxel_id )h_free_space_wo_duplicates[i + 1]);
//
//  time = getCPUTime();
//  uint32_t num_packed_voxel = 0;
//  uint32_t last_level_sum = 0;
//  uint32_t total_num_voxel = num_free_wo_duplicates;
//  for (uint32_t l = 0; l < level_count; ++l)
//  {
//    //numBlocks = 1;
//    // ###### pack voxel of level i ######
//    thrust::device_vector<voxel_count> d_num_voxel_this_level(numBlocks + 1);
//    thrust::device_vector<voxel_count> d_num_voxel_next_level(numBlocks + 1);
//    d_num_voxel_this_level.back() = 0;
//    d_num_voxel_next_level.back() = 0;
//
//    //printf("\nnum_free_wo_duplicates: %u \n", num_free_wo_duplicates);
////    for (uint32_t i = 0; i < num_free_wo_duplicates; ++i)
////      printf("d_free_space_wo_duplicates %u: %lu\n", i, (voxel_id) d_free_space_wo_duplicates[i]);
//
//    //printf("RUN %u\n", l);
//    //printf("Check sorting 1\n");
//    //assert(checkSorting(D_PTR(d_free_space_wo_duplicates), num_free_wo_duplicates));
//    timespec time_loop = getCPUTime();
//    kernel_packVoxel<branching_factor, true> <<<numBlocks, numThreadsPerBlock>>>(
//        D_PTR(d_free_space_wo_duplicates), num_free_wo_duplicates,
//        D_PTR(d_num_voxel_this_level), D_PTR(d_num_voxel_next_level), l, NULL, NULL);
//    CHECK_CUDA_ERROR();
//    HANDLE_CUDA_ERROR(cudaDeviceSynchronize());
//    // TODO use only one array to do the scan for both and then use an offset since it's more efficient
//    thrust::exclusive_scan(d_num_voxel_this_level.begin(), d_num_voxel_this_level.end(),
//                           d_num_voxel_this_level.begin());
//    thrust::exclusive_scan(d_num_voxel_next_level.begin(), d_num_voxel_next_level.end(),
//                           d_num_voxel_next_level.begin());
//    HANDLE_CUDA_ERROR(cudaDeviceSynchronize());
//    voxel_count num_this_level = (voxel_count) d_num_voxel_this_level.back();
//    voxel_count num_next_level = (voxel_count) d_num_voxel_next_level.back();
//    printf("num_this_level: %u num_next_level: %u \n", num_this_level, num_next_level);
//    printf("kernel_packVoxel_count() level %u: %f ms\n", l, timeDiff(time_loop, getCPUTime()));
//    time_loop = getCPUTime();
//
//    assert(num_free_wo_duplicates == num_this_level + num_next_level * branching_factor);
//
//    last_level_sum += num_this_level * uint32_t(std::pow(branching_factor, l));
//    assert(last_level_sum + num_next_level * uint32_t(std::pow(branching_factor, l + 1)) == total_num_voxel);
//
//    //break;
//    // move data
//    OctreeVoxelID* d_free_space_this_level = NULL;
//    HANDLE_CUDA_ERROR(cudaMalloc(&d_free_space_this_level, num_this_level * sizeof(OctreeVoxelID)));
//    thrust::device_vector<OctreeVoxelID> d_free_space_next_level(num_next_level);
//
//    //printf("Check sorting 2\n");
//    // assert(checkSorting(D_PTR(d_free_space_wo_duplicates), num_free_wo_duplicates));
//
//    kernel_packVoxel<branching_factor, false> <<<numBlocks, numThreadsPerBlock>>>(
//        D_PTR(d_free_space_wo_duplicates), num_free_wo_duplicates,
//        D_PTR(d_num_voxel_this_level), D_PTR(d_num_voxel_next_level),
//        l, d_free_space_this_level, D_PTR(d_free_space_next_level));
//    CHECK_CUDA_ERROR();
//    HANDLE_CUDA_ERROR(cudaDeviceSynchronize());
//    printf("kernel_packVoxel() level %u: %f ms\n", l, timeDiff(time_loop, getCPUTime()));
//
//    d_free_space_wo_duplicates.clear();
//    d_free_space_wo_duplicates.shrink_to_fit();
//    d_free_space_wo_duplicates.swap(d_free_space_next_level);
//    num_free_wo_duplicates = num_next_level;
//
//    //printf("Check sorting 3\n");
//    //assert(checkSorting(d_free_space_this_level, num_this_level));
//
//    // store level pointer
//    h_packed_levels[l] = thrust::make_pair<OctreeVoxelID*, voxel_count>(d_free_space_this_level, num_this_level);
//
//    num_packed_voxel += num_this_level;
//
//    if (num_next_level == 0)
//      break;
//  }
//  printf("num_packed_voxel: %u\n", num_packed_voxel);
//  printf("kernel_packVoxel(): %f ms\n", timeDiff(time, getCPUTime()));
//}

template<std::size_t branching_factor, std::size_t level_count, typename InnerNode, typename LeafNode>
void NTree<branching_factor, level_count, InnerNode, LeafNode>::packVoxel_Map_and_List(
    MapProperties<typename InnerNode::RayCastType, branching_factor>& map_properties,
    thrust::host_vector<thrust::pair<OctreeVoxelID*, voxel_count> >& h_packed_levels, voxel_count num_free_voxel,
    uint32_t min_level)
{
  // ### pack voxel - compute needed space ###
  timespec time = getCPUTime();
  thrust::device_vector<voxel_count> d_num_voxel_this_level(numBlocks + 1);
  thrust::device_vector<voxel_count> d_num_voxel_next_level(numBlocks + 1);
  d_num_voxel_this_level.back() = 0;
  d_num_voxel_next_level.back() = 0;
  // cudaProfilerStart();
  //  kernel_packByteMap_MemEfficient_Coa2<branching_factor, true> <<<numBlocks, 128>>>(
  //      D_PTR(d_num_voxel_this_level), D_PTR(d_num_voxel_next_level), map_properties);
  // CHECK_CUDA_ERROR();
  kernel_packMortonL0Map<NUM_THREADS_PER_BLOCK, branching_factor, true, false, PACKING_OF_VOXEL, InnerNode> <<<
      numBlocks,
      NUM_THREADS_PER_BLOCK>>>(D_PTR(d_num_voxel_this_level), D_PTR(d_num_voxel_next_level), map_properties);
  CHECK_CUDA_ERROR();
  HANDLE_CUDA_ERROR(cudaDeviceSynchronize());
  //  cudaProfilerStop();
#ifdef FREESPACE_MESSAGES
  LOGGING_DEBUG(OctreeFreespaceLog, "kernel_packByteMap(): " << timeDiff(time, getCPUTime() << " ms" << endl);
#endif
  voxel_count num_this_level =
      (min_level > 0) ? 0 : thrust::reduce(d_num_voxel_this_level.begin(), d_num_voxel_this_level.end());
  voxel_count num_next_level = thrust::reduce(d_num_voxel_next_level.begin(), d_num_voxel_next_level.end());
  HANDLE_CUDA_ERROR(cudaDeviceSynchronize());
#ifdef FREESPACE_MESSAGES
  LOGGING_DEBUG(OctreeFreespaceLog, "num_this_level: " << num_this_level << " num_next_level: " << num_next_level << endl);
#endif
  assert(num_this_level + num_next_level * branching_factor == num_free_voxel);

  // ### pack voxel with ByteMap ###
  time = getCPUTime();
  thrust::device_vector<voxel_count> d_this_level_index(1, 0);
  thrust::device_vector<voxel_count> d_next_level_index(1, 0);
  OctreeVoxelID* d_free_space_this_level = NULL;
  if (min_level == 0)
    HANDLE_CUDA_ERROR(cudaMalloc(&d_free_space_this_level, num_this_level * sizeof(OctreeVoxelID)));
  thrust::device_vector<OctreeVoxelID> d_free_space_next_level(num_next_level);
  //  kernel_packByteMap_MemEfficient_Coa2<branching_factor, false> <<<numBlocks, 128>>>(
  //      D_PTR(d_num_voxel_this_level), D_PTR(d_num_voxel_next_level),map_properties, D_PTR(d_this_level_index), D_PTR(d_next_level_index),d_free_space_this_level,D_PTR(d_free_space_next_level));
  //  CHECK_CUDA_ERROR();
  kernel_packMortonL0Map<NUM_THREADS_PER_BLOCK, branching_factor, false, false, PACKING_OF_VOXEL, InnerNode> <<<
      numBlocks,
      NUM_THREADS_PER_BLOCK>>>(D_PTR(d_num_voxel_this_level),
  D_PTR(d_num_voxel_next_level), map_properties, D_PTR(d_this_level_index), D_PTR(d_next_level_index),
  d_free_space_this_level, D_PTR(d_free_space_next_level), NULL, NULL);
  CHECK_CUDA_ERROR();
  HANDLE_CUDA_ERROR(cudaDeviceSynchronize());

#ifdef FREESPACE_MESSAGES
  LOGGING_DEBUG(OctreeFreespaceLog, "kernel_packByteMap(): " << timeDiff(time, getCPUTime() << " ms" << endl);
  LOGGING_DEBUG(OctreeFreespaceLog, "this level: " << (voxel_count) d_this_level_index.back() << endl);
  LOGGING_DEBUG(OctreeFreespaceLog, "next level: " << (voxel_count) d_next_level_index.back() << endl);
#endif

  HANDLE_CUDA_ERROR(cudaFree(map_properties.d_ptr));

  // TODO: eliminate the sorting step with the following brute force method (MIGHT BE FASTER):
  // compute the smallest InnerNode (Super-Voxel) fot the ByteMap
  // split the InnerNode in small Morton-Cubes; Each Block interates over it's morton codes, invert them an check whether the position still in the ByteMap
  // count needed space, make prefix sum, move data (morton codes) in parallel, try to use memory coalescing for the memory reads

  // #### sort packed voxel ####
  time = getCPUTime();
  if (min_level == 0)
  {
    thrust::device_ptr<OctreeVoxelID> ptr(d_free_space_this_level);
    thrust::sort(ptr, ptr + num_this_level);
    HANDLE_CUDA_ERROR(cudaDeviceSynchronize());
  }
#ifdef FREESPACE_MESSAGES
  LOGGING_DEBUG(OctreeFreespaceLog, "thrust::sort(): " << timeDiff(time, getCPUTime() << " ms" << endl);
#endif
  time = getCPUTime();
  thrust::sort(d_free_space_next_level.begin(), d_free_space_next_level.end());
  HANDLE_CUDA_ERROR(cudaDeviceSynchronize());
#ifdef FREESPACE_MESSAGES
  LOGGING_DEBUG(OctreeFreespaceLog, "thrust::sort(): " << timeDiff(time, getCPUTime() << " ms" << endl);
#endif

  // check for duplicates in both sets
#ifndef NDEBUG
  if (min_level == 0)
  {
#ifdef FREESPACE_MESSAGES
    LOGGING_DEBUG(OctreeFreespaceLog, "checking for duplicates in this and next level..." << endl);
#endif
    thrust::device_ptr<OctreeVoxelID> ptr(d_free_space_this_level);
    thrust::device_vector<OctreeVoxelID> result(max(num_this_level, num_next_level));
    thrust::device_vector<OctreeVoxelID>::iterator result_end = thrust::set_intersection(
        d_free_space_next_level.begin(), d_free_space_next_level.end(), ptr, ptr + num_this_level,
        result.begin());
    if (result_end != result.begin())
    {
#ifdef FREESPACE_MESSAGES
      LOGGING_ERROR(OctreeFreespaceLog, "voxel_id " << (OctreeVoxelID) result[0] << " in both this and next level" << endl);
#endif
      assert(false);
    }
  }
#endif

  h_packed_levels[0] = thrust::make_pair<OctreeVoxelID*, voxel_count>(d_free_space_this_level, num_this_level);

//  OctreeVoxelID* tmp_ptr = NULL;
//  HANDLE_CUDA_ERROR(cudaMalloc(&tmp_ptr, num_next_level * sizeof(OctreeVoxelID)));
//  HANDLE_CUDA_ERROR(
//      cudaMemcpy(tmp_ptr, D_PTR(d_free_space_next_level), num_next_level * sizeof(OctreeVoxelID), cudaMemcpyDeviceToDevice));
//  h_packed_levels[1] = thrust::make_pair<OctreeVoxelID*, voxel_count>(tmp_ptr, num_next_level);
//  return;

  thrust::device_vector<OctreeVoxelID> d_free_space;
  d_free_space.swap(d_free_space_next_level);
  uint32_t num_free_space = num_next_level;

  time = getCPUTime();
  uint32_t num_packed_voxel = num_this_level;
  uint32_t last_level_sum = num_this_level;
  uint32_t total_num_voxel = num_this_level + num_next_level * branching_factor;
  for (uint32_t l = 1; l < level_count; ++l)
  {
    //numBlocks = 1;
    // ###### pack voxel of level i ######
    d_num_voxel_this_level.back() = 0;
    d_num_voxel_next_level.back() = 0;

    //printf("\nnum_free_wo_duplicates: %u \n", num_free_wo_duplicates);
    //    for (uint32_t i = 0; i < num_free_wo_duplicates; ++i)
    //      printf("d_free_space_wo_duplicates %u: %lu\n", i, (voxel_id) d_free_space_wo_duplicates[i]);

    //printf("RUN %u\n", l);
    //printf("Check sorting 1\n");
    assert(checkSorting(D_PTR(d_free_space), num_free_space));
    timespec time_loop = getCPUTime();

    kernel_packVoxel<branching_factor, true> <<<numBlocks, 32>>>(D_PTR(d_free_space),num_free_space,
    D_PTR(d_num_voxel_this_level), D_PTR(
        d_num_voxel_next_level), l, NULL, NULL);
    CHECK_CUDA_ERROR();
    HANDLE_CUDA_ERROR(cudaDeviceSynchronize());
    // TODO use only one array to do the scan for both and then use an offset since it's more efficient
    thrust::exclusive_scan(d_num_voxel_this_level.begin(), d_num_voxel_this_level.end(),
                           d_num_voxel_this_level.begin());
    thrust::exclusive_scan(d_num_voxel_next_level.begin(), d_num_voxel_next_level.end(),
                           d_num_voxel_next_level.begin());
    HANDLE_CUDA_ERROR(cudaDeviceSynchronize());
    num_this_level = (min_level > l) ? 0 : (voxel_count) d_num_voxel_this_level.back();
    num_next_level = (voxel_count) d_num_voxel_next_level.back();
#ifdef FREESPACE_MESSAGES
    LOGGING_DEBUG(OctreeFreespaceLog, "num_this_level: " << num_this_level << " num_next_level: "<< num_next_level << endl);
    LOGGING_DEBUG(OctreeFreespaceLog, "kernel_packVoxel_count() level " << l << ": " << timeDiff(time_loop, getCPUTime()) << " ms" << endl);
#endif
    time_loop = getCPUTime();

    assert(num_free_space == num_this_level + num_next_level * branching_factor);

    last_level_sum += num_this_level * uint32_t(std::pow(branching_factor, l));
    unused(total_num_voxel); // gets used in assert
    assert(last_level_sum + num_next_level * uint32_t(std::pow(branching_factor, l + 1)) == total_num_voxel);

    //break;
    // move data
    OctreeVoxelID* d_free_space_this_level = NULL;
    if (min_level <= l)
      HANDLE_CUDA_ERROR(cudaMalloc(&d_free_space_this_level, num_this_level * sizeof(OctreeVoxelID)));
    thrust::device_vector<OctreeVoxelID> d_free_space_next_level(num_next_level);

    //printf("Check sorting 2\n");
    assert(checkSorting(D_PTR(d_free_space), num_free_space));

    kernel_packVoxel<branching_factor, false> <<<numBlocks, 32>>>(D_PTR(d_free_space),num_free_space,
    D_PTR(d_num_voxel_this_level), D_PTR(
        d_num_voxel_next_level),
    l, d_free_space_this_level, D_PTR(d_free_space_next_level));
    CHECK_CUDA_ERROR();
    HANDLE_CUDA_ERROR(cudaDeviceSynchronize());
#ifdef FREESPACE_MESSAGES
    LOGGING_DEBUG(OctreeFreespaceLog, "kernel_packVoxel_count() level " << l << ": " << timeDiff(time_loop, getCPUTime()) << " ms" << endl);
#endif

    d_free_space.clear();
    d_free_space.shrink_to_fit();
    d_free_space.swap(d_free_space_next_level);
    num_free_space = num_next_level;

    //printf("Check sorting 3\n");
    assert(checkSorting(d_free_space_this_level, num_this_level));

    // store level pointer
    h_packed_levels[l] = thrust::make_pair<OctreeVoxelID*, voxel_count>(d_free_space_this_level, num_this_level);

    num_packed_voxel += num_this_level;

    if (num_next_level == 0)
      break;
  }
#ifdef FREESPACE_MESSAGES
  LOGGING_DEBUG(OctreeFreespaceLog, "num_packed_voxel: " << num_packed_voxel << endl);
  LOGGING_DEBUG(OctreeFreespaceLog, "kernel_packVoxel(): " << timeDiff(time, getCPUTime()) << " ms" << endl);
#endif
}

template<std::size_t branching_factor, std::size_t level_count, typename InnerNode, typename LeafNode>
void NTree<branching_factor, level_count, InnerNode, LeafNode>::packVoxel_Map(
    MapProperties<typename InnerNode::RayCastType, branching_factor>& map_properties,
    thrust::host_vector<ComputeFreeSpaceData>& h_packed_levels, voxel_count num_free_voxel,
    uint32_t min_level)
{
  const std::string prefix = __FUNCTION__;
  const std::string temp_timer = prefix + "_temp";
  const std::string loop_timer = prefix + "_loop";
  PERF_MON_START(prefix);
  PERF_MON_START(temp_timer);

#if defined FREESPACE_MESSAGES && defined _IC_DEBUG_
  timespec time_total = getCPUTime();
  timespec time;
#endif
  thrust::device_vector<voxel_count> d_num_voxel_this_level(numBlocks + 1);
  thrust::device_vector<voxel_count> d_num_voxel_next_level(numBlocks + 1);
  voxel_count num_next_level = 1;
  MapProperties<typename InnerNode::RayCastType, branching_factor> this_level_map = map_properties;
  double total_sort_time = 0.0, total_count_kernel_time = 0.0, total_malloc_time = 0.0, total_kernel_time = 0.0;

  uint32_t num_packed_voxel = 0;
  uint32_t last_level_sum = 0;
  uint32_t num_last_level = num_free_voxel;
  uint32_t l = map_properties.level;
  for (; l < level_count && num_next_level != 0; ++l)
  {
    PERF_MON_START(loop_timer);

    // ### pack voxel - compute needed space ###
    d_num_voxel_this_level.back() = 0;
    d_num_voxel_next_level.back() = 0;

#if defined FREESPACE_MESSAGES && defined _IC_DEBUG_
    time = getCPUTime();
#endif
    kernel_packMortonL0Map<NUM_THREADS_PER_BLOCK, branching_factor, true, true, PACKING_OF_VOXEL, InnerNode>
    <<<numBlocks, NUM_THREADS_PER_BLOCK>>>(
        D_PTR(d_num_voxel_this_level),
        D_PTR(d_num_voxel_next_level),
        this_level_map);
    CHECK_CUDA_ERROR();
    HANDLE_CUDA_ERROR(cudaDeviceSynchronize());

#ifdef IC_PERFORMANCE_MONITOR
// we need the return value, so we need to call the function directly
    total_count_kernel_time += icl_core::perf_mon::PerformanceMonitor::measurement(loop_timer, "PackCountKernelL" + to_string(l, "%02d"), prefix);
#endif

    //  cudaProfilerStop();
#ifdef FREESPACE_MESSAGES
    LOGGING_DEBUG(OctreeFreespaceLog, "level " << l << endl);
    LOGGING_DEBUG(OctreeFreespaceLog, "kernel_packByteMap(counting): " << timeDiff(time, getCPUTime()) << " ms" << endl);
#ifdef _IC_DEBUG_
    time = getCPUTime();
#endif
#endif
    uint32_t num_this_level =
        (min_level > l) ? 0 : thrust::reduce(d_num_voxel_this_level.begin(), d_num_voxel_this_level.end());
    num_next_level = thrust::reduce(d_num_voxel_next_level.begin(), d_num_voxel_next_level.end());
    HANDLE_CUDA_ERROR(cudaDeviceSynchronize());
#ifdef FREESPACE_MESSAGES
    LOGGING_DEBUG(OctreeFreespaceLog, "num_this_level: " << num_this_level << "num_next_level: " << num_next_level << endl);
    LOGGING_DEBUG(OctreeFreespaceLog, "thrust::reduce(): " << timeDiff(time, getCPUTime()) << " ms" << endl);
#ifdef _IC_DEBUG_
    time = getCPUTime();
#endif
#endif
    unused(num_last_level);
    assert(num_last_level == (num_this_level + num_next_level * branching_factor));
    last_level_sum += num_this_level * uint32_t(std::pow(branching_factor, l));
    assert(last_level_sum + num_next_level * uint32_t(std::pow(branching_factor, l + 1)) == num_free_voxel);

    // ### pack voxel with ByteMap ###
    thrust::device_vector<voxel_count> d_this_level_index(1, 0);
    OctreeVoxelID* d_this_level_voxel_id = NULL;
    BasicData* d_this_level_basic_data = NULL;
    if (min_level <= l)
    {
      HANDLE_CUDA_ERROR(cudaMalloc(&d_this_level_voxel_id, num_this_level * sizeof(OctreeVoxelID)));
      HANDLE_CUDA_ERROR(cudaMalloc(&d_this_level_basic_data, num_this_level * sizeof(BasicData)));
    }

    // create new map, alloc and init mem
    MapProperties<typename InnerNode::RayCastType, branching_factor> next_level_map =
        this_level_map.createNextLevelMap();
    HANDLE_CUDA_ERROR(
        cudaMalloc((void** ) &next_level_map.d_ptr,
                   next_level_map.size_v * sizeof(typename InnerNode::RayCastType)));

    typename InnerNode::RayCastType init;
    getRayCastInit(&init);
    thrust::device_ptr<typename InnerNode::RayCastType> d_ptr(next_level_map.d_ptr);
    thrust::fill(d_ptr, d_ptr + next_level_map.size_v, init);

#ifdef IC_PERFORMANCE_MONITOR
// we need the return value, so we need to call the function directly
    total_malloc_time += icl_core::perf_mon::PerformanceMonitor::measurement(loop_timer, "MallocL" + to_string(l, "%02d"), prefix);
#endif

#ifdef FREESPACE_MESSAGES
    LOGGING_DEBUG(OctreeFreespaceLog, "malloc/memset: " << timeDiff(time, getCPUTime() << " ms" << endl);
    LOGGING_DEBUG(OctreeFreespaceLog, "next_level_map" <<  endl);
#ifdef _IC_DEBUG_
    time = getCPUTime();
#endif
#endif

    kernel_packMortonL0Map<NUM_THREADS_PER_BLOCK, branching_factor, false, true, PACKING_OF_VOXEL, InnerNode>
    <<<numBlocks, NUM_THREADS_PER_BLOCK>>>(
        D_PTR(d_num_voxel_this_level),
        D_PTR(d_num_voxel_next_level),
        this_level_map,
        D_PTR(d_this_level_index),
        NULL,
        d_this_level_voxel_id,
        NULL,
        d_this_level_basic_data,
        NULL,
        next_level_map);
    CHECK_CUDA_ERROR();
    HANDLE_CUDA_ERROR(cudaDeviceSynchronize());

#ifdef IC_PERFORMANCE_MONITOR
// we need the return value, so we need to call the function directly
    total_kernel_time += icl_core::perf_mon::PerformanceMonitor::measurement(loop_timer, "PackKernelL" + to_string(l, "%02d"), prefix);
#endif

    num_this_level = (voxel_count) d_this_level_index.back();

    HANDLE_CUDA_ERROR(cudaFree(this_level_map.d_ptr));
    this_level_map = next_level_map;

#ifdef FREESPACE_MESSAGES
    LOGGING_DEBUG(OctreeFreespaceLog, "kernel_packByteMap(): " << timeDiff(time, getCPUTime() << " ms" << endl);
    LOGGING_DEBUG(OctreeFreespaceLog, "this level: " << num_this_level << endl);
#ifdef _IC_DEBUG_
    time = getCPUTime();
#endif
#endif

    if (min_level <= l)
    {
      const std::string sort_timer = prefix + "_sort";
      PERF_MON_START(sort_timer);
#if defined FREESPACE_MESSAGES && defined _IC_DEBUG_
      timespec time_sort = getCPUTime();
#endif

      // sort with CUB for small problem sizes
      // thrust performs better for large ones
      if(num_this_level < 300000)
      {
         OctreeVoxelID* key_tmp = NULL;
         BasicData* value_tmp = NULL;
         HANDLE_CUDA_ERROR(cudaMalloc(&key_tmp, num_this_level * sizeof(OctreeVoxelID)));
         HANDLE_CUDA_ERROR(cudaMalloc(&value_tmp, num_this_level * sizeof(BasicData)));

         OctreeVoxelID *d_key_buf = d_this_level_voxel_id;
         OctreeVoxelID *d_key_alt_buf = key_tmp;
         BasicData *d_value_buf = d_this_level_basic_data;
         BasicData *d_value_alt_buf = value_tmp;
         cub::DoubleBuffer<OctreeVoxelID> d_keys(d_key_buf, d_key_alt_buf);
         cub::DoubleBuffer<BasicData> d_values(d_value_buf, d_value_alt_buf);

         // Determine temporary device storage requirements
         void *d_temp_storage = NULL;
         size_t temp_storage_bytes = 0;
         cub::DeviceRadixSort::SortPairs(d_temp_storage, temp_storage_bytes, d_keys, d_values, num_this_level);
         // Allocate temporary storage
         HANDLE_CUDA_ERROR(cudaMalloc(&d_temp_storage, temp_storage_bytes));
         // Run sorting operation
         cub::DeviceRadixSort::SortPairs(d_temp_storage, temp_storage_bytes, d_keys, d_values, num_this_level);
         HANDLE_CUDA_ERROR(cudaFree(d_temp_storage));

         if(d_keys.Current() == key_tmp)
           HANDLE_CUDA_ERROR(cudaFree(d_this_level_voxel_id));
         else
           HANDLE_CUDA_ERROR(cudaFree(key_tmp));
         if(d_values.Current() == value_tmp)
           HANDLE_CUDA_ERROR(cudaFree(d_this_level_basic_data));
         else
           HANDLE_CUDA_ERROR(cudaFree(value_tmp));
         d_this_level_voxel_id = d_keys.Current();
         d_this_level_basic_data = d_values.Current();
       }
       else
       {
        thrust::device_ptr<OctreeVoxelID> ptr_voxel(d_this_level_voxel_id);
        thrust::device_ptr<BasicData> ptr_data(d_this_level_basic_data);
        thrust::sort_by_key(ptr_voxel, ptr_voxel + num_this_level, ptr_data);
        HANDLE_CUDA_ERROR(cudaDeviceSynchronize());
       }

#ifdef IC_PERFORMANCE_MONITOR
// we need the return value, so we need to call the function directly
    total_sort_time += icl_core::perf_mon::PerformanceMonitor::measurement(sort_timer, "SortL" + to_string(l, "%02d"), prefix);
#endif

#ifdef FREESPACE_MESSAGES
      LOGGING_DEBUG(OctreeFreespaceLog, "sort(): " << timeDiff(time_sort, getCPUTime()) << " ms" << endl);
#endif
    }

    // store level pointer
    h_packed_levels[l] = ComputeFreeSpaceData(d_this_level_voxel_id, d_this_level_basic_data, num_this_level);
    num_packed_voxel += num_this_level;

#ifdef FREESPACE_MESSAGES
    LOGGING_DEBUG(OctreeFreespaceLog, "num_this_level: " << num_this_level << "num_next_level: " << num_next_level << endl);
    // printf("kernel_packVoxel_count() level %u: %f ms\n", l, timeDiff(time_loop, getCPUTime()));
#endif
    num_last_level = num_next_level;

    PERF_MON_PRINT_INFO_P(temp_timer, "PackL" + to_string(l, "%02d"), prefix);

    if (!PACKING_OF_VOXEL)
      break;
  }
  HANDLE_CUDA_ERROR(cudaFree(this_level_map.d_ptr));

  // timings for skipped level to be complete
  for (size_t i = l; i < level_count; ++i)
  {
    PERF_MON_ADD_DATA_P("PackL" + to_string(i, "%02d"), 0, prefix);
    PERF_MON_ADD_DATA_P("SortL" + to_string(i, "%02d"), 0, prefix);
    PERF_MON_ADD_DATA_P("PackCountKernelL" + to_string(i, "%02d"), 0, prefix);
    PERF_MON_ADD_DATA_P("MallocL" + to_string(i, "%02d"), 0, prefix);
    PERF_MON_ADD_DATA_P("PackKernelL" + to_string(i, "%02d"), 0, prefix);
  }
  PERF_MON_ADD_DATA_P("SortALL", total_sort_time, prefix);
  PERF_MON_ADD_DATA_P("PackCountKernelALL", total_count_kernel_time, prefix);
  PERF_MON_ADD_DATA_P("MallocALL", total_malloc_time, prefix);
  PERF_MON_ADD_DATA_P("PackKernelALL", total_kernel_time, prefix);

  PERF_MON_PRINT_INFO_P(prefix, "", prefix);

#ifdef FREESPACE_MESSAGES
  LOGGING_DEBUG(OctreeFreespaceLog, "num_packed_voxel: " << num_packed_voxel << endl);
  LOGGING_DEBUG(OctreeFreespaceLog, "kernel_packVoxel(total): " << timeDiff(time_total, getCPUTime() << " ms" << endl);
#endif
}

//template<std::size_t branching_factor, std::size_t level_count, typename InnerNode, typename LeafNode>
//void NTree<branching_factor, level_count, InnerNode, LeafNode>::computeFreeSpaceViaRayCast(
//    thrust::device_vector<Voxel>& d_occupied_voxel, gpu_voxels::Vector3ui sensor_origin,
//    thrust::host_vector<thrust::pair<OctreeVoxelID*, voxel_count> >& h_packed_levels, uint32_t min_level)
//{
//  const bool MODE_MAP_ONLY = true;
//
//  assert(sizeof(typename InnerNode::RayCastType) == sizeof(typename InnerNode::RayCastType::Type));
//
//// ### find min/max coordinates ###
//  timespec time = getCPUTime();
//  voxel_count num_voxel = d_occupied_voxel.size();
//  thrust::device_vector<uint32_t> d_x(num_voxel);
//  thrust::device_vector<uint32_t> d_y(num_voxel);
//  thrust::device_vector<uint32_t> d_z(num_voxel);
//  kernel_split_voxel_vector<false, false, false, true> <<<numBlocks, numThreadsPerBlock>>>(
//      D_PTR(d_occupied_voxel), num_voxel, NULL,
//      NULL, NULL, D_PTR(d_x), D_PTR(d_y), D_PTR(d_z));
//  CHECK_CUDA_ERROR();
//  HANDLE_CUDA_ERROR(cudaDeviceSynchronize());
//#ifdef FREESPACE_MESSAGES
//  printf("kernel_split_voxel_vector(): %f ms\n", timeDiff(time, getCPUTime()));
//#endif
//
//  time = getCPUTime();
//  thrust::pair<thrust::device_vector<uint32_t>::iterator, thrust::device_vector<uint32_t>::iterator> res_x =
//      thrust::minmax_element(d_x.begin(), d_x.end());
//  thrust::pair<thrust::device_vector<uint32_t>::iterator, thrust::device_vector<uint32_t>::iterator> res_y =
//      thrust::minmax_element(d_y.begin(), d_y.end());
//  thrust::pair<thrust::device_vector<uint32_t>::iterator, thrust::device_vector<uint32_t>::iterator> res_z =
//      thrust::minmax_element(d_z.begin(), d_z.end());
//  MapProperties<typename InnerNode::RayCastType, branching_factor> map_properties(0);
//  map_properties.coordinate_x = D_PTR(d_x);
//  map_properties.coordinate_y = D_PTR(d_y);
//  map_properties.coordinate_z = D_PTR(d_z);
//  map_properties.coordinates_size = num_voxel;
//  map_properties.min_x = std::min((uint32_t) *res_x.first, sensor_origin.x);
//  map_properties.max_x = std::max((uint32_t) *res_x.second, sensor_origin.x);
//  map_properties.min_y = std::min((uint32_t) *res_y.first, sensor_origin.y);
//  map_properties.max_y = std::max((uint32_t) *res_y.second, sensor_origin.y);
//  map_properties.min_z = std::min((uint32_t) *res_z.first, sensor_origin.z);
//  map_properties.max_z = std::max((uint32_t) *res_z.second, sensor_origin.z);
//
//// ### align map at morton code and sizeof(ByteType) ###
////  uint32_t branching_factor_third_root = (uint32_t) std::pow(branching_factor, 1.0 / 3);
//// align x for WARP access
//  //  ###################################################
//  // TODO: align according to branching_factor
//  //  ###################################################
////  uint32_t x_alignment = 128; //sizeof(ByteType) * 8 / branching_factor_third_root / branching_factor_third_root;
////  map_properties.min_x -= map_properties.min_x % x_alignment;
////  map_properties.min_y -= map_properties.min_y % branching_factor_third_root;
////  map_properties.min_z -= map_properties.min_z % branching_factor_third_root;
////  map_properties.max_x += x_alignment - 1 - (map_properties.max_x % x_alignment);
////  map_properties.max_y += branching_factor_third_root - 1
////      - (map_properties.max_y % branching_factor_third_root);
////  map_properties.max_z += branching_factor_third_root - 1
////      - (map_properties.max_z % branching_factor_third_root);
//
//  map_properties.align();
//
//  HANDLE_CUDA_ERROR(cudaDeviceSynchronize());
//#ifdef FREESPACE_MESSAGES
//  printf("thrust::minmax_element(): %f ms\n", timeDiff(time, getCPUTime()));
//  cout << map_properties;
//#endif
//
//// ### malloc array ###
//  time = getCPUTime();
//
//  HANDLE_CUDA_ERROR(
//      cudaMalloc((void** ) &map_properties.d_ptr,
//                 map_properties.size * sizeof(typename InnerNode::RayCastType)));
//  HANDLE_CUDA_ERROR(cudaDeviceSynchronize());
//
//// ##### init free space #####
//  time = getCPUTime();
//  HANDLE_CUDA_ERROR(
//      cudaMemset(map_properties.d_ptr, getRayCastInitByte(typename InnerNode::RayCastType()),
//                 sizeof(typename InnerNode::RayCastType) * map_properties.size));
////  HANDLE_CUDA_ERROR(
////      cudaMemset(map_properties.d_ptr, 0, sizeof(typename InnerNode::RayCastType) * map_properties.size));
//  HANDLE_CUDA_ERROR(cudaDeviceSynchronize());
//#ifdef FREESPACE_MESSAGES
//  printf("cudaMemset : %f ms\n", timeDiff(time, getCPUTime()));
//#endif
//
//  thrust::device_ptr<typename InnerNode::RayCastType> d_ptr(map_properties.d_ptr);
//
//  // ##### ray cast #####
//// bit vector for ray casting is slow compared to byte array due to the need for an atomic operation
//// using morton code for 8 neighbors makes it even slower -> there might be more memory conflicts due to higher memory locality
//  thrust::device_vector<uint32_t> d_voxel_count(numBlocks * numThreadsPerBlock);
//#ifdef FREESPACE_MESSAGES
//  printf("sensor origin %u %u %u\n", sensor_origin.x, sensor_origin.y, sensor_origin.z);
//#endif
//
//  time = getCPUTime();
//  kernel_rayInsert<branching_factor, InnerNode> <<<numBlocks, numThreadsPerBlock>>>(sensor_origin,
//                                                                                    D_PTR(d_voxel_count),map_properties);
//  CHECK_CUDA_ERROR();
//  HANDLE_CUDA_ERROR(cudaDeviceSynchronize());
//
//#ifdef FREESPACE_MESSAGES
//  printf("Ray cast: %f ms\n", timeDiff(time, getCPUTime()));
//#endif
//
//  uint32_t set_to_free = thrust::reduce(d_voxel_count.begin(), d_voxel_count.end());
//  HANDLE_CUDA_ERROR(cudaDeviceSynchronize());
//  d_voxel_count.clear();
//  d_voxel_count.shrink_to_fit();
//  uint32_t num_free_voxel = thrust::count_if(d_ptr, d_ptr + map_properties.size, Comp_is_valid<InnerNode>());
//#ifdef FREESPACE_MESSAGES
//  printf("set_to_free: %u\n", set_to_free);
//  printf("num_free_voxel: %u\n", num_free_voxel);
//#endif
//
//  if (MODE_MAP_ONLY)
//    packVoxel_Map(map_properties, h_packed_levels, num_free_voxel, min_level);
//  else
//    packVoxel_Map_and_List(map_properties, h_packed_levels, num_free_voxel, min_level);
//}

template<std::size_t branching_factor, std::size_t level_count, typename InnerNode, typename LeafNode>
void NTree<branching_factor, level_count, InnerNode, LeafNode>::computeFreeSpaceViaRayCast(
    thrust::device_vector<Voxel>& d_occupied_voxel, gpu_voxels::Vector3ui sensor_origin,
    thrust::host_vector<ComputeFreeSpaceData>& h_packed_levels, uint32_t min_level)
{
  assert(sizeof(typename InnerNode::RayCastType) == sizeof(typename InnerNode::RayCastType::Type));

  const std::string prefix = __FUNCTION__;
  const std::string temp_timer = prefix + "_temp";
  PERF_MON_START(prefix);
  PERF_MON_START(temp_timer);

// ### find min/max coordinates ###
  timespec time = getCPUTime();
  voxel_count num_voxel = d_occupied_voxel.size();
  thrust::device_vector<uint32_t> d_x(num_voxel);
  thrust::device_vector<uint32_t> d_y(num_voxel);
  thrust::device_vector<uint32_t> d_z(num_voxel);
  uint32_t num_threads = 128;
  uint32_t num_blocks = num_voxel / num_threads + 1;
  kernel_split_voxel_vector<false, false, false, true> <<<num_blocks, num_threads>>>(
      D_PTR(d_occupied_voxel), num_voxel, NULL,
      NULL, NULL, D_PTR(d_x), D_PTR(d_y), D_PTR(d_z));
  CHECK_CUDA_ERROR();
  HANDLE_CUDA_ERROR(cudaDeviceSynchronize());
#ifdef FREESPACE_MESSAGES
  LOGGING_DEBUG(OctreeFreespaceLog, "kernel_split_voxel_vector(): " << timeDiff(time_total, getCPUTime() << " ms" << endl);
#endif

  time = getCPUTime();
  thrust::pair<thrust::device_vector<uint32_t>::iterator, thrust::device_vector<uint32_t>::iterator> res_x =
      thrust::minmax_element(d_x.begin(), d_x.end());
  thrust::pair<thrust::device_vector<uint32_t>::iterator, thrust::device_vector<uint32_t>::iterator> res_y =
      thrust::minmax_element(d_y.begin(), d_y.end());
  thrust::pair<thrust::device_vector<uint32_t>::iterator, thrust::device_vector<uint32_t>::iterator> res_z =
      thrust::minmax_element(d_z.begin(), d_z.end());
  MapProperties<typename InnerNode::RayCastType, branching_factor> map_properties(0);
  map_properties.coordinate_x = D_PTR(d_x);
  map_properties.coordinate_y = D_PTR(d_y);
  map_properties.coordinate_z = D_PTR(d_z);
  map_properties.coordinates_size = num_voxel;
  map_properties.min_x = std::min((uint32_t) *res_x.first, sensor_origin.x);
  map_properties.max_x = std::max((uint32_t) *res_x.second, sensor_origin.x);
  map_properties.min_y = std::min((uint32_t) *res_y.first, sensor_origin.y);
  map_properties.max_y = std::max((uint32_t) *res_y.second, sensor_origin.y);
  map_properties.min_z = std::min((uint32_t) *res_z.first, sensor_origin.z);
  map_properties.max_z = std::max((uint32_t) *res_z.second, sensor_origin.z);

  map_properties.align();

  HANDLE_CUDA_ERROR(cudaDeviceSynchronize());
#ifdef FREESPACE_MESSAGES
  LOGGING_DEBUG(OctreeFreespaceLog, "thrust::minmax_element(): " << timeDiff(time, getCPUTime() << " ms" << endl);
  LOGGING_DEBUG(OctreeFreespaceLog, map_properties << endl);
#endif

// ### malloc array ###
  time = getCPUTime();

  HANDLE_CUDA_ERROR(
      cudaMalloc((void** ) &map_properties.d_ptr,
                 map_properties.size * sizeof(typename InnerNode::RayCastType)));
  HANDLE_CUDA_ERROR(cudaDeviceSynchronize());

// ##### init free space #####
  time = getCPUTime();
  typename InnerNode::RayCastType init;
  getRayCastInit(&init);
  thrust::device_ptr<typename InnerNode::RayCastType> d_ptr(map_properties.d_ptr);
  thrust::fill(d_ptr, d_ptr + map_properties.size, init);
#ifdef FREESPACE_MESSAGES
  LOGGING_DEBUG(OctreeFreespaceLog, "cudaMemset : " << timeDiff(time, getCPUTime() << " ms" << endl);
#endif

  // ##### ray cast #####
// bit vector for ray casting is slow compared to byte array due to the need for an atomic operation
// using morton code for 8 neighbors makes it even slower -> there might be more memory conflicts due to higher memory locality
  thrust::device_vector<uint32_t> d_voxel_count(numBlocks * numThreadsPerBlock);
  HANDLE_CUDA_ERROR(cudaDeviceSynchronize());
#ifdef FREESPACE_MESSAGES
  LOGGING_DEBUG(OctreeFreespaceLog, "sensor origin " << sensor_origin << endl);
#endif

  PERF_MON_PRINT_AND_RESET_INFO_P(temp_timer, "RayCastPreparations", prefix);

  time = getCPUTime();
  kernel_rayInsert<branching_factor, InnerNode> <<<numBlocks, numThreadsPerBlock>>>(sensor_origin,
                                                                                    D_PTR(d_voxel_count),map_properties);
  CHECK_CUDA_ERROR();
  HANDLE_CUDA_ERROR(cudaDeviceSynchronize());

  PERF_MON_PRINT_AND_RESET_INFO_P(temp_timer, "RayCast", prefix);

#if defined FREESPACE_MESSAGES && defined _IC_DEBUG_
  uint32_t set_to_free = thrust::reduce(d_voxel_count.begin(), d_voxel_count.end());
#endif
  HANDLE_CUDA_ERROR(cudaDeviceSynchronize());
  d_voxel_count.clear();
  d_voxel_count.shrink_to_fit();
  uint32_t num_free_voxel = thrust::count_if(d_ptr, d_ptr + map_properties.size, Comp_is_valid<InnerNode>());

  PERF_MON_ADD_DATA_NONTIME_P("NumFreeVoxel", num_free_voxel, prefix);

#ifdef FREESPACE_MESSAGES
  LOGGING_DEBUG(OctreeFreespaceLog, "set_to_free: " << set_to_free << endl);
  LOGGING_DEBUG(OctreeFreespaceLog, "num_free_voxel: " << num_free_voxel << endl);
#endif

  packVoxel_Map(map_properties, h_packed_levels, num_free_voxel, min_level);
}

template<std::size_t branching_factor, std::size_t level_count, typename InnerNode, typename LeafNode>
template<bool SET_UPDATE_FLAG, typename BasicData, typename Iterator1, typename Iterator2>
void NTree<branching_factor, level_count, InnerNode, LeafNode>::insertVoxel(OctreeVoxelID* d_voxel_vector,
                                                                            Iterator1 d_set_basic_data,
                                                                            Iterator2 d_reset_basic_data,
                                                                            voxel_count num_voxel,
                                                                            uint32_t target_level)
{
  if (num_voxel == 0)
    return;
//
//  assert((SET_STATUS && d_status_vector != NULL) || (SET_OCCUPANCY && d_occupancy_vector != NULL));
//  assert((!SET_STATUS && d_status_vector == NULL) || (!SET_OCCUPANCY && d_occupancy_vector == NULL));
  assert(checkSorting(d_voxel_vector, num_voxel));

  timespec time = getCPUTime();
  thrust::device_vector<voxel_count> d_neededNodesPerLevel((numBlocks + 1) * level_count, 0);
  thrust::device_vector<void*> d_traversalNodes(num_voxel);
  thrust::device_vector<uint32_t> d_traversalLevels(num_voxel);

// count number of needed inner and leaf nodes
  kernel_insert_countNeededNodes<branching_factor, level_count, InnerNode, LeafNode, SET_UPDATE_FLAG> <<<
      numBlocks, NUM_THREADS_PER_BLOCK>>>(this->m_root, d_voxel_vector, num_voxel,
                                          D_PTR(d_neededNodesPerLevel),
                                          D_PTR(d_traversalNodes),
                                          D_PTR(d_traversalLevels),
                                          target_level
                                          );
  CHECK_CUDA_ERROR();
  HANDLE_CUDA_ERROR(cudaDeviceSynchronize());

#ifdef INSERT_MESSAGES
  LOGGING_DEBUG(OctreeInsertLog, "kernel_insert_countNeededNodes(): " <<  timeDiff(time, getCPUTime()) << " ms" << endl);
#endif
  time = getCPUTime();

// prefix sum
  thrust::exclusive_scan(d_neededNodesPerLevel.begin(), d_neededNodesPerLevel.end(),
                         d_neededNodesPerLevel.begin());
  HANDLE_CUDA_ERROR(cudaDeviceSynchronize()); // sync just like for plain kernel calls
//  for (uint32_t i = 0; i < d_neededNodesPerLevel.size(); ++i)
//    printf("d_neededNodesPerLevel %u %u\n", i % numBlocks, (uint32_t) d_neededNodesPerLevel[i]);

#ifdef INSERT_MESSAGES
  LOGGING_DEBUG(OctreeInsertLog, "thrust::exclusive_scan(): " <<  timeDiff(time, getCPUTime()) << " ms" << endl);
#endif
  time = getCPUTime();

  thrust::device_vector<voxel_count> neededNodesPerLevel_h = d_neededNodesPerLevel;
  HANDLE_CUDA_ERROR(cudaDeviceSynchronize()); // sync just like for plain kernel calls
  const voxel_count nLeafNodes = neededNodesPerLevel_h[numBlocks];
  const voxel_count nInnerNodes = neededNodesPerLevel_h.back() - nLeafNodes;
#ifdef INSERT_MESSAGES
  LOGGING_DEBUG(OctreeInsertLog, "new leaf nodes: " << nLeafNodes << endl << "new inner nodes: " <<  nInnerNodes << endl);
#endif

  void* d_newNodes = NULL;
  const size_t leafLevel_size = size_t(nLeafNodes) * sizeof(LeafNode);
  const uint32_t off = (leafLevel_size % 128);
  const uint32_t alignment = (off == 0) ? 0 : 128 - off;
  size_t nSize = leafLevel_size + alignment + size_t(nInnerNodes) * sizeof(InnerNode);
  HANDLE_CUDA_ERROR(cudaMalloc(&d_newNodes, nSize));
  m_allocation_list.push_back(d_newNodes);
#ifdef INSERT_MESSAGES
  LOGGING_DEBUG(OctreeInsertLog, "cudaMalloc() for " << nSize * cBYTE2MBYTE << " MB" << endl);
  LOGGING_DEBUG(OctreeInsertLog, "cudaMalloc(): " << timeDiff(time, getCPUTime()) << " ms" << endl);
#endif
  time = getCPUTime();

// init nodes
  LeafNode* leafNodes = (LeafNode*) d_newNodes;
//  printf("leafNodes: %p\n", leafNodes);
  InnerNode* const innerNodes = (InnerNode*) (((char*) d_newNodes) + leafLevel_size + alignment);
  InnerNode* innerNodes_ptr = innerNodes;
  //  printf("innerNodes: %p\n", innerNodes);
  voxel_count numNodes = neededNodesPerLevel_h[2 * numBlocks] - neededNodesPerLevel_h[numBlocks];
//printf("init level 0\n");

//level 0
  kernel_insert_initNeededNodes<branching_factor, level_count, LeafNode, false> <<<numBlocks,
                                                                                   numThreadsPerBlock>>>(
      leafNodes, nLeafNodes);
  CHECK_CUDA_ERROR();
  HANDLE_CUDA_ERROR(cudaDeviceSynchronize()); //TODO: safe to remove synchronization?
  //HANDLE_CUDA_ERROR(cudaDeviceSynchronize());
//printf("init level 1\n");

//level 1
  kernel_insert_initNeededNodes<branching_factor, level_count, InnerNode, true> <<<numBlocks,
                                                                                   numThreadsPerBlock>>>(
      innerNodes_ptr, numNodes);
  CHECK_CUDA_ERROR();
  HANDLE_CUDA_ERROR(cudaDeviceSynchronize()); //TODO: safe to remove synchronization?
  //HANDLE_CUDA_ERROR(cudaDeviceSynchronize());
  innerNodes_ptr += numNodes;

//level 2-n
  for (uint32_t i = 2; i < level_count; ++i)
  {
    //    printf("init level %i\n", i);
    numNodes = neededNodesPerLevel_h[numBlocks * (i + 1)] - neededNodesPerLevel_h[numBlocks * i];
    if (numNodes > 0)
    {
//printf("init level %u numNodes %u\n", i, numNodes);
      kernel_insert_initNeededNodes<branching_factor, level_count, InnerNode, false> <<<numBlocks,
                                                                                        numThreadsPerBlock>>>(
          innerNodes_ptr, numNodes);
      CHECK_CUDA_ERROR();
      HANDLE_CUDA_ERROR(cudaDeviceSynchronize()); //TODO: safe to remove synchronization?
      //HANDLE_CUDA_ERROR(cudaDeviceSynchronize());
    }
    innerNodes_ptr = (InnerNode*) &innerNodes_ptr[numNodes];
  }
  HANDLE_CUDA_ERROR(cudaDeviceSynchronize());

#ifdef INSERT_MESSAGES
  LOGGING_DEBUG(OctreeInsertLog, "kernel_insert_initNeededNodes(): " << timeDiff(time, getCPUTime()) << " ms" << endl);
#endif
  time = getCPUTime();

// set nodes
  kernel_insert_setNodes<branching_factor, level_count, InnerNode, LeafNode, NUM_THREADS_PER_BLOCK, Iterator1,
      Iterator2, BasicData, SET_UPDATE_FLAG> <<<numBlocks, NUM_THREADS_PER_BLOCK>>>(
      this->m_root, d_voxel_vector, d_set_basic_data, d_reset_basic_data, num_voxel,
      D_PTR(d_neededNodesPerLevel),
      leafNodes,
      innerNodes,
      D_PTR(d_traversalNodes),
      D_PTR(d_traversalLevels),
      target_level);
  CHECK_CUDA_ERROR();
  HANDLE_CUDA_ERROR(cudaDeviceSynchronize());

// update counter
  allocLeafNodes += nLeafNodes;
  allocInnerNodes += nInnerNodes;
  m_has_data = true; // indicate that the NTree holds some data

#ifdef INSERT_MESSAGES
  LOGGING_DEBUG(OctreeInsertLog, "kernel_insert_setNodes(): " << timeDiff(time, getCPUTime()) << " ms" << endl);
  LOGGING_DEBUG(OctreeInsertLog, "insert finished!" << endl);
#endif
}

template<std::size_t branching_factor, std::size_t level_count, typename InnerNode, typename LeafNode>
void NTree<branching_factor, level_count, InnerNode, LeafNode>::insertVoxel(
    thrust::device_vector<Voxel>& d_voxel_vector, bool set_free, bool propagate_up)
{
  typedef typename NodeData::BasicData BasicData;

  timespec time = getCPUTime();
  voxel_count num_voxel = d_voxel_vector.size();
  thrust::device_vector<OctreeVoxelID> d_voxel_id(num_voxel);
  thrust::device_vector<Probability> d_occupancy(num_voxel);

  kernel_split_voxel_vector<true, true, false, false> <<<numBlocks, numThreadsPerBlock>>>(
      D_PTR(d_voxel_vector),num_voxel, D_PTR(d_voxel_id),
      D_PTR(d_occupancy), NULL, NULL, NULL, NULL);
  CHECK_CUDA_ERROR();
  HANDLE_CUDA_ERROR(cudaDeviceSynchronize());
#ifdef INSERT_MESSAGES
  LOGGING_DEBUG(OctreeInsertLog, "kernel_split_voxel_vector(): " << timeDiff(time, getCPUTime()) << " ms" << endl);
#endif

// ##### insert voxel ######
#ifdef LOAD_BALANCING_PROPAGATE
  const bool update_Flag = true;
  unused(update_Flag);
#else
  const bool update_Flag = false;
  unused(update_Flag);
#endif
  LOGGING_INFO(OctreeInsertLog, "\ninsert voxel" << endl);
  time = getCPUTime();

  BasicData tmp;
  getHardInsertResetData(tmp);
  thrust::constant_iterator<BasicData> reset_basic_data(tmp);

  if (set_free)
  {
    getFreeData(tmp);
  }
  else
  {
    getOccupiedData(tmp);
  }
  thrust::constant_iterator<BasicData> set_basic_data(tmp);
  insertVoxel<update_Flag, BasicData>(D_PTR(d_voxel_id), set_basic_data, reset_basic_data, num_voxel, 0);

#ifdef INSERT_MESSAGES
  LOGGING_DEBUG(OctreeInsertLog, "insertVoxel(): " << timeDiff(time, getCPUTime()) << " ms" << endl);
#endif

  if (propagate_up)
  {
    time = getCPUTime();
#ifdef LOAD_BALANCING_PROPAGATE
    propagate();
#ifdef INSERT_MESSAGES
    LOGGING_DEBUG(OctreeInsertLog, "propagate load balancing: " << timeDiff(time, getCPUTime()) << " ms" << endl);
#endif
#else
    kernel_propagate_bottom_up_simple<branching_factor, level_count, InnerNode, LeafNode> <<<1, 1>>>(
        this->m_root, D_PTR(d_voxel_id), num_voxel, 0);
    CHECK_CUDA_ERROR();
    HANDLE_CUDA_ERROR(cudaDeviceSynchronize());
#ifdef INSERT_MESSAGES
    LOGGING_DEBUG(OctreeInsertLog, "kernel_propagate_bottom_up_simple(): " << timeDiff(time, getCPUTime()) << " ms" << endl);
#endif
#endif

  }
}

template<std::size_t branching_factor, std::size_t level_count, typename InnerNode, typename LeafNode>
void NTree<branching_factor, level_count, InnerNode, LeafNode>::insertVoxel(
    thrust::device_vector<Voxel>& d_free_space_voxel, thrust::device_vector<Voxel>& d_object_voxel,
    gpu_voxels::Vector3ui sensor_origin, const uint32_t free_space_resolution, const uint32_t object_resolution)
{
  const std::string prefix = __FUNCTION__;
  const std::string temp_timer = prefix + "_temp";
  PERF_MON_START(prefix);
  PERF_MON_START(temp_timer);

  timespec time = getCPUTime();
#if defined(INSERT_MESSAGES) || defined(FEW_MESSAGES)
#ifdef _IC_DEBUG_
  timespec total_time = getCPUTime();
#endif
#endif

  assert((free_space_resolution % m_resolution) == 0);
  assert(
      uint32_t(pow(2, uint32_t(log2(float(free_space_resolution / m_resolution)))))
          == uint32_t(free_space_resolution / m_resolution));
  assert((object_resolution % m_resolution) == 0);
  assert(
      uint32_t(pow(2, uint32_t(log2(float(object_resolution / m_resolution)))))
          == uint32_t(object_resolution / m_resolution));

//  // check for duplicates
//  for (uint32_t i = 0; i < d_voxelVector.size(); ++i)
//  {
//    assert(i == 0 || ((Voxel )d_voxelVector[i - 1]).voxelId < ((Voxel )d_voxelVector[i]).voxelId);
//    for (uint32_t j = 0; j < d_voxelVector.size(); ++j)
//      assert((i == j) || (((Voxel )d_voxelVector[i]).voxelId != ((Voxel )d_voxelVector[j]).voxelId));
//  }

#ifdef INSERT_MESSAGES
  LOGGING_DEBUG(OctreeInsertLog, "[Check for duplicates: " << timeDiff(time, getCPUTime()) << " ms]" << endl);
#endif
  time = getCPUTime();

  voxel_count num_voxel_object = d_object_voxel.size();
#ifdef INSERT_MESSAGES
  LOGGING_DEBUG(OctreeInsertLog, "num_voxel_object_voxel: " << num_voxel_object << endl);
  LOGGING_DEBUG(OctreeInsertLog, "num_free_space_voxel: " << d_free_space_voxel.size() << endl);
#endif

// ##### compute free space #####
#ifdef INSERT_MESSAGES
  LOGGING_DEBUG(OctreeInsertLog, "\n## computeFreeSpaceViaRayCast ###" << endl);
#endif
  thrust::host_vector<ComputeFreeSpaceData> h_packed_levels(level_count, ComputeFreeSpaceData(NULL, NULL, 0));
//computeFreeSpaceViaRayCast_VoxelList(d_voxel_vector, sensor_origin, h_packed_levels);
  const uint32_t free_space_scale = free_space_resolution / m_resolution;
  gpu_voxels::Vector3ui sensor_origin_scaled = gpu_voxels::Vector3ui(sensor_origin.x / free_space_scale,
                                                       sensor_origin.y / free_space_scale,
                                                       sensor_origin.z / free_space_scale);
  PERF_MON_START(temp_timer);

  if (d_free_space_voxel.size() > 0)
    computeFreeSpaceViaRayCast(d_free_space_voxel, sensor_origin_scaled, h_packed_levels);

  PERF_MON_PRINT_AND_RESET_INFO_P(temp_timer, "FreeSpaceComputation", prefix);

#ifdef INSERT_MESSAGES
  LOGGING_DEBUG(OctreeInsertLog, "## computeFreeSpaceViaRayCast(): " << timeDiff(time, getCPUTime()) << " ms ##" << endl);
#endif
  time = getCPUTime();

//  thrust::host_vector<thrust::pair<voxel_id*, voxel_count> > h_packed_levels2(
//      level_count, thrust::make_pair<voxel_id*, voxel_count>(NULL, 0));
//  computeFreeSpaceViaRayCast(d_voxel_vector, sensor_origin, h_packed_levels2);

//  kernel_checkBlub<<<1,1>>>(h_packed_levels[0].first, h_packed_levels2[0].second, h_packed_levels2[0].first);
//  CHECK_CUDA_ERROR();
//  HANDLE_CUDA_ERROR(cudaDeviceSynchronize());

#ifdef LOAD_BALANCING_PROPAGATE
  const bool update_Flag = true;
  unused(update_Flag);
#else
  const bool update_Flag = false;
  unused(update_Flag);
#endif

// ##### insert free space #####
#ifdef INSERT_MESSAGES
  LOGGING_DEBUG(OctreeInsertLog, "\n ## insert free space ##" << endl);
#endif
  time = getCPUTime();
  const uint32_t free_space_min_level = uint32_t(log2(float(free_space_scale)));
  //thrust::device_vector<uint8_t> d_status(1, uint8_t(ns_FREE));
  uint32_t free_space_voxel = 0;
  for (std::size_t l = 0; l < level_count; ++l)
  {
    //printf("\ninsertVoxel level: %i #voxel: %u\n", l, h_packed_levels[l].second);
    if (h_packed_levels[l].m_count != 0 && free_space_scale != 1)
    {
      // scale voxel data if necessary
      thrust::device_ptr<OctreeVoxelID> ptr(h_packed_levels[l].m_voxel_id);
      thrust::transform(ptr, ptr + h_packed_levels[l].m_count, ptr, Trafo_OctreeVoxelID(free_space_scale));
      HANDLE_CUDA_ERROR(cudaDeviceSynchronize());
    }

    const uint32_t my_level = free_space_min_level + l;

//    if (my_level == 1)
//    {
    BasicData tmp;
    getSoftInsertResetData(tmp);
    //getOccupancyResetData(tmp);
    thrust::constant_iterator<BasicData> reset_data(tmp);
//    getFreeData(tmp);
//    thrust::constant_iterator<BasicData> set_data(tmp);
//    insertVoxel<update_Flag, BasicData>(h_packed_levels[l].first, set_data, reset_data,
//                                        h_packed_levels[l].second, my_level);
    insertVoxel<update_Flag, BasicData>(
        h_packed_levels[l].m_voxel_id,
        h_packed_levels[l].m_basic_data,
        reset_data,
        h_packed_levels[l].m_count,
        my_level);

    free_space_voxel += h_packed_levels[l].m_count;

//    }

//    if (my_level == 0)
//    {
//      thrust::device_vector<Probability> d_occupancy(1, MIN_PROBABILITY);
//#ifdef INSERT
//      insertVoxel<update_Flag>(h_packed_levels[l].first, d_occupancy, NULL, h_packed_levels[l].second,
//                               my_level);
//#endif
//    }
//    else
//    {
//#ifdef INSERT
//
//      insertVoxel<false, false, true, update_Flag>(h_packed_levels[l].first, NULL, D_PTR(d_status),h_packed_levels[l].second, my_level);
//#endif
//    }
  }

  PERF_MON_PRINT_AND_RESET_INFO_P(temp_timer, "InsertFreeSpaceVoxel", prefix);

#ifdef INSERT_MESSAGES
  LOGGING_DEBUG(OctreeInsertLog, "## insertVoxel(free space): " << timeDiff(time, getCPUTime()) << " ms ##" << endl);
#endif

// ##### insert occupied voxel ######
  thrust::device_vector<OctreeVoxelID> d_voxel_id_object(num_voxel_object);
  thrust::device_vector<Probability> d_occupancy_object(num_voxel_object);

// split object data
  kernel_split_voxel_vector<true, true, false, false> <<<numBlocks, numThreadsPerBlock>>>(
      D_PTR(d_object_voxel), num_voxel_object, D_PTR(d_voxel_id_object),
      D_PTR(d_occupancy_object), NULL, NULL, NULL, NULL);
  CHECK_CUDA_ERROR();
  HANDLE_CUDA_ERROR(cudaDeviceSynchronize());

// scale voxel data if necessary
  const uint32_t object_scale = object_resolution / m_resolution;
  if (object_scale != 1)
  {
    thrust::transform(d_voxel_id_object.begin(), d_voxel_id_object.end(), d_voxel_id_object.begin(),
                      Trafo_OctreeVoxelID(object_scale));
    HANDLE_CUDA_ERROR(cudaDeviceSynchronize());
  }
#ifdef INSERT_MESSAGES
  LOGGING_DEBUG(OctreeInsertLog, "kernel_split_voxel_vector(): " << timeDiff(time, getCPUTime()) << " ms" << endl);
#endif
#ifdef INSERT_MESSAGES
  LOGGING_DEBUG(OctreeInsertLog, "\n## insert occupied voxel ###" << endl);
#endif
  time = getCPUTime();

  BasicData tmp;
  //getOccupancyResetData(tmp);
  getHardInsertResetData(tmp);
  thrust::constant_iterator<BasicData> reset_data(tmp);

  //  getOccupiedData(tmp);
  //  thrust::constant_iterator<BasicData> set_data(tmp);
  //insertVoxel<update_Flag, BasicData>(D_PTR(d_voxel_id_object), set_data, reset_data, num_voxel_object, uint32_t(log2(float(object_scale)))) ;

  thrust::device_vector<BasicData> set_data;
  getBasicData(d_object_voxel, set_data);
  insertVoxel<update_Flag, BasicData>(
      D_PTR(d_voxel_id_object),
      D_PTR(set_data),
      reset_data,
      num_voxel_object,
      uint32_t(log2(float(object_scale))));

  PERF_MON_ADD_DATA_NONTIME_P("NewVoxel", free_space_voxel + num_voxel_object, prefix);
  PERF_MON_PRINT_AND_RESET_INFO_P(temp_timer, "InsertObjectVoxel", prefix);

#ifdef INSERT_MESSAGES
  LOGGING_DEBUG(OctreeInsertLog, "## insertVoxel(occupied): " << timeDiff(time, getCPUTime()) << " ms ##" << endl);
#endif
  time = getCPUTime();

//  // simple propagate bottom-up SINGLE THREAD
//  printf("\n ## propagate bottom-up ##\n");
//  time = getCPUTime();
//  kernel_propagate_bottom_up_simple<branching_factor, level_count, InnerNode, LeafNode> <<<1,1>>>(this->root, D_PTR(d_voxel_id), num_voxel, 0);
//  CHECK_CUDA_ERROR();
//  HANDLE_CUDA_ERROR(cudaDeviceSynchronize());
//  for (int32_t l = 0; l < level_count - 1; ++l)
//  {
//    if (h_packed_levels[l].second > 0)
//    {
//      kernel_propagate_bottom_up_simple<branching_factor, level_count, InnerNode, LeafNode> <<<1,1>>>(this->root, h_packed_levels[l].first, h_packed_levels[l].second, l);
//      CHECK_CUDA_ERROR();
//      HANDLE_CUDA_ERROR(cudaDeviceSynchronize());
//    }
//  }
//  printf("## kernel_propagate_bottom_up_simple(): %f ms ##\n", timeDiff(time, getCPUTime()));

#ifdef LOAD_BALANCING_PROPAGATE
#ifdef INSERT_MESSAGES
  LOGGING_DEBUG(OctreeInsertLog,"\n ## load balancing propagate ##" << endl);
#endif

  cudaProfilerStart();
  propagate(free_space_voxel + num_voxel_object);
  cudaProfilerStop();

  PERF_MON_PRINT_INFO_P(temp_timer, "Propagate", prefix);

#ifdef INSERT_MESSAGES
  LOGGING_DEBUG(OctreeInsertLog, "## load balancing propagate: " << timeDiff(time, getCPUTime()) << " ms ##" << endl);
#endif
#else

#ifdef INSERT_MESSAGES
  LOGGING_DEBUG(OctreeInsertLog,"\n ## propagate bottom-up ##" << endl);
#endif
  time = getCPUTime();
  propagate_bottom_up(D_PTR(d_voxel_id_object), num_voxel_object, 0);
  for (int32_t l = 0; l < int32_t(level_count) - 1; ++l)
  {
    if (h_packed_levels[l].second > 0)
    {
      propagate_bottom_up(h_packed_levels[l].first, h_packed_levels[l].second, l);
    }
  }
#ifdef INSERT_MESSAGES
  LOGGING_DEBUG(OctreeInsertLog, "## kernel_propagate_bottom(): " << timeDiff(time, getCPUTime()) << " ms ##" << endl);
#endif
#endif

// free memory
  for (int32_t l = 0; l < int32_t(level_count) - 1; ++l)
  {
    if (h_packed_levels[l].m_count > 0)
    {
      HANDLE_CUDA_ERROR(cudaFree(h_packed_levels[l].m_voxel_id));
      HANDLE_CUDA_ERROR(cudaFree(h_packed_levels[l].m_basic_data));
    }
  }

  PERF_MON_PRINT_INFO_P(temp_timer, "FreeMem", prefix);

  PERF_MON_ADD_DATA_NONTIME_P("UsedMemOctree", getMemUsage(), prefix);

#if defined(INSERT_MESSAGES) || defined(FEW_MESSAGES)
  LOGGING_DEBUG(OctreeInsertLog, "### insertVoxel(total): " << timeDiff(total_time, getCPUTime()) << " ms ###" << endl);
#endif
}

template<std::size_t branching_factor, std::size_t level_count, typename InnerNode, typename LeafNode>
void NTree<branching_factor, level_count, InnerNode, LeafNode>::propagate_bottom_up(
    thrust::device_vector<Voxel>& d_voxel_vector, uint32_t level)
{
#ifdef _IC_DEBUG_
  timespec time = getCPUTime();
#endif
  voxel_count num_voxel = d_voxel_vector.size();
  thrust::device_vector<OctreeVoxelID> d_voxel_id(num_voxel);

  kernel_split_voxel_vector<true, false, false, false> <<<numBlocks, numThreadsPerBlock>>>(
      D_PTR(d_voxel_vector),num_voxel, D_PTR(d_voxel_id),
      NULL, NULL,NULL,NULL,NULL);
  CHECK_CUDA_ERROR();
  HANDLE_CUDA_ERROR(cudaDeviceSynchronize());
  LOGGING_DEBUG(OctreeLog, "kernel_split_voxel_vector(): " << timeDiff(time, getCPUTime()) << " ms" << endl);

  propagate_bottom_up(D_PTR(d_voxel_id), num_voxel, level);
}

// Has the bug of setting already free voxel to unknown due to a missing top-down propagate step,
// which sets the status of the new nodes to it's parent node's status
// TODO: does not call the kernel. investigate!
template<std::size_t branching_factor, std::size_t level_count, typename InnerNode, typename LeafNode>
void NTree<branching_factor, level_count, InnerNode, LeafNode>::propagate_bottom_up(OctreeVoxelID* d_voxel_id,
                                                                                    voxel_count num_voxel,
                                                                                    uint32_t level)
{
  if (num_voxel == 0)
    return;

#ifdef PROPAGATE_MESSAGES
  LOGGING_DEBUG(OctreePropagateLog, "## propagate_bottom_up ##" << endl);
  LOGGING_DEBUG(OctreePropagateLog,"num_voxel " << num_voxel << " level " << level << endl);
#endif

  // propagate bottom up level by level
#if defined PROPAGATE_MESSAGES && defined _IC_DEBUG_
  timespec time = getCPUTime();
#endif
  for (uint32_t l = level; l < level_count - 1; ++l)
  {
#if defined PROPAGATE_MESSAGES && defined _IC_DEBUG_
    timespec time_loop = getCPUTime();
#endif
//    kernel_propagate_bottom_up<branching_factor, level_count, InnerNode, LeafNode> <<<numBlocks,
//                                                                                      32>>>(
//        m_root, d_voxel_id, num_voxel, l);
//  CHECK_CUDA_ERROR();
    HANDLE_CUDA_ERROR(cudaDeviceSynchronize());
#ifdef PROPAGATE_MESSAGES
    LOGGING_DEBUG(OctreePropagateLog, "kernel_propagate_bottom_up(level = " << l << "): " << timeDiff(time_loop, getCPUTime()) << " ms" << endl);
#endif
  }
#ifdef PROPAGATE_MESSAGES
  LOGGING_DEBUG(OctreePropagateLog, "## kernel_propagate_bottom_up(total): " << timeDiff(time, getCPUTime()) << " ms ##" << endl);
#endif
}

template<std::size_t branching_factor, std::size_t level_count, typename InnerNode, typename LeafNode>
bool NTree<branching_factor, level_count, InnerNode, LeafNode>::checkTree()
{
  LOGGING_INFO(OctreeLog, "checking tree . . ." << endl);
//thrust::device_vector<uint8_t> error(1);
  uint8_t* ptr = NULL;
  HANDLE_CUDA_ERROR(cudaMalloc(&ptr, 128));
  kernel_checkTree<branching_factor, level_count, InnerNode, LeafNode> <<<1, 1>>>(m_root, ptr);
  CHECK_CUDA_ERROR();
  HANDLE_CUDA_ERROR(cudaDeviceSynchronize());

//bool e = (uint8_t) error[0];
  uint8_t h_e;
  HANDLE_CUDA_ERROR(cudaMemcpy(&h_e, ptr, 1, cudaMemcpyDeviceToHost));
  bool e = h_e;
  if (!e)
    LOGGING_INFO(OctreeLog, "checkTree() OK" << endl);
  else
    LOGGING_ERROR(OctreeLog, "##### ERROR checkTree() FAILED #####" << endl);
  HANDLE_CUDA_ERROR(cudaFree(ptr));
  return e;
}

template<std::size_t branching_factor, std::size_t level_count, typename InnerNode, typename LeafNode>
uint32_t NTree<branching_factor, level_count, InnerNode, LeafNode>::extractCubes(
    std::vector<Vector3f> &points, uint8_t* d_status_selection, uint32_t min_level)
{
  //TODO: Make this more performant! Use thrust transform...
  thrust::device_vector<Cube> *d_tmp = NULL;
  thrust::host_vector<Cube> h_tmp;

  extractCubes(d_tmp, d_status_selection, min_level);

  h_tmp = *d_tmp;

  //points.resize(h_tmp.size());



  for(size_t i = 0; i < h_tmp.size(); ++i)
  {
    if(h_tmp[i].m_type_vector.getBit(eBVM_OCCUPIED))
    {
      if(h_tmp[i].m_side_length == 1)
      {
        points.push_back(Vector3f(h_tmp[i].m_position.x + m_resolution / 2.0,
                                  h_tmp[i].m_position.y + m_resolution / 2.0,
                                  h_tmp[i].m_position.z + m_resolution / 2.0));
      }else{
        LOGGING_WARNING(OctreeLog, "Ignoring extracted voxel larger than 1" << endl);
      }
    }
  }
  return points.size();
}

// static int num_extract_call = -1;

template<std::size_t branching_factor, std::size_t level_count, typename InnerNode, typename LeafNode>
uint32_t NTree<branching_factor, level_count, InnerNode, LeafNode>::extractCubes(
    thrust::device_vector<Cube> *&d_cubes, uint8_t* d_status_selection, uint32_t min_level)
{
//  std::size_t size = allocInnerNodes + allocLeafNodes;
//  d_cubes->resize(size);

  if(!d_cubes) d_cubes = new thrust::device_vector<Cube>(1);

  if (d_status_selection == NULL)
    d_status_selection = m_extract_status_selection;

#if defined   _IC_DEBUG_ && (defined COUNT_BEFORE_EXTRACT || defined EXTRACTCUBE_MESSAGES)
 timespec time = getCPUTime();
#endif
  uint32_t needed_size = m_extract_buffer_size;
#ifdef COUNT_BEFORE_EXTRACT
  {
  typedef LoadBalancer::Extract<
        branching_factor,
        level_count,
        InnerNode,
        LeafNode,
        false,
        true> MyLoadBalancer;
  MyLoadBalancer load_balancer(
        this,
        NULL,
        0,
        d_status_selection,
        min_level);
  load_balancer.run();
  needed_size = load_balancer.m_num_elements;
  }
  LOGGING_DEBUG(OctreeCountBeforeExtractLog, "count: " << timeDiff(time, getCPUTime()) << " ms" << endl);
#endif
  thrust::device_vector<NodeData> d_node_data(needed_size);
  typedef LoadBalancer::Extract<
        branching_factor,
        level_count,
        InnerNode,
        LeafNode,
        true,
        false> MyLoadBalancer;
  MyLoadBalancer load_balancer(
        this,
        D_PTR(d_node_data),
        needed_size,
        d_status_selection,
        min_level);
  load_balancer.run();
  uint32_t used_size = load_balancer.m_num_elements;

  //LOGGING_INFO(OctreeLog, "needed size " << used_size << endl);
  if (used_size > d_node_data.size())
  {
    // increase buffer and try another time
    d_node_data.clear();
    d_node_data.shrink_to_fit();
    m_extract_buffer_size = used_size*5/4;
    d_node_data.resize(used_size);
    MyLoadBalancer load_balancer(
          this,
          D_PTR(d_node_data),
          used_size,
          d_status_selection,
          min_level);
    load_balancer.run();
    uint32_t used_size = load_balancer.m_num_elements;

    if (used_size > d_node_data.size())
    {
      LOGGING_ERROR(OctreeLog, "ERROR in extractCubes(). d_node_data is too small!" << endl);
      //exit(0); No need to exit. We will only miss the visualization of some nodes.
    }
  }
  //LOGGING_INFO(OctreeLog, "used min level: " << min_level << endl);

  d_cubes->resize(used_size);

  uint8_t* mapping = m_status_mapping;
//  if (num_extract_call == 1)
//  {
//    const uint32_t mapping_size = 256;
//    uint8_t* m_status_mapping = NULL;
//    HANDLE_CUDA_ERROR(cudaMalloc((void**) &m_status_mapping, mapping_size * sizeof(uint8_t)));
//
//// create default status to VoxelType mapping
//    uint8_t mapping[mapping_size];
//    memset(&mapping, 0, mapping_size * sizeof(uint8_t));
//    mapping[ns_FREE] = gpu_voxels::eBVM_SWEPT_VOLUME_START;
//    mapping[ns_FREE | ns_UNKNOWN] = gpu_voxels::eBVM_SWEPT_VOLUME_START;
//    mapping[ns_UNKNOWN] = gpu_voxels::eBVM_UNDEFINED;
//    mapping[ns_OCCUPIED] = gpu_voxels::eBVM_OCCUPIED;
//    mapping[ns_OCCUPIED | ns_FREE] = gpu_voxels::eBVM_OCCUPIED;
//    mapping[ns_OCCUPIED | ns_FREE | ns_UNKNOWN] = gpu_voxels::eBVM_OCCUPIED;
//    mapping[ns_OCCUPIED | ns_UNKNOWN] = gpu_voxels::eBVM_OCCUPIED;
//
//    HANDLE_CUDA_ERROR(
//        cudaMemcpy((void*) m_status_mapping, (void*) &mapping, mapping_size * sizeof(uint8_t),
//                   cudaMemcpyHostToDevice));
//    num_extract_call = -1;
//  }
  thrust::transform(d_node_data.begin(), d_node_data.begin() + used_size, d_cubes->begin(),
                    Trafo_NodeData_to_Cube(mapping));
  HANDLE_CUDA_ERROR(cudaDeviceSynchronize());

//++num_extract_call;

  if (4*used_size < m_extract_buffer_size) // decrease buffer
    m_extract_buffer_size = max(m_extract_buffer_size/2, INITIAL_EXTRACT_BUFFER_SIZE);

#ifdef EXTRACTCUBE_MESSAGES
  LOGGING_INFO(OctreeExtractCubeLog, "cubes buffer size " << d_cubes->size() << endl);
  LOGGING_INFO(OctreeExtractCubeLog, "used_size " << used_size << endl);
  LOGGING_INFO(OctreeExtractCubeLog, "extractCubes total: " << timeDiff(time, getCPUTime()) << " ms" << endl);
#endif
  return used_size;
}

template<std::size_t branching_factor, std::size_t level_count, typename InnerNode, typename LeafNode>
void NTree<branching_factor, level_count, InnerNode,
    LeafNode>::internal_rebuild(thrust::device_vector<NodeData>& d_node_data, const uint32_t num_cubes)
{

// This throws a -Wunsused-variable at compile time, since this is only used as template parameter.
#ifdef LOAD_BALANCING_PROPAGATE
  const bool update_Flag = true;
  unused(update_Flag);
#else
  const bool update_Flag = false;
  unused(update_Flag);
#endif

  const std::string prefix = "rebuild";
  const std::string temp_timer = prefix + "_temp";
  PERF_MON_START(temp_timer);

  ++m_rebuild_counter;
  PERF_MON_ADD_STATIC_DATA_P("RebuildCount", m_rebuild_counter, prefix);

#if defined(REBUILD_MESSAGES) || defined(FEW_MESSAGES)
  LOGGING_INFO(OctreeRebuildLog, "\n\n\n ##### rebuild() #####" << endl);
  LOGGING_INFO(OctreeRebuildLog, "alloc inner " << allocInnerNodes  << " alloc leaf " << allocLeafNodes << endl);
#endif
  std::cout << gpu_voxels::getDeviceMemoryInfo();

#if defined REBUILD_MESSAGES && defined _IC_DEBUG_
  timespec total_time = getCPUTime();
  timespec time = getCPUTime();
#endif
  thrust::host_vector<voxel_count> num_per_level(level_count);
  thrust::device_vector<OctreeVoxelID> h_voxel_lists[level_count];
//thrust::device_vector<thrust::pair<NodeStatus, Probability> > d_last_level;
  thrust::device_vector<BasicData>h_basic_data[level_count];

  {
#if defined REBUILD_MESSAGES && defined _IC_DEBUG_
    time = getCPUTime();
#endif
    thrust::device_vector<NodeData> d_node_data_tmp(num_cubes);

    // compute list of voxel_ids for each tree level
    for (uint32_t l = 0; l < level_count - 1; ++l)
    {
      voxel_count num_items;
      num_per_level[l] = num_items = voxel_count(
          thrust::copy_if(d_node_data.begin(), d_node_data.begin() + num_cubes, d_node_data_tmp.begin(),
                          Comp_has_level(l)) - d_node_data_tmp.begin());
#ifdef REBUILD_MESSAGES
      LOGGING_DEBUG(OctreeRebuildLog, "level " << l << " num_items: " << num_items << endl);
#endif

      // transform to OctreeVoxelID
      h_voxel_lists[l]=thrust::device_vector<OctreeVoxelID>(num_items);
      thrust::transform(d_node_data_tmp.begin(), d_node_data_tmp.begin() + num_items,
                        h_voxel_lists[l].begin(), Trafo_NodeData_to_OctreeVoxelID());

//      if (l == 0)
//      {
//        // transform to Pair
//        d_last_level.resize(num_items);
//        thrust::transform(d_node_data_tmp.begin(), d_node_data_tmp.begin() + num_items, d_last_level.begin(),
//                          Trafo_NodeData_to_Pair());
//        thrust::sort_by_key(h_voxel_lists[l].begin(), h_voxel_lists[l].end(), d_last_level.begin());
//      }
//      else
//      {
      // transform to NodeStatus
      h_basic_data[l]=thrust::device_vector<BasicData>(num_items);
      thrust::transform(d_node_data_tmp.begin(), d_node_data_tmp.begin() + num_items, h_basic_data[l].begin(),
                        Trafo_to_BasicData());
      thrust::sort_by_key(h_voxel_lists[l].begin(), h_voxel_lists[l].end(), h_basic_data[l].begin());
      //}
      //thrust::sort(h_voxel_lists[l].begin(), h_voxel_lists[l].end());
    }
    HANDLE_CUDA_ERROR(cudaDeviceSynchronize());
    std::cout << gpu_voxels::getDeviceMemoryInfo();
  }

  PERF_MON_PRINT_AND_RESET_INFO_P(temp_timer, "ProcessData", prefix);

#ifdef REBUILD_MESSAGES
  LOGGING_DEBUG(OctreeRebuildLog, "preprocess voxelList(): " <<  timeDiff(time, getCPUTime()) << " ms" << endl);
#ifdef _IC_DEBUG_
  time = getCPUTime();
#endif
#endif

  clear();

// insert InnerNodes
  for (std::size_t l = 0; l < level_count - 1; ++l)
  {
    BasicData tmp;
    getRebuildResetData(tmp);
    thrust::constant_iterator<BasicData> reset_data(tmp);

    insertVoxel<update_Flag, BasicData>(
        D_PTR(h_voxel_lists[l]),
        D_PTR(h_basic_data[l]),
        reset_data,
        voxel_count(h_voxel_lists[l].size()),
        l);

#ifndef LOAD_BALANCING_PROPAGATE
    propagate_bottom_up(D_PTR(h_voxel_lists[l]), voxel_count(h_voxel_lists[l].size()), l);
#endif
  }

  PERF_MON_PRINT_AND_RESET_INFO_P(temp_timer, "InsertVoxel", prefix);

//// insert LeafNodes
//  thrust::device_vector<Probability> d_ll_probability(d_last_level.size());
//  thrust::device_vector<NodeStatus> d_ll_status(d_last_level.size());
//  thrust::transform(d_last_level.begin(), d_last_level.end(), d_ll_probability.begin(),
//                    Trafo_Pair_to_Probability());
//  thrust::transform(d_last_level.begin(), d_last_level.end(), d_ll_status.begin(), Trafo_Pair_to_Status());
//  HANDLE_CUDA_ERROR(cudaDeviceSynchronize());
//
//  // insertVoxel<true, true, true, update_Flag>(D_PTR(h_voxel_lists[0]), D_PTR(d_ll_probability), D_PTR(d_ll_status), voxel_count(h_voxel_lists[0].size()), 0);
//
//  thrust::constant_iterator<NodeStatus> reset_status(STATUS_OCCUPANCY_MASK | ns_DYNAMIC_MAP | ns_STATIC_MAP);
//  insertVoxel<update_Flag>(D_PTR(h_voxel_lists[0]), D_PTR(d_ll_probability), reset_status, D_PTR(d_ll_status),voxel_count(h_voxel_lists[0].size()), 0);

#ifndef LOAD_BALANCING_PROPAGATE
  propagate_bottom_up(D_PTR(h_voxel_lists[0]), voxel_count(h_voxel_lists[0].size()), 0);
#else
  propagate();
#endif

  PERF_MON_PRINT_AND_RESET_INFO_P(temp_timer, "Propagate", prefix);
  PERF_MON_PRINT_INFO_P(prefix, "", prefix);

#ifdef REBUILD_MESSAGES
  LOGGING_DEBUG(OctreeRebuildLog, "insertVoxel(): " <<  timeDiff(time, getCPUTime()) << " ms" << endl);
  LOGGING_DEBUG(OctreeRebuildLog, "allocLeafNodes: " << allocLeafNodes << " allocInnerNodes: " << allocInnerNodes << endl);
#endif

  std::cout << gpu_voxels::getDeviceMemoryInfo();

#if defined(REBUILD_MESSAGES) || defined(FEW_MESSAGES)
  LOGGING_DEBUG(OctreeRebuildLog, "#### rebuild(): " <<  timeDiff(total_time, getCPUTime()) << " ms ####\n\n " << endl);
#endif
}

template<std::size_t branching_factor, std::size_t level_count, typename InnerNode, typename LeafNode>
void NTree<branching_factor, level_count, InnerNode,
    LeafNode>::rebuild()
{
  const std::string prefix = __FUNCTION__;
  const std::string temp_timer = prefix + "_temp";
  PERF_MON_START(prefix);
  PERF_MON_START(temp_timer);

  ++m_rebuild_counter;
  PERF_MON_ADD_STATIC_DATA_P("RebuildCount", m_rebuild_counter, prefix);

#if defined(REBUILD_MESSAGES) || defined(FEW_MESSAGES)
  LOGGING_INFO(OctreeRebuildLog, "\n\n\n ##### rebuild() #####\n" << endl);
  LOGGING_INFO(OctreeRebuildLog, "alloc inner " << allocInnerNodes << " alloc leaf " << allocLeafNodes << endl);
#endif
  std::cout << gpu_voxels::getDeviceMemoryInfo();

  timespec time = getCPUTime();

  uint32_t needed_size = m_rebuild_buffer_size;
#ifdef COUNT_BEFORE_EXTRACT
  {
  typedef LoadBalancer::Extract<
         branching_factor,
         level_count,
         InnerNode,
         LeafNode,
         false,
         true> MyLoadBalancer;
  MyLoadBalancer load_balancer(
         this,
         NULL,
         0,
         m_extract_status_selection,
         0);
  load_balancer.run();
  needed_size = load_balancer.m_num_elements;
  }

#endif

  PERF_MON_PRINT_AND_RESET_INFO_P(temp_timer, "ExtractCount", prefix);

  thrust::device_vector<NodeData> d_node_data(needed_size);

  PERF_MON_PRINT_AND_RESET_INFO_P(temp_timer, "Malloc", prefix);

  typedef LoadBalancer::Extract<
         branching_factor,
         level_count,
         InnerNode,
         LeafNode,
         false,
         false> MyLoadBalancer;
  MyLoadBalancer load_balancer(
         this,
         D_PTR(d_node_data),
         needed_size,
         m_extract_status_selection,
         0);
  load_balancer.run();
  uint32_t num_cubes = load_balancer.m_num_elements;

  PERF_MON_PRINT_AND_RESET_INFO_P(temp_timer, "Extract", prefix);

#ifdef REBUILD_MESSAGES
  LOGGING_INFO(OctreeRebuildLog, "num_cubes " << num_cubes << endl);
#endif

  if (num_cubes > d_node_data.size())
   {
     // increase buffer an try another time
     d_node_data.clear();
     d_node_data.shrink_to_fit();
     needed_size = num_cubes;
     m_rebuild_buffer_size = num_cubes * 5/4;
     d_node_data.resize(num_cubes);
     MyLoadBalancer load_balancer(
            this,
            D_PTR(d_node_data),
            needed_size,
            m_extract_status_selection,
            0);
     load_balancer.run();
     num_cubes = load_balancer.m_num_elements;

     if (num_cubes > d_node_data.size())
     {
       LOGGING_ERROR(OctreeRebuildLog, "ERROR in extractCubes(): d_node_data is too small!" << endl);
       exit(0);
     }
   }
  if (4*needed_size < m_rebuild_buffer_size) // decrease buffer
    m_rebuild_buffer_size = max(m_rebuild_buffer_size/2, INITIAL_REBUILD_BUFFER_SIZE);

#ifdef REBUILD_MESSAGES
  LOGGING_INFO(OctreeRebuildLog, "extractTreeData(): " << timeDiff(time, getCPUTime()) << endl);
#endif

  internal_rebuild(d_node_data, num_cubes);
}


template<std::size_t branching_factor, std::size_t level_count, typename InnerNode, typename LeafNode>
std::size_t NTree<branching_factor, level_count, InnerNode, LeafNode>::getMemUsage() const
{
  return allocLeafNodes * sizeof(LeafNode) + allocInnerNodes * sizeof(InnerNode);
}

template<std::size_t branching_factor, std::size_t level_count, typename InnerNode, typename LeafNode>
bool NTree<branching_factor, level_count, InnerNode, LeafNode>::needsRebuild() const
{
  return m_max_memory_usage != 0 && getMemUsage() >= m_max_memory_usage;
}

template<std::size_t branching_factor, std::size_t level_count, typename InnerNode, typename LeafNode>
void NTree<branching_factor, level_count, InnerNode, LeafNode>::propagate(const uint32_t num_changed_nodes)
{
  const std::string prefix = __FUNCTION__;
  const std::string temp_timer = prefix + "_temp";
  PERF_MON_START(prefix);

  uint32_t blocks = DEFAULT_PROPAGATE_QUEUE_NTASKS;

  if(num_changed_nodes != 0)
  {
    float blocks_1 = 1024; // first reference point
    float nodes_1 = 3000000;
    float blocks_2 = 4096; // second reference point
    float nodes_2 = 13000000;
    blocks = linearApprox(blocks_1, nodes_1, blocks_2, nodes_2, num_changed_nodes);

    // more suitable linear approx for inserting small point sets
    // overhead of parallelization bigger than profit
    const uint32_t thresh = 100000;
    if(num_changed_nodes < thresh)
    {
      // get connection point
      blocks_2 =  linearApprox(blocks_1, nodes_1, blocks_2, nodes_2, thresh);
      nodes_2 = thresh;

      // next measurment point
      blocks_1 = 1;
      nodes_1 = 128; //10000;
      blocks = linearApprox(blocks_1, nodes_1, blocks_2, nodes_2, num_changed_nodes);
    }
    PERF_MON_ADD_DATA_NONTIME_P("LinearApprox", blocks, prefix);
  }

  //blocks = numBlocks;

  typedef LoadBalancer::Propagate<
          branching_factor,
          level_count,
          InnerNode,
          LeafNode> MyLoadBalancer;
  MyLoadBalancer load_balancer(
      this,
      blocks);
  load_balancer.run();
  PERF_MON_PRINT_INFO_P(temp_timer, "", prefix);
}

template<std::size_t branching_factor, std::size_t level_count, typename InnerNode, typename LeafNode>
void NTree<branching_factor, level_count, InnerNode, LeafNode>::init_const_memory()
{
  OctreeVoxelID temp[const_voxel_at_level_size];
  for (uint32_t i = 0; i < const_voxel_at_level_size; ++i)
    temp[i] = OctreeVoxelID(pow(branching_factor, i));

// copy selection lookup table to constant memory
  HANDLE_CUDA_ERROR(
      cudaMemcpyToSymbol(const_voxel_at_level, temp, const_voxel_at_level_size * sizeof(OctreeVoxelID), 0, cudaMemcpyHostToDevice));
  HANDLE_CUDA_ERROR(cudaDeviceSynchronize());

  uint32_t temp2[const_voxel_at_level_size];
  for (uint32_t i = 0; i < const_voxel_at_level_size; ++i)
    temp2[i] = uint32_t(pow(pow(branching_factor, 1.0 / 3), i));

// copy selection lookup table to constant memory
  HANDLE_CUDA_ERROR(
      cudaMemcpyToSymbol(const_cube_side_length, temp2, const_voxel_at_level_size * sizeof(uint32_t), 0, cudaMemcpyHostToDevice));
  HANDLE_CUDA_ERROR(cudaDeviceSynchronize());

//  uint16_t temp3[MORTON_LOOKUP_SIZE];
//  for (uint32_t x = 0; x < MORTON_LOOKUP_SIZE; ++x)
//    temp3[x] = morton_code(x, 0, 0);
//  // copy morton lookup table to constant memory
//  HANDLE_CUDA_ERROR(
//      cudaMemcpyToSymbol(const_motron_lookup, temp3, branching_factor * sizeof(lookUpType), 0,
//                         cudaMemcpyHostToDevice));
//  HANDLE_CUDA_ERROR(cudaDeviceSynchronize());
}

template<std::size_t branching_factor, std::size_t level_count, typename InnerNode, typename LeafNode>
void NTree<branching_factor, level_count, InnerNode, LeafNode>::free_bounding_box(
    thrust::device_vector<Vector3ui>& d_points)
{
  timespec time = getCPUTime();
  uint32_t level = 1;

  MapProperties<typename InnerNode::RayCastType, branching_factor> map_properties(level);
  voxel_count num_voxel = d_points.size();
  {
    thrust::device_vector<uint32_t> d_x(num_voxel);
    thrust::device_vector<uint32_t> d_y(num_voxel);
    thrust::device_vector<uint32_t> d_z(num_voxel);
    uint32_t num_threads = 128;
    uint32_t num_blocks = num_voxel / num_threads + 1;
    kernel_splitCoordinates<<<num_blocks, num_threads>>>(D_PTR(d_points), num_voxel, D_PTR(d_x), D_PTR(d_y), D_PTR(d_z));
    CHECK_CUDA_ERROR();
    HANDLE_CUDA_ERROR(cudaDeviceSynchronize());
    d_points.clear();
    d_points.shrink_to_fit();

    thrust::pair<thrust::device_vector<uint32_t>::iterator, thrust::device_vector<uint32_t>::iterator> res_x =
        thrust::minmax_element(d_x.begin(), d_x.end());
    thrust::pair<thrust::device_vector<uint32_t>::iterator, thrust::device_vector<uint32_t>::iterator> res_y =
        thrust::minmax_element(d_y.begin(), d_y.end());
    thrust::pair<thrust::device_vector<uint32_t>::iterator, thrust::device_vector<uint32_t>::iterator> res_z =
        thrust::minmax_element(d_z.begin(), d_z.end());
    HANDLE_CUDA_ERROR(cudaDeviceSynchronize());

    map_properties.min_x = *res_x.first;
    map_properties.max_x = *res_x.second;
    map_properties.min_y = *res_y.first;
    map_properties.max_y = *res_y.second;
    map_properties.min_z = *res_z.first;
    map_properties.max_z = *res_z.second;
  }

//  map_properties.min_x = map_properties.max_x = h_points[0].x;
//  map_properties.min_y = map_properties.max_y = h_points[0].y;
//  map_properties.min_z = map_properties.max_z = h_points[0].z;
//
//  for (uint32_t i = 0; i < num_voxel; ++i)
//  {
//    Vector3ui tmp = h_points[i];
//    if (tmp.x < map_properties.min_x)
//      map_properties.min_x = tmp.x;
//    if (tmp.x > map_properties.max_x)
//      map_properties.max_x = tmp.x;
//    if (tmp.y < map_properties.min_y)
//      map_properties.min_y = tmp.y;
//    if (tmp.y > map_properties.max_y)
//      map_properties.max_y = tmp.y;
//    if (tmp.z < map_properties.min_z)
//      map_properties.min_z = tmp.z;
//    if (tmp.z > map_properties.max_z)
//      map_properties.max_z = tmp.z;
//  }
#ifdef FREE_BOUNDING_BOX_MESSAGES
  LOGGING_DEBUG(OctreeFreeBoundingBoxLog, "compute min/max: " <<  timeDiff(time, getCPUTime()) << " ms" << endl);
#endif
  time = getCPUTime();

  map_properties.align();

  // determine level of free space computation
  const uint64_t max_mem = 200 * cMBYTE2BYTE; // 200 MB
  const uint64_t mem_needed = map_properties.size_v * sizeof(typename InnerNode::RayCastType);
  if (mem_needed >= max_mem)
  {
    double factor = double(mem_needed) / max_mem;
    uint32_t levels = ceil(log(factor) / log(branching_factor));
    level += levels;
    map_properties = map_properties.createNextLevelMap(level);
  }

#ifdef FREE_BOUNDING_BOX_MESSAGES
  LOGGING_DEBUG(OctreeFreeBoundingBoxLog, map_properties << endl);
#endif

  // ### malloc array ###
  time = getCPUTime();

  HANDLE_CUDA_ERROR(
      cudaMalloc((void** ) &map_properties.d_ptr,
                 map_properties.size_v * sizeof(typename InnerNode::RayCastType)));
  HANDLE_CUDA_ERROR(cudaDeviceSynchronize());

  // ##### init free space #####
  typename InnerNode::RayCastType init;
  getFreeValue(&init);
  thrust::device_ptr<typename InnerNode::RayCastType> d_ptr(map_properties.d_ptr);
  thrust::fill(d_ptr, d_ptr + map_properties.size_v, init);
  HANDLE_CUDA_ERROR(cudaDeviceSynchronize());
#ifdef FREE_BOUNDING_BOX_MESSAGES
  LOGGING_DEBUG(OctreeFreeBoundingBoxLog, "cudaMalloc + cudaMemset: " <<  timeDiff(time, getCPUTime()) << " ms" << endl);
#endif

  time = getCPUTime();
  thrust::host_vector<ComputeFreeSpaceData> h_packed_levels(level_count, ComputeFreeSpaceData(NULL, NULL, 0));
  packVoxel_Map(map_properties, h_packed_levels, map_properties.size_v, level);

#ifdef FREE_BOUNDING_BOX_MESSAGES
  LOGGING_DEBUG(OctreeFreeBoundingBoxLog, "pack bounding box: " <<  timeDiff(time, getCPUTime()) << " ms" << endl);
#endif

  // ##### insert free space #####
  time = getCPUTime();
  for (int32_t l = level_count - 1; l >= 0; --l)
  {
    // if(l == 7){
    BasicData tmp;
    getFreeBoxResetData(tmp);
    //getOccupancyResetData(tmp);
    thrust::constant_iterator<BasicData> reset_data(tmp);
    insertVoxel<true, BasicData>(h_packed_levels[l].m_voxel_id, h_packed_levels[l].m_basic_data, reset_data,
                                 h_packed_levels[l].m_count, l);
    //}
  }

#ifdef FREE_BOUNDING_BOX_MESSAGES
  LOGGING_DEBUG(OctreeFreeBoundingBoxLog, "insert voxel of bounding box: " <<  timeDiff(time, getCPUTime()) << " ms" << endl);
#endif
}

template<std::size_t branching_factor, std::size_t level_count, typename InnerNode, typename LeafNode>
void NTree<branching_factor, level_count, InnerNode, LeafNode>::clearCollisionFlags()
{
  typedef LoadBalancer::Extract<
        branching_factor,
        level_count,
        InnerNode,
        LeafNode,
        true,
        false> MyLoadBalancer;

    MyLoadBalancer load_balancer(
        this,
        NULL,
        0,
        NULL,
        0);

    load_balancer.run();

//  double balance_overhead;
//  int num_balance_tasks;
}

template<std::size_t branching_factor, std::size_t level_count, typename InnerNode, typename LeafNode>
void NTree<branching_factor, level_count, InnerNode, LeafNode>::serialize(std::ostream& out, const bool bin_mode)
{
  LOGGING_DEBUG(OctreeDebugLog, "Serialize..." << endl);
  uint32_t needed_size = m_rebuild_buffer_size;
  typedef LoadBalancer::Extract<
          branching_factor,
          level_count,
          InnerNode,
          LeafNode,
          false,
          true> MyLBCounter;
  typedef LoadBalancer::Extract<
          branching_factor,
          level_count,
          InnerNode,
          LeafNode,
          false,
          false> MyLoadBalancer;
  MyLBCounter load_balancer(
          this,
          NULL,
          0,
          m_extract_status_selection,
          0);
  load_balancer.run();
  needed_size = load_balancer.m_num_elements;

  thrust::host_vector<NodeData> h_node_data;
  uint32_t num_cubes;
  {
    thrust::device_vector<NodeData> d_node_data(needed_size);
    MyLoadBalancer load_balancer(
            this,
            D_PTR(d_node_data),
            needed_size,
            m_extract_status_selection,
            0);
    load_balancer.run();
    num_cubes = load_balancer.m_num_elements;
    h_node_data = d_node_data;
  }

  LOGGING_DEBUG(OctreeDebugLog, "Extract done." << endl);
  if (bin_mode)
  {
    out.write((char*) &numBlocks, sizeof(uint32_t));
    out.write((char*) &numThreadsPerBlock, sizeof(uint32_t));
    out.write((char*) &m_resolution, sizeof(uint32_t));
    out.write((char*) &num_cubes, sizeof(uint32_t));
    out.write((char*) &h_node_data[0], num_cubes * sizeof(NodeData));
  }
  else
  {
    out << numThreadsPerBlock << "\n";
    out << m_resolution << "\n";
    out << num_cubes << "\n";
    for (uint32_t i = 0; i < num_cubes; ++i)
      out << h_node_data[i] << "\n";
  }

  LOGGING_DEBUG(OctreeDebugLog, "Serialize done: Extracted " << num_cubes << " Voxels" << endl);
}

template<std::size_t branching_factor, std::size_t level_count, typename InnerNode, typename LeafNode>
bool NTree<branching_factor, level_count, InnerNode, LeafNode>::deserialize(std::istream& in, const bool bin_mode)
{
  LOGGING_DEBUG(OctreeDebugLog, "Deserialize..." << endl);
  uint32_t numBlocks, numThreadsPerBlock, resolution, size;
  if(bin_mode)
  {
    in.read((char*)&numBlocks, sizeof(uint32_t));
    in.read((char*)&numThreadsPerBlock, sizeof(uint32_t));
    in.read((char*)&resolution, sizeof(uint32_t));
    in.read((char*)&size, sizeof(uint32_t));
  }
  else
  {
    in >> numBlocks;
    in >> numThreadsPerBlock;
    in >> resolution;
    in >> size;
  }

  thrust::host_vector<NodeData> h_node_data(size);
  thrust::device_vector<NodeData> d_node_data(size);

  if(bin_mode)
  {
    in.read((char*)&h_node_data[0], size * sizeof(NodeData));
  }
  else
  {
    for(uint32_t i = 0; i < size; ++i)
      in >> h_node_data[i];
  }
  d_node_data = h_node_data;

  const std::string prefix = "rebuild";
  const std::string temp_timer = prefix + "_temp";
  PERF_MON_START(prefix);

  this->numBlocks = numBlocks;
  this->numThreadsPerBlock = numThreadsPerBlock;
  this->m_resolution = resolution;
  internal_rebuild(d_node_data, size);
  LOGGING_DEBUG(OctreeDebugLog, "Deserialize done. Loaded " << size << " Cubes (h_node_data size = " << h_node_data.size() << " )" << endl);
  return true;
}

template<std::size_t branching_factor, std::size_t level_count, typename InnerNode, typename LeafNode>
void NTree<branching_factor, level_count, InnerNode, LeafNode>::clear()
{
  for (uint32_t i = 0; i < m_allocation_list.size(); ++i)
    HANDLE_CUDA_ERROR(cudaFree((m_allocation_list)[i]));
  m_allocation_list.clear();

  InnerNode* r = new InnerNode();
  initRoot(*r);
  r->setStatus(r->getStatus() | ns_STATIC_MAP | ns_DYNAMIC_MAP);
  HANDLE_CUDA_ERROR(cudaMalloc(&m_root, sizeof(InnerNode)));
  HANDLE_CUDA_ERROR(cudaMemcpy(m_root, r, sizeof(InnerNode), cudaMemcpyHostToDevice));
  m_allocation_list.push_back(m_root);
  delete r;

  allocInnerNodes = 1;
  allocLeafNodes = 0;
}

} // end of ns
} // end of ns

#endif /* NTREE_HPP_ */
