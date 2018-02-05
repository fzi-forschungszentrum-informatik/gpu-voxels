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
 * \date    2013-11-07
 *
 */
//----------------------------------------------------------------------/*
#ifndef GPU_VOXELS_OCTREE_NTREEDATA_H_INCLUDED
#define GPU_VOXELS_OCTREE_NTREEDATA_H_INCLUDED

#include <iostream>

#include <gpu_voxels/octree/Nodes.h>
//#include <gpu_voxels/octree/Voxel.h>
#include <gpu_voxels/octree/EnvironmentNodes.h>
#include <gpu_voxels/octree/RobotNodes.h>

//including cub.h will place all of cub in "thrust::system::cuda::detail::cub_"
//we don't want to include all of cub here, because gcc will get confused by device code
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

namespace gpu_voxels {
namespace NTree {

enum NodeType
{
  isInnerNode = 0, isLeafNode = 1
};

template<typename InnerNode>
struct Work_Item
{
  InnerNode* x;
  InnerNode* y;

  __host__ __device__ Work_Item()
  {

  }

  __host__ __device__ Work_Item(InnerNode* x, InnerNode* y)
  {
    this->x = x;
    this->y = y;
  }

  __host__ __device__ Work_Item(bool init)
  {
    x = NULL;
    y = NULL;
  }
};

template<typename InnerNode>
struct Work_Item_Small
{
  InnerNode* x;
  InnerNode* y;

  __host__ __device__ Work_Item_Small()
  {

  }

  __host__ __device__ Work_Item_Small(InnerNode* x, InnerNode* y)
  {
    this->x = x;
    this->y = y;
  }

  __host__ __device__ Work_Item_Small(bool init)
  {
    x = NULL;
    y = NULL;
  }
};

/**
 * Work item used for intersecting two NTree.
 */
template<typename a_InnerNode, typename b_InnerNode>
struct WorkItemIntersect
{
  a_InnerNode* a;
  b_InnerNode* b;
  uint8_t level;
  bool a_active;
  bool b_active;

  __host__ __device__ WorkItemIntersect()
  {

  }

  __host__ __device__ WorkItemIntersect(a_InnerNode* a, b_InnerNode* b, uint8_t level, bool a_active,
                                        bool b_active)
  {
    this->a = a;
    this->b = b;
    this->level = level;
    this->a_active = a_active;
    this->b_active = b_active;
  }
};

/**
 * Work item used for extracting data out of an NTree.
 */
template<typename InnerNode>
struct WorkItemExtract
{
  InnerNode* node;
  OctreeVoxelID nodeId;
  uint8_t level;

  __host__ __device__ WorkItemExtract()
  {
  }

  __host__ __device__ WorkItemExtract(InnerNode* node, OctreeVoxelID nodeId, uint8_t level)
  {
    this->node = node;
    this->nodeId = nodeId;
    this->level = level;
  }
};

/**
 * Work item used for propagating the status flags of the NTree and therefore restore the NTree invariant.
 */
template<typename InnerNode>
struct WorkItemPropagate
{
  InnerNode* node;
  InnerNode* parent_node;
  uint8_t level;
  //NodeStatus parent_status;
  bool is_top_down;
  bool update_subtree;

  __host__ __device__ WorkItemPropagate()
  {
  }

  __host__ __device__
  //WorkItemPropagate(InnerNode* node, InnerNode* parent_node, NodeStatus parent_status, uint8_t level)
  WorkItemPropagate(InnerNode* const node, InnerNode* const parent_node, const bool is_top_down,
                    const uint8_t level, const bool update_subtree)
  {
    this->node = node;
    this->parent_node = parent_node;
    this->level = level;
    //this->parent_status = parent_status;
    this->is_top_down = is_top_down;
    this->update_subtree = update_subtree;
  }
};

/**
 * Work item used for intersecting a NTree with a VoxelMap.
 */
template<typename InnerNode>
struct WorkItemIntersectVoxelMap
{
  InnerNode* node;
  gpu_voxels::Vector3ui coordinates;
  uint8_t level;
  bool check_border;
  bool active;

  __host__ __device__ WorkItemIntersectVoxelMap()
  {

  }

  __host__ __device__ WorkItemIntersectVoxelMap(InnerNode* node, gpu_voxels::Vector3ui coordinates, uint8_t level,
                                                bool check_border, bool active)
  {
    this->node = node;
    this->coordinates = coordinates;
    this->level = level;
    this->check_border = check_border;
    this->active = active;
  }
};

struct Work_Cache
{
  void* a;
  void* b;

  __host__ __device__ Work_Cache()
  {
  }
};

//__host__ __device__ __forceinline__
//OctreeVoxelID computeVoxelID(const uint32_t index, const uint32_t size_x, const uint32_t size_y)
//{
//  gpu_voxels::Vector3ui coordinates = computeCoordinates(index, size_x, size_y);
//  return morton_code60(coordinates);
//}
//
//__host__ __device__ __forceinline__
//OctreeVoxelID computeVoxelID_morton(const uint32_t index, const uint32_t size_x, const uint32_t size_y,
//                              const uint32_t log_branching_factor, const uint32_t branching_factor,
//                              const gpu_voxels::Vector3ui coordinates_offset)
//{
//  const uint32_t num_bits = log_branching_factor / 3;
//  const gpu_voxels::Vector3ui coordinates = (coordinates_offset >> num_bits)
//      + computeCoordinates(index >> log_branching_factor, size_x >> num_bits, size_y >> num_bits);
//  return (morton_code60(coordinates) << log_branching_factor) | (index & (branching_factor - 1));
//}

template<typename T, std::size_t branching_factor>
struct MapProperties
{
  uint32_t* coordinate_x;
  uint32_t* coordinate_y;
  uint32_t* coordinate_z;
  uint32_t coordinates_size;
  uint32_t min_x, min_y, min_z;
  uint32_t max_x, max_y, max_z;
  uint32_t size_x, size_y, size_z;
  uint32_t size_vx, size_vy, size_vz;
  uint32_t morton_size_vx, morton_size_vy, morton_size_vz;
  std::size_t size, size_v;

  cudaPitchedPtr devPitchedPtr;
  T* d_ptr;

  uint32_t level;
  uint32_t log_branching_factor;
  uint32_t coordinate_shift;
  uint32_t log_branching_factor3;
  uint16_t morton_mask;
  gpu_voxels::Vector3ui next_map_shift;

//  __host__ __device__ MapProperties()
//  {
//
//  }

  __host__
   friend std::ostream& operator<<(std::ostream& out, const MapProperties& p)
  {
    out << "coordinates_size: " << p.coordinates_size << "  min_x: " << p.min_x << "  min_y: " << p.min_y
        << "  min_z: " << p.min_z << std::endl << "max_x: " << p.max_x << "  max_y: " << p.max_y
        << "  max_z: " << p.max_z << std::endl << "size_x: " << p.size_x << "  size_y: " << p.size_y
        << "  size_z: " << p.size_z << std::endl << "size_vx: " << p.size_vx << "  size_vy: " << p.size_vy
        << "  size_vz: " << p.size_vz << std::endl << "morton_size_vx: " << p.morton_size_vx
        << "  morton_size_vy: " << p.morton_size_vy << "  morton_size_vz: " << p.morton_size_vz << std::endl
        << "size: " << p.size << "  size_v: " << p.size_v << "  level: " << p.level << std::endl
        << "log_branching_factor: " << p.log_branching_factor << "  coordinate_shift: " << p.coordinate_shift
        << "  log_branching_factor3: " << p.log_branching_factor3 << "  morton_mask: " << p.morton_mask
        << std::endl;
    return out;
  }

  __host__ __device__ MapProperties()
  {

  }

//  __host__ __device__ MapProperties() :
//      level(0), log_branching_factor(0), log_branching_factor3(0), morton_mask(0), coordinate_shift(0)
//  {
//
//  }

  __host__ __device__ MapProperties(const uint32_t level)
  {
    this->level = level;
    log_branching_factor = cub::Log2<branching_factor>::VALUE;
    log_branching_factor3 = log_branching_factor / 3;
    morton_mask = (1 << log_branching_factor3) - 1;
    coordinate_shift = level * log_branching_factor3;
  }

//  template<typename T>
//  __host__ __device__
//  __forceinline__ T* computePtr(uint32_t x, uint32_t y, uint32_t z)
//  {
//    if (min_x <= x && x <= max_x && min_y <= y && y <= max_y && min_z <= z && z <= max_z)
//      return (((T*) devPitchedPtr.ptr) + (z - min_z) * devPitchedPtr.pitch * size_y
//          + (y - min_y) * devPitchedPtr.pitch + (x - min_x));
//    else
//      return NULL;
//  }

  __host__  __device__
   __forceinline__ T* computePtr(uint32_t x, uint32_t y, uint32_t z)
  {
    assert(min_x <= x && x <= max_x);
    assert(min_y <= y && y <= max_y);
    assert(min_z <= z && z <= max_z);
    return d_ptr + ((x - min_x) >> coordinate_shift) + ((y - min_y) >> coordinate_shift) * size_vx
        + ((z - min_z) >> coordinate_shift) * size_vy * size_vx;
  }

//  template<typename T>
//  __host__ __device__
//  __forceinline__ T* computeOffsetPtr(uint32_t offset_x, uint32_t offset_y, uint32_t offset_z)
//  {
////    if (offset_x <= size_x && offset_y <= size_y && offset_z <= size_z)
//    return (((T*) devPitchedPtr.ptr) + offset_z * devPitchedPtr.pitch * size_y
//        + offset_y * devPitchedPtr.pitch + offset_x);
////    else
////      return NULL;
//  }

  __host__  __device__
   __forceinline__ T* computeOffsetPtr(uint32_t offset_x, uint32_t offset_y,
                                                          uint32_t offset_z)
  {
    assert(offset_x < size_x && offset_y < size_y && offset_z < size_z);
    return d_ptr + (offset_x >> coordinate_shift) + (offset_y >> coordinate_shift) * size_vx
        + (offset_z >> coordinate_shift) * size_vy * size_vx;
  }

  __host__  __device__
   __forceinline__ T* computeRelativeMortonOffsetPtr(uint32_t offset_x, uint32_t offset_y,
                                                                        uint32_t offset_z)
  {
    const uint16_t morton = morton_code12(uint16_t(offset_x & morton_mask), uint16_t(offset_y & morton_mask),
                                          uint16_t(offset_z & morton_mask));
    offset_x = offset_x >> log_branching_factor3;
    offset_y = offset_y >> log_branching_factor3;
    offset_z = offset_z >> log_branching_factor3;
    return d_ptr
        + (((offset_x + offset_y * morton_size_vx + offset_z * morton_size_vx * morton_size_vy)
            << log_branching_factor) | morton);
  }

  __host__  __device__
   __forceinline__ T* computeMortonOffsetPtr(uint32_t offset_x, uint32_t offset_y,
                                                                uint32_t offset_z)
  {
    offset_x = offset_x >> coordinate_shift;
    offset_y = offset_y >> coordinate_shift;
    offset_z = offset_z >> coordinate_shift;
    return computeRelativeMortonOffsetPtr(offset_x, offset_y, offset_z);
  }

  __host__  __device__
   __forceinline__ T* computeMortonOffsetPtr(gpu_voxels::Vector3ui offset)
  {
    return computeMortonOffsetPtr(offset.x, offset.y, offset.z);
  }

  __host__  __device__
   __forceinline__ T* computeRelativeMortonOffsetPtr(gpu_voxels::Vector3ui offset)
  {
    return computeRelativeMortonOffsetPtr(offset.x, offset.y, offset.z);
  }

  __host__ __device__
  void align()
  {
    uint32_t alignment = 1 << ((level + 1) * log_branching_factor3);
    min_x -= min_x % alignment;
    min_y -= min_y % alignment;
    min_z -= min_z % alignment;
    max_x += alignment - 1 - (max_x % alignment);
    max_y += alignment - 1 - (max_y % alignment);
    max_z += alignment - 1 - (max_z % alignment);

    next_map_shift = ((gpu_voxels::Vector3ui(min_x, min_y, min_z) >> ((level + 1) * log_branching_factor3))
        & morton_mask);

    size_x = max_x - min_x + 1;
    size_y = max_y - min_y + 1;
    size_z = max_z - min_z + 1;
    size = size_x * size_y * size_z;

    size_vx = size_x >> coordinate_shift;
    size_vy = size_y >> coordinate_shift;
    size_vz = size_z >> coordinate_shift;
    size_v = size_vx * size_vy * size_vz;

    morton_size_vx = size_vx >> log_branching_factor3;
    morton_size_vy = size_vy >> log_branching_factor3;
    morton_size_vz = size_vz >> log_branching_factor3;

//    morton_size_x = size_x >> log_bf3;
//    morton_size_y = size_y >> log_bf3;
//    morton_size_z = size_z >> log_bf3;
//
//    assert((size_x & ((1 << log_bf3) - 1)) == 0);
//    assert((size_y & ((1 << log_bf3) - 1)) == 0);
//    assert((size_z & ((1 << log_bf3) - 1)) == 0);
  }

  __host__   __device__ MapProperties createNextLevelMap(uint32_t l)
  {
    MapProperties new_map(l);

    new_map.coordinate_x = new_map.coordinate_y = new_map.coordinate_z = NULL;
    new_map.d_ptr = NULL;
    new_map.coordinates_size = 0;

    new_map.min_x = min_x;
    new_map.min_y = min_y;
    new_map.min_z = min_z;

    new_map.max_x = max_x;
    new_map.max_y = max_y;
    new_map.max_z = max_z;

    new_map.align();
    return new_map;
  }

  __host__   __device__ MapProperties createNextLevelMap()
  {
    return createNextLevelMap(level + 1);
  }

  __host__  __device__  __forceinline__ gpu_voxels::Vector3ui computeNextRelativeCoordinates(uint32_t index)
  {
    index = index >> log_branching_factor;
    const uint32_t x_y_plane = morton_size_vx * morton_size_vy;
    const uint32_t tmp = index % x_y_plane;
    return gpu_voxels::Vector3ui(tmp % morton_size_vx, tmp / morton_size_vx, index / x_y_plane);
  }

  __host__  __device__  __forceinline__ T* computeNextMapPtr(const uint32_t this_map_index,
                                                           MapProperties next_map)
  {
    gpu_voxels::Vector3ui rel_coordinates = computeNextRelativeCoordinates(this_map_index);
    // fix-up to translate between the maps origins (min_x, ...)
    rel_coordinates = rel_coordinates + next_map_shift;
    return next_map.computeRelativeMortonOffsetPtr(rel_coordinates);
  }

//  __host__ __device__ __forceinline__
//  gpu_voxels::Vector3ui trafoRelativeToNextMap(gpu_voxels::Vector3ui rel_coordinates)
//  {
//    return (rel_coordinates >> log_branching_factor3);
//  }
//
//  __host__ __device__ __forceinline__
//  gpu_voxels::Vector3ui trafoRelativeToAbsolute(gpu_voxels::Vector3ui rel_coordinates)
//  {
//    return (rel_coordinates << (log_branching_factor3 * level));
//  }

  __host__  __device__  __forceinline__ OctreeVoxelID computeVoxelID_morton(const uint32_t index)
  {
    const uint32_t m = index & (branching_factor - 1);
    const gpu_voxels::Vector3ui offset(min_x, min_y, min_z);
    const gpu_voxels::Vector3ui coordinates = offset
        + (computeNextRelativeCoordinates(index) << (log_branching_factor3 * (level + 1)));
    return morton_code60(coordinates) | (m << (level * log_branching_factor));
  }

};

} // end of ns
} // end of ns

#endif /* NTREEDATA_H_ */
