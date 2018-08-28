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
* \author  Felix Mauch <mauch@fzi.de>
* \date    2015-05-05
*
*/
//----------------------------------------------------------------------

#include "VoxelListOperations.h"
#include <gpu_voxels/voxelmap/kernels/VoxelMapOperations.h>
#include <gpu_voxels/octree/Morton.h>


namespace gpu_voxels {
namespace voxellist {


// ============================================================================
// All Kernels that take MapVoxelIDs (uint_32) are used for Voxel-Map-Adressing
// ============================================================================


// used to avoid "non-empty default constructor" problems in shared memory arrays
extern __shared__ int dynamic_shared_mem[];

template<class Voxel>
__global__
void kernelInsertGlobalPointCloud(MapVoxelID* id_list, Vector3ui* coord_list, Voxel* voxel_list,
                                  const Vector3ui ref_map_dim, const float voxel_side_length,
                                  const Vector3f* points, const std::size_t sizePoints,
                                  const uint32_t offset_new_points, const BitVoxelMeaning voxel_meaning)
{
  //printf("RefMapDim = [%d, %d, %d] \n",ref_map_dim.x, ref_map_dim.y, ref_map_dim.z);

  Voxel new_voxel = Voxel();
  new_voxel.insert(voxel_meaning);
  Vector3ui uint_coords;
  for (uint32_t i = blockIdx.x * blockDim.x + threadIdx.x; i < sizePoints; i += blockDim.x * gridDim.x)
  {
    uint_coords = voxelmap::mapToVoxels(voxel_side_length, points[i]);
    coord_list[i+offset_new_points] = uint_coords;
    voxel_list[i+offset_new_points] = new_voxel;
    id_list[i+offset_new_points] = voxelmap::getVoxelIndexUnsigned(ref_map_dim, uint_coords);

  }
}

template<class Voxel>
__global__
void kernelInsertCoordinateTuples(MapVoxelID* id_list, Vector3ui* coord_list, Voxel* voxel_list,
                                  const Vector3ui ref_map_dim, const Vector3ui* coordinates, const std::size_t sizeVoxels,
                                  const uint32_t offset_new_voxels, const BitVoxelMeaning voxel_meaning)
{
  //printf("RefMapDim = [%d, %d, %d] \n",ref_map_dim.x, ref_map_dim.y, ref_map_dim.z);

  Voxel new_voxel = Voxel();
  new_voxel.insert(voxel_meaning);

  for (uint32_t i = blockIdx.x * blockDim.x + threadIdx.x; i < sizeVoxels; i += blockDim.x * gridDim.x)
  {
    Vector3ui uint_coords = coordinates[i];
    coord_list[i+offset_new_voxels] = uint_coords;
    voxel_list[i+offset_new_voxels] = new_voxel;
    id_list[i+offset_new_voxels] = voxelmap::getVoxelIndexUnsigned(ref_map_dim, uint_coords);

  }
}

template<class Voxel>
__global__
void kernelInsertMetaPointCloud(MapVoxelID* id_list, Vector3ui* coord_list, Voxel* voxel_list,
                                const Vector3ui ref_map_dim, const float voxel_side_length,
                                const MetaPointCloudStruct *meta_point_cloud,
                                const uint32_t offset_new_points, const BitVoxelMeaning voxel_meaning)
{
  for (uint32_t i = blockIdx.x * blockDim.x + threadIdx.x; i < meta_point_cloud->accumulated_cloud_size;
      i += blockDim.x * gridDim.x)
  {
    Voxel new_voxel;
    new_voxel.insert(voxel_meaning);
    Vector3ui uint_coords;
    //printf("Inserting a voxel with meaning %d \n", voxel_meanings[sub_cloud]);
    uint_coords = voxelmap::mapToVoxels(voxel_side_length,
                                                      meta_point_cloud->clouds_base_addresses[0][i]);

    coord_list[i+offset_new_points] = uint_coords;
    voxel_list[i+offset_new_points] = new_voxel;
    id_list[i+offset_new_points] = voxelmap::getVoxelIndexUnsigned(ref_map_dim, uint_coords);
  }
}


template<class Voxel>
__global__
void kernelInsertMetaPointCloud(MapVoxelID* id_list, Vector3ui* coord_list, Voxel* voxel_list,
                                const Vector3ui ref_map_dim, const float voxel_side_length,
                                const MetaPointCloudStruct *meta_point_cloud,
                                const uint32_t offset_new_points, const BitVoxelMeaning* voxel_meanings)
{

  u_int16_t sub_cloud = 0;
  u_int32_t sub_cloud_upper_bound = meta_point_cloud->cloud_sizes[sub_cloud];
  Vector3ui uint_coords;
  for (uint32_t i = blockIdx.x * blockDim.x + threadIdx.x; i < meta_point_cloud->accumulated_cloud_size;
      i += blockDim.x * gridDim.x)
  {
    // find out, to which sub_cloud our point belongs
    while(i >= sub_cloud_upper_bound)
    {
      sub_cloud++;
      sub_cloud_upper_bound += meta_point_cloud->cloud_sizes[sub_cloud];
    }
    //printf("i = %d, sub_cloud_upper_bound = %d \n", i, sub_cloud_upper_bound);

    Voxel new_voxel;
    new_voxel.insert(voxel_meanings[sub_cloud]);
    //printf("Inserting a voxel with meaning %d \n", voxel_meanings[sub_cloud]);
    uint_coords = voxelmap::mapToVoxels(voxel_side_length,
                                                      meta_point_cloud->clouds_base_addresses[0][i]);

    coord_list[i+offset_new_points] = uint_coords;
    voxel_list[i+offset_new_points] = new_voxel;
    id_list[i+offset_new_points] = voxelmap::getVoxelIndexUnsigned(ref_map_dim, uint_coords);
  }
}


/*!
 * Counting collision results.
 */
template<class Voxel, class OtherVoxel, class Collider>
__global__
void kernelCollideWithVoxelMap(const MapVoxelID* this_id_list, Voxel* this_voxel_list, uint32_t this_list_size,
                               const OtherVoxel* other_map, Vector3ui other_map_dim, Collider collider,
                               Vector3i offset, uint16_t* results)
{
  __shared__ uint16_t cache[cMAX_THREADS_PER_BLOCK];
  uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
  uint32_t cache_index = threadIdx.x;
  cache[cache_index] = 0;

  OtherVoxel* max_index = getHighestVoxelPtr(other_map, other_map_dim);
  while (i < this_list_size)
  {
    const OtherVoxel* other_voxel = (other_map + this_id_list[i]) + voxelmap::getVoxelIndexSigned(other_map_dim, offset);
    if(other_voxel >= other_map && other_voxel <= max_index)
    {
      const bool collision = collider.collide(this_voxel_list[i], *other_voxel);
      if (collision) // store collision info
      {
        cache[cache_index] += 1;
        // Mark the Voxel as colliding.
        this_voxel_list[i].insert(eBVM_COLLISION);
      }
    }
//    else
//    {
//      printf("Ignoring out of map voxel.");
//    }
    i += blockDim.x * gridDim.x;
  }

  __syncthreads();

  uint32_t j = blockDim.x / 2;

  while (j != 0)
  {
    if (cache_index < j)
    {
      cache[cache_index] = cache[cache_index] + cache[cache_index + j];
    }
    __syncthreads();
    j /= 2;
  }

  // copy results from this block to global memory
  if (cache_index == 0)
  {
    results[blockIdx.x] = cache[0];
  }
}

/*!
 * Calculating Bitvector results and counting collisions
 */
template<class VoxelType>
__global__
void kernelCollideWithVoxelMap(const MapVoxelID* this_id_list, BitVectorVoxel *this_voxel_list, uint32_t this_list_size,
                               const VoxelType *other_map, Vector3ui other_map_dim, float col_threshold,
                               Vector3i offset, uint16_t* coll_counter_results, BitVectorVoxel* bitvoxel_results)
{
  __shared__ uint16_t coll_counter_cache[cMAX_THREADS_PER_BLOCK];

  // points to dynamic shared memory; memory is uninitialised
  BitVectorVoxel* bitvoxel_cache = (BitVectorVoxel*)dynamic_shared_mem; //size: cMAX_THREADS_PER_BLOCK

  uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
  uint32_t cache_index = threadIdx.x;
  coll_counter_cache[cache_index] = 0;
  bitvoxel_cache[cache_index] = BitVectorVoxel();

  VoxelType* max_index = voxelmap::getHighestVoxelPtr(other_map, other_map_dim);
  while (i < this_list_size)
  {
    const VoxelType* other_voxel = (other_map + this_id_list[i]) + voxelmap::getVoxelIndexSigned(other_map_dim, offset);
    if(other_voxel >= other_map && other_voxel <= max_index)
    {
      if(other_voxel->isOccupied(col_threshold))
      {
        coll_counter_cache[cache_index] += 1;
        bitvoxel_cache[cache_index].bitVector() |= this_voxel_list[i].bitVector();
        // Mark the Voxel as colliding.
        this_voxel_list[i].insert(eBVM_COLLISION);
      }
    }
//    else
//    {
//      printf("Ignoring out of map voxel.");
//    }
    i += blockDim.x * gridDim.x;
  }

  __syncthreads();

  uint32_t j = blockDim.x / 2;

  while (j != 0)
  {
    if (cache_index < j)
    {
      coll_counter_cache[cache_index] = coll_counter_cache[cache_index] + coll_counter_cache[cache_index + j];
      bitvoxel_cache[cache_index].bitVector() = bitvoxel_cache[cache_index].bitVector() | bitvoxel_cache[cache_index + j].bitVector();
    }
    __syncthreads();
    j /= 2;
  }

  // copy results from this block to global memory
  if (cache_index == 0)
  {
    coll_counter_results[blockIdx.x] = coll_counter_cache[0];
    bitvoxel_results[blockIdx.x] = bitvoxel_cache[0];
  }
}

template<class VoxelType>
__global__
void kernelCollideWithVoxelMap(const OctreeVoxelID* this_id_list, BitVectorVoxel *this_voxel_list, uint32_t this_list_size,
                               const VoxelType *other_map, Vector3ui other_map_dim, float col_threshold,
                               Vector3i offset, uint16_t* coll_counter_results, BitVectorVoxel* bitvoxel_results)
{}

template<class VoxelType>
__global__
void kernelCollideWithVoxelMapBitMask(const OctreeVoxelID* this_id_list, BitVectorVoxel *this_voxel_list, uint32_t this_list_size,
                                      const VoxelType *other_map, Vector3ui other_map_dim, float col_threshold,
                                      Vector3i offset, const BitVectorVoxel* bitvoxel_mask, uint16_t* coll_counter_results)
{}

/*!
 * Counts collisions only if at least one of this lists Bitvector Bits matches the given Bitvector mask.
 * The other map voxel are checked for eBVM_OCCUPIED.
 */
template<class VoxelType>
__global__
void kernelCollideWithVoxelMapBitMask(const MapVoxelID* this_id_list, BitVectorVoxel *this_voxel_list, uint32_t this_list_size,
                                      const VoxelType *other_map, Vector3ui other_map_dim, float col_threshold,
                                      Vector3i offset, const BitVectorVoxel* bitvoxel_mask, uint16_t* coll_counter_results)
{
  __shared__ uint16_t coll_counter_cache[cMAX_THREADS_PER_BLOCK];

  uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
  uint32_t cache_index = threadIdx.x;
  coll_counter_cache[cache_index] = 0;

  VoxelType* max_index = voxelmap::getHighestVoxelPtr(other_map, other_map_dim);
  while (i < this_list_size)
  {
    const VoxelType* other_voxel = (other_map + this_id_list[i]) + voxelmap::getVoxelIndexSigned(other_map_dim, offset);
    if(other_voxel >= other_map && other_voxel <= max_index)
    {
      if(other_voxel->isOccupied(col_threshold))
      {
        if ( !(bitvoxel_mask->bitVector() & this_voxel_list[i].bitVector()).isZero() )
        {
          coll_counter_cache[cache_index] += 1;
          // Mark the Voxel as colliding.
          this_voxel_list[i].insert(eBVM_COLLISION);
        }
      }
    }
//    else
//    {
//      printf("Ignoring out of map voxel.");
//    }
    i += blockDim.x * gridDim.x;
  }

  __syncthreads();

  uint32_t j = blockDim.x / 2;

  while (j != 0)
  {
    if (cache_index < j)
    {
      coll_counter_cache[cache_index] = coll_counter_cache[cache_index] + coll_counter_cache[cache_index + j];
    }
    __syncthreads();
    j /= 2;
  }

  // copy results from this block to global memory
  if (cache_index == 0)
  {
    coll_counter_results[blockIdx.x] = coll_counter_cache[0];
  }
}

// ================================================================================
// All Kernels that take OctreeVoxelID (uint64_t) IDs are used for Morton-Adressing
// ================================================================================

// used to avoid "non-empty default constructor" problems in shared memory arrays
extern __shared__ int dynamic_shared_mem[];

template<class Voxel>
__global__
void kernelInsertGlobalPointCloud(OctreeVoxelID* id_list, Vector3ui* coord_list, Voxel* voxel_list,
                                  const Vector3ui ref_map_dim, const float voxel_side_length,
                                  const Vector3f* points, const std::size_t sizePoints,
                                  const uint32_t offset_new_points, const BitVoxelMeaning voxel_meaning)
{  
  Voxel new_voxel = Voxel();
  new_voxel.insert(voxel_meaning);
  Vector3ui integer_coordinates;
  for (uint32_t i = blockIdx.x * blockDim.x + threadIdx.x; i < sizePoints; i += blockDim.x * gridDim.x)
  {
    integer_coordinates = voxelmap::mapToVoxels(voxel_side_length, points[i]);
    coord_list[i+offset_new_points] = integer_coordinates;
    voxel_list[i+offset_new_points] = new_voxel;
    id_list[i+offset_new_points] = NTree::morton_code60(integer_coordinates);
  }
}

template<class Voxel>
__global__
void kernelInsertCoordinateTuples(OctreeVoxelID* id_list, Vector3ui* coord_list, Voxel* voxel_list,
                                  const Vector3ui ref_map_dim, const Vector3ui* coordinates, const std::size_t sizeVoxels,
                                  const uint32_t offset_new_points, const BitVoxelMeaning voxel_meaning)
{  
  Voxel new_voxel = Voxel();
  new_voxel.insert(voxel_meaning);
  
  for (uint32_t i = blockIdx.x * blockDim.x + threadIdx.x; i < sizeVoxels; i += blockDim.x * gridDim.x)
  {
    Vector3ui integer_coordinates = coordinates[i];
    coord_list[i+offset_new_points] = integer_coordinates;
    voxel_list[i+offset_new_points] = new_voxel;
    id_list[i+offset_new_points] = NTree::morton_code60(integer_coordinates);
  }
}

template<class Voxel>
__global__
void kernelInsertMetaPointCloud(OctreeVoxelID* id_list, Vector3ui* coord_list, Voxel* voxel_list,
                                const Vector3ui ref_map_dim, const float voxel_side_length,
                                const MetaPointCloudStruct *meta_point_cloud,
                                const uint32_t offset_new_points, const BitVoxelMeaning voxel_meaning)
{
  for (uint32_t i = blockIdx.x * blockDim.x + threadIdx.x; i < meta_point_cloud->accumulated_cloud_size;
      i += blockDim.x * gridDim.x)
  {
    Voxel new_voxel;
    new_voxel.insert(voxel_meaning);
    Vector3ui integer_coordinates;
    //printf("Inserting a voxel with meaning %d \n", voxel_meanings[sub_cloud]);
    integer_coordinates = voxelmap::mapToVoxels(voxel_side_length,
                                                      meta_point_cloud->clouds_base_addresses[0][i]);

    coord_list[i+offset_new_points] = integer_coordinates;
    voxel_list[i+offset_new_points] = new_voxel;
    id_list[i+offset_new_points] = NTree::morton_code60(integer_coordinates);
  }
}


template<class Voxel>
__global__
void kernelInsertMetaPointCloud(OctreeVoxelID* id_list, Vector3ui* coord_list, Voxel* voxel_list,
                                const Vector3ui ref_map_dim, const float voxel_side_length,
                                const MetaPointCloudStruct *meta_point_cloud,
                                const uint32_t offset_new_points, const BitVoxelMeaning* voxel_meanings)
{

  u_int16_t sub_cloud = 0;
  u_int32_t sub_cloud_upper_bound = meta_point_cloud->cloud_sizes[sub_cloud];
  Vector3ui integer_coordinates;
  for (uint32_t i = blockIdx.x * blockDim.x + threadIdx.x; i < meta_point_cloud->accumulated_cloud_size;
      i += blockDim.x * gridDim.x)
  {
    // find out, to which sub_cloud our point belongs
    while(i >= sub_cloud_upper_bound)
    {
      sub_cloud++;
      sub_cloud_upper_bound += meta_point_cloud->cloud_sizes[sub_cloud];
    }
    //printf("i = %d, sub_cloud_upper_bound = %d \n", i, sub_cloud_upper_bound);

    Voxel new_voxel;
    new_voxel.insert(voxel_meanings[sub_cloud]);
    //printf("Inserting a voxel with meaning %d \n", voxel_meanings[sub_cloud]);
    integer_coordinates = voxelmap::mapToVoxels(voxel_side_length,
                                                      meta_point_cloud->clouds_base_addresses[0][i]);

    coord_list[i+offset_new_points] = integer_coordinates;
    voxel_list[i+offset_new_points] = new_voxel;
    id_list[i+offset_new_points] = NTree::morton_code60(integer_coordinates);
  }
}


/*!
 * Counting collision results.
 */
template<class Voxel, class OtherVoxel, class Collider>
__global__
void kernelCollideWithVoxelMap(const OctreeVoxelID* this_id_list, Voxel* this_voxel_list, uint32_t this_list_size,
                               const OtherVoxel* other_map, Vector3ui other_map_dim, Collider collider,
                               Vector3i offset, uint16_t* results)
{
  // NOP
  printf("kernelCollideWithVoxelMap not implemented for Octreee!");
}

/*!
 * Calculating Bitvector results and counting collisions
 */
__global__
void kernelCollideWithVoxelMap(const OctreeVoxelID* this_id_list, BitVectorVoxel *this_voxel_list, uint32_t this_list_size,
                               const ProbabilisticVoxel *other_map, Vector3ui other_map_dim, float col_threshold,
                               Vector3i offset, uint16_t* coll_counter_results, BitVectorVoxel* bitvoxel_results)
{
  // NOP
  printf("kernelCollideWithVoxelMap not implemented for Octreee!");
}

/*!
 * Calculating Bitvector results and counting collisions
 */
__global__
void kernelCollideWithVoxelMap(const OctreeVoxelID* this_id_list, BitVectorVoxel *this_voxel_list, uint32_t this_list_size,
                               const BitVectorVoxel *other_map, Vector3ui other_map_dim,
                               Vector3i offset, uint16_t* coll_counter_results, BitVectorVoxel* bitvoxel_results)
{
  // NOP
  printf("kernelCollideWithVoxelMap not implemented for Octreee!");
}

__global__
void kernelCollideWithVoxelMapBitMask(const OctreeVoxelID* this_id_list, BitVectorVoxel *this_voxel_list, uint32_t this_list_size,
                                      const ProbabilisticVoxel *other_map, Vector3ui other_map_dim, float col_threshold,
                                      Vector3i offset, const BitVectorVoxel* bitvoxel_mask, uint16_t* coll_counter_results)
{
  // NOP
  printf("kernelCollideWithVoxelMapBitMask not implemented for Octreee!");
}

__global__
void kernelCollideWithVoxelMapBitMask(const OctreeVoxelID* this_id_list, BitVectorVoxel *this_voxel_list, uint32_t this_list_size,
                                      const BitVectorVoxel *other_map, Vector3ui other_map_dim,
                                      Vector3i offset, const BitVectorVoxel* bitvoxel_mask, uint16_t* coll_counter_results)
{
  // NOP
  printf("kernelCollideWithVoxelMapBitMask not implemented for Octreee!");
}

} // end of namespace voxellist
} // end of namespace gpu_voxels
