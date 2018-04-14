// this is for emacs file handling -*- mode: c++; indent-tabs-mode: nil -*-

// -- BEGIN LICENSE BLOCK ----------------------------------------------
// -- END LICENSE BLOCK ------------------------------------------------

//----------------------------------------------------------------------
/*!\file
 *
 * \author  Sebastian Klemm
 * \date    2012-09-13
 *
 */
//----------------------------------------------------------------------
//#define LOCAL_DEBUG
#undef LOCAL_DEBUG

#include "VoxelMapOperations.h"
#include <stdio.h>

namespace gpu_voxels {
namespace voxelmap {

//
//void kernelCalculateBoundingBox(Voxel* voxelmap, const uint32_t voxelmap_size, )

//__global__
//void kernelInsertKinematicLinkBitvector(Voxel* voxelmap, const uint32_t voxelmap_size,
//                                        const Vector3ui dimensions, const float voxel_side_length,
//                                        uint32_t link_nr, uint32_t* point_cloud_sizes,
//                                        Vector3f** point_clouds, uint64_t bit_number)
//{
//  const uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
//  const Vector3ui map_dim = (*dimensions);
//
//  if (i < point_cloud_sizes[link_nr])
//  {
//    const Vector3ui integer_coordinates = mapToVoxels(voxel_side_length, point_clouds[link_nr][i]);
//    if ((integer_coordinates.x < map_dim.x) && (integer_coordinates.y < map_dim.y)
//        && (integer_coordinates.z < map_dim.z))
//    {
//
//      Voxel* voxel = getVoxelPtr(voxelmap, dimensions, integer_coordinates.x, integer_coordinates.y,
//                                 integer_coordinates.z);
//      voxel->setBitvector(voxel->getBitvector() | bit_number);
//
//    }
//  }
//}

//__global__
//void kernelInsertRobotKinematicLinkOverwritingSensorData(Voxel* voxelmap, const uint32_t voxelmap_size,
//                                                         const Vector3ui dimensions,
//                                                         const float voxel_side_length,
//                                                         const MetaPointCloudStruct *robot_links,
//                                                         uint32_t link_nr, const Voxel* environment_map)
//{
//  const Vector3ui map_dim = (*dimensions);
//
//  for (uint32_t i = blockIdx.x * blockDim.x + threadIdx.x; i < robot_links->cloud_sizes[link_nr];
//      i += gridDim.x * blockDim.x)
//  {
//    const Vector3ui integer_coordinates = mapToVoxels(voxel_side_length,
//                                                      robot_links->clouds_base_addresses[link_nr][i]);
//    if ((integer_coordinates.x < map_dim.x) && (integer_coordinates.y < map_dim.y)
//        && (integer_coordinates.z < map_dim.z))
//    {
//      Voxel* voxel = getVoxelPtr(voxelmap, dimensions, integer_coordinates.x, integer_coordinates.y,
//                                 integer_coordinates.z);
//      Voxel* env_voxel = getVoxelPtr(environment_map, dimensions, integer_coordinates.x,
//                                     integer_coordinates.y, integer_coordinates.z);
//      voxel->voxelmeaning = eBVM_OCCUPIED;
//      voxel->occupancy = 255;
//      env_voxel->occupancy = 0;
//    }
//  }
//}

///*! Insert a configuration for a kinematic link with self-collision check.
// *  Always set self_ to false before calling this function because
// *  it only indicates if there was a collision and not if there was none!
// */
//__global__
//void kernelInsertRobotKinematicLinkWithSelfCollisionCheck(Voxel* voxelmap, const uint32_t voxelmap_size,
//                                                          const Vector3ui dimensions,
//                                                          const float voxel_side_length,
//                                                          const MetaPointCloudStruct *robot_links,
//                                                          uint32_t link_nr, bool* self_collision)
//{
//  const Vector3ui map_dim = (*dimensions);
//
//  for (uint32_t i = blockIdx.x * blockDim.x + threadIdx.x; i < robot_links->cloud_sizes[link_nr];
//      i += gridDim.x * blockDim.x)
//  {
//    const Vector3ui integer_coordinates = mapToVoxels(voxel_side_length,
//                                                      robot_links->clouds_base_addresses[link_nr][i]);
//    if ((integer_coordinates.x < map_dim.x) && (integer_coordinates.y < map_dim.y)
//        && (integer_coordinates.z < map_dim.z))
//    {
//      Voxel* voxel = getVoxelPtr(voxelmap, dimensions, integer_coordinates.x, integer_coordinates.y,
//                                 integer_coordinates.z);
//
//      if (voxel->occupancy != 0)
//      {
//        (*self_collision) = true;
//      }
//      else
//      {
//        voxel->voxelmeaning = eBVM_OCCUPIED;
//        voxel->occupancy = 255;
//      }
//    }
//  }
//}



//__global__
//void kernelCollideVoxelMapsBoundingBox(Voxel* voxelmap, const uint32_t voxelmap_size, const uint8_t threshold,
//                                       Voxel* other_map, const uint8_t other_threshold, bool* results,
//                                       uint32_t offset_x, uint32_t offset_y, uint32_t offset_z,
//                                       uint32_t size_x, Vector3ui* dimensions)
//{
////  extern __shared__ bool cache[];//[cMAX_THREADS_PER_BLOCK];			//define Cache size in kernel call
////  uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
//
//  //calculate i:
//  uint32_t i = (offset_z + threadIdx.x) * dimensions->x * dimensions->y + //every Column on the y-axis is one block and
//      (offset_y + blockIdx.x) * dimensions->x + offset_x; // the threads are going from z = 0 till dim_z
//  uint32_t counter = 0;
////  printf("thread idx: %i, blockIdx: %i, dimx: %i, dimy: %i", threadIdx.x, blockIdx.x, dimensions->x, dimensions->y);
//
////  uint32_t cache_index = threadIdx.x;
////  int32_t _size_x = size_x;
////  int32_t counter = 0;
//
////  cache[cache_index] = false;
//  bool temp = false;
//
//  while (counter < size_x)
//  {
//    // todo / note: at the moment collision check is only used for DYNAMIC and SWEPT VOLUME meaning, static is used for debugging
//    temp = temp
//        || ((voxelmap[i].occupancy >= threshold) && (voxelmap[i].voxelmeaning != eBVM_OCCUPIED)
//            && (other_map[i].occupancy >= other_threshold) && (other_map[i].voxelmeaning != eBVM_OCCUPIED));
//
//    counter += 1;
//    i += 1;
////      i += blockDim.x * gridDim.x;
//  }
//
////  if(true)//(i == 30050600)
////  {
////	  printf("thread %i, collision %i \n", i, temp);
////	  printf("--- occupation planning: %i, voxelmeaning planning: %i \n",
////			  other_map[i].occupancy_planning, other_map[i].voxelmeaning_planning);
////  }
//
//  results[blockIdx.x * blockDim.x + threadIdx.x] = temp; //
//
////  cache[cache_index] = temp;
////  __syncthreads();
////
////  uint32_t j = blockDim.x / 2;
////
////  while (j!=0)
////  {
////    if (cache_index < j)
////    {
////      cache[cache_index] = cache[cache_index] || cache[cache_index + j];
////    }
////    __syncthreads();
////    j /= 2;
////  }
////
////  // copy results from this block to global memory
////  if (cache_index == 0)
////  {
//////    // FOR MEASUREMENT TEMPORARILY EDITED:
//////    results[blockIdx.x] = true;
////    results[blockIdx.x] = cache[0];
////  }
//}



//__global__
//void kernelShrinkCopyVoxelMapBitvector(Voxel* destination_map, const uint32_t destination_map_size,
//                                       Vector3ui* dest_map_dim, Voxel* source_map,
//                                       const uint32_t source_map_size, Vector3ui* source_map_dim,
//                                       uint8_t factor)
//{
//  const uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
//  if (i < destination_map_size)
//  {
//    Voxel* v = destination_map + i;
//    //getting indices for the Destination Voxel i
//    Vector3ui dest_voxel_index = mapToVoxels(destination_map, dest_map_dim, v);
//
//    uint64_t bitvector = 0;
//    //loop over every axis and get the value of the voxel
//    Voxel* index_z = getVoxelPtr(source_map, source_map_dim, dest_voxel_index.x * factor,
//                                 dest_voxel_index.y * factor, dest_voxel_index.z * factor);
//    Voxel* index_y;
//    Voxel* index_x;
//    for (uint8_t z = 0; z < factor; ++z)
//    {
//      index_y = index_z; //resetting the index
//      for (uint8_t y = 0; y < factor; ++y)
//      {
//        index_x = index_y;
//        for (uint8_t x = 0; x < factor; ++x)
//        {
//          bitvector |= index_x->getBitvector();
//          index_x += 1;
//        }
//        index_y += source_map_dim->x;
//      }
//      index_z += source_map_dim->y * source_map_dim->x;
//    }
//    v->setBitvector(bitvector);
//  }
//}


////for different sized voxelmaps
//__global__
//void kernelShrinkCopyVoxelMap(Voxel* destination_map, const uint32_t destination_map_size,
//                              Vector3ui* dest_map_dim, Voxel* source_map, const uint32_t source_map_size,
//                              Vector3ui* source_map_dim, uint8_t factor)
//{
//  const uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
//  if (i < destination_map_size)
//  {
//    Voxel* v = destination_map + i;
//    //getting indices for the Destination Voxel i
//    Vector3ui dest_voxel_index = mapToVoxels(destination_map, dest_map_dim, v);
//
//    uint8_t occupancy = 0;
//    uint8_t voxelmeaning = 0;
//    //loop over every axis and get the value of the voxel
//    Voxel* index_z = getVoxelPtr(source_map, source_map_dim, dest_voxel_index.x * factor,
//                                 dest_voxel_index.y * factor, dest_voxel_index.z * factor);
//    Voxel* index_y;
//    Voxel* index_x;
//    for (uint8_t z = 0; z < factor; ++z)
//    {
//      index_y = index_z; //resetting the index
//      for (uint8_t y = 0; y < factor; ++y)
//      {
//        index_x = index_y;
//        for (uint8_t x = 0; x < factor; ++x)
//        {
//          if (index_x->occupancy > occupancy)
//          {
//            occupancy = index_x->occupancy;
//            voxelmeaning = index_x->voxelmeaning;
//          }
//          index_x += 1;
//        }
//        index_y += source_map_dim->x;
//      }
//      index_z += source_map_dim->y * source_map_dim->x;
//    }
//    v->occupancy = occupancy;
//    v->voxelmeaning = voxelmeaning;
//  }
//}

} // end of namespace voxelmap
} // end of namespace voxellist

#ifdef LOCAL_DEBUG
#undef LOCAL_DEBUG
#endif
