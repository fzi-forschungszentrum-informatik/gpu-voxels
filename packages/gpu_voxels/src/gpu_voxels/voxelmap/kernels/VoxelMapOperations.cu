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
//__global__
//void kernelClearVoxelMap(Voxel* voxelmap, const uint32_t voxelmap_size)
//{
//  for (uint32_t i = blockIdx.x * blockDim.x + threadIdx.x; i < voxelmap_size; i += gridDim.x * blockDim.x)
//  {
//    Voxel* voxel = &(voxelmap[i]);
//    voxel->occupancy = 0;
//    voxel->setBitvector(0);
//  }
//}

//
//__global__
//void kernelDumpVoxelMap(const Voxel* voxelmap, const Vector3ui* dimensions, const uint32_t voxelmap_size)
//{
//  for (uint32_t i = blockIdx.x * blockDim.x + threadIdx.x; i < voxelmap_size; i += gridDim.x * blockDim.x)
//  {
//    Vector3ui coords = mapToVoxels(voxelmap, dimensions, &(voxelmap[i]));
//    Voxel voxel = voxelmap[i];
//    printf("Voxel(%u) = Voxel(%u, %u, %u) = (voxeltype, occupancy) = (%u, %u)\n", i, coords.x, coords.y,
//           coords.z, voxel.voxeltype, voxel.occupancy);
//  }
//}

/* Transform sensor data from sensor coordinate system
 * to world system. Needs extrinsic calibration of sensor.
 */

__global__
void kernelTransformSensorData(Sensor* sensor, Vector3f* raw_sensor_data, Vector3f* transformed_sensor_data)
{
  for (uint32_t i = blockIdx.x * blockDim.x + threadIdx.x; i < sensor->data_size; i += gridDim.x * blockDim.x)
  {
    Matrix4f sensor_pose;

    const Matrix3f mat_in = sensor->orientation;
    const Vector3f vec_in = sensor->position;

    sensor_pose.a11 = mat_in.a11;
    sensor_pose.a12 = mat_in.a12;
    sensor_pose.a13 = mat_in.a13;
    sensor_pose.a14 = vec_in.x;
    sensor_pose.a21 = mat_in.a21;
    sensor_pose.a22 = mat_in.a22;
    sensor_pose.a23 = mat_in.a23;
    sensor_pose.a24 = vec_in.y;
    sensor_pose.a31 = mat_in.a31;
    sensor_pose.a32 = mat_in.a32;
    sensor_pose.a33 = mat_in.a33;
    sensor_pose.a34 = vec_in.z;
    sensor_pose.a41 = 0;
    sensor_pose.a42 = 0;
    sensor_pose.a43 = 0;
    sensor_pose.a44 = 1;

    transformed_sensor_data[i] = sensor_pose * raw_sensor_data[i];
    // Todo remove later:
    if (transformed_sensor_data[i].z < 0) //This code sets the points that lie outside of the map to z=0
    {
      // Vector3f start = transformed_sensor_data[i];
      double factor = vec_in.z / (vec_in.z - transformed_sensor_data[i].z); //the vector from the sensor to the world point is scaled with this factor, then its z-coordinate will be zero
      Vector3f raw_data = raw_sensor_data[i];
      raw_data.x *= factor;
      raw_data.y *= factor;
      raw_data.z *= factor;
      transformed_sensor_data[i] = sensor_pose * raw_data;
      // if(i % 100 == 0)
      // {
      //     printf("factor: %f, start sensor data: x: %f, y: %f, z: %f \n new sensor data: x: %f, y: %f, z: %f\n\n",factor, start.x, start.y, start.z, transformed_sensor_data[i].x, transformed_sensor_data[i].y, transformed_sensor_data[i].z);
      // }
    }
  }
}

///* Insert static data into voxel map.
// * Data must be in world coordinate system.
// * Static data is considered 100% certain.
// */
//__global__
//void kernelInsertStaticData(Voxel* voxelmap, const uint32_t voxelmap_size, const Vector3ui* dimensions,
//                            const float voxel_side_length, uint32_t static_data_size,
//                            const Vector3f* static_data)
//{
//  const Vector3ui map_dim = (*dimensions);
//
//  for (uint32_t i = blockIdx.x * blockDim.x + threadIdx.x; (i < voxelmap_size) && (i < static_data_size);
//      i += gridDim.x * blockDim.x)
//  {
//    const Vector3ui integer_coordinates = mapToVoxels(voxel_side_length, static_data[i]);
//    if ((integer_coordinates.x < map_dim.x) && (integer_coordinates.y < map_dim.y)
//        && (integer_coordinates.z < map_dim.z))
//    {
//      Voxel* voxel = getVoxelPtr(voxelmap, dimensions, integer_coordinates.x, integer_coordinates.y,
//                                 integer_coordinates.z);
//      voxel->voxeltype = eVT_OCCUPIED;
//      voxel->occupancy = 255;
//    }
//  }
//}

///* Inserts a link of a kinematic chain into a map.
// * See also function with self collision check.
// * Kinematic data is considered 100% certain.
// */
//__global__
//void kernelInsertRobotKinematicLink(Voxel* voxelmap, const uint32_t voxelmap_size,
//                                    const Vector3ui* dimensions, const float voxel_side_length,
//                                    const MetaPointCloudStruct* robot_links, uint32_t link_nr)
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
//      //printf("inserting robot voxel to position (%u, %u, %u)\n", integer_coordinates.x, integer_coordinates.y, integer_coordinates.z);
//      voxel->voxeltype = eVT_OCCUPIED;
//      voxel->occupancy = 255;
//    }
//  }
//}

//
//void kernelCalculateBoundingBox(Voxel* voxelmap, const uint32_t voxelmap_size, )

//__global__
//void kernelInsertKinematicLinkBitvector(Voxel* voxelmap, const uint32_t voxelmap_size,
//                                        const Vector3ui* dimensions, const float voxel_side_length,
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
//void kernelClearBitvector(Voxel* voxelmap, uint32_t voxelmap_size, const Vector3ui* dimensions,
//                          uint8_t bit_number)
//{
//  const uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
//  const Vector3ui map_dim = (*dimensions);
//  if (bit_number == 0xff) //clear all bits
//  {
//    if (i < voxelmap_size)
//    {
//      Voxel* voxel = &(voxelmap[i]);
//      voxel->setBitvector(0);
//    }
//  }
//  else //clear only one bit
//  {
//    if (i < voxelmap_size)
//    {
//      Voxel* voxel = &(voxelmap[i]);
//      voxel->setBitvector(voxel->getBitvector() & (0x1 << bit_number));
//    }
//  }
//}

//__global__
//void kernelInsertRobotKinematicLinkOverwritingSensorData(Voxel* voxelmap, const uint32_t voxelmap_size,
//                                                         const Vector3ui* dimensions,
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
//      voxel->voxeltype = eVT_OCCUPIED;
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
//                                                          const Vector3ui* dimensions,
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
//        voxel->voxeltype = eVT_OCCUPIED;
//        voxel->occupancy = 255;
//      }
//    }
//  }
//}

///*! Insert a link of a kinematic chain that will be
// * treated as swept volume into the voxel map.
// * Different configurations may be identified by the
// * swept_volume_index, that is limited by values defined
// * in Voxel.h.
// * Swept volume data is considered 100% certain.
// */
//__global__
//void kernelInsertSweptVolumeConfiguration(Voxel* voxelmap, const uint32_t voxelmap_size,
//                                          const Vector3ui* dimensions, const float voxel_side_length,
//                                          uint32_t link_nr, uint32_t* point_cloud_sizes,
//                                          Vector3f** point_clouds, uint8_t swept_volume_index)
//{
//  const Vector3ui map_dim = (*dimensions);
//
//  for (uint32_t i = blockIdx.x * blockDim.x + threadIdx.x; i < point_cloud_sizes[link_nr];
//      i += gridDim.x * blockDim.x)
//  {
//    const Vector3ui integer_coordinates = mapToVoxels(voxel_side_length, point_clouds[link_nr][i]);
//    if ((integer_coordinates.x < map_dim.x) && (integer_coordinates.y < map_dim.y)
//        && (integer_coordinates.z < map_dim.z))
//    {
//      Voxel* voxel = getVoxelPtr(voxelmap, dimensions, integer_coordinates.x, integer_coordinates.y,
//                                 integer_coordinates.z);
//      voxel->voxeltype = limitSweptVolumeIndex(swept_volume_index);
//      voxel->occupancy = 255;
//    }
//  }
//}

///*! Remove swept volume from voxel map.
// * Different configurations may be identified by the
// * swept_volume_index, that is limited by values defined
// * in Voxel.h.
// */
//__global__
//void kernelRemoveSweptVolumeConfiguration(Voxel* voxelmap, const uint32_t voxelmap_size,
//                                          const Vector3ui* dimensions, const float voxel_side_length,
//                                          uint32_t link_nr, uint32_t* point_cloud_sizes,
//                                          Vector3f** point_clouds, uint8_t swept_volume_index)
//{
//  const Vector3ui map_dim = (*dimensions);
//
//  for (uint32_t i = blockIdx.x * blockDim.x + threadIdx.x; i < point_cloud_sizes[link_nr];
//      i += gridDim.x * blockDim.x)
//  {
//    const Vector3ui integer_coordinates = mapToVoxels(voxel_side_length, point_clouds[link_nr][i]);
//    if ((integer_coordinates.x < map_dim.x) && (integer_coordinates.y < map_dim.y)
//        && (integer_coordinates.z < map_dim.z))
//    {
//      Voxel* voxel = getVoxelPtr(voxelmap, dimensions, integer_coordinates.x, integer_coordinates.y,
//                                 integer_coordinates.z);
//      // check if voxel is belonging to the swept volume index
//      if (voxel->voxeltype == limitSweptVolumeIndex(swept_volume_index))
//      {
//        // if so, mark as free
//        voxel->occupancy = 0;
//      }
//    }
//  }
//}

///*! Insert data that will treated as
// * swept volume into the voxel map.
// * Different data sets may be identified by the
// * swept_volume_index, that is limited by values defined
// * in Voxel.h.
// * Swept volume data is considered 100% certain.
// */
//__global__
//void kernelInsertSweptVolume(Voxel* voxelmap, const uint32_t voxelmap_size, const Vector3ui* dimensions,
//                             uint32_t swept_volume_data_size, const Vector3f* swept_volume_data,
//                             uint8_t swept_volume_index)
//{
//  for (uint32_t i = blockIdx.x * blockDim.x + threadIdx.x; i < voxelmap_size && i < swept_volume_data_size;
//      i += gridDim.x * blockDim.x)
//  {
//    Voxel* voxel = getVoxelPtr(voxelmap, dimensions, swept_volume_data[i].x, swept_volume_data[i].y,
//                               swept_volume_data[i].z);
//    voxel->voxeltype = eVT_SWEPT_VOLUME_START + limitSweptVolumeIndex(swept_volume_index);
//    voxel->occupancy = 255;
//  }
//}

//__global__
//void kernelCollideVoxelMapsIndices(Voxel* voxelmap, uint8_t threshold, bool* results, uint32_t* index_list,
//                                   uint32_t index_number)
//{
//  __shared__ bool cache[cMAX_NR_OF_THREADS_PER_BLOCK];
//  uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
//  uint32_t cache_index = threadIdx.x;
//
//  bool temp = false;
//
//  if (i < index_number)
//  {
//    uint32_t index = index_list[i];
//    // todo / note: at the moment collision check is only used for DYNAMIC and SWEPT VOLUME type, static is used for debugging
//    temp = temp || ((voxelmap[index].occupancy >= threshold) && (voxelmap[index].voxeltype != eVT_OCCUPIED));
//
////	&& (other_map[i].occupancy_execution >= other_threshold) && (other_map[i].voxeltype_execution != eVT_OCCUPIED));
//  }
//  cache[cache_index] = temp;
//  __syncthreads();
//
//  uint32_t j = blockDim.x / 2;
//
//  while (j != 0)
//  {
//    if (cache_index < j)
//    {
//      cache[cache_index] = cache[cache_index] || cache[cache_index + j];
//    }
//    __syncthreads();
//    j /= 2;
//  }
//
//  // copy results from this block to global memory
//  if (cache_index == 0)
//  {
//    //    // FOR MEASUREMENT TEMPORARILY EDITED:
//    //    results[blockIdx.x] = true;
////		printf("blockid: %i, result: %i \n", blockIdx.x, cache[0]);
////		  if(i == 1328128)
////		  {
////			  printf("---------------------- cache: %i\n", cache[0]);
////		  }
//    results[blockIdx.x] = cache[0];
//  }
//}

//__global__
//void kernelCollideVoxelMapsIndicesBitmap(Voxel* voxelmap, uint8_t threshold, uint64_t* results,
//                                         uint32_t* index_list, uint32_t index_number, uint64_t* bitmap_list,
//                                         Vector3ui* dimensions)
//{
//  __shared__ uint64_t cache[cMAX_NR_OF_THREADS_PER_BLOCK];
//  uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
//  uint32_t cache_index = threadIdx.x;
//
//  uint64_t temp = 0;
//
//  uint32_t index = index_list[i];
//  uint32_t temp_index = index;
//
//  uint32_t z = index / (dimensions->x * dimensions->y); //testing whether voxel lies in map  Vector3ui integer_coordinates. First getting position in map
//  uint32_t y = (temp_index -= z * (dimensions->x * dimensions->y)) / dimensions->x;
//  uint32_t x = (temp_index -= y * dimensions->x);
//
//  if (i < index_number)
//  {
//
//    if (!((x < dimensions->x) && (y < dimensions->y) && (z < dimensions->z)))
//    {
//      temp = ~temp; //temp was 0 here, now it is 0xfff... => Collision because outside of map
//      printf("voxel %i out of map\n", i);
//    }
//    else
//    {
//      if (voxelmap[index].occupancy >= threshold)
//      {
//        temp |= bitmap_list[i];
//        voxelmap[index].voxeltype = 1; // This value marks a collision at this voxel, should be set to zero by the visualizer
//      }
//    }
//  }
//  cache[cache_index] = temp;
//  __syncthreads();
//
//  uint32_t j = blockDim.x / 2;
//
//  while (j != 0)
//  {
//    if (cache_index < j)
//    {
//      cache[cache_index] = cache[cache_index] | cache[cache_index + j];
//    }
//    __syncthreads();
//    j /= 2;
//  }
////
////	  // copy results from this block to global memory
//  if (cache_index == 0)
//  {
//    //    // FOR MEASUREMENT TEMPORARILY EDITED:
//    //    results[blockIdx.x] = true;
////		printf("blockid: %i, result: %i \n", blockIdx.x, cache[0]);
////		  if(cache[0] != 0)
////		  {
////			  printf("---------------------- cache: %i\n", cache[0]);
////		  }
//    results[blockIdx.x] = cache[0];
//  }
//}

//__global__
//void kernelCollideVoxelMapsBoundingBox(Voxel* voxelmap, const uint32_t voxelmap_size, const uint8_t threshold,
//                                       Voxel* other_map, const uint8_t other_threshold, bool* results,
//                                       uint32_t offset_x, uint32_t offset_y, uint32_t offset_z,
//                                       uint32_t size_x, Vector3ui* dimensions)
//{
////  extern __shared__ bool cache[];//[cMAX_NR_OF_THREADS_PER_BLOCK];			//define Cache size in kernel call
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
//    // todo / note: at the moment collision check is only used for DYNAMIC and SWEPT VOLUME type, static is used for debugging
//    temp = temp
//        || ((voxelmap[i].occupancy >= threshold) && (voxelmap[i].voxeltype != eVT_OCCUPIED)
//            && (other_map[i].occupancy >= other_threshold) && (other_map[i].voxeltype != eVT_OCCUPIED));
//
//    counter += 1;
//    i += 1;
////      i += blockDim.x * gridDim.x;
//  }
//
////  if(true)//(i == 30050600)
////  {
////	  printf("thread %i, collision %i \n", i, temp);
////	  printf("--- occupation planning: %i, voxeltype planning: %i \n",
////			  other_map[i].occupancy_planning, other_map[i].voxeltype_planning);
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

///* Collide two voxel maps without plain parallelization.
// * According to hardware this may increase performance.
// * The performance increase is dependent on loop_size.
// * Voxels are considered occupied for values
// * greater or equal given thresholds.
// */
//__global__
//void kernelCollideVoxelMapsAlternative(Voxel* voxelmap, const uint32_t voxelmap_size, const uint8_t threshold,
//                                       Voxel* other_map, const uint8_t other_threshold, uint32_t loop_size,
//                                       bool* results)
//{
//  __shared__ bool cache[cMAX_NR_OF_THREADS_PER_BLOCK];
//  const uint32_t i = (blockIdx.x * blockDim.x + threadIdx.x) * loop_size;
//  uint32_t cache_index = threadIdx.x;
//  cache[cache_index] = false;
//  bool temp = false;
//
//// ToDo: replace if with while and increment:     i += blockDim.x * gridDim.x;
//  if (i < voxelmap_size)
//  {
//    for (uint32_t k = 0; (k < loop_size && (i + k < voxelmap_size)); k++)
//    {
//      temp = temp
//          || ((voxelmap[i + k].occupancy >= threshold) && (other_map[i + k].occupancy >= other_threshold));
//    }
//
//    cache[cache_index] = temp;
//  }
//  __syncthreads();
//
//  uint32_t j = blockDim.x / 2;
//
//  while (j != 0)
//  {
//    if (cache_index < j)
//    {
//      cache[cache_index] = cache[cache_index] || cache[cache_index + j];
//    }
//    __syncthreads();
//    j /= 2;
//  }
//
//// copy results from this block to global memory
//  if (cache_index == 0)
//  {
//    // FOR MEASUREMENT TEMPORARILY EDITED:
//    //results[blockIdx.x] = true;
//    results[blockIdx.x] = cache[0];
//  }
//}


//__global__
//void kernelInsertBox(Voxel* voxelmap, const uint32_t voxelmap_size, const Vector3ui* dimensions,
//                     const uint32_t from_voxel_x, const uint32_t from_voxel_y, const uint32_t from_voxel_z,
//                     const uint32_t to_voxel_x, const uint32_t to_voxel_y, const uint32_t to_voxel_z,
//                     uint8_t voxeltype, uint8_t occupancy)
//{
//  for (uint32_t i = blockIdx.x * blockDim.x + threadIdx.x; i < voxelmap_size; i += gridDim.x * blockDim.x)
//  {
//    Voxel* voxel = &(voxelmap[i]);
//    Vector3ui voxel_coordinates = mapToVoxels(voxelmap, dimensions, voxel);
//
//    // check if voxel is within bounding box
//    if ((voxel_coordinates.x >= from_voxel_x) && (voxel_coordinates.y >= from_voxel_y)
//        && (voxel_coordinates.z >= from_voxel_z) && (voxel_coordinates.x <= to_voxel_x)
//        && (voxel_coordinates.y <= to_voxel_y) && (voxel_coordinates.z <= to_voxel_z))
//    {
//      voxel->voxeltype = voxeltype;
//      voxel->occupancy = occupancy;
//    }
//  }
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

//__global__
//void kernelInsertVoxelVector(Voxel* destination_map, uint32_t* voxel_list, uint32_t list_size)
//{
//  const uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
//  if (i < list_size)
//  {
//    Voxel* v = destination_map + voxel_list[i];
//    v->voxeltype = eVT_OCCUPIED;
//    v->occupancy = 255;
//  }
//}

//__global__
//void kernelInsertVoxelVectorBitmap(Voxel* destination_map, uint32_t* voxel_list, uint32_t list_size,
//                                   uint64_t mask)
//{
//  const uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
//  if (i < list_size)
//  {
//    Voxel* v = destination_map + voxel_list[i];
//    v->setBitvector(v->getBitvector() | mask);
//  }
//}

//inserts a bitvector by the indices
//__global__
//void kernelInsertBitmapByIndices(Voxel* destination_map, uint32_t* voxel_list, uint32_t list_size,
//                                 uint64_t* bitvector)
//{
//  const uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
//  if (i < list_size)
//  {
//    Voxel* v = destination_map + voxel_list[i];
//    v->setBitvector(bitvector[i]);
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
//    uint8_t voxeltype = 0;
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
//            voxeltype = index_x->voxeltype;
//          }
//          index_x += 1;
//        }
//        index_y += source_map_dim->x;
//      }
//      index_z += source_map_dim->y * source_map_dim->x;
//    }
//    v->occupancy = occupancy;
//    v->voxeltype = voxeltype;
//  }
//}

} // end of namespace voxelmap
} // end of namespace voxellist

#ifdef LOCAL_DEBUG
#undef LOCAL_DEBUG
#endif
