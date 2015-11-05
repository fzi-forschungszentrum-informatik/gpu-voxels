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
 * \author  Sebastian Klemm
 * \date    2012-09-13
 *
 */
//----------------------------------------------------------------------
#ifndef ICL_PLANNING_GPU_KERNELS_VOXELMAP_OPERATIONS_HPP_INCLUDED
#define ICL_PLANNING_GPU_KERNELS_VOXELMAP_OPERATIONS_HPP_INCLUDED

#include "VoxelMapOperations.h"
#include <gpu_voxels/voxel/BitVoxel.hpp>

namespace gpu_voxels {
namespace voxelmap {

template<class Voxel>
__global__
void kernelClearVoxelMap(Voxel* voxelmap, const uint32_t voxelmap_size)
{
  for (uint32_t i = blockIdx.x * blockDim.x + threadIdx.x; i < voxelmap_size; i += gridDim.x * blockDim.x)
  {
    voxelmap[i] = Voxel();
  }
}

template<std::size_t bit_length>
__global__
void kernelClearVoxelMap(BitVoxel<bit_length>* voxelmap, const uint32_t voxelmap_size, const uint32_t bit_index)
{
  for (uint32_t i = blockIdx.x * blockDim.x + threadIdx.x; i < voxelmap_size; i += gridDim.x * blockDim.x)
  {
    BitVector<bit_length>& bit_vector = voxelmap[i].bitVector();
    if (bit_vector.getBit(bit_index))
      bit_vector.clearBit(bit_index);
  }
}

template<std::size_t bit_length>
__global__
void kernelClearVoxelMap(BitVoxel<bit_length>* voxelmap, uint32_t voxelmap_size,
                         BitVector<bit_length> bits)
{
  for (uint32_t i = blockIdx.x * blockDim.x + threadIdx.x; i < voxelmap_size; i += gridDim.x * blockDim.x)
  {
    BitVector<bit_length>& bit_vector = voxelmap[i].bitVector();
    if ((!(bit_vector & bits).isZero()))
    {
      BitVector<bit_length> tmp = bit_vector;
      tmp = tmp & (~bits);
      bit_vector = tmp;
    }
  }
}

/*! Collide two voxel maps.
 * Voxels are considered occupied for values
 * greater or equal given thresholds.
 */
template<class Voxel, class OtherVoxel, class Collider>
__global__
void kernelCollideVoxelMaps(Voxel* voxelmap, const uint32_t voxelmap_size, OtherVoxel* other_map,
                            Collider collider, bool* results)
{
  __shared__ bool cache[cMAX_NR_OF_THREADS_PER_BLOCK];
  uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
  uint32_t cache_index = threadIdx.x;
  cache[cache_index] = false;
  bool temp = false;

  while (i < voxelmap_size)
  {
    if (collider.collide(voxelmap[i], other_map[i]))
    {
      temp = true;
    }
    i += blockDim.x * gridDim.x;
  }

  cache[cache_index] = temp;
  __syncthreads();

  uint32_t j = blockDim.x / 2;

  while (j != 0)
  {
    if (cache_index < j)
    {
      cache[cache_index] = cache[cache_index] || cache[cache_index + j];
    }
    __syncthreads();
    j /= 2;
  }

// copy results from this block to global memory
  if (cache_index == 0)
  {
//    // FOR MEASUREMENT TEMPORARILY EDITED:
//    results[blockIdx.x] = true;
    results[blockIdx.x] = cache[0];
  }
}

/* Collide two voxel maps with storing collision info (for debugging only)
 * Voxels are considered occupied for values
 * greater or equal given thresholds.
 *
 * Collision info is stored within eBVM_COLLISION model for 'other_map'.
 * Warning: Original model is modified!
 */
template<class Voxel, class OtherVoxel, class Collider>
__global__
void kernelCollideVoxelMapsDebug(Voxel* voxelmap, const uint32_t voxelmap_size, OtherVoxel* other_map,
                                 Collider collider, uint16_t* results)
{
//#define DISABLE_STORING_OF_COLLISIONS
  __shared__ uint16_t cache[cMAX_NR_OF_THREADS_PER_BLOCK];
  uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
  uint32_t cache_index = threadIdx.x;
  cache[cache_index] = 0;

  while (i < voxelmap_size)
  {
    // todo / note: at the moment collision check is only used for DYNAMIC and SWEPT VOLUME meaning, static is used for debugging
    const bool collision = collider.collide(voxelmap[i], other_map[i]);
    if (collision) // store collision info
    {
#ifndef DISABLE_STORING_OF_COLLISIONS
//      other_map[i].occupancy = 255;
//      other_map[i].insert(eBVM_COLLISION); // sets m_occupancy = MAX_PROBABILITY for prob voxels
      voxelmap[i].insert(eBVM_COLLISION); // sets m_occupancy = MAX_PROBABILITY for prob voxels
#endif
      cache[cache_index] += 1;
    }
    i += blockDim.x * gridDim.x;
  }

  // debug: print collision coordinates

//  if (temp)
//  {
//    Vector3ui col_coord = mapToVoxels(voxelmap, dimensions, &(voxelmap[i]));
//    printf("Collision at voxel (%u) = (%u, %u, %u). Memory addresses are %p and %p.\n",
//           i, col_coord.x, col_coord.y, col_coord.z, (void*)&(voxelmap[i]), (void*)&(other_map[i]));
//  }
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
#undef DISABLE_STORING_OF_COLLISIONS
}

template<std::size_t length, class Collider>
__global__
void kernelCollideVoxelMapsBitvector(BitVoxel<length>* voxelmap, const uint32_t voxelmap_size,
                                     BitVoxel<length>* other_map, Collider collider,
                                     BitVector<length>* results, uint16_t* num_collisions, const uint16_t sv_offset)
{
  extern __shared__ BitVector<length> cache[]; //[cMAX_NR_OF_THREADS_PER_BLOCK];
  __shared__ uint16_t cache_num[cMAX_NR_OF_THREADS_PER_BLOCK];
  uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
  uint32_t cache_index = threadIdx.x;
  cache[cache_index] = BitVector<length>();
  cache_num[cache_index] = 0;
  BitVector<length> temp;

  while (i < voxelmap_size)
  {
    const bool collision = collider.collide(voxelmap[i], other_map[i], &temp, sv_offset);
    if (collision) // store collision info
    {
      #ifndef DISABLE_STORING_OF_COLLISIONS
      //      other_map[i].occupancy = 255;
      //      other_map[i].insert(eBVM_COLLISION); // sets m_occupancy = MAX_PROBABILITY for prob voxels
            voxelmap[i].insert(eBVM_COLLISION); // sets m_occupancy = MAX_PROBABILITY for prob voxels
      #endif
      cache[cache_index] = temp;
      cache_num[cache_index] += 1;
    }
    i += blockDim.x * gridDim.x;
  }
  __syncthreads();

  uint32_t j = blockDim.x / 2;

  while (j != 0)
  {
    if (cache_index < j)
    {
      cache[cache_index] = cache[cache_index] | cache[cache_index + j];
      cache_num[cache_index] = cache_num[cache_index] + cache_num[cache_index + j];
    }
    __syncthreads();
    j /= 2;
  }

  // copy results from this block to global memory
  if (cache_index == 0)
  {
    // FOR MEASUREMENT TEMPORARILY EDITED:
    //results[blockIdx.x] = true;
    results[blockIdx.x] = cache[0];
    num_collisions[blockIdx.x] = cache_num[0];
  }
}

template<class Voxel>
__global__
void kernelInsertGlobalPointCloud(Voxel* voxelmap, const Vector3ui dimensions, const float voxel_side_length,
                                  Vector3f* points, const std::size_t sizePoints, const BitVoxelMeaning voxel_meaning)
{
  for (uint32_t i = blockIdx.x * blockDim.x + threadIdx.x; i < sizePoints; i += blockDim.x * gridDim.x)
  {
    const Vector3ui integer_coordinates = mapToVoxels(voxel_side_length, points[i]);
    //check if point is in the range of the voxel map
    if ((integer_coordinates.x < dimensions.x) && (integer_coordinates.y < dimensions.y)
        && (integer_coordinates.z < dimensions.z))
    {
      Voxel* voxel = &voxelmap[getVoxelIndex(dimensions, integer_coordinates.x, integer_coordinates.y,
                                             integer_coordinates.z)];
      voxel->insert(voxel_meaning);
    }
    else
    {
      printf("Point (%u,%u,%u) is not in the range of the voxel map \n", points[i].x, points[i].y,
             points[i].z);
    }
  }
}

template<class Voxel>
__global__
void kernelInsertMetaPointCloud(Voxel* voxelmap, const MetaPointCloudStruct* meta_point_cloud,
                                BitVoxelMeaning voxel_meaning, const Vector3ui dimensions, const float voxel_side_length)
{
  for (uint32_t i = blockIdx.x * blockDim.x + threadIdx.x; i < meta_point_cloud->accumulated_cloud_size;
      i += blockDim.x * gridDim.x)
  {
    const Vector3ui integer_coordinates = mapToVoxels(voxel_side_length,
                                                      meta_point_cloud->clouds_base_addresses[0][i]);

//        printf("Point @(%f,%f,%f)\n",
//               meta_point_cloud->clouds_base_addresses[0][i].x,
//               meta_point_cloud->clouds_base_addresses[0][i].y,
//               meta_point_cloud->clouds_base_addresses[0][i].z);

    //check if point is in the range of the voxel map
    if ((integer_coordinates.x < dimensions.x) && (integer_coordinates.y < dimensions.y)
        && (integer_coordinates.z < dimensions.z))
    {
      Voxel* voxel = &voxelmap[getVoxelIndex(dimensions, integer_coordinates.x, integer_coordinates.y,
                                             integer_coordinates.z)];
      voxel->insert(voxel_meaning);

//        printf("Inserted Point @(%u,%u,%u) into the voxel map \n",
//               integer_coordinates.x,
//               integer_coordinates.y,
//               integer_coordinates.z);

    }
    else
    {
      printf("Point (%f,%f,%f) is not in the range of the voxel map \n",
             meta_point_cloud->clouds_base_addresses[0][i].x, meta_point_cloud->clouds_base_addresses[0][i].y,
             meta_point_cloud->clouds_base_addresses[0][i].z);
    }
  }
}

template<class Voxel>
__global__
void kernelInsertMetaPointCloud(Voxel* voxelmap, const MetaPointCloudStruct* meta_point_cloud,
                                BitVoxelMeaning* voxel_meanings, const Vector3ui dimensions,
                                const float voxel_side_length)
{
  u_int16_t sub_cloud = 0;
  u_int32_t sub_cloud_upper_bound = meta_point_cloud->cloud_sizes[sub_cloud];

  for (uint32_t i = blockIdx.x * blockDim.x + threadIdx.x; i < meta_point_cloud->accumulated_cloud_size;
      i += blockDim.x * gridDim.x)
  {
    // find out, to which sub_cloud our point belongs
    while(i >= sub_cloud_upper_bound)
    {
      sub_cloud++;
      sub_cloud_upper_bound += meta_point_cloud->cloud_sizes[sub_cloud];
    }


    const Vector3ui integer_coordinates = mapToVoxels(voxel_side_length,
                                                      meta_point_cloud->clouds_base_addresses[0][i]);

//        printf("Point @(%f,%f,%f)\n",
//               meta_point_cloud->clouds_base_addresses[0][i].x,
//               meta_point_cloud->clouds_base_addresses[0][i].y,
//               meta_point_cloud->clouds_base_addresses[0][i].z);

    //check if point is in the range of the voxel map
    if ((integer_coordinates.x < dimensions.x) && (integer_coordinates.y < dimensions.y)
        && (integer_coordinates.z < dimensions.z))
    {
      Voxel* voxel = &voxelmap[getVoxelIndex(dimensions, integer_coordinates.x, integer_coordinates.y,
                                             integer_coordinates.z)];
      voxel->insert(voxel_meanings[sub_cloud]);

//        printf("Inserted Point @(%u,%u,%u) with meaning %u into the voxel map \n",
//               integer_coordinates.x,
//               integer_coordinates.y,
//               integer_coordinates.z,
//               voxel_meanings[voxel_meaning_index]);

    }
    else
    {
      printf("Point (%f,%f,%f) is not in the range of the voxel map \n",
             meta_point_cloud->clouds_base_addresses[0][i].x, meta_point_cloud->clouds_base_addresses[0][i].y,
             meta_point_cloud->clouds_base_addresses[0][i].z);
    }
  }
}

//template<std::size_t length>
//__global__
//void kernelInsertSensorDataWithRayCasting(ProbabilisticVoxel* voxelmap, const uint32_t voxelmap_size,
//                                          const Vector3ui dimensions, const float voxel_side_length,
//                                          Sensor* sensor, const Vector3f* sensor_data,
//                                          const bool cut_real_robot, BitVoxel<length>* robotmap,
//                                          const uint32_t bit_index)
//{
//  for (uint32_t i = blockIdx.x * blockDim.x + threadIdx.x; (i < voxelmap_size) && (i < sensor->data_size);
//      i += gridDim.x * blockDim.x)
//  {
//    if (!(isnan(sensor_data[i].x) || isnan(sensor_data[i].y) || isnan(sensor_data[i].z)))
//    {
//      const Vector3ui integer_coordinates = mapToVoxels(voxel_side_length, sensor_data[i]);
//      const Vector3ui sensor_coordinates = mapToVoxels(voxel_side_length, sensor->position);
//
//      /* both data and sensor coordinates must
//       be within boundaries for raycasting to work */
//      if ((integer_coordinates.x < dimensions.x) && (integer_coordinates.y < dimensions.y)
//          && (integer_coordinates.z < dimensions.z) && (sensor_coordinates.x < dimensions.x)
//          && (sensor_coordinates.y < dimensions.y) && (sensor_coordinates.z < dimensions.z))
//      {
//        bool update = false;
//        if (robotmap && cut_real_robot)
//        {
//          BitVoxel<length>* robot_voxel = getVoxelPtr(robotmap, dimensions, integer_coordinates.x,
//                                                      integer_coordinates.y, integer_coordinates.z);
//
////          if (!((robot_voxel->occupancy > 0) && (robot_voxel->voxelmeaning == eBVM_OCCUPIED))) // not occupied by robot
////           {
//          update = !robot_voxel->bitVector().getBit(bit_index); // not occupied by robot
////          else // else: sensor sees robot, no need to insert data.
////          {
////            printf("cutting robot from sensor data in kernel %u\n", i);
////          }
//        }
//        else
//          update = true;
//
//        if (update)
//        {
//          // sensor does not see robot, so insert data into voxelmap
//          // raycasting
//          rayCast(voxelmap, dimensions, sensor, sensor_coordinates, integer_coordinates);
//
//          // insert measured data itself:
//          ProbabilisticVoxel* voxel = getVoxelPtr(voxelmap, dimensions, integer_coordinates.x,
//                                                  integer_coordinates.y, integer_coordinates.z);
//          voxel->updateOccupancy(cSENSOR_MODEL_OCCUPIED);
////            voxel->voxelmeaning = eBVM_OCCUPIED;
////            increaseOccupancy(voxel, cSENSOR_MODEL_OCCUPIED); // todo: replace with "occupied" of sensor model
//        }
//      }
//    }
//  }
//}

/* Insert sensor data into voxel map.
 * Assumes sensor data is already transformed
 * into world coordinate system.
 * If cut_real_robot is enabled one has to
 * specify pointer to the robot voxel map.
 * The robot voxels will be assumed 100% certain
 * and cut from sensor data.
 * See also function with ray casting.
 */
template<std::size_t length, class RayCasting>
__global__
void kernelInsertSensorData(ProbabilisticVoxel* voxelmap, const uint32_t voxelmap_size,
                            const Vector3ui dimensions, const float voxel_side_length, Sensor* sensor,
                            const Vector3f* sensor_data, const bool cut_real_robot,
                            BitVoxel<length>* robotmap, const uint32_t bit_index, RayCasting rayCaster)
{
  for (uint32_t i = blockIdx.x * blockDim.x + threadIdx.x; (i < voxelmap_size) && (i < sensor->data_size);
      i += gridDim.x * blockDim.x)
  {
    if (!(isnan(sensor_data[i].x) || isnan(sensor_data[i].y) || isnan(sensor_data[i].z)))
    {
      const Vector3ui integer_coordinates = mapToVoxels(voxel_side_length, sensor_data[i]);
      const Vector3ui sensor_coordinates = mapToVoxels(voxel_side_length, sensor->position);

      /* both data and sensor coordinates must
       be within boundaries for raycasting to work */
      if ((integer_coordinates.x < dimensions.x) && (integer_coordinates.y < dimensions.y)
          && (integer_coordinates.z < dimensions.z) && (sensor_coordinates.x < dimensions.x)
          && (sensor_coordinates.y < dimensions.y) && (sensor_coordinates.z < dimensions.z))
      {
        bool update = false;
        if (cut_real_robot)
        {
          BitVoxel<length>* robot_voxel = getVoxelPtr(robotmap, dimensions, integer_coordinates.x,
                                                      integer_coordinates.y, integer_coordinates.z);

          //          if (!((robot_voxel->occupancy > 0) && (robot_voxel->voxelmeaning == eBVM_OCCUPIED))) // not occupied by robot
          //           {
          update = !robot_voxel->bitVector().getBit(bit_index); // not occupied by robot
          //          else // else: sensor sees robot, no need to insert data.
          //          {
          //            printf("cutting robot from sensor data in kernel %u\n", i);
          //          }
        }
        else
          update = true;

        if (update)
        {
          // sensor does not see robot, so insert data into voxelmap
          // raycasting
          rayCaster.rayCast(voxelmap, dimensions, sensor, sensor_coordinates, integer_coordinates);

          // insert measured data itself:
          ProbabilisticVoxel* voxel = getVoxelPtr(voxelmap, dimensions, integer_coordinates.x,
                                                  integer_coordinates.y, integer_coordinates.z);
          voxel->updateOccupancy(cSENSOR_MODEL_OCCUPIED);
          //            voxel->voxelmeaning = eBVM_OCCUPIED;
          //            increaseOccupancy(voxel, cSENSOR_MODEL_OCCUPIED); // todo: replace with "occupied" of sensor model
        }
      }
    }
  }
}

template<class Voxel>
__global__
void kernelAddressingTest(const Voxel* voxelmap_base_address, const Vector3ui dimensions, const float voxel_side_length,
                          const Vector3f *testpoints, const size_t testpoints_size, bool* success)
{
  for (uint32_t i = blockIdx.x * blockDim.x + threadIdx.x; i < testpoints_size; i += gridDim.x * blockDim.x)
  {
    Vector3ui test_ccords = mapToVoxels(voxel_side_length, testpoints[i]);
    Voxel* testvoxel = getVoxelPtr(voxelmap_base_address, dimensions, test_ccords.x, test_ccords.y, test_ccords.z);
    Vector3ui int_coords = mapToVoxels(voxelmap_base_address, dimensions, testvoxel);
    Vector3f center = getVoxelCenter(voxel_side_length, int_coords);

  //  printf("TestCoord    (%f,%f,%f)\n",testpoints[i].x, testpoints[i].y, testpoints[i].z);
  //  printf("TestIntCoord (%d,%d,%d)\n",int_coords.x, int_coords.y, int_coords.z);
  //  printf("ReturnCoord  (%f,%f,%f)\n",center.x, center.y, center.z);

    if ((abs(center.x - testpoints[i].x) > voxel_side_length / 2.0) ||
        (abs(center.y - testpoints[i].y) > voxel_side_length / 2.0) ||
        (abs(center.z - testpoints[i].z) > voxel_side_length / 2.0))
    {
      *success = false;
    }
  }
}

template<std::size_t length>
__global__
void kernelShiftBitVector(BitVoxel<length>* voxelmap,
                          const uint32_t voxelmap_size, uint8_t shift_size)
{
  for (uint32_t i = blockIdx.x * blockDim.x + threadIdx.x; i < voxelmap_size; i += gridDim.x * blockDim.x)
  {
    performLeftShift(voxelmap[i].bitVector(), shift_size);
  }
}


} // end of namespace voxelmap
} // end of namespace gpu_voxels
#endif
