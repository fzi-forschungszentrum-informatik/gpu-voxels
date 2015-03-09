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
#include <gpu_voxels/voxelmap/BitVoxel.hpp>

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
//      voxelmap[i].voxeltype = eVT_COLLISION;
//      voxelmap[i].occupancy = 255;
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
 * Collision info is stored within eVT_COLLISION model for 'other_map'.
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
    // todo / note: at the moment collision check is only used for DYNAMIC and SWEPT VOLUME type, static is used for debugging
    const bool collision = collider.collide(voxelmap[i], other_map[i]);
    if (collision) // store collision info
    {
//#ifndef DISABLE_STORING_OF_COLLISIONS
//      other_map[i].occupancy = 255;
//      other_map[i].voxeltype = eVT_COLLISION;
//#endif
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
void kernelCollideVoxelMapsBitvector(ProbabilisticVoxel* voxelmap, const uint32_t voxelmap_size,
                                     BitVoxel<length>* other_map, uint32_t loop_size, Collider collider,
                                     BitVector<length>* results)
{
  extern __shared__ BitVector<length> cache[]; //[cMAX_NR_OF_THREADS_PER_BLOCK];
  const uint32_t i = (blockIdx.x * blockDim.x + threadIdx.x) * loop_size;
  uint32_t cache_index = threadIdx.x;
  cache[cache_index] = BitVector<length>();
  BitVector<length> temp;

  // ToDo: replace if with while and increment:     i += blockDim.x * gridDim.x;
  if (i < voxelmap_size)
  {
    for (uint32_t k = 0; (k < loop_size && (i + k < voxelmap_size)); k++)
    {
      if (collider.collide(voxelmap[i + k]))
      {
        temp |= other_map[i + k].bitVector;
      }
    }

    cache[cache_index] = temp;
  }
  __syncthreads();

  uint32_t j = blockDim.x / 2;

  while (j != 0)
  {
    if (cache_index < j)
    {
      cache[cache_index] = cache[cache_index] | cache[cache_index + j];
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
  }
}

template<class Voxel>
__global__
void kernelInsertGlobalPointCloud(Voxel* voxelmap, const Vector3ui *map_dim, const float voxel_side_length,
                                  Vector3f* points, const std::size_t sizePoints, const uint32_t voxel_type)
{

  const Vector3ui dimensions = (*map_dim);
  for (uint32_t i = blockIdx.x * blockDim.x + threadIdx.x; i < sizePoints; i += blockDim.x * gridDim.x)
  {
    const Vector3ui integer_coordinates = mapToVoxels(voxel_side_length, points[i]);
    //check if point is in the range of the voxel map
    if ((integer_coordinates.x < dimensions.x) && (integer_coordinates.y < dimensions.y)
        && (integer_coordinates.z < dimensions.z))
    {
      Voxel* voxel = &voxelmap[getVoxelIndex(map_dim, integer_coordinates.x, integer_coordinates.y,
                                             integer_coordinates.z)];
      voxel->insert(voxel_type);
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
                                VoxelType voxelType, const Vector3ui *map_dim, const float voxel_side_length)
{
  const Vector3ui dimensions = (*map_dim);
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
      Voxel* voxel = &voxelmap[getVoxelIndex(map_dim, integer_coordinates.x, integer_coordinates.y,
                                             integer_coordinates.z)];
      voxel->insert(voxelType);

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
                                VoxelType* voxel_types, const Vector3ui *map_dim,
                                const float voxel_side_length)
{
  const Vector3ui dimensions = (*map_dim);
  for (uint32_t i = blockIdx.x * blockDim.x + threadIdx.x; i < meta_point_cloud->accumulated_cloud_size;
      i += blockDim.x * gridDim.x)
  {
    uint32_t voxel_type_index = i / meta_point_cloud->cloud_sizes[0];
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
      Voxel* voxel = &voxelmap[getVoxelIndex(map_dim, integer_coordinates.x, integer_coordinates.y,
                                             integer_coordinates.z)];
      voxel->insert(voxel_types[voxel_type_index]);

//        printf("Inserted Point @(%u,%u,%u) with type %u into the voxel map \n",
//               integer_coordinates.x,
//               integer_coordinates.y,
//               integer_coordinates.z,
//               voxel_types[voxel_type_index]);

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
//                                          const Vector3ui* dimensions, const float voxel_side_length,
//                                          Sensor* sensor, const Vector3f* sensor_data,
//                                          const bool cut_real_robot, BitVoxel<length>* robotmap,
//                                          const uint32_t bit_index)
//{
//  const Vector3ui map_dim = (*dimensions);
//
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
//      if ((integer_coordinates.x < map_dim.x) && (integer_coordinates.y < map_dim.y)
//          && (integer_coordinates.z < map_dim.z) && (sensor_coordinates.x < map_dim.x)
//          && (sensor_coordinates.y < map_dim.y) && (sensor_coordinates.z < map_dim.z))
//      {
//        bool update = false;
//        if (robotmap && cut_real_robot)
//        {
//          BitVoxel<length>* robot_voxel = getVoxelPtr(robotmap, dimensions, integer_coordinates.x,
//                                                      integer_coordinates.y, integer_coordinates.z);
//
////          if (!((robot_voxel->occupancy > 0) && (robot_voxel->voxeltype == eVT_OCCUPIED))) // not occupied by robot
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
////            voxel->voxeltype = eVT_OCCUPIED;
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
                            const Vector3ui* dimensions, const float voxel_side_length, Sensor* sensor,
                            const Vector3f* sensor_data, const bool cut_real_robot,
                            BitVoxel<length>* robotmap, const uint32_t bit_index, RayCasting rayCaster)
{
  const Vector3ui map_dim = (*dimensions);

  for (uint32_t i = blockIdx.x * blockDim.x + threadIdx.x; (i < voxelmap_size) && (i < sensor->data_size);
      i += gridDim.x * blockDim.x)
  {
    if (!(isnan(sensor_data[i].x) || isnan(sensor_data[i].y) || isnan(sensor_data[i].z)))
    {
      const Vector3ui integer_coordinates = mapToVoxels(voxel_side_length, sensor_data[i]);
      const Vector3ui sensor_coordinates = mapToVoxels(voxel_side_length, sensor->position);

      /* both data and sensor coordinates must
       be within boundaries for raycasting to work */
      if ((integer_coordinates.x < map_dim.x) && (integer_coordinates.y < map_dim.y)
          && (integer_coordinates.z < map_dim.z) && (sensor_coordinates.x < map_dim.x)
          && (sensor_coordinates.y < map_dim.y) && (sensor_coordinates.z < map_dim.z))
      {
        bool update = false;
        if (cut_real_robot)
        {
          BitVoxel<length>* robot_voxel = getVoxelPtr(robotmap, dimensions, integer_coordinates.x,
                                                      integer_coordinates.y, integer_coordinates.z);

          //          if (!((robot_voxel->occupancy > 0) && (robot_voxel->voxeltype == eVT_OCCUPIED))) // not occupied by robot
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
          //            voxel->voxeltype = eVT_OCCUPIED;
          //            increaseOccupancy(voxel, cSENSOR_MODEL_OCCUPIED); // todo: replace with "occupied" of sensor model
        }
      }
    }
  }
}

template<class Voxel>
__global__
void kernelAddressingTest(const Voxel* voxelmap, const Vector3ui* dimensions, const float *voxel_side_length,
                          const Vector3f *testpoint, bool* success)
{
//  const uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;

  Vector3ui test_ccords = mapToVoxels(*voxel_side_length, *testpoint);
  Voxel* testvoxel = getVoxelPtr(voxelmap, dimensions, test_ccords.x, test_ccords.y, test_ccords.z);
  Vector3ui int_coords = mapToVoxels(voxelmap, dimensions, testvoxel);
  Vector3f center = getVoxelCenter(voxelmap, dimensions, *voxel_side_length, &int_coords);

//  printf("TestCoord    (%f,%f,%f)\n",testpoint->x, testpoint->y, testpoint->z);
//  printf("TestIntCoord (%d,%d,%d)\n",int_coords.x, int_coords.y, int_coords.z);
//  printf("ReturnCoord  (%f,%f,%f)\n",center.x, center.y, center.z);

  if ((abs(center.x - testpoint->x) > *voxel_side_length / 2.0)
      || (abs(center.y - testpoint->y) > *voxel_side_length / 2.0)
      || (abs(center.z - testpoint->z) > *voxel_side_length / 2.0))
  {
    *success = false;
  }
  else
  {
    *success = true;
  }
}


} // end of namespace voxelmap
} // end of namespace gpu_voxels
#endif
