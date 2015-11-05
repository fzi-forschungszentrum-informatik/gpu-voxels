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
#ifndef ICL_PLANNING_GPU_KERNELS_VOXELMAP_OPERATIONS_H_INCLUDED
#define ICL_PLANNING_GPU_KERNELS_VOXELMAP_OPERATIONS_H_INCLUDED

#include <cuda_runtime.h>
#include <gpu_voxels/helpers/cuda_datatypes.h>
#include <gpu_voxels/voxel/BitVoxel.h>
#include <gpu_voxels/voxel/ProbabilisticVoxel.h>

namespace gpu_voxels {
namespace voxelmap {

/* ------------------ Temporary Sensor Model ------------ */
static const probability cSENSOR_MODEL_FREE = -10;
static const probability cSENSOR_MODEL_OCCUPIED = 72;

const uint32_t cMAX_NR_OF_THREADS_PER_BLOCK = 1024;

/* ------------------ DEVICE FUNCTIONS ------------------ */

// VoxelMap addressing
//! Maps discrete voxel coordinates to a voxel address
__device__      __forceinline__
uint32_t getVoxelIndex(const Vector3ui &dimensions, const uint32_t x,
                                                  const uint32_t y, const uint32_t z)
{
  return (z * dimensions.x * dimensions.y + y * dimensions.x + x);
}

//! Maps discrete voxel coordinates to a voxel adress
__host__ __device__      __forceinline__
uint32_t getVoxelIndex(const Vector3ui &dimensions, const Vector3ui &coords)
{
  return (coords.z * dimensions.x * dimensions.y + coords.y * dimensions.x + coords.x);
}

template<class Voxel>
__device__      __forceinline__
Voxel* getVoxelPtr(const Voxel* voxelmap, const Vector3ui &dimensions,
                   const uint32_t x, const uint32_t y, const uint32_t z)
{
  return (Voxel*) (voxelmap + getVoxelIndex(dimensions, x, y, z));
}

//! Maps a voxel address to discrete voxel coordinates
template<class Voxel>
__device__ __forceinline__
Vector3ui mapToVoxels(const Voxel* voxelmap, const Vector3ui &dimensions,
                      const Voxel* voxel)
{
  Vector3ui integer_coordinates;
  int voxel_index = voxel - voxelmap;
  integer_coordinates.z = voxel_index / (dimensions.x * dimensions.y);
  integer_coordinates.y = (voxel_index -= integer_coordinates.z * (dimensions.x * dimensions.y)) / dimensions.x;
  integer_coordinates.x = (voxel_index -= integer_coordinates.y * dimensions.x);
  return integer_coordinates;
}

//! Partitioning of continuous data into voxels. Maps float coordinates to dicrete voxel coordinates.
__device__ __host__     __forceinline__
Vector3ui mapToVoxels(const float voxel_side_length, const Vector3f &coordinates)
{
  Vector3ui integer_coordinates;
  integer_coordinates.x = static_cast<uint32_t>(floor(coordinates.x / voxel_side_length));
  integer_coordinates.y = static_cast<uint32_t>(floor(coordinates.y / voxel_side_length));
  integer_coordinates.z = static_cast<uint32_t>(floor(coordinates.z / voxel_side_length));

  // printf("coordinates.x = %f -> integer_coordinates.x = %u \n", coordinates.x, integer_coordinates.x);
  return integer_coordinates;
}

template<class Voxel>
__device__      __forceinline__
Voxel* getHighestVoxelPtr(const Voxel* base_addr, const Vector3ui &dimensions)
{
  return getVoxelPtr(base_addr, dimensions, dimensions.x -1, dimensions.y -1, dimensions.z -1);
}

//! Returns the center of a voxel as float coordinates. Mainly for boost test purposes!
__device__      __forceinline__
Vector3f getVoxelCenter(float voxel_side_length, const Vector3ui &voxel_coords)
{

  return Vector3f(voxel_coords.x * voxel_side_length + voxel_side_length / 2.0,
                  voxel_coords.y * voxel_side_length + voxel_side_length / 2.0,
                  voxel_coords.z * voxel_side_length + voxel_side_length / 2.0);
}

struct RayCaster
{
public:

//! raycasting from one point to another marks decreases voxel occupancy along the ray
  __device__ __forceinline__
  void rayCast(ProbabilisticVoxel* voxelmap, const Vector3ui &dimensions, const Sensor* sensor,
               const Vector3ui& from, const Vector3ui& to)
  {
    int32_t difference_x = 0;
    int32_t difference_y = 0;
    int32_t difference_z = 0;

    int32_t x = 0;
    int32_t y = 0;
    int32_t z = 0;

    int32_t n;

    int8_t x_increment = 0;
    int8_t y_increment = 0;
    int8_t z_increment = 0;

    int32_t error_xy = 0;
    int32_t error_xz = 0;
    int32_t error_yz = 0;
    // -------------------------------------------  //

    // differences
    if (to.x > from.x)
    {
      difference_x = to.x - from.x;
      x_increment = 1;
    }
    else
    {
      difference_x = from.x - to.x;
      x_increment = -1;
    }

    if (to.y > from.y)
    {
      difference_y = to.y - from.y;
      y_increment = 1;
    }
    else
    {
      difference_y = from.y - to.y;
      y_increment = -1;
    }

    if (to.z > from.z)
    {
      difference_z = to.z - from.z;
      z_increment = 1;
    }
    else
    {
      difference_z = from.z - to.z;
      z_increment = -1;
    }

    // pointer to currently visited voxel
    ProbabilisticVoxel* voxel = NULL;

    // start values
    x = from.x;
    y = from.y;
    z = from.z;

    // number of cells to visit
    n = 1 + difference_x + difference_y + difference_z;

    // error between x- and y- difference
    error_xy = difference_x - difference_y;

    // error between x- and z- difference
    error_xz = difference_x - difference_z;

    // error between y- and z- difference
    error_yz = difference_y - difference_z;

    // double differences to avoid float values
    difference_x *= 2;
    difference_y *= 2;
    difference_z *= 2;

    for (; n > 0; n--)
    {
      //printf("visiting cell (%d, %d, %d)\n", x, y, z);
      voxel = getVoxelPtr(voxelmap, dimensions, x, y, z);
      //voxel->voxelmeaning = eBVM_OCCUPIED;

      voxel->updateOccupancy(cSENSOR_MODEL_FREE);

      //decreaseOccupancy(voxel, cSENSOR_MODEL_FREE); // todo: replace with "free" of sensor model


      if ((error_xy > 0) && (error_xz > 0))
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
};

struct DummyRayCaster: public RayCaster
{
public:
  __device__ __forceinline__
  void rayCast(ProbabilisticVoxel* voxelmap, const Vector3ui &dimensions, const Sensor* sensor,
               const Vector3ui& from, const Vector3ui& to)
  {
    // override and do nothing
  }
};

///* ------------------ KERNELS ------------------ */

//! Clear voxel occupancy
template<class Voxel>
__global__
void kernelClearVoxelMap(Voxel* voxelmap, const uint32_t voxelmap_size);

//! Clear voxel occupancy for specific voxel_meaning
template<std::size_t bit_length>
__global__
void kernelClearVoxelMap(BitVoxel<bit_length>* voxelmap, const uint32_t voxelmap_size, const uint32_t bit_index);

template<std::size_t bit_length>
__global__
void kernelClearVoxelMap(BitVoxel<bit_length>* voxelmap, uint32_t voxelmap_size,
                         BitVector<bit_length> bits);

///*! Print voxel info from within kernel.
// */
//__global__
//void kernelDumpVoxelMap(const Voxel* voxelmap, const Vector3ui dimensions, const uint32_t voxelmap_size);

/*! Transform sensor data from sensor coordinate system
 * to world system. Needs extrinsic calibration of sensor.
 */
__global__
void kernelTransformSensorData(Sensor* sensor, Vector3f* raw_sensor_data, Vector3f* transformed_sensor_data);

/*! Insert sensor data into voxel map.
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
                            BitVoxel<length>* robotmap, const uint32_t bit_index, RayCasting rayCaster);

/*!
 * Collide two voxel maps.
 */
template<class Voxel, class OtherVoxel, class Collider>
__global__
void kernelCollideVoxelMaps(Voxel* voxelmap, const uint32_t voxelmap_size, OtherVoxel* other_map,
                            Collider collider, bool* results);



/*!
 * Collide two Voxelmaps but test against the Bitvector of the other voxelmap
 *
 */
template<std::size_t length, class Collider>
__global__
void kernelCollideVoxelMapsBitvector(BitVoxel<length>* voxelmap, const uint32_t voxelmap_size,
                                     BitVoxel<length>* other_map, Collider collider,
                                     BitVector<length>* results, uint16_t* num_collisions, const uint16_t sv_offset);

////! Function that tests 3d -> 1d mapping of voxel map storage
template<class Voxel>
__global__
void kernelAddressingTest(const Voxel* voxelmap_base_address, const Vector3ui dimensions, const float voxel_side_length,
                          const Vector3f *testpoints, const size_t testpoints_size, bool* success);


/*! Collide two voxel maps with storing collision info (for debugging only)
 * Voxels are considered occupied for values
 * greater or equal given thresholds.
 *
 * Collision info is stored within static model for 'other_map'.
 * Warning: Original model is modified!
 */
template<class Voxel, class OtherVoxel, class Collider>
__global__
void kernelCollideVoxelMapsDebug(Voxel* voxelmap, const uint32_t voxelmap_size, OtherVoxel* other_map,
                                 Collider collider, uint16_t* results);

/*!
 * Inserts pointcloud with global coordinates.
 *
 */
template<class Voxel>
__global__
void kernelInsertGlobalPointCloud(Voxel* voxelmap, const Vector3ui map_dim, const float voxel_side_length,
                                  Vector3f* points, const std::size_t sizePoints, const BitVoxelMeaning voxel_meaning);


template<class Voxel>
__global__
void kernelInsertMetaPointCloud(Voxel *voxelmap, const MetaPointCloudStruct *meta_point_cloud,
                                BitVoxelMeaning voxel_meaning, const Vector3ui map_dim, const float voxel_side_length);

template<class Voxel>
__global__
void kernelInsertMetaPointCloud(Voxel *voxelmap, const MetaPointCloudStruct *meta_point_cloud,
                                BitVoxelMeaning* voxel_meanings, const Vector3ui map_dim,
                                const float voxel_side_length);

/**
 * Shifts all swept-volume-IDs by shift_size towards lower IDs.
 * Currently this is limited to a shift size <64
 */
template<std::size_t length>
__global__
void kernelShiftBitVector(BitVoxel<length>* voxelmap, const uint32_t voxelmap_size, uint8_t shift_size);

} // end of namespace voxelmap
} // end of namespace gpu_voxels
#endif
