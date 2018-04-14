// this is for emacs file handling -*- mode: c++; indent-tabs-mode: nil -*-

// -- BEGIN LICENSE BLOCK ----------------------------------------------
// -- END LICENSE BLOCK ------------------------------------------------

//----------------------------------------------------------------------
/*!\file
 *
 * \author  Matthias Wagner
 * \date    2013-12-17
 *
 */
//----------------------------------------------------------------------/*
#include "VoxelMapVisualizerOperations.h"
#include <gpu_voxels/voxelmap/VoxelMap.hpp>

namespace gpu_voxels {
namespace visualization {

//////////////////////////////////// CUDA device functions /////////////////////////////////////////

//////////////////////////////////// CUDA kernel functions /////////////////////////////////////////

/**
 * Search the voxel map for occupied voxels and write the position into the VBO.
 * A voxel is occupied if its occupancy is greater than the occupancy_threshold.
 * start_voxel and end_voxel define a cuboid in the voxel map, which will be traversed (the rest of the map will be ignored).
 *
 * @param voxelMap: the device pointer of the voxel map
 * @param dim_voxel_map: the dimension of the voxel map
 * @param start_voxel: the position of the first voxel of the voxel map
 * @param end_voxel: the position of the last voxel of the voxel map
 * @param occupancy_threshold: the minimum occupancy value.
 * @param vbo: the device pointer of the VBO.
 * @param vbo_offsets: the offsets of the imaginary VBO segments.
 * @param vbo_limits: the maximum number of elements in the VBO segments.
 * @param write_index: the atomic counters for each voxel type (should be initialized with 0).
 * @param draw_voxel_type: if 0 the corresponding type at this index will not be drawn.
 * @param prefixes: stores the index of the VBO segment for each voxel type.
 */
__global__ void fill_vbo_without_precounting(ProbabilisticVoxel* voxelMap, Vector3ui dim_voxel_map,
                                              Vector3ui dim_super_voxel, Vector3ui start_voxel,
                                              Vector3ui end_voxel, Probability occupancy_threshold, float4* vbo,
                                              uint32_t* vbo_offsets, uint32_t* vbo_limits,
                                              uint32_t* write_index, uint8_t* draw_voxel_type,
                                              uint8_t* prefixes)
{
  // Grid-Stride Loops
  for (uint32_t x = dim_super_voxel.x * (blockIdx.x * blockDim.x + threadIdx.x) + start_voxel.x;
      x < dim_voxel_map.x && x < end_voxel.x; x += blockDim.x * gridDim.x * dim_super_voxel.x)
  {
    for (uint32_t y = dim_super_voxel.y * (blockIdx.y * blockDim.y + threadIdx.y) + start_voxel.y;
        y < dim_voxel_map.y && y < end_voxel.y; y += blockDim.y * gridDim.y * dim_super_voxel.y)
    {
      for (uint32_t z = dim_super_voxel.z * (blockIdx.z * blockDim.z + threadIdx.z) + start_voxel.z;
          z < dim_voxel_map.z && z < end_voxel.z; z += blockDim.z * gridDim.z * dim_super_voxel.z)
      {
        bool found = false;

        // check if one of the voxel of the super voxel is occupied
        // these 3 loop are were slow for big super voxel sizes
        for (uint32_t i = x; i < dim_super_voxel.x + x && i < dim_voxel_map.x && !found; i++)
        {
          for (uint32_t j = y; j < dim_super_voxel.y + y && j < dim_voxel_map.y && !found; j++)
          {
            for (uint32_t k = z; k < dim_super_voxel.z + z && k < dim_voxel_map.z && !found; k++)
            {
              uint32_t index = 0xffff;
              uint8_t prefix;
              ProbabilisticVoxel voxel = voxelMap[k * dim_voxel_map.x * dim_voxel_map.y
                  + j * dim_voxel_map.x + i];

              if (voxel.getOccupancy() >= occupancy_threshold)
              {
                // map the occupancy on the SweptVolume types, which is a bit fuzzy, was only 250 Types are available
                // so we have to cap it. Also we shift the prob by +127 to get rid of negative values:
                uint8_t type = MAX( MIN((eBVM_SWEPT_VOLUME_START + voxel.getOccupancy() - MIN_PROBABILITY), eBVM_SWEPT_VOLUME_END), eBVM_SWEPT_VOLUME_START);
                if (draw_voxel_type[type])
                {
                  prefix = prefixes[type];
                  index = atomicAdd(write_index + prefix, 1);
                  found = true;
                }
                if (index != 0xffff && index < vbo_limits[prefix])
                {
                  index = index + vbo_offsets[prefix];

                  vbo[index] = make_float4(x, y, z, dim_super_voxel.x);
                  // write the lower left front corner of the super voxel into the vbo as its translation
                  found = true;
                }
              }
            }
          }
        }
      }
    }
  }
}

/**
 * Search the voxel map for occupied voxels and write the position into the VBO.
 * A voxel is occupied if its occupancy is greater than the occupancy_threshold.
 * start_voxel and end_voxel define a cuboid in the voxel map, which will be traversed (the rest of the map will be ignored).
 *
 * @param voxelMap: the device pointer of the voxel map
 * @param dim_voxel_map: the dimension of the voxel map
 * @param start_voxel: the position of the first voxel of the voxel map
 * @param end_voxel: the position of the last voxel of the voxel map
 * @param occupancy_threshold: the minimum occupancy value.
 * @param vbo: the device pointer of the VBO.
 * @param vbo_offsets: the offsets of the imaginary VBO segments.
 * @param vbo_limits: the maximum number of elements in the VBO segments.
 * @param write_index: the atomic counters for each voxel type (should be initialized with 0).
 * @param draw_voxel_type: if 0 the corresponding type at this index will not be drawn.
 * @param prefixes: stores the index of the VBO segment for each voxel type.
 */__global__ void fill_vbo_without_precounting(BitVectorVoxel* voxelMap, Vector3ui dim_voxel_map,
                                              Vector3ui dim_super_voxel, Vector3ui start_voxel,
                                              Vector3ui end_voxel, uint8_t occupancy_threshold, float4* vbo,
                                              uint32_t* vbo_offsets, uint32_t* vbo_limits,
                                              uint32_t* write_index, uint8_t* draw_voxel_type,
                                              uint8_t* prefixes)
{
  // Grid-Stride Loops
  for (uint32_t x = dim_super_voxel.x * (blockIdx.x * blockDim.x + threadIdx.x) + start_voxel.x;
      x < dim_voxel_map.x && x < end_voxel.x; x += blockDim.x * gridDim.x * dim_super_voxel.x)
  {
    for (uint32_t y = dim_super_voxel.y * (blockIdx.y * blockDim.y + threadIdx.y) + start_voxel.y;
        y < dim_voxel_map.y && y < end_voxel.y; y += blockDim.y * gridDim.y * dim_super_voxel.y)
    {
      for (uint32_t z = dim_super_voxel.z * (blockIdx.z * blockDim.z + threadIdx.z) + start_voxel.z;
          z < dim_voxel_map.z && z < end_voxel.z; z += blockDim.z * gridDim.z * dim_super_voxel.z)
      {
        bool found = false;

        // check if one of the voxel of the super voxel is occupied
        // these 3 loop are were slow for big super voxel sizes
        for (uint32_t i = x; i < dim_super_voxel.x + x && i < dim_voxel_map.x && !found; i++)
        {
          for (uint32_t j = y; j < dim_super_voxel.y + y && j < dim_voxel_map.y && !found; j++)
          {
            for (uint32_t k = z; k < dim_super_voxel.z + z && k < dim_voxel_map.z && !found; k++)
            {
              BitVectorVoxel voxel = voxelMap[k * dim_voxel_map.x * dim_voxel_map.y + j * dim_voxel_map.x + i];

              if (!voxel.bitVector().isZero())
              {
                for (uint32_t t = 0 ; t < min((unsigned long long) BIT_VECTOR_LENGTH, (unsigned long long) MAX_DRAW_TYPES); ++t)
                {
                  uint32_t index = 0xffff;
                  uint8_t prefix;

                 if (draw_voxel_type[t] && voxel.bitVector().getBit(t))
                  {
                    prefix = prefixes[t];
                    index = atomicAdd(write_index + prefix, 1);
                    found = true;
                  }

                  if (index != 0xffff && index < vbo_limits[prefix])
                  {
                    index = index + vbo_offsets[prefix];

                    vbo[index] = make_float4(x, y, z, dim_super_voxel.x);
                    // write the lower left front corner of the super voxel into the vbo as its translation
                    found = true;
                  }
                  if (found)
                  { // if a set bit in the bit vector was found leave this loop
                    break;
                  }
                }
              }
            }
          }
        }
      }
    }
  }
}

/**

TODO: what to change for distancemap?

prefix is index into vbo_offsets and vbo_limits!
write_index is global array containing first free vbo index of given type

 * Search the voxel map for occupied voxels and write the position into the VBO.
 * A voxel is occupied if its occupancy is greater than the occupancy_threshold.
 * start_voxel and end_voxel define a cuboid in the voxel map, which will be traversed (the rest of the map will be ignored).
 *
 * @param voxelMap: the device pointer of the voxel map
 * @param dim_voxel_map: the dimension of the voxel map
 * @param start_voxel: the position of the first voxel of the voxel map
 * @param end_voxel: the position of the last voxel of the voxel map
 * @param occupancy_threshold: the minimum occupancy value.
 * @param vbo: the device pointer of the VBO.
 * @param vbo_offsets: the offsets of the imaginary VBO segments.
 * @param vbo_limits: the maximum number of elements in the VBO segments.
 * @param write_index: the atomic counters for each voxel type (should be initialized with 0).
 * @param draw_voxel_type: if 0 the corresponding type at this index will not be drawn.
 * @param prefixes: stores the index of the VBO segment for each voxel type.
 */
__global__ void fill_vbo_without_precounting(DistanceVoxel* voxelMap, Vector3ui dim_voxel_map,
                                              Vector3ui dim_super_voxel, Vector3ui start_voxel,
                                              Vector3ui end_voxel, visualizer_distance_drawmodes distance_drawmode, float4* vbo,
                                              uint32_t* vbo_offsets, uint32_t* vbo_limits,
                                              uint32_t* write_index, uint8_t* draw_voxel_type,
                                              uint8_t* prefixes)
{
  // Grid-Stride Loops
  for (uint32_t x = dim_super_voxel.x * (blockIdx.x * blockDim.x + threadIdx.x) + start_voxel.x;
      x < dim_voxel_map.x && x < end_voxel.x; x += blockDim.x * gridDim.x * dim_super_voxel.x)
  {
    for (uint32_t y = dim_super_voxel.y * (blockIdx.y * blockDim.y + threadIdx.y) + start_voxel.y;
        y < dim_voxel_map.y && y < end_voxel.y; y += blockDim.y * gridDim.y * dim_super_voxel.y)
    {
      for (uint32_t z = dim_super_voxel.z * (blockIdx.z * blockDim.z + threadIdx.z) + start_voxel.z;
          z < dim_voxel_map.z && z < end_voxel.z; z += blockDim.z * gridDim.z * dim_super_voxel.z)
      {
        bool found = false;

        //TODO: only look at the center voxel (or average the 8 center voxels) for supervoxels?

        // check if one of the voxel of the super voxel is occupied
        // these 3 loop are were slow for big super voxel sizes
        for (uint32_t i = x; i < dim_super_voxel.x + x && i < dim_voxel_map.x && !found; i++)
        {
          for (uint32_t j = y; j < dim_super_voxel.y + y && j < dim_voxel_map.y && !found; j++)
          {
            for (uint32_t k = z; k < dim_super_voxel.z + z && k < dim_voxel_map.z && !found; k++)
            {
              const uint32_t UNINITIALIZED = 0xffffffff;
              uint32_t index = UNINITIALIZED;
              uint8_t prefix;

              DistanceVoxel voxel = voxelMap[k * dim_voxel_map.x * dim_voxel_map.y
                  + j * dim_voxel_map.x + i];
              int32_t squared_distance = voxel.squaredObstacleDistance(Vector3i(i, j, k));
              if (squared_distance < 0) squared_distance = 0;
              float distance = sqrtf((float)squared_distance);

              //VBO speichert nur offset und cube size, also intensität hier nur relevant für ">0?"

              //always draw recognized obstacles, regardless of distance_drawmode
              //mark obstacles as COLLISION(1)
              if (distance <= 0)
              {
                uint8_t type = 1;

                if (draw_voxel_type[type])
                {
                  prefix = prefixes[type];
                  index = atomicAdd(write_index + prefix, 1);
                  found = true;

                  if (index != UNINITIALIZED && index < vbo_limits[prefix])
                  {
                    index = index + vbo_offsets[prefix];

                    vbo[index] = make_float4(x, y, z, dim_super_voxel.x);
                    // write the lower left front corner of the super voxel into the vbo as its translation
                    // use the z as height so switch z and y
                  }
                }
              }

              if (distance_drawmode == DISTANCE_DRAW_PBA_INTERMEDIATE)
              {
                //mark voxels populated by PBA as SWEPT_VOLUME (10)
                // TODO: remove after PBA debugging? needed because PBA will fill in distance exclusively in phase 3
                if ((voxel.getObstacle().x != PBA_UNINITIALISED_COORD) || (voxel.getObstacle().y != PBA_UNINITIALISED_COORD) || (voxel.getObstacle().z != PBA_UNINITIALISED_COORD))
                {
                  uint8_t type = 10;

                  if (draw_voxel_type[type])
                  {
                    prefix = prefixes[type];
                    index = atomicAdd(write_index + prefix, 1);
                    found = true;

                    if (index != UNINITIALIZED && index < vbo_limits[prefix])
                    {
                      index = index + vbo_offsets[prefix];

                      vbo[index] = make_float4(x, y, z, dim_super_voxel.x);
                      // write the lower left front corner of the super voxel into the vbo as its translation
                      // use the z as height so switch z and y
                    }
                  }
                }

                //mark obstacles processed by PBA as SWEPT_VOLUME (11)
                // TODO: remove after PBA debugging? needed because PBA will fill in distance exclusively in phase 3
                if ((voxel.getObstacle().x == i) && (voxel.getObstacle().y == j) && (voxel.getObstacle().z == k))
                {
                  uint8_t type = 11;

                  if (draw_voxel_type[type])
                  {
                    prefix = prefixes[type];
                    index = atomicAdd(write_index + prefix, 1);
                    found = true;

                    if (index != UNINITIALIZED && index < vbo_limits[prefix])
                    {
                      index = index + vbo_offsets[prefix];

                      vbo[index] = make_float4(x, y, z, dim_super_voxel.x);
                      // write the lower left front corner of the super voxel into the vbo as its translation
                      // use the z as height so switch z and y
                    }
                  }
                }


                //mark obstacles processed by PBA as SWEPT_VOLUME (12)
                if ((voxel.getObstacle().x != PBA_UNINITIALISED_COORD) && (voxel.getObstacle().y != PBA_UNINITIALISED_COORD) && (voxel.getObstacle().z != PBA_UNINITIALISED_COORD))
                {
                  uint8_t type = 12;

                  if (draw_voxel_type[type])
                  {
                    prefix = prefixes[type];
                    index = atomicAdd(write_index + prefix, 1);
                    found = true;

                    if (index != UNINITIALIZED && index < vbo_limits[prefix])
                    {
                      index = index + vbo_offsets[prefix];

                      vbo[index] = make_float4(x, y, z, dim_super_voxel.x);
                      // write the lower left front corner of the super voxel into the vbo as its translation
                      // use the z as height so switch z and y
                    }
                  }
                }


                //mark obstacles processed by PBA as SWEPT_VOLUME (12)
                if ((voxel.getObstacle().x != PBA_UNINITIALISED_COORD) && (voxel.getObstacle().y != PBA_UNINITIALISED_COORD) && (voxel.getObstacle().z != PBA_UNINITIALISED_COORD))
                {
                  if (distance <= 5)
                  {
                    uint8_t type = 13;

                    if (draw_voxel_type[type])
                    {
                      prefix = prefixes[type];
                      index = atomicAdd(write_index + prefix, 1);
                      found = true;

                      if (index != UNINITIALIZED && index < vbo_limits[prefix])
                      {
                        index = index + vbo_offsets[prefix];

                        vbo[index] = make_float4(x, y, z, dim_super_voxel.x);
                        // write the lower left front corner of the super voxel into the vbo as its translation
                        // use the z as height so switch z and y
                      }
                    }
                  }
                }

                if (true) {
                  uint8_t type = 14;

                  if (draw_voxel_type[type])
                  {
                    prefix = prefixes[type];
                    index = atomicAdd(write_index + prefix, 1);
                    found = true;

                    if (index != UNINITIALIZED && index < vbo_limits[prefix])
                    {
                      index = index + vbo_offsets[prefix];

                      vbo[index] = make_float4(x, y, z, dim_super_voxel.x);
                      // write the lower left front corner of the super voxel into the vbo as its translation
                    }
                  }
                }
              } //DRAW_PBA_INTERMEDIATE


              if (distance_drawmode == DISTANCE_DRAW_MULTICOLOR_GRADIENT)
              {
                //mark distance <= threshold as Swept Volume (10-20)
                if (distance >= 1 && distance < 11) {
                  uint8_t type = 10 + (int)distance; //use swept volume types 11-20

                  if (draw_voxel_type[type])
                  {
                    prefix = prefixes[type];
                    index = atomicAdd(write_index + prefix, 1);
                    found = true;
                    if (index != UNINITIALIZED && index < vbo_limits[prefix])
                    {
                      index = index + vbo_offsets[prefix];
                      vbo[index] = make_float4(x, y, z, dim_super_voxel.x);
                    }
                  }
                }
              }


              if (distance_drawmode == DISTANCE_DRAW_TWOCOLOR_GRADIENT)
              {
                if (distance < 50)
                {
                  uint8_t type = 21 + distance;

                  if (draw_voxel_type[type])
                  {
                    prefix = prefixes[type];
                    index = atomicAdd(write_index + prefix, 1);
                    found = true;
                    if (index != UNINITIALIZED && index < vbo_limits[prefix])
                    {
                      index = index + vbo_offsets[prefix];
                      vbo[index] = make_float4(x, y, z, dim_super_voxel.x);
                    }
                  }
                }

                if (distance >= 50)
                {
                  uint8_t type = 250;

                  if (draw_voxel_type[type])
                  {
                    prefix = prefixes[type];
                    index = atomicAdd(write_index + prefix, 1);
                    found = true;
                    if (index != UNINITIALIZED && index < vbo_limits[prefix])
                    {
                      index = index + vbo_offsets[prefix];
                      vbo[index] = make_float4(x, y, z, dim_super_voxel.x);
                    }
                  }
                }
              } //DRAW_TWOCOLOR_GRADIENT


              if (distance_drawmode == DISTANCE_DRAW_VORONOI_LINEAR)
              {
                const Vector3ui obstacle = voxel.getObstacle();
                uint linear_id = obstacle.x + dim_voxel_map.x*(obstacle.y + dim_voxel_map.y * (obstacle.z));

                uint8_t type = 21 + (linear_id % 50); //use swept ids 20-219

                if (draw_voxel_type[type])
                {
                  prefix = prefixes[type];
                  index = atomicAdd(write_index + prefix, 1);
                  found = true;
                  if (index != UNINITIALIZED && index < vbo_limits[prefix])
                  {
                    index = index + vbo_offsets[prefix];
                    vbo[index] = make_float4(x, y, z, dim_super_voxel.x);
                  }
                }
              }

              if (distance_drawmode == DISTANCE_DRAW_VORONOI_SCRAMBLE)
              {
                const Vector3ui obstacle = voxel.getObstacle();
                uint linear_id = obstacle.x + dim_voxel_map.x*(obstacle.y + dim_voxel_map.y * (obstacle.z));
                uint scrambled_id = linear_id * 137 + 241; //use swept ids 20-219

  //              uint8_t type = 20 + (scrambled_id % 200); //use swept ids 20-219
                uint8_t type = 20 + (scrambled_id % 196); //use swept ids 20-215

                if (draw_voxel_type[type])
                {
                  prefix = prefixes[type];
                  index = atomicAdd(write_index + prefix, 1);
                  found = true;
                  if (index != UNINITIALIZED && index < vbo_limits[prefix])
                  {
                    index = index + vbo_offsets[prefix];
                    vbo[index] = make_float4(x, y, z, dim_super_voxel.x);
                  }
                }
              }

            }
          }
        }
      }
    }
  }
}

/**
 * Write the position of each cube into the VBO.
 *
 * @param cubes: the device pointer of the cube list.
 * @param size: the size of cubes.
 * @param vbo: the device pointer of the VBO.
 * @param vbo_offsets: the offsets of the imaginary VBO segments.
 * @param write_index: the atomic counters for each type (should be initialized with 0).
 * @param draw_voxel_type: if 0 the corresponding type at this index will not be drawn.
 * @param prefixes: stores the index of the VBO segment for each voxel type.
 */
__global__ void fill_vbo_with_cubelist(Cube* cubes, uint32_t size, float4* vbo, uint32_t* vbo_offsets,
                                      uint32_t* write_index, uint8_t* draw_voxel_type, uint8_t* prefixes)
{
  //use Grid-Stride Loops
  for (uint32_t i = blockIdx.x * blockDim.x + threadIdx.x; i < size; i += blockDim.x * gridDim.x)
  {
    Cube* cube = cubes + i;

    uint32_t index = 0xffff;

    bool found = false;
    for (size_t t = 0 ; t < MAX_DRAW_TYPES; ++t)
    {
      // TODO: Create a bitmask outside the kernel and just do a bit comparison in here! Instead of for-loop
      if (draw_voxel_type[t] && cube->m_type_vector.getBit(t))
      {
        uint8_t prefix = prefixes[t];
        index = atomicAdd(write_index + prefix, 1);
        index = index + vbo_offsets[prefix];
        // write the position and the scale factor into the vbo
        float x = cube->m_position.x;
        float y = cube->m_position.y;
        float z = cube->m_position.z;
        float w = cube->m_side_length;
        //            printf("Found voxel at (%f,%f,%f) with voxel type %lu, and sidelenth %f\n", x, y, z, t, w);
        vbo[index] = make_float4(x, y, z, w);
        found = true;
      }

      if (found)
      { // if a set bit in the bit vector was found leave this loop
        break;
      }
    }
  }
}

/**
 * Calculate the amount of cubes per type in the cubes list.
 *
 * @param cubes: the device pointer of the cube list.
 * @param size: the size of cubes.
 * @param cubes_per_type: Will contain the number of cubes per type afterwards (should be initialized with 0).
 * @param draw_voxel_type: if 0 the corresponding type at this index will not be drawn.
 * @param prefixes: stores the index of the VBO segment for each voxel type.
 */
__global__ void calculate_cubes_per_type_list(Cube* cubes, uint32_t size, uint32_t* cubes_per_type, uint8_t* draw_voxel_type, uint8_t* prefixes)
{
  for (uint32_t i = blockIdx.x * blockDim.x + threadIdx.x; i < size; i += blockDim.x * gridDim.x)
  {
    Cube* cube = cubes + i;

    bool found = false;
    // TODO: Create a bitmask outside the kernel and just do a bit comparison in here! Instead of for-loop
    for (size_t t = 0; t < MAX_DRAW_TYPES; ++t)
    {
      uint8_t prefix;
      if (draw_voxel_type[t] && cube->m_type_vector.getBit(t))
      {
        prefix = prefixes[t];
        atomicAdd(cubes_per_type + prefix, 1);
        //printf("Found voxel with type %lu. Now drawing %u voxels of type %lu\n", t, *(cubes_per_type+prefix), t);
        found = true;
      }

      if (found)
      { // if a set bit in the bit vector was found leave this loop
        break;
      }
    }
  }
}



/**
 * Find the matching Cube and return its position in the array.
 * This is used in the "click voxel to get its info" functionality.
 *
 * @param cubes: the device pointer of the cube list.
 * @param num_cubes: the number of cubes to be searched
 * @param coords: the search criteria coordinates.
 * @param found_cube: found voxel will be written into this.
 * @param found_flag: True if found, not modified if not found. So set to false before calling.
 */
__global__ void find_cubes_by_coordinates(const Cube* cubes, size_t num_cubes, Vector3ui coords, Cube* found_cube, bool* found_flag)
{
  for (uint32_t i = blockIdx.x * blockDim.x + threadIdx.x; i < num_cubes; i += blockDim.x * gridDim.x)
  {
    const Cube* cube = cubes + i;
    if(cube->m_position == coords)
    {
      *found_flag = true;
      *found_cube = *cube;
    }
  }
}

} // end of ns
} // end of ns
