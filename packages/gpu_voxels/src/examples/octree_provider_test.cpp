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
 * \author  Matthias Wagner
 * \date    2014-02-02
 *
 *  \brief Camera class for the voxel map visualizer on GPU
 *
 */
//----------------------------------------------------------------------
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <time.h>       /* time */
#include <math.h>
#include <iostream>
#include <fstream>

#include <boost/interprocess/managed_shared_memory.hpp>
#include <boost/lexical_cast.hpp>

#include <gpu_voxels/helpers/cuda_datatypes.h>
#include <gpu_voxels/helpers/cuda_handling.h>

#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/device_ptr.h>

#include <icl_core_config/Config.h>
#include <gpu_voxels/voxelmap/VoxelMap.h>

#include <gpu_voxels/vis_interface/VisualizerInterface.h>

using namespace gpu_voxels;
using namespace boost::interprocess;

int32_t main(int32_t argc, char* argv[])
{
  ////////////////////////////create some test data ////////////////////////////
  //thrust::device_vector<Cube> cubes;
  uint32_t num_cubes = 1;
  //uint32_t num_cubes = 500;
  srand(time(NULL));
  Cube* cubes;
  cudaMalloc(&cubes, num_cubes * sizeof(Cube));
  Cube* h_cubes = (Cube*) malloc(num_cubes * sizeof(Cube));

  Cube c = Cube(3, Vector3ui(3, 10, 10), eVT_OCCUPIED);
  h_cubes[0] = c;
//  c = Cube(3, Vector3ui(2, 2, 2), eVT_OCCUPIED);
//  h_cubes[1] = c;
//  c = Cube(3, Vector3ui(0, 0, 10), eVT_OCCUPIED);
//  h_cubes[2] = c;
//  c = Cube(1, Vector3ui(0, 1, 0), eVT_OCCUPIED);
//  h_cubes[3] = c;
//  c = Cube(1, Vector3ui(0, 0, 0), eVT_OCCUPIED);
//  h_cubes[4] = c;

  for (uint32_t i = 5; i < num_cubes; i++)
  {
    uint32_t x, y, z, length;

    x = rand() % 1250;
    y = rand() % 809;
    z = rand() % 1004;
    length = rand() % 1;
    int ty = rand() % 5;

    VoxelType t = eVT_OCCUPIED;
    if (ty == 0)
      t = eVT_COLLISION;
    else if (ty == 1)
      t = eVT_OCCUPIED;
    else if (ty == 2)
      t = eVT_SWEPT_VOLUME_START;
    else if (ty == 3)
      t = eVT_UNDEFINED;

    Cube c = Cube(length + 1, Vector3ui(x, y, z), eVT_OCCUPIED);
    // h_cubes[i] = c;
  }

  cudaMemcpy(cubes, h_cubes, num_cubes * sizeof(Cube), cudaMemcpyHostToDevice);

  uint32_t num_cubes_2 = 1;
  Cube* cubes_2;
  cudaMalloc(&cubes_2, num_cubes_2 * sizeof(Cube));
  Cube* h_cubes_2 = (Cube*) malloc(num_cubes_2 * sizeof(Cube));

  Cube cu = Cube(3, Vector3ui(0, 10, 0), eVT_COLLISION);
  h_cubes_2[0] = cu;

  for (uint32_t i = 0; i < num_cubes_2; i++)
  {
    uint32_t x, y, z, length;

    x = rand() % 50;
    y = rand() % 89;
    z = rand() % 14;
    length = rand() % 4 + 3;
    int ty = rand() % 5;

    VoxelType t = eVT_OCCUPIED;
    if (ty == 0)
      t = eVT_COLLISION;
    else if (ty == 1)
      t = eVT_OCCUPIED;
    else if (ty == 2)
      t = eVT_SWEPT_VOLUME_START;
    else if (ty == 3)
      t = eVT_UNDEFINED;

    Cube c = Cube(length + 1, Vector3ui(x, y, z), eVT_COLLISION);
    //h_cubes_2[i] = c;
  }

  cudaMemcpy(cubes_2, h_cubes_2, num_cubes_2 * sizeof(Cube), cudaMemcpyHostToDevice);

//
//  Cube* h_cubes_t = (Cube*) malloc(num_cubes * sizeof(Cube));
//
//  cudaMemcpy(h_cubes_t, cubes, num_cubes * sizeof(Cube), cudaMemcpyDeviceToHost);
//
//  for (uint32_t i = 0; i < num_cubes; i++)
//  {
//    std::cout << "length:  " << h_cubes_t[i].m_side_length << " x:  " << h_cubes_t[i].m_position.x << " y:  "
//        << h_cubes_t[i].m_position.y << " z:  " << h_cubes_t[i].m_position.z << "  type:  "
//        << h_cubes_t[i].m_type << "\n";
//  }

  ////////////////////////////write the stuff in the shared memory///////////////////////////

  //Construct managed shared memory
  shared_memory_object::remove(shm_segment_name_octrees.c_str());
  shared_memory_object::remove(shm_segment_name_visualizer.c_str());
  managed_shared_memory segment(create_only, shm_segment_name_octrees.c_str(), 65536);

  segment.construct<uint32_t>(shm_variable_name_number_of_octrees.c_str())(2);
  uint32_t svoxel_size = 3;
  uint32_t* super_voxel_size_ptr = segment.construct<uint32_t>(shm_variable_name_super_voxel_size.c_str())(
      svoxel_size);

  std::string octree_name_0 = "MyOctree";
  segment.construct_it<char>((shm_variable_name_octree_name + "0").c_str())[octree_name_0.size()](
      octree_name_0.data());

  bool* sh_buffer_swapped = segment.construct<bool>((shm_variable_name_buffer_swapped + "0").c_str())(false);
  /////Buffer1 for the octree
  cudaIpcMemHandle_t handler;
  HANDLE_CUDA_ERROR(cudaIpcGetMemHandle((cudaIpcMemHandle_t * ) &handler, cubes));
  /////Buffer2 for the octree
  cudaIpcMemHandle_t handler_2;
  HANDLE_CUDA_ERROR(cudaIpcGetMemHandle((cudaIpcMemHandle_t * ) &handler_2, cubes_2));

  cudaIpcMemHandle_t* sh_handler = segment.construct<cudaIpcMemHandle_t>(
      (shm_variable_name_octree_handler_dev_pointer + "0").c_str())(handler);
  *sh_buffer_swapped = true;
  segment.construct<uint32_t>((shm_variable_name_number_cubes + "0").c_str())(num_cubes);

  bool* sh_buffer_swapped_2 = segment.construct<bool>((shm_variable_name_buffer_swapped + "1").c_str())(
      false);
  cudaIpcMemHandle_t* sh_handler_2 = segment.construct<cudaIpcMemHandle_t>(
      (shm_variable_name_octree_handler_dev_pointer + "1").c_str())(handler_2);
  *sh_buffer_swapped_2 = false;
  uint32_t* sh_number_cubes_2 = segment.construct<uint32_t>((shm_variable_name_number_cubes + "1").c_str())(
      num_cubes_2);

  //Stuff for the primitives and target point
  managed_shared_memory vis_segment(create_only, shm_segment_name_visualizer.c_str(), 65536);
  vis_segment.construct<Vector3f>(shm_variable_name_target_point.c_str())(10, 0, 20);

  uint32_t zahl = 1;
  uint32_t num_primitives_1 = zahl * zahl * zahl;
  Vector4f* primitives_1;
  PrimitiveTypes prim_type_1 = primitive_Cuboid;
  // PrimitiveTypes prim_type_1 = primitive_Sphere;
  cudaMalloc(&primitives_1, num_primitives_1 * sizeof(Vector4f));
  Vector4f* h_primitves_1 = (Vector4f*) malloc(num_primitives_1 * sizeof(Vector4f));

//  for (uint32_t i = 0; i < num_primitives_1; i++)
//  {
//
//    uint32_t x = i % zahl, y = (i / zahl) % zahl, z = i / (zahl * zahl);
//
//    h_primitves_1[i] = Vector4f(x, y, z);
//
//  }

  srand(time(NULL));
  for (uint32_t i = 0; i < num_primitives_1; i++)
  {
    uint32_t max_dim = 1000;
    float x, y, z, scale;
    x = rand() % max_dim;

    y = rand() % max_dim;

    z = rand() % max_dim;

    scale = (rand() % 2) + 1;
    h_primitves_1[i] = Vector4f(x, y, z, scale);

  }

  cudaMemcpy(primitives_1, h_primitves_1, num_primitives_1 * sizeof(Vector4f), cudaMemcpyHostToDevice);
  cudaIpcMemHandle_t handler_primtives;
  HANDLE_CUDA_ERROR(cudaIpcGetMemHandle((cudaIpcMemHandle_t * ) &handler_primtives, primitives_1));

  vis_segment.construct<cudaIpcMemHandle_t>(shm_variable_name_primitive_handler_dev_pointer.c_str())(
      handler_primtives);
  vis_segment.construct<uint32_t>(shm_variable_name_number_of_primitives.c_str())(num_primitives_1);
  vis_segment.construct<PrimitiveTypes>(shm_variable_name_primitive_type.c_str())(prim_type_1);
  bool* sh_primitive_buffer_changed = vis_segment.construct<bool>(
      shm_variable_name_primitive_buffer_changed.c_str())(true);

  std::string exit = "";
  while (true)
  {
    std::cout << "To quit the program enter \"exit\"" << std::endl;
    std::cout << "To change primitive array \"t\"" << std::endl;
    std::cin >> exit;
    if (exit.compare("exit") == 0)
    {
      break;
    }
    else if (exit.compare("t") == 0)
    {
      // *target_point = *target_point + Vector3f(1, 0, 0);
      if (!*sh_primitive_buffer_changed)
        *sh_primitive_buffer_changed = true;
    }
    else if (exit.compare("s") == 0)
    {
      uint32_t count = 0;
      while (count < 20)
      {
        count++;
        if (svoxel_size != *super_voxel_size_ptr)
        {
          svoxel_size = *super_voxel_size_ptr;
          std::cout << "calculate new buffer ... " << *super_voxel_size_ptr << std::endl;
          if (*sh_buffer_swapped_2 == false)
          {
            uint32_t num_cubes_3 = 10000000 / *super_voxel_size_ptr;

            std::cout << "Number of cubes: " << num_cubes_3 << std::endl;
            Cube* cubes_3;
            cudaFree(cubes_3);
            cudaMalloc(&cubes_3, num_cubes_3 * sizeof(Cube));
            Cube* h_cubes_3 = (Cube*) malloc(num_cubes_3 * sizeof(Cube));

            for (uint32_t i = 0; i < num_cubes_3; i++)
            {
              uint32_t x, y, z, length;

              x = rand() % 500;
              y = rand() % 689;
              z = rand() % 854;
              length = rand() % 2;
              int ty = rand() % 5;

              VoxelType t = eVT_OCCUPIED;
              if (ty == 0)
                t = eVT_COLLISION;
              else if (ty == 1)
                t = eVT_OCCUPIED;
              else if (ty == 2)
                t = eVT_SWEPT_VOLUME_START;
              else if (ty == 3)
                t = eVT_UNDEFINED;

              t = eVT_OCCUPIED;
              Cube c = Cube(length + svoxel_size, Vector3ui(x, y, z), t);
              h_cubes_3[i] = c;
            }

            cudaMemcpy(cubes_3, h_cubes_3, num_cubes_3 * sizeof(Cube), cudaMemcpyHostToDevice);
            free(h_cubes_3);
            cudaDeviceSynchronize();
            HANDLE_CUDA_ERROR(cudaIpcGetMemHandle((cudaIpcMemHandle_t * ) &handler_2, cubes_3));

            *sh_handler_2 = handler_2;
            *sh_number_cubes_2 = num_cubes_3;
            *sh_buffer_swapped_2 = true;
          }
        }
        sleep(1);
      }
    }
    else
    {
      uint32_t count = 0;
      while (count < 250)
      {

        if (!*sh_buffer_swapped)
        {
          *sh_buffer_swapped_2 = true;
          *sh_buffer_swapped = true;
          std::cout << "Buffer swapped!!!!!!!!" << std::endl;
        }
        count++;
        usleep(100000);
      }
    }

  }

  shared_memory_object::remove(shm_segment_name_octrees.c_str());
  shared_memory_object::remove(shm_segment_name_visualizer.c_str());
  return 0;
}
