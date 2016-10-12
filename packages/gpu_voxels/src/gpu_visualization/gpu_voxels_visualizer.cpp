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
 * \date    2013-11-6
 *
 *  Wrapper for the Visualizer class.
 *
 *
 */

#include <gpu_visualization/gpu_voxels_visualizer.h>

using namespace gpu_voxels::visualization;
using namespace gpu_voxels;
using namespace boost::interprocess;

void runVisualisation(int32_t* argc, char* argv[])
{
  //register callback functions
  glutReshapeFunc(resizeFunctionWrapper);
  glutDisplayFunc(renderFunctionWrapper);
  glutIdleFunc(idleFunctionWrapper);
  glutTimerFunc(0, timerFunctionWrapper, 0);
  glutCloseFunc(cleanupFunctionWrapper);
  glutKeyboardFunc(keyboardFunctionWrapper);
  glutSpecialFunc(keyboardSpecialFunctionWrapper);
  glutMotionFunc(mouseMotionFunctionWrapper);
  glutPassiveMotionFunc(mousePassiveMotionFunctionWrapper);
  glutMouseFunc(mouseClickFunctionWrapper);

  glutMainLoop();
  return;
}

uint32_t getNumberOfOctreesFromSharedMem()
{
  uint32_t res = 0;
  try
  {
    SharedMemoryManagerOctrees shm_manager_octrees;
    res = shm_manager_octrees.getNumberOfOctreesToDraw();
  } catch (interprocess_exception& e)
  {
    LOGGING_DEBUG(Visualization, "Couldn't open the shared memory segment of the octrees!" << endl);
    return 0;
  }
  return res;
}
uint32_t getNumberOfVoxelmapsFromSharedMem()
{
  uint32_t res = 0;
  try
  {
    SharedMemoryManagerVoxelMaps shm_manager_voxelmaps;
    res = shm_manager_voxelmaps.getNumberOfVoxelMapsToDraw();
  } catch (interprocess_exception& e)
  {
    LOGGING_DEBUG(Visualization, "Couldn't open the shared memory segment of the voxel maps!" << endl);
    return 0;
  }
  return res;
}

uint32_t getNumberOfVoxellistsFromSharedMem()
{
  uint32_t res = 0;
  try
  {
    SharedMemoryManagerVoxelLists shm_manager_voxellists;
    res = shm_manager_voxellists.getNumberOfVoxelListsToDraw();
  } catch (interprocess_exception& e)
  {
    LOGGING_DEBUG(Visualization, "Couldn't open the shared memory segment of the voxel lists!" << endl);
    return 0;
  }
  return res;
}

uint32_t getNumberOfPrimitiveArraysFromSharedMem()
{
  uint32_t res = 0;
  try
  {
    SharedMemoryManagerPrimitiveArrays shm_manager_prim_arrays;
    res = shm_manager_prim_arrays.getNumberOfPrimitiveArraysToDraw();
  } catch (interprocess_exception& e)
  {
    LOGGING_DEBUG(Visualization, "Couldn't open the shared memory segment of the Primitive Arrays!" << endl);
    return 0;
  }
  return res;
}

void registerVoxelmapFromSharedMemory(uint32_t index)
{
  try
  {
    SharedMemoryManagerVoxelMaps shm_manager_voxelmaps;
    void* dev_data_pointer;
    Vector3ui dim;
    float voxel_side_length;
    MapType map_type;
    std::string map_name;

    bool error = false;
    if (!shm_manager_voxelmaps.getDevicePointer(dev_data_pointer, index))
    {
      error = true;
      LOGGING_ERROR(
          Visualization,
          "Couldn't find mem_segment "<< shm_variable_name_voxelmap_handler_dev_pointer << index << endl);
    }
    if (!shm_manager_voxelmaps.getVoxelMapDimension(dim, index))
    {
      error = true;
      LOGGING_ERROR(Visualization,
                    "Couldn't find mem_segment "<< shm_variable_name_voxelmap_dimension << index << endl);
    }
    if (!shm_manager_voxelmaps.getVoxelMapSideLength(voxel_side_length, index))
    {
      error = true;
      LOGGING_ERROR(Visualization,
                    "Couldn't find mem_segment "<< shm_variable_name_voxel_side_length << index << endl);
    }
    if (!shm_manager_voxelmaps.getVoxelMapType(map_type, index))
    {
      error = true;
      LOGGING_ERROR(Visualization,
                    "Couldn't find mem_segment "<< shm_variable_name_voxelmap_type << index << endl);
    }
    if (!shm_manager_voxelmaps.getVoxelMapName(map_name, index))
    {
      LOGGING_WARNING(
          Visualization,
          "Couldn't find mem_segment "<< shm_variable_name_voxelmap_name << index << " or the name is empty! Using a default name."<< endl);
    }
    if (!error)
    {
      voxelmap::AbstractVoxelMap* voxel_map;
      switch (map_type)
      {
        case MT_BITVECTOR_VOXELMAP:
          voxel_map = new voxelmap::BitVectorVoxelMap((BitVectorVoxel*) dev_data_pointer, dim,
                                                      voxel_side_length, MT_BITVECTOR_VOXELMAP);
          break;
      case MT_PROBAB_VOXELMAP:
        voxel_map = new voxelmap::ProbVoxelMap((ProbabilisticVoxel*) dev_data_pointer, dim,
                                               voxel_side_length, MT_PROBAB_VOXELMAP);
        break;
      case MT_DISTANCE_VOXELMAP:
        voxel_map = new voxelmap::DistanceVoxelMap((DistanceVoxel*) dev_data_pointer, dim,
                                               voxel_side_length, MT_DISTANCE_VOXELMAP);
        break;
        default:
          LOGGING_ERROR(
              Visualization,
              "Used map type ("<< typeToString(map_type) << ") for voxel map (" << map_name <<") not supported!" << endl);
          return;
      }
      LOGGING_INFO(
          Visualization,
          "Providing a "<< typeToString(map_type) << " called \""<< map_name << "\" with side_lenght " << voxel_side_length << " and dimension [" << dim.x << ", " << dim.y << ", " << dim.z << "]" << endl);
      vis->registerVoxelMap(voxel_map, index, map_name);
    }
  } catch (interprocess_exception& e)
  {
    LOGGING_DEBUG(Visualization, "Couldn't open the shared memory segment of the voxel maps!" << endl);
    return;
  }
}

void registerVoxellistFromSharedMemory(uint32_t index)
{
  try
  {
    SharedMemoryManagerVoxelLists shm_manager_voxellists;
    std::string map_name;

    if (!shm_manager_voxellists.getVoxelListName(map_name, index))
    {
      LOGGING_WARNING(
          Visualization,
          "Couldn't find mem_segment "<< shm_variable_name_voxellist_name << index << " or the name is empty! Using a default name."<< endl);
    }

    LOGGING_INFO(
          Visualization,
          "Providing a voxellist called \""<< map_name << "\"" << endl);
    vis->registerVoxelList(index, map_name);
  } catch (interprocess_exception& e)
  {
    LOGGING_DEBUG(Visualization, "Couldn't open the shared memory segment of the voxel lists!" << endl);
    return;
  }
}

void registerOctreeFromSharedMemory(uint32_t index)
{
  try
  {
    SharedMemoryManagerOctrees shm_manager_octrees;
    std::string map_name = shm_manager_octrees.getNameOfOctree(index);
    LOGGING_INFO(Visualization, "Providing a Octree called \""<< map_name << "\"." << endl);
    vis->registerOctree(index, map_name);
  } catch (interprocess_exception& e)
  {
    LOGGING_DEBUG(Visualization, "Couldn't open the shared memory segment of the octrees!" << endl);
    return;
  }
}

void registerPrimitiveArrayFromSharedMemory(uint32_t index)
{
  try
  {
    SharedMemoryManagerPrimitiveArrays shm_manager_prim_arrays;
    std::string prim_array_name = shm_manager_prim_arrays.getNameOfPrimitiveArray(index);
    LOGGING_INFO(Visualization, "Providing a Primitive Array called \""<< prim_array_name << "\"." << endl);
    vis->registerPrimitiveArray(index, prim_array_name);
  } catch (interprocess_exception& e)
  {
    LOGGING_DEBUG(Visualization, "Couldn't open the shared memory segment of the Primitive Arrays!" << endl);
    return;
  }
}

int32_t main(int32_t argc, char* argv[])
{
// Initialize the logging
  icl_core::logging::initialize(argc, argv);

  LOGGING_INFO(Visualization, "Starting the gpu_voxels Visualizer." << endl);
  vis->initalizeVisualizer(argc, argv);

  uint32_t num_voxelmaps = getNumberOfVoxelmapsFromSharedMem();
  LOGGING_INFO(Visualization, "Number of voxel maps that will be drawn: " << num_voxelmaps << endl);
  for (uint32_t i = 0; i < num_voxelmaps; i++)
  {
    registerVoxelmapFromSharedMemory(i);
  }

  uint32_t num_voxellists = getNumberOfVoxellistsFromSharedMem();
  LOGGING_INFO(Visualization, "Number of voxel lists that will be drawn: " << num_voxellists << endl);
  for (uint32_t i = 0; i < num_voxellists; i++)
  {
    registerVoxellistFromSharedMemory(i);
  }

  uint32_t num_octrees = getNumberOfOctreesFromSharedMem();
  LOGGING_INFO(Visualization, "Number of octrees that will be drawn: " << num_octrees << endl);
  for (uint32_t i = 0; i < num_octrees; i++)
  {
    registerOctreeFromSharedMemory(i);
  }

  uint32_t num_prim_arrays = getNumberOfPrimitiveArraysFromSharedMem();
  LOGGING_INFO(Visualization, "Number of Primitive Arrays that will be drawn: " << num_prim_arrays << endl);
  for (uint32_t i = 0; i < num_prim_arrays; i++)
  {
    registerPrimitiveArrayFromSharedMemory(i);
  }

  runVisualisation(&argc, argv);

  LOGGING_INFO(Visualization, "Exiting..\n");
  delete vis;
  icl_core::logging::LoggingManager::instance().shutdown();
  return EXIT_SUCCESS;
}
