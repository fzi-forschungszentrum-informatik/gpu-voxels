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

  createRightClickMenu();

  glutMainLoop();
  return;
}

void createRightClickMenu()
{
  int textMenu = glutCreateMenu(&menuFunctionWrapper);
  glutAddMenuEntry("All", MENU_TEXT_ALL);
  glutAddMenuEntry("Points Count", MENU_TEXT_POINTS);
  glutAddMenuEntry("VBO Info", MENU_TEXT_VBO);
  glutAddMenuEntry("VoxelMaps", MENU_TEXT_VOXELMAPS);
  glutAddMenuEntry("VoxelLists", MENU_TEXT_VOXELLISTS);
  glutAddMenuEntry("Octrees", MENU_TEXT_OCTREES);
  glutAddMenuEntry("PrimitiveArrays", MENU_TEXT_PRIMITIVEARRAYS);
  glutAddMenuEntry("Types", MENU_TEXT_TYPES);
  glutAddMenuEntry("Clicked Voxel Info", MENU_TEXT_CLICKEDVOXELINFO);

  int mapMenu = glutCreateMenu(&menuFunctionWrapper);

  std::vector<std::string> voxelmapNames = vis->getVoxelMapNames();
  for(size_t i = 0; i < voxelmapNames.size(); i++)
  {
    glutSetMenu(mapMenu);
    std::stringstream tmp;
    tmp << "Toggle Draw " << voxelmapNames[i];
    glutAddMenuEntry(tmp.str().data(), 300 + i);
  }

  std::vector<std::string> voxellistNames = vis->getVoxelListNames();
  for(size_t i = 0; i < voxellistNames.size(); i++)
  {
    glutSetMenu(mapMenu);
    std::stringstream tmp;
    tmp << "Toggle Draw " << voxellistNames[i];
    glutAddMenuEntry(tmp.str().data(), 400 + i);
  }

  std::vector<std::string> octreeNames = vis->getOctreeNames();
  for(size_t i = 0; i < octreeNames.size(); i++)
  {
    glutSetMenu(mapMenu);
    std::stringstream tmp;
    tmp << "Toggle Draw " << octreeNames[i];
    glutAddMenuEntry(tmp.str().data(), 500 + i);
  }

  std::vector<std::string> primArrayNames = vis->getPrimitiveArrayNames();
  for(size_t i = 0; i < primArrayNames.size(); i++)
  {
    glutSetMenu(mapMenu);
    std::stringstream tmp;
    tmp << "Toggle Draw " << primArrayNames[i];
    glutAddMenuEntry(tmp.str().data(), 600 + i);
  }


  //define sub menus
  int cameraMenu = glutCreateMenu(&menuFunctionWrapper);
  glutAddMenuEntry("Free", MENU_CAMERA_FREE);
  glutAddMenuEntry("Orbit", MENU_CAMERA_ORBIT);
  glutAddMenuEntry("Reset", MENU_CAMERA_RESET);
  glutAddMenuEntry("Toggle Info", MENU_CAMERA_TOGGLETEXT);

  int gridMenu = glutCreateMenu(&menuFunctionWrapper);
  glutAddMenuEntry("On", MENU_GRID_ON);
  glutAddMenuEntry("Off", MENU_GRID_OFF);

    // sub sub menu of rendermodes
    int rendermodeSubMenuDist = glutCreateMenu(&menuFunctionWrapper);
    glutAddMenuEntry("Default", MENU_RENDERMODE_DIST_DEFAULT);
    glutAddMenuEntry("Two-color gradient", MENU_RENDERMODE_DIST_TWOCOLOR_GRADIENT);
    glutAddMenuEntry("Multicolor gradient", MENU_RENDERMODE_DIST_MULTICOLOR_GRADIENT);
    glutAddMenuEntry("Voronoi linear", MENU_RENDERMODE_DIST_VORONOI_LINEAR);
    glutAddMenuEntry("Voronoi scrambled", MENU_RENDERMODE_DIST_VORONOI_SCRAMBLE);
    glutAddMenuEntry("!Press 2x 's' afterwards!", MENU_NONSENSE);
    int rendermodeSubMenuSlice = glutCreateMenu(&menuFunctionWrapper);
    glutAddMenuEntry("No slicing", MENU_RENDERMODE_SLICING_OFF);
    glutAddMenuEntry("Slice X", MENU_RENDERMODE_SLICING_X);
    glutAddMenuEntry("Slice Y", MENU_RENDERMODE_SLICING_Y);
    glutAddMenuEntry("Slice Z", MENU_RENDERMODE_SLICING_Z);
  int rendermodeMenu = glutCreateMenu(&menuFunctionWrapper);
  glutAddMenuEntry("Solid", MENU_RENDERMODE_SOLID);
  glutAddMenuEntry("Wireframe", MENU_RENDERMODE_WIREFRAME);
  glutAddMenuEntry("Solid+Wireframe", MENU_RENDERMODE_SOLIDWIREFRAME);
  glutAddSubMenu("Distance Maps Rendermode", rendermodeSubMenuDist);
  glutAddSubMenu("Slicing", rendermodeSubMenuSlice);

  int drawmapMenu = glutCreateMenu(&menuFunctionWrapper);
  glutAddMenuEntry("All", MENU_DRAWMAP_ALL);
  glutAddMenuEntry("View", MENU_DRAWMAP_VIEW);

  int depthMenu = glutCreateMenu(&menuFunctionWrapper);
  glutAddMenuEntry("Always", MENU_DEPTHTEST_ALWAYS);
  glutAddMenuEntry("LEqual", MENU_DEPTHTEST_LEQUAL);

  int lightMenu = glutCreateMenu(&menuFunctionWrapper);
  glutAddMenuEntry("On", MENU_LIGHT_ON);
  glutAddMenuEntry("Off", MENU_LIGHT_OFF);

  int visibilityMenu = glutCreateMenu(&menuFunctionWrapper);
  glutAddMenuEntry("Activated", MENU_VISIBILITYTRIGGER_ACTIVATED);
  glutAddMenuEntry("Deacvivated", MENU_VISIBILITYTRIGGER_DEACTIVATED);

  //define main menu
  glutCreateMenu(&menuFunctionWrapper);

  glutAddMenuEntry("Print Help in Console", MENU_HELP);

  glutAddSubMenu("Toggle Text", textMenu);

  if((voxelmapNames.size() + voxellistNames.size() + octreeNames.size() + primArrayNames.size()) > 0)
  glutAddSubMenu("Maps", mapMenu);

  glutAddSubMenu("Camera", cameraMenu);
  glutAddSubMenu("Grid", gridMenu);
  glutAddSubMenu("Render Mode", rendermodeMenu);
  glutAddSubMenu("Draw Map", drawmapMenu);
  glutAddSubMenu("Depth Test", depthMenu);
  glutAddSubMenu("Light", lightMenu);
  glutAddSubMenu("Visibility Trigger", visibilityMenu);

  glutAttachMenu(GLUT_RIGHT_BUTTON);
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
          "Providing a "<< typeToString(map_type) << " called \""<< map_name << "\" with side_length " << voxel_side_length << " and dimension [" << dim.x << ", " << dim.y << ", " << dim.z << "]" << endl);
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
    LOGGING_INFO(Visualization, "Providing an Octree called \""<< map_name << "\"." << endl);
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
// Initialize the logging framework. Also calls icl_core::config::initialize
  icl_core::logging::initialize(argc, argv);

  LOGGING_INFO(Visualization, "Starting the gpu_voxels Visualizer." << endl);
  vis->initializeVisualizer(argc, argv);

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
  vis->initializeDrawTextFlags();

  // TODO: This is catching an arbitrary exception, that is rather a symptom than the cause of the error.
  // If the implementation of the visualization changes, it might throw another exception.
  // This case should rather be caught before this point is reached and the visualizer works with dangling pointers.
  try
  {
    runVisualisation(&argc, argv);
  } catch (const thrust::system::system_error& e)
  {
    // TODO: Is this path portable?
    std::string shmPath("/dev/shm/");
    LOGGING_ERROR(Visualization, "Visualization failed, possibly caused by corrupted shared memory files!" << endl
                                 << "Error message: '" << e.what() << "'. Removing shared memory files in " << shmPath << "..." << endl);

    boost::filesystem::remove(boost::filesystem::path(shmPath + shm_segment_name_octrees));
    boost::filesystem::remove(boost::filesystem::path(shmPath + shm_segment_name_voxelmaps));
    boost::filesystem::remove(boost::filesystem::path(shmPath + shm_segment_name_voxellists));
    boost::filesystem::remove(boost::filesystem::path(shmPath + shm_segment_name_primitive_array));
    boost::filesystem::remove(boost::filesystem::path(shmPath + shm_segment_name_visualizer));
    LOGGING_ERROR(Visualization, "Cleaned up shared memory. Please try to restart your GPU-Voxels processes and then the visualizer." << endl);

    LOGGING_INFO(Visualization, "Exiting.." << endl);
    delete vis;
    icl_core::logging::LoggingManager::instance().shutdown();
    return EXIT_FAILURE;
  }

  LOGGING_INFO(Visualization, "Exiting.." << endl);
  delete vis;
  icl_core::logging::LoggingManager::instance().shutdown();
  return EXIT_SUCCESS;
}
