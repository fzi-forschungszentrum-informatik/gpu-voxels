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
* \author  Darius Pietsch <pietsch@fzi.de|
* \date    2016-07-05
*
*
* This example program shows the different map types available in gpu-voxels
* and their usage.
*
*/
//----------------------------------------------------------------------

#include <signal.h>


#include <gpu_voxels/GpuVoxels.h>
#include <gpu_voxels/logging/logging_gpu_voxels.h>

gpu_voxels::GpuVoxelsSharedPtr gvl;

void ctrlchandler(int)
{
  gvl.reset();
  exit(EXIT_SUCCESS);
}
void killhandler(int)
{
  gvl.reset();
  exit(EXIT_SUCCESS);
}

int main(int argc, char* argv[])
{
  signal(SIGINT, ctrlchandler);
  signal(SIGTERM, killhandler);

  icl_core::logging::initialize(argc, argv);

  /*
   * First, we generate an API class, which defines the
   * volume of our space and the resolution.
   * Be careful here! The size is limited by the memory
   * of your GPU. Even if an empty Octree is small, a
   * Voxelmap will always require the full memory.
   */
  gvl = gpu_voxels::GpuVoxels::getInstance();
  gvl->initialize(200, 200, 200, 0.01);



  // Now we can add different types of maps and assign a name.
  gvl->addMap(gpu_voxels::MT_BITVECTOR_VOXELMAP, "myBitmapVoxmap");           // 3D-Array of deterministic Voxels (identified by their Voxelmap-like Pointer adress) that hold a Bitvector
  gvl->addMap(gpu_voxels::MT_BITVECTOR_VOXELLIST, "myBitmapVoxlist");         // List of     deterministic Voxels (identified by their Voxelmap-like Pointer adress) that hold a Bitvector
  gvl->addMap(gpu_voxels::MT_BITVECTOR_OCTREE, "myBitmapOctree");             // Octree of   deterministic Voxels (identified by a Morton Code)                      that hold a Bitvector


  gvl->addMap(gpu_voxels::MT_PROBAB_VOXELMAP, "myProbabVoxmap");              // 3D-Array of probabilistic Voxels (identified by their Voxelmap-like Pointer adress) that hold a Probability
  gvl->addMap(gpu_voxels::MT_PROBAB_OCTREE, "myProbabOctree");                // Octree of   probabilistic Voxels (identified by a Morton Code)                      that hold a Probability




  /*
   * At this point we can add geometries to the maps in different ways.
   *
   * We can add a simple box.
   */
  gpu_voxels::Vector3f center_box1_min(0.5,0.5,0.5);
  gpu_voxels::Vector3f center_box1_max(0.6,0.6,0.6);
  gvl->insertBoxIntoMap(center_box1_min,center_box1_max,"myBitmapVoxmap",gpu_voxels::eBVM_OCCUPIED);

  /*
   * We can add a pointcloud from file (must end in .xyz for XYZ files, .pcd for PCD files or .binvox for Binvox files).
   * The GPU_VOXELS_MODEL_PATH environment variable holds the directory of the files and needs to be set to ../gpu_voxels/models/
   */
  if (!gvl->insertPointCloudFromFile("myProbabVoxmap", "coordinate_system_100.binvox", true,
                                     gpu_voxels::eBVM_OCCUPIED, true, gpu_voxels::Vector3f(0, 0, 0),0.5))
  {
    LOGGING_WARNING(gpu_voxels::Gpu_voxels, "Could not insert the pointcloud..." << gpu_voxels::endl );
  }

  /*
   * With
   * gvl->insertRobotIntoMap
   * we can insert a whole robot into a map, which is shown in
   * RobotVsEnvironment
   */

  /*
   * To see our maps we have to visualize them.
   * You can determin if you want to force a redpaint of the map. Not working?
   */
  gvl->visualizeMap("myBitmapVoxmap");
  gvl->visualizeMap("myProbabVoxmap");

  //take a couple seconds to look at the result.
  usleep(10000000);

  //We can delete all the data from a map:
  gvl->clearMap("myBitmapVoxmap");








  /*
   * Let's animate the scene with a moving box.
   */
  gpu_voxels::Vector3f corner_box1_min;
  gpu_voxels::Vector3f corner_box1_max;

  float t = 0.0;
  LOGGING_INFO(gpu_voxels::Gpu_voxels, "start loop" << gpu_voxels::endl);
  while(true)
  {
    t += 0.03;


    float x = sin(t);
    float y = cos(t);

    corner_box1_min = center_box1_min + gpu_voxels::Vector3f(0.2 * x, 0.2 * y, 0);
    corner_box1_max = center_box1_max + gpu_voxels::Vector3f(0.2 * x, 0.2 * y, 0);


    // We cleared the old map and can now insert a transformed box.
    gvl->insertBoxIntoMap(corner_box1_min,corner_box1_max,"myBitmapVoxmap",gpu_voxels::eBVM_OCCUPIED, 2);

    // Tell the visualier that maps have changed.
    gvl->visualizeMap("myBitmapVoxmap");

    usleep(100000);

    //Reset the map:
    gvl->clearMap("myBitmapVoxmap");
  }

}

