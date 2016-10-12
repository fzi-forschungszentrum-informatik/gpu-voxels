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
 * \author  Andreas Hermann
 * \date    2016-11-27
 *
 * This little example program shows how to populate a VoxelMap
 * with height-fields from a bitmap graphics file.
 *
 */
//----------------------------------------------------------------------
#include <cstdlib>
#include <signal.h>

#include <gpu_voxels/GpuVoxels.h>
#include <gpu_voxels/helpers/MetaPointCloud.h>
#include <gpu_voxels/helpers/HeightMapLoader.h>
#include <gpu_voxels/logging/logging_gpu_voxels.h>

using namespace gpu_voxels;

GpuVoxelsSharedPtr gvl;

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
   */
  gvl = GpuVoxels::getInstance();
  gvl->initialize(1120, 910, 270, 0.04);

  // Now we add some maps
  gvl->addMap(MT_PROBAB_OCTREE, "myProbabVoxmap");

  // First we generate a pointcloud out of the height-map:
  gpu_voxels::PointCloud cloud;
  file_handling::HeightMapLoader("fzi_4cm_per_pixel.png", "", true, 1, 255, 0.04, 0.01, gpu_voxels::Vector3f(0), cloud);
  
  // Then we insert it into the VoxelMap:
  gvl->getMap("myProbabVoxmap")->insertPointCloud(cloud, eBVM_OCCUPIED);
  
  std::cout << "Stuff loaded..." << std::endl;

  // Now we start the main loop, to visualize the scene.
  while(true)
  {
    // tell the visualier that the maps have changed
    gvl->visualizeMap("myProbabVoxmap");
    
    usleep(30000);
  }

}
