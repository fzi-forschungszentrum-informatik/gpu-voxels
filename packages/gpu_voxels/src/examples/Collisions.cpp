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
* \date    2016-07-18
*
*
* This example program shows different ways to collide two objects with gpu voxels.
*
*/
//----------------------------------------------------------------------
#include <signal.h>

#include <gpu_voxels/GpuVoxels.h>
#include <gpu_voxels/logging/logging_gpu_voxels.h>

gpu_voxels::GpuVoxelsSharedPtr gvl;


// We define exit handlers to quit the program:
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

  // First we initialize gpu voxels
  icl_core::logging::initialize(argc, argv);
  gvl = gpu_voxels::GpuVoxels::getInstance();
  gvl->initialize(200, 200, 200, 0.01);

  // We add different maps with objects, to collide them
  gvl->addMap(gpu_voxels::MT_BITVECTOR_VOXELLIST,"myPointcloud");
  gvl->addMap(gpu_voxels::MT_PROBAB_VOXELMAP,"myObstacles");
  gvl->addMap(gpu_voxels::MT_BITVECTOR_VOXELLIST, "mySweptVolume");

  // We load a pointcloud
  if (!gvl->insertPointCloudFromFile("myPointcloud", "robot4cmRes.pcd", true,
                                     gpu_voxels::eBVM_OCCUPIED, true, gpu_voxels::Vector3f(0.3, 0.2, 0.0),0.5))
  {
    LOGGING_WARNING(gpu_voxels::Gpu_voxels, "Could not insert the pointcloud..." << gpu_voxels::endl);
  }
  gvl->visualizeMap("myPointcloud");

  /*
   * We add a SWEPT VOLUME.
   * With a Swept Volume we can "voxelize" a moving object (like a robot) in every step and insert it into a map.
   * As the map is not cleared, this will generate a sweep.
   * The ID within the sweep is incremented with the single poses
   * so we can later identify, which pose created a collision.
   */
  const int num_swept_volumes = 50;// < eBVM_SWEPT_VOLUME_END;
  gpu_voxels::Vector3f center0_min(0.3,0.7,0.3);
  gpu_voxels::Vector3f center0_max(0.4,0.8,0.7);
  for (int i = 0; i < num_swept_volumes; ++i)
  {
    gpu_voxels::Vector3f corner0_min = center0_min + gpu_voxels::Vector3f(0.025*i, 0.0, 0.0);
    gpu_voxels::Vector3f corner0_max = center0_max + gpu_voxels::Vector3f(0.025*i, 0.0, 0.0);
    gpu_voxels::BitVoxelMeaning v = gpu_voxels::BitVoxelMeaning(gpu_voxels::eBVM_SWEPT_VOLUME_START + i);
    gvl->insertBoxIntoMap(corner0_min, corner0_max, "mySweptVolume", v);
  }
  gvl->visualizeMap("mySweptVolume");

  // These coordinates are used for two boxes, wich will represent our obstacles
  gpu_voxels::Vector3f center1_min(0.5,0.5,0.5);
  gpu_voxels::Vector3f center1_max(0.6,0.6,0.6);
  gpu_voxels::Vector3f center2_min(0.5,0.5,0.3);
  gpu_voxels::Vector3f center2_max(0.6,0.6,0.4);
  gpu_voxels::Vector3f corner1_min;
  gpu_voxels::Vector3f corner2_min;
  gpu_voxels::Vector3f corner1_max;
  gpu_voxels::Vector3f corner2_max;

  // Now we can start the main loop, wich will move some boxes and collide them with the pointcloud and the Swept Volume
  float t = 0.0;
  while(true)
  {
    t += 0.03;

    // Move the boxes
    float x = sin(t);
    float y = cos(t);

    corner1_min = center1_min + gpu_voxels::Vector3f(0.2 * x, 0.2 * y, 0);
    corner1_max = center1_max + gpu_voxels::Vector3f(0.2 * x, 0.2 * y, 0);
    gvl->insertBoxIntoMap(corner1_min, corner1_max, "myObstacles", gpu_voxels::eBVM_OCCUPIED, 2);

    corner2_min = center2_min + gpu_voxels::Vector3f(0 , 0.4 * x, 0);
    corner2_max = center2_max + gpu_voxels::Vector3f(0 , 0.4 * x, 0);
    gvl->insertBoxIntoMap(corner2_min, corner2_max, "myObstacles", gpu_voxels::eBVM_OCCUPIED, 2);

    /*
     * Now we can collide different Objects in different ways.
     *
     * The "collideWith" function will return the number of collisions between the objects.
     */
    size_t num_colls_pc = gvl->getMap("myPointcloud")->as<gpu_voxels::voxellist::BitVectorVoxelList>()->collideWith(gvl->getMap("myObstacles")->as<gpu_voxels::voxelmap::ProbVoxelMap>());
    LOGGING_INFO(gpu_voxels::Gpu_voxels, "collisions with pointcloud: " << num_colls_pc << gpu_voxels::endl );

    // You can also get the colliding bits. This is especially useful when working with a Swept Volume. The bits will tell you wich step of the sweep is colliding.
    gpu_voxels::BitVectorVoxel collision_bits;
    size_t num_colls_r = gvl->getMap("mySweptVolume")->as<gpu_voxels::voxellist::BitVectorVoxelList>()->collideWithTypes(gvl->getMap("myObstacles")->as<gpu_voxels::voxelmap::ProbVoxelMap>(), collision_bits, 1.0);
    LOGGING_INFO(gpu_voxels::Gpu_voxels, "collisions with SWEPT VOLUME: " << num_colls_r  << " with bits: \n" << collision_bits << gpu_voxels::endl);

    // don't forget to keep your varying Maps up to date
    gvl->visualizeMap("myObstacles");
    usleep(100000); // give the visualizer some time to render the maps
    gvl->clearMap("myObstacles");

  }

}
