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
* \author  Herbert Pietrzyk <pietrzyk@fzi.de|
* \date    2017-11-20
*
*
* This example program tests the CountingVoxelList
*
*/
//----------------------------------------------------------------------

#include <cstdlib>
#include <signal.h>
#include <typeinfo>

#include <icl_core_config/Config.h>

#include <gpu_voxels/GpuVoxels.h>
#include <gpu_voxels/helpers/MetaPointCloud.h>
#include <gpu_voxels/logging/logging_gpu_voxels.h>

GpuVoxelsSharedPtr gvl;

void ctrlchandler(int)
{
  //delete gvl;
  gvl.reset();
  exit(EXIT_SUCCESS);
}
void killhandler(int)
{
  //delete gvl;
  gvl.reset();
  exit(EXIT_SUCCESS);
}

int main(int argc, char* argv[])
{
  signal(SIGINT, ctrlchandler);
  signal(SIGTERM, killhandler);

  icl_core::logging::initialize(argc, argv);

  gvl = GpuVoxels::getInstance();
  gvl->initialize(200, 200, 200, 0.01);

  gvl->addMap(MT_COUNTING_VOXELLIST, "CountingMap");
  gvl->addMap(MT_COUNTING_VOXELLIST, "CountingFilter0Map");
  gvl->addMap(MT_COUNTING_VOXELLIST, "CountingFilter1Map");
  gvl->addMap(MT_COUNTING_VOXELLIST, "CountingFilter2Map");
  gvl->addMap(MT_BITVECTOR_VOXELLIST, "BitvectorMap");
  gvl->addMap(MT_BITVECTOR_VOXELLIST, "ObstacleBitvectorMap");

  // We load a pointcloud
  if (!gvl->insertPointCloudFromFile("BitvectorMap", "robot4cmRes.pcd", true,
                                     gpu_voxels::eBVM_OCCUPIED, true, gpu_voxels::Vector3f(0.3, 0.2, 0.0),0.5))
  {
    LOGGING_WARNING(gpu_voxels::Gpu_voxels, "Could not insert the pointcloud..." << gpu_voxels::endl);
  }
  if (!gvl->insertPointCloudFromFile("CountingMap", "robot4cmRes.pcd", true,
                                     gpu_voxels::eBVM_OCCUPIED, true, gpu_voxels::Vector3f(0.3, 0.2, 0.0),0.5))
  {
    LOGGING_WARNING(gpu_voxels::Gpu_voxels, "Could not insert the pointcloud..." << gpu_voxels::endl);
  }

  Vector3f center1_min = Vector3f(0.0, 0.0, 0.2);
  Vector3f center1_max = Vector3f(2.0, 2.0, 0.5);
  gvl->insertBoxIntoMap(center1_min, center1_max, "ObstacleBitvectorMap", gpu_voxels::eBVM_OCCUPIED, 1);
//   gvl->insertPointCloudFromFile("ObstacleBitvectorMap", "robot4cmRes.pcd", true,
//                                      gpu_voxels::eBVM_OCCUPIED, true, gpu_voxels::Vector3f(0.3, 0.2, 0.0),0.5);
  // cut the robot:
  gvl->getMap("CountingMap")->as<gpu_voxels::voxellist::CountingVoxelList>()->subtractFromCountingVoxelList(
      gvl->getMap("ObstacleBitvectorMap")->as<gpu_voxels::voxellist::BitVectorVoxelList>(),
      Vector3f());
  gvl->getMap("BitvectorMap")->as<gpu_voxels::voxellist::BitVectorVoxelList>()->subtract(
      gvl->getMap("ObstacleBitvectorMap")->as<gpu_voxels::voxellist::BitVectorVoxelList>(),
      Vector3f());

   //insert an obstacle to set each Counting voxel to 1
   //TODO: fix bug in insertBoxIntoMap resulting in voxels appearing up to 15 times!
   //TODO: test by building vector of point that fall into voxel centers
   Vector3f center2_min = Vector3f(0.1, 0.5, 0.7);
   Vector3f center2_max = Vector3f(0.5, 1.0, 0.9);
   gvl->insertBoxIntoMap(center2_min, center2_max, "CountingFilter0Map", gpu_voxels::eBVM_OCCUPIED, 1);
   gvl->insertBoxIntoMap(center2_min, center2_max, "CountingFilter1Map", gpu_voxels::eBVM_OCCUPIED, 1);
   gvl->insertBoxIntoMap(center2_min, center2_max, "CountingFilter2Map", gpu_voxels::eBVM_OCCUPIED, 1);

  // insert an obstacle to set each Counting voxel to 1
  std::vector<Vector3f> listPoints;
  listPoints.push_back(Vector3f(0.2f, 0.7f, 0.5f));
  listPoints.push_back(Vector3f(0.2f, 0.7f, 0.4f));
  listPoints.push_back(Vector3f(0.2f, 0.7f, 0.4f));
  listPoints.push_back(Vector3f(0.2f, 0.7f, 0.3f));
  listPoints.push_back(Vector3f(0.2f, 0.7f, 0.3f));
  listPoints.push_back(Vector3f(0.2f, 0.7f, 0.3f));
  //TODO WTFBBQ: results in voxel content 6, 4 and 2 !?!
  gvl->insertPointCloudIntoMap(listPoints, "CountingFilter0Map", gpu_voxels::eBVM_OCCUPIED);
  gvl->insertPointCloudIntoMap(listPoints, "CountingFilter1Map", gpu_voxels::eBVM_OCCUPIED);
  gvl->insertPointCloudIntoMap(listPoints, "CountingFilter2Map", gpu_voxels::eBVM_OCCUPIED);

  // remove underpopulated voxels: should result in one empty voxellist and two untouched ones
  gvl->getMap("CountingFilter0Map")->as<gpu_voxels::voxellist::CountingVoxelList>()->remove_underpopulated(0);
  gvl->getMap("CountingFilter1Map")->as<gpu_voxels::voxellist::CountingVoxelList>()->remove_underpopulated(1);
  gvl->getMap("CountingFilter2Map")->as<gpu_voxels::voxellist::CountingVoxelList>()->remove_underpopulated(2);

  std::cout << "CountingFilter0Map size: " <<
    gvl->getMap("CountingFilter0Map")->as<gpu_voxels::voxellist::CountingVoxelList>()->m_dev_id_list.size() << std::endl;
  std::cout << "CountingFilter1Map size: " <<
    gvl->getMap("CountingFilter1Map")->as<gpu_voxels::voxellist::CountingVoxelList>()->m_dev_id_list.size() << std::endl;
  std::cout << "CountingFilter2Map size: " <<
    gvl->getMap("CountingFilter2Map")-> as<gpu_voxels::voxellist::CountingVoxelList> ()->m_dev_id_list.size() << std::endl;

//  gvl->getMap("CountingFilter2Map")-> as<gpu_voxels::voxellist::CountingVoxelList> ()->screendump(true);

  std::cout << "START DRAWING" << std::endl;

  gvl->visualizeMap("CountingFilter0Map");
  gvl->visualizeMap("CountingFilter1Map");
  gvl->visualizeMap("CountingFilter2Map");

  std::string input;
  while(true)
  {
    std::cout << "Set filter threshold to? : " << std::endl;
    std::getline(std::cin, input);
    float f = atof(input.c_str());
    size_t collisions = gvl->getMap("CountingMap")->as<gpu_voxels::voxellist::CountingVoxelList>()->collideWith(gvl->getMap("BitvectorMap")->as<gpu_voxels::voxellist::BitVectorVoxelList>(), f);
    std::cout << "Number of Collisions: " << collisions << std::endl;

    gvl->visualizeMap("CountingMap");
    gvl->visualizeMap("BitvectorMap");
    gvl->visualizeMap("ObstacleBitvectorMap");
  }

  gvl.reset();
  std::cout << "PROGRAMM FINISHED" << std::endl;
}
