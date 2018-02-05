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
* \author  Felix Mauch <mauch@fzi.de>
* \date    2015-05-04
*
* This program demonstrates various collision functions of VoxelLists.
*
*/
//----------------------------------------------------------------------
#include <cstdlib>
#include <signal.h>

#include <gpu_voxels/GpuVoxels.h>
#include <gpu_voxels/helpers/MetaPointCloud.h>
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

  gvl = GpuVoxels::getInstance();
  gvl->initialize(100, 100, 100, 0.1);

  gvl->addMap(MT_BITVECTOR_VOXELLIST, "myVoxelList");
  gvl->addMap(MT_PROBAB_VOXELMAP, "myProbVoxelMap");


//  gvl->addMap(MT_BITVECTOR_VOXELLIST, "myVoxelList2");
//  gvl->addMap(MT_BITVECTOR_OCTREE, "myOctree");
//  gvl->addMap(MT_BITVECTOR_VOXELMAP, "myVoxelMap");


  Vector3f center1_min(0.09,0.09,0.09);
  Vector3f center1_max(0.91,0.41,0.41);
  gvl->insertBoxIntoMap(center1_min, center1_max, "myProbVoxelMap", gpu_voxels::eBVM_OCCUPIED, 1);

  center1_min = Vector3f(0.09,0.29,0.09);
  center1_max = Vector3f(0.31,0.61,0.41);
  gvl->insertBoxIntoMap(center1_min, center1_max, "myVoxelList", gpu_voxels::BitVoxelMeaning(34), 1);

  center1_min = Vector3f(0.29,0.29,0.09);
  center1_max = Vector3f(0.61,0.61,0.41);
  gvl->insertBoxIntoMap(center1_min, center1_max, "myVoxelList", gpu_voxels::BitVoxelMeaning(63), 1);

  center1_min = Vector3f(0.59,0.29,0.09);
  center1_max = Vector3f(0.91,0.61,0.41);
  gvl->insertBoxIntoMap(center1_min, center1_max, "myVoxelList", gpu_voxels::BitVoxelMeaning(102), 1);

  std::cout << "Voxellist1 size: " << gvl->getMap("myVoxelList")->getDimensions().x << " voxels" << std::endl;

/*  // We load the model of a coordinate system.
  if (!gvl->insertPointCloudFromFile("myVoxelList2", "coordinate_system_100.binvox", true,
                                     gpu_voxels::BitVoxelMeaning(24), true, Vector3f(0, 0, 0),1.0))
  {
    LOGGING_WARNING(Gpu_voxels, "Could not insert the PCD file..." << endl);
  }

  //std::cout << "Voxellist2 size: " << gvl->getMap("myVoxelList2")->getDimensions().x << " voxels" << std::endl;
*/

  size_t num_colls;

  BitVectorVoxel collision_types_map;
//  num_colls = gvl->getMap("myVoxelList")->as<voxellist::BitVectorVoxelList>()->collideWithTypes(gvl->getMap("myProbVoxelMap")->as<voxelmap::ProbVoxelMap>(), collision_types_map, 1.0f);
//  std::cout << "Voxellist1 collided with Probab Voxelmap. Bitcheck gives: " << num_colls << std::endl;
//  std::cout << "In voxeltypes: " << collision_types_map << std::endl;


  BitVectorVoxel types_to_check;
  num_colls = gvl->getMap("myVoxelList")->as<voxellist::BitVectorVoxelList>()->collideWithTypeMask(gvl->getMap("myProbVoxelMap")->as<voxelmap::ProbVoxelMap>(), types_to_check, 1.0f);
  std::cout << "Voxellist1 collided with Probab Voxelmap. No Bits in mask set: " << num_colls << std::endl;

//  types_to_check.bitVector().setBit(34);
//  num_colls = gvl->getMap("myVoxelList")->as<voxellist::BitVectorVoxelList>()->collideWithTypeMask(gvl->getMap("myProbVoxelMap")->as<voxelmap::ProbVoxelMap>(), types_to_check, 1.0f);
//  std::cout << "Voxellist1 collided with Probab Voxelmap. Bit 34 in mask set: " << num_colls << std::endl;

  types_to_check.bitVector().setBit(63);
  num_colls = gvl->getMap("myVoxelList")->as<voxellist::BitVectorVoxelList>()->collideWithTypeMask(gvl->getMap("myProbVoxelMap")->as<voxelmap::ProbVoxelMap>(), types_to_check, 1.0f, Vector3i(0,-1,0));
  std::cout << "Voxellist1 collided with Probab Voxelmap. Bit 63 in mask set: " << num_colls << std::endl;

//  types_to_check.bitVector().setBit(66);
//  num_colls = gvl->getMap("myVoxelList")->as<voxellist::BitVectorVoxelList>()->collideWithTypeMask(gvl->getMap("myProbVoxelMap")->as<voxelmap::ProbVoxelMap>(), types_to_check, 1.0f);
//  std::cout << "Voxellist1 collided with Probab Voxelmap. Bit 66 in mask set: " << num_colls << std::endl;

//  types_to_check.bitVector().setBit(102);
//  num_colls = gvl->getMap("myVoxelList")->as<voxellist::BitVectorVoxelList>()->collideWithTypeMask(gvl->getMap("myProbVoxelMap")->as<voxelmap::ProbVoxelMap>(), types_to_check, 1.0f);
//  std::cout << "Voxellist1 collided with Probab Voxelmap. Bit 102 in mask set: " << num_colls << std::endl;

//  bool bin_coll = gvl->getMap("myVoxelList")->collideWith(gvl->getMap("myVoxelList2"));
//  std::cout << "Voxellist1 collided with Voxellist2: " << bin_coll << std::endl;


//  size_t num_colls = gvl->getMap("myVoxelList")->collideWithBitcheck(gvl->getMap("myVoxelList2"));
//  std::cout << "Voxellist1 collided with Voxellist2. Bitcheck gives: " << num_colls << std::endl;

//  BitVectorVoxel collision_types_map;
//  num_colls = gvl->getMap("myVoxelList")->collideWithTypes(gvl->getMap("myVoxelList2"), collision_types_map, 1.0f);
//  std::cout << "Voxellist1 collided with Voxellist2. Types in collision gives: " << num_colls << std::endl;
//  std::cout << collision_types_map << std::endl;

//  BitVectorVoxel collision_types_list;
//  uint32_t num_collisions_list = gvl->getMap("myOctree")->collideWithTypes(gvl->getMap("myVoxelList"), collision_types_list, 1.0f);
//  BitVectorVoxel collision_types_map;
//  uint32_t num_collisions_map = gvl->getMap("myOctree")->collideWithTypes(gvl->getMap("myVoxelMap"), collision_types_map, 1.0f);

//  std::cout << "Number of collisions octree with list: " << num_collisions_list << "\n";
//  std::cout << "Number of collisions octree with map: " << num_collisions_map << "\n";

//  num_collisions_list = gvl->getMap("myVoxelList2")->collideWith(gvl->getMap("myVoxelList"));
//  std::cout << "Number of collisions list with list: " << num_collisions_list << "\n";
//  num_collisions_list = gvl->getMap("myVoxelList2")->collideWith(gvl->getMap("myVoxelMap"));
//  std::cout << "Number of collisions list with map: " << num_collisions_list << "\n";

//  BitVectorVoxel collisions_list_list;
//  gvl->getMap("myVoxelList")->collideWithTypes(gvl->getMap("myVoxelList2"), collisions_list_list);

//  std::cout << "Collinding types within lists:" << collisions_list_list << std::endl;

  while(true)
  {
    gvl->visualizeMap("myVoxelList");
    gvl->visualizeMap("myProbVoxelMap");
    sleep(1);
  }

  return 0;
}
