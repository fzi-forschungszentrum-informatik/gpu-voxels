// this is for emacs file handling -*- mode: c++; indent-tabs-mode: nil -*-

// -- BEGIN LICENSE BLOCK ----------------------------------------------
// -- END LICENSE BLOCK ------------------------------------------------

//----------------------------------------------------------------------
/*!\file
*
* \author  Felix Mauch <mauch@fzi.de>
* \date    2015-05-04
*
*/
//----------------------------------------------------------------------
#include <cstdlib>
#include <signal.h>

#include <gpu_voxels/GpuVoxels.h>
#include <gpu_voxels/helpers/MetaPointCloud.h>
#include <gpu_voxels/logging/logging_gpu_voxels.h>

using namespace gpu_voxels;

GpuVoxels* gvl;

void ctrlchandler(int)
{
  delete gvl;
  exit(EXIT_SUCCESS);
}
void killhandler(int)
{
  delete gvl;
  exit(EXIT_SUCCESS);
}

int main(int argc, char* argv[])
{
  signal(SIGINT, ctrlchandler);
  signal(SIGTERM, killhandler);

  icl_core::logging::initialize(argc, argv);

  gvl = new GpuVoxels(100, 100, 100, 0.1);

  gvl->addMap(MT_BITVECTOR_VOXELLIST, "myVoxelList");
  gvl->addMap(MT_BITVECTOR_VOXELLIST, "myVoxelList2");
  gvl->addMap(MT_BITVECTOR_OCTREE, "myOctree");
  gvl->addMap(MT_BITVECTOR_VOXELMAP, "myVoxelMap");
  gvl->addMap(MT_PROBAB_VOXELMAP, "myProbVoxelMap");

  Vector3f center1_min(0.1,0.1,0.1);
  Vector3f center1_max(0.7,0.7,0.7);

//  gvl->insertBoxIntoMap(center1_min, center1_max, "myVoxelList", eBVM_OCCUPIED, 1);
  gvl->insertBoxIntoMap(center1_min, center1_max, "myVoxelMap", gpu_voxels::BitVoxelMeaning(24), 1);

  Vector3f center2_min(0.5,0.6,0.6);
  Vector3f center2_max(3.7,3.7,3.7);
  gvl->insertBoxIntoMap(center2_min, center2_max, "myOctree", gpu_voxels::BitVoxelMeaning(24), 1);

  gvl->insertBoxIntoMap(center2_min, center2_max, "myProbVoxelMap", gpu_voxels::eBVM_OCCUPIED, 1);

  gvl->insertBoxIntoMap(center1_min, center1_max, "myVoxelList", gpu_voxels::BitVoxelMeaning(23), 1);
  std::cout << "Voxellist1 size: " << gvl->getMap("myVoxelList")->getDimensions().x << " voxels" << std::endl;

  // We load the model of a coordinate system.
  if (!gvl->insertPointcloudFromFile("myVoxelList2", "coordinate_system_100.binvox", true,
                                     gpu_voxels::BitVoxelMeaning(24), true, Vector3f(0, 0, 0),1.0))
  {
    LOGGING_WARNING(Gpu_voxels, "Could not insert the PCD file..." << endl);
  }

  //std::cout << "Voxellist2 size: " << gvl->getMap("myVoxelList2")->getDimensions().x << " voxels" << std::endl;

  BitVectorVoxel collision_types_map;
  size_t num_colls = gvl->getMap("myVoxelList")->collideWithTypes(gvl->getMap("myProbVoxelMap"), collision_types_map, 1.0f);
  std::cout << "Voxellist1 collided with Probab Voxelmap. Bitcheck gives: " << num_colls << std::endl;
  std::cout << collision_types_map << std::endl;

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
