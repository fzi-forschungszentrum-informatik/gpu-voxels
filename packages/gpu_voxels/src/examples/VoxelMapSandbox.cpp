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
* This program demonstrates collisions between two VoxelMaps.
*
*/
//----------------------------------------------------------------------
#include <cstdlib>
#include <signal.h>

#include <gpu_voxels/GpuVoxels.h>
#include <gpu_voxels/helpers/GeometryGeneration.h>
#include <gpu_voxels/logging/logging_gpu_voxels.h>

using namespace gpu_voxels;
using namespace voxelmap;
using namespace geometry_generation;

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

  Vector3ui dim(89, 123, 74);
  float side_length = 1.f; // voxel side length
  gvl->initialize(dim.x, dim.y, dim.z, side_length);

  gvl->addMap(MT_PROBAB_VOXELMAP, "myProbVoxelMap1");
  gvl->addMap(MT_PROBAB_VOXELMAP, "myProbVoxelMap2");
  boost::shared_ptr<ProbVoxelMap> map_1(gvl->getMap("myProbVoxelMap1")->as<ProbVoxelMap>());
  boost::shared_ptr<ProbVoxelMap> map_2(gvl->getMap("myProbVoxelMap2")->as<ProbVoxelMap>());

  std::vector<Vector3f> this_testpoints1;
  std::vector<Vector3f> this_testpoints2;

  this_testpoints1 = createBoxOfPoints( Vector3f(2.1, 2.1, 2.1), Vector3f(4.1, 4.1, 4.1), 0.5);
  this_testpoints2 = createBoxOfPoints( Vector3f(3.1, 3.1, 3.1), Vector3f(5.1, 5.1, 5.1), 0.5);

  map_1->insertPointCloud(this_testpoints1, eBVM_OCCUPIED);
  map_2->insertPointCloud(this_testpoints2, eBVM_OCCUPIED);

  std::cout << "Collisions w offset: " << map_1->collideWith(map_2.get(), 0.1, Vector3i(-1,-0,-1)) << std::endl;
  std::cout << "Collisions w/o offset: " << map_1->collideWith(map_2.get(), 0.1) << std::endl;

  while(true)
  {
    gvl->visualizeMap("myProbVoxelMap1");
    gvl->visualizeMap("myProbVoxelMap2");
    sleep(1);
  }

  return 0;
}
