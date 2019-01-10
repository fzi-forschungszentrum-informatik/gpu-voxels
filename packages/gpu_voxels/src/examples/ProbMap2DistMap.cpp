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
* \author  Andreas Hermann <hermann@fzi.de>
* \date    2018-03-13
*
* This program demonstrates how to insert sensor data into a probabilistic map,
* and then using PBA to compute the free space around each voxel.
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

  uint32_t m1 = 1;
  uint32_t m2 = 1;
  uint32_t m3 = 4;

  signal(SIGINT, ctrlchandler);
  signal(SIGTERM, killhandler);

  icl_core::logging::initialize(argc, argv);

  gvl = GpuVoxels::getInstance();

  Vector3ui dim(128, 128, 128);
  float side_length = 1.0; // voxel side length
  gvl->initialize(dim.x, dim.y, dim.z, side_length);

  gvl->addMap(MT_PROBAB_VOXELMAP, "myProbVoxelMap");
  gvl->addMap(MT_DISTANCE_VOXELMAP, "myDistanceMap");
  boost::shared_ptr<ProbVoxelMap> prob_map(gvl->getMap("myProbVoxelMap")->as<ProbVoxelMap>());
  boost::shared_ptr<DistanceVoxelMap> dist_map(gvl->getMap("myDistanceMap")->as<DistanceVoxelMap>());

  std::vector<Vector3f> this_testpoints1;
  std::vector<Vector3f> this_testpoints2;

  // create two partly overlapping cubes
  this_testpoints1 = createBoxOfPoints( Vector3f(40, 40, 40), Vector3f(60, 60, 60), 0.9);
  this_testpoints2 = createBoxOfPoints( Vector3f(50, 50, 50), Vector3f(70, 70, 70), 0.9);

  // insert a cube with high occupancy
  prob_map->insertPointCloud(this_testpoints1, eBVM_MAX_OCC_PROB);  // Prob = 0.9843 due to available encoding precision
  // insert a cube with a value that will be interpreted as free space
  prob_map->insertPointCloud(this_testpoints2, eBVM_MAX_FREE_PROB); // Prob = 0.0157 due to available encoding precision
  // the overlapping section of the two cubes will have a probability of 0.5

  // add only voxels with high occupancy
  float occupancy_threshold = 0.55f;
  dist_map->mergeOccupied(prob_map, Vector3ui(0), occupancy_threshold); // merge only those voxels with occupancy exceeding this threshold

  dist_map->parallelBanding3D(m1, m2, m3, PBA_DEFAULT_M1_BLOCK_SIZE, PBA_DEFAULT_M2_BLOCK_SIZE, PBA_DEFAULT_M3_BLOCK_SIZE, 1);

  while(true)
  {
    gvl->visualizeMap("myProbVoxelMap");
    gvl->visualizeMap("myDistanceMap");
    sleep(1);
  }

  return 0;
}
