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
* while using raycasting to mark the free space.
* The robot will get cut out.
* You will see some slight unrealistic effects in the results, which are caused by the
* unrealistic "sensor point cloud", as the sensor can see through the spheres.
* Such effects will not occur when using a real sensor.
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

  Vector3ui dim(128, 128, 128);
  float side_length = 1.0; // voxel side length
  gvl->initialize(dim.x, dim.y, dim.z, side_length);

  gvl->addMap(MT_PROBAB_VOXELMAP, "myProbVoxelMap");
  gvl->addMap(MT_PROBAB_VOXELMAP, "myProbVoxelMap2");
  gvl->addMap(MT_BITVECTOR_VOXELMAP, "myRobotMap");
  boost::shared_ptr<ProbVoxelMap> prob_map(gvl->getMap("myProbVoxelMap")->as<ProbVoxelMap>());
  boost::shared_ptr<ProbVoxelMap> prob_map2(gvl->getMap("myProbVoxelMap2")->as<ProbVoxelMap>());
  boost::shared_ptr<BitVectorVoxelMap> rob_map(gvl->getMap("myRobotMap")->as<BitVectorVoxelMap>());

  std::vector<Vector3f> testpoints1;
  std::vector<Vector3f> testpoints2;
  std::vector<Vector3f> testpoints3;

  std::vector<Vector3f> testpoints_rob1;
  std::vector<Vector3f> testpoints_rob2;

  testpoints1 = createSphereOfPoints(Vector3f(50, 65, 50), 10.0, 0.9);
  testpoints2 = createSphereOfPoints(Vector3f(65, 50, 50), 8.0, 0.9);

  testpoints_rob1 = createBoxOfPoints( Vector3f(35,50,35), Vector3f(50, 90, 65), 0.9);
  testpoints_rob2 = createBoxOfPoints( Vector3f(35,65,35), Vector3f(65, 80, 65), 0.9);

  PointCloud obstacle;
  obstacle.add(testpoints1);
  obstacle.add(testpoints2);
  obstacle.add(testpoints3);

  rob_map->insertPointCloud(testpoints_rob1, eBVM_OCCUPIED);
  rob_map->insertPointCloud(testpoints_rob2, eBVM_OCCUPIED);


  for(size_t pos = 50; pos < 100; ++pos)
  {
    // We perform the insertion and the raycasting twice, so that in the visualizer we can display either all probabilistic voxels or only the occupied ones (without the free ones)
    prob_map->insertSensorData(obstacle, Vector3f(pos, 2, pos/2.0), true, true, eBVM_OCCUPIED, rob_map->getDeviceDataPtr());
    prob_map2->insertSensorData(obstacle, Vector3f(pos, 2, pos/2.0), true, false, eBVM_OCCUPIED, rob_map->getDeviceDataPtr());
  }
  
  while(true)
  {
    gvl->visualizeMap("myProbVoxelMap");
    gvl->visualizeMap("myProbVoxelMap2");
    gvl->visualizeMap("myRobotMap");
    sleep(1);
  }

  return 0;
}
