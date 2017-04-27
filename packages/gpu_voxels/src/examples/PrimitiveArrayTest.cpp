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
* \author  Andreas Hermann <hermann@fzi.de|
* \date    2017-02-08
*
*
* This example program shows the two APIs for creating primitives:
* One accepting metric coordinates, the other Voxel coordinates.
*
*/
//----------------------------------------------------------------------

#include <cstdlib>
#include <signal.h>
#include <typeinfo>

#include <icl_core_config/Config.h>

#include <gpu_voxels/GpuVoxels.h>
#include <gpu_voxels/primitive_array/PrimitiveArray.h>
#include <gpu_voxels/logging/logging_gpu_voxels.h>


using namespace gpu_voxels;

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
  gvl->initialize(200, 200, 100, 3.0);

  gvl->addMap(MT_BITVECTOR_VOXELLIST, "voxellist");

  // This should result in two lines of points with equal spacing inbetween the points of each line.
  // The Primitives hover a bit above the pointcloud points.

  std::vector<Vector3f> listPoints;
  listPoints.push_back(Vector3f(30.0f, 9.0f, 12.0f));
  listPoints.push_back(Vector3f(30.0f,15.0f, 12.0f));
  listPoints.push_back(Vector3f(30.0f,21.0f, 12.0f));
  listPoints.push_back(Vector3f(30.0f,27.0f, 12.0f));

  std::vector<Vector3f> metric3Point;
  metric3Point.push_back(Vector3f(30.0f,9.0f,9.0f));

  std::vector<Vector4f> metric4Point;
  metric4Point.push_back(Vector4f(30.0f,15.0f,9.0f,3.0f));

  std::vector<Vector3i> voxel3Point;
  voxel3Point.push_back(Vector3i(10,7,3));

  std::vector<Vector4i> voxel4Point;
  voxel4Point.push_back(Vector4i(10,9,3,1));

  gvl->insertPointCloudIntoMap(listPoints, "voxellist", BitVoxelMeaning(1));

  gvl->addPrimitives(primitive_array::ePRIM_SPHERE, "metric3Sphere");
  bool prim = gvl->modifyPrimitives("metric3Sphere", metric3Point, 3.0f);

  gvl->addPrimitives(primitive_array::ePRIM_SPHERE, "metric4Sphere");
  prim = gvl->modifyPrimitives("metric4Sphere", metric4Point);

  gvl->addPrimitives(primitive_array::ePRIM_SPHERE, "voxel3Sphere");
  prim = gvl->modifyPrimitives("voxel3Sphere", voxel3Point, 1);

  gvl->addPrimitives(primitive_array::ePRIM_SPHERE, "voxel4Sphere");
  prim = gvl->modifyPrimitives("voxel4Sphere", voxel4Point);

  std::cout << "Entering Draw Loop : " << prim << std::endl;
  while(true)
  {
    gvl->visualizeMap("voxellist");
    gvl->visualizePrimitivesArray("metric3Sphere");
    gvl->visualizePrimitivesArray("metric4Sphere");
    gvl->visualizePrimitivesArray("voxel3Sphere");
    gvl->visualizePrimitivesArray("voxel4Sphere");
  }
}
