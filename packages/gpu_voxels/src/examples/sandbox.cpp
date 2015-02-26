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
 * \date    2014-11-27
 *
 */
//----------------------------------------------------------------------
#include <cstdlib>
#include <signal.h>

#include <boost/filesystem/operations.hpp>
#include <boost/filesystem/path.hpp>

#include <gpu_voxels/GpuVoxels.h>
#include <gpu_voxels/helpers/MetaPointCloud.h>
#include <gpu_voxels/logging/logging_gpu_voxels.h>

using namespace gpu_voxels;
namespace bfs = boost::filesystem;

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

  /*!
   * First, we generate an API class, which defines the
   * volume of our space and the resolution.
   * Be careful here! The size is limited by the memory
   * of your GPU. Even if an empty Octree is small, a
   * Voxelmap will always require the full memory.
   */
  gvl = new GpuVoxels(200, 200, 200, 0.01);

  gvl->addMap(MT_PROBAB_VOXELMAP, "myProbabVoxmap");
  gvl->addMap(MT_BIT_VOXELMAP, "myBitmapVoxmap");
  gvl->addMap(MT_OCTREE, "myOctree");

  gvl->addPrimitives(primitive_array::primitive_Sphere, "myPrims");
  gvl->addPrimitives(primitive_array::primitive_Cuboid, "mySecondPrims");

  std::vector<Vector4f> prim_positions(1000);
  std::vector<Vector4f> prim_positions2(1000);

  Vector3f center1_min(0.5,0.5,0.5);
  Vector3f center1_max(0.6,0.6,0.6);
  Vector3f center2_min(0.5,0.5,0.5);
  Vector3f center2_max(0.6,0.6,0.6);
  Vector3f center3_min(0.5,0.5,0.5);
  Vector3f center3_max(0.6,0.6,0.6);
  Vector3f corner1_min;
  Vector3f corner2_min;
  Vector3f corner3_min;
  Vector3f corner1_max;
  Vector3f corner2_max;
  Vector3f corner3_max;

  float t = 0.0;
  int j = 0;
  while(true)
  {
    float x = sin(t);
    float y = cos(t);
    t += 0.03;
    corner1_min = center1_min + Vector3f(0.2 * x, 0.2 * y, 0);
    corner1_max = center1_max + Vector3f(0.2 * x, 0.2 * y, 0);
    gvl->insertBoxIntoMap(corner1_min, corner1_max, "myProbabVoxmap", eVT_OCCUPIED, 2);
    corner2_min = center2_min + Vector3f(0.0, 0.2 * x, 0.2 * y);
    corner2_max = center2_max + Vector3f(0.0, 0.2 * x, 0.2 * y);
    gvl->insertBoxIntoMap(corner3_min, corner3_max, "myBitmapVoxmap", eVT_OCCUPIED, 2);
    corner3_min = center3_min + Vector3f(0.2 * x, 0.0, 0.2 * y);
    corner3_max = center3_max + Vector3f(0.2 * x, 0.0, 0.2 * y);
    gvl->insertBoxIntoMap(corner2_min, corner2_max, "myOctree", eVT_OCCUPIED, 2);


//    LOGGING_INFO(
//        Gpu_voxels,
    std::cout << "Collsions myProbabVoxmap + myBitmapVoxmap: " << gvl->getMap("myProbabVoxmap")->collideWith(gvl->getMap("myBitmapVoxmap")) << std::endl;
    std::cout << "Collsions myOctree + myBitmapVoxmap: " << gvl->getMap("myOctree")->collideWith(gvl->getMap("myBitmapVoxmap")) << std::endl;
    std::cout << "Collsions myOctree + myProbabVoxmap: " << gvl->getMap("myOctree")->collideWith(gvl->getMap("myProbabVoxmap")) << std::endl;
    //visualize both maps
    gvl->visualizeMap("myProbabVoxmap");
    gvl->visualizeMap("myBitmapVoxmap");
    gvl->visualizeMap("myOctree");

    for(size_t i = 0; i < prim_positions.size(); i++)
    {
      // x, y, z, size
      prim_positions[i] = Vector4f(i / 10.0, sin(i/5.0), sin(j++ / 5.0), 0.4);
      prim_positions2[i] = Vector4f(sin(i/5.0), sin(j++ / 5.0), i / 10.0, 0.4);
    }
    gvl->modifyPrimitives("myPrims", prim_positions);
    gvl->modifyPrimitives("mySecondPrims", prim_positions2);

    gvl->visualizePrimitivesArray("myPrims");
    gvl->visualizePrimitivesArray("mySecondPrims");

    usleep(30000);

    gvl->clearMap("myProbabVoxmap");
    gvl->clearMap("myBitmapVoxmap");
    gvl->clearMap("myOctree");
  }

}
