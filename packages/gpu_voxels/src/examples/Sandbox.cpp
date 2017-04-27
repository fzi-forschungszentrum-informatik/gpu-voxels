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
 * This little example program shows how to use the primitive_array
 * datatype.
 * It also collides different maps with each other and loads
 * a scaled coordinate system.
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

  /*
   * First, we generate an API class, which defines the
   * volume of our space and the resolution.
   * Be careful here! The size is limited by the memory
   * of your GPU. Even if an empty Octree is small, a
   * Voxelmap will always require the full memory.
   */
  gvl = GpuVoxels::getInstance();
  gvl->initialize(200, 200, 200, 0.01);

  // Now we add some maps
  gvl->addMap(MT_PROBAB_VOXELMAP, "myProbabVoxmap");
  gvl->addMap(MT_BITVECTOR_VOXELMAP, "myBitmapVoxmap");
  gvl->addMap(MT_BITVECTOR_OCTREE, "myOctree");
  gvl->addMap(MT_PROBAB_VOXELMAP, "myCoordinateSystemMap");

  // And two different primitive types
  gvl->addPrimitives(primitive_array::ePRIM_SPHERE, "myPrims");
  gvl->addPrimitives(primitive_array::ePRIM_CUBOID, "mySecondPrims");
  std::vector<Vector4f> prim_positions(1000);
  std::vector<Vector4i> prim_positions2(1000);

  // These coordinates are used for three boxes that are inserted into the maps
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

  // We load the model of a coordinate system.
  if (!gvl->insertPointCloudFromFile("myCoordinateSystemMap", "coordinate_system_100.binvox", true,
                                     eBVM_OCCUPIED, true, Vector3f(0, 0, 0),0.5))
  {
    LOGGING_WARNING(Gpu_voxels, "Could not insert the PCD file..." << endl);
  }

  /*
   * Now we start the main loop, that will animate the scene.
   */
  float t = 0.0;
  int j = 0;
  while(true)
  {
    // Calculate new positions for the boxes
    float x = sin(t);
    float y = cos(t);
    t += 0.03;
    corner1_min = center1_min + Vector3f(0.2 * x, 0.2 * y, 0);
    corner1_max = center1_max + Vector3f(0.2 * x, 0.2 * y, 0);
    gvl->insertBoxIntoMap(corner1_min, corner1_max, "myProbabVoxmap", eBVM_OCCUPIED, 2);
    corner2_min = center2_min + Vector3f(0.0, 0.2 * x, 0.2 * y);
    corner2_max = center2_max + Vector3f(0.0, 0.2 * x, 0.2 * y);
    gvl->insertBoxIntoMap(corner3_min, corner3_max, "myBitmapVoxmap", eBVM_OCCUPIED, 2);
    corner3_min = center3_min + Vector3f(0.2 * x, 0.0, 0.2 * y);
    corner3_max = center3_max + Vector3f(0.2 * x, 0.0, 0.2 * y);
    gvl->insertBoxIntoMap(corner2_min, corner2_max, "myOctree", eBVM_OCCUPIED, 2);

    // generate info on the occuring collisions:
    LOGGING_INFO(
        Gpu_voxels, "Collsions myProbabVoxmap + myBitmapVoxmap: " << gvl->getMap("myProbabVoxmap")->as<voxelmap::ProbVoxelMap>()->collideWith(gvl->getMap("myBitmapVoxmap")->as<voxelmap::BitVectorVoxelMap>()) << endl <<
        "Collsions myOctree + myBitmapVoxmap: " << gvl->getMap("myOctree")->as<NTree::GvlNTreeDet>()->collideWith(gvl->getMap("myBitmapVoxmap")->as<voxelmap::BitVectorVoxelMap>()) << endl <<
        "Collsions myOctree + myProbabVoxmap: " << gvl->getMap("myOctree")->as<NTree::GvlNTreeDet>()->collideWith(gvl->getMap("myProbabVoxmap")->as<voxelmap::ProbVoxelMap>()) << endl);

    // tell the visualier that the maps have changed
    gvl->visualizeMap("myProbabVoxmap");
    gvl->visualizeMap("myBitmapVoxmap");
    gvl->visualizeMap("myOctree");
    gvl->visualizeMap("myCoordinateSystemMap");

    // update the primitves:
    for(size_t i = 0; i < prim_positions.size(); i++)
    {
      // x, y, z, size
      prim_positions[i] = Vector4f(0.2 + (i / 250.0), 0.2 + (sin(i/5.0)/50.0), (sin(j/5.0) / 50.0), 0.01);
      prim_positions2[i] = Vector4i(20 + (sin(i/5.0)/0.5), 20 + (sin(j/5.0) / 0.5), i / 2.5, 1);
      j++;
    }
    gvl->modifyPrimitives("myPrims", prim_positions);
    gvl->modifyPrimitives("mySecondPrims", prim_positions2);

    // tell the visualizier that the data has changed:
    gvl->visualizePrimitivesArray("myPrims");
    gvl->visualizePrimitivesArray("mySecondPrims");

    usleep(30000);

    // Reset the maps:
    gvl->clearMap("myProbabVoxmap");
    gvl->clearMap("myBitmapVoxmap");
    gvl->clearMap("myOctree");
  }

}
