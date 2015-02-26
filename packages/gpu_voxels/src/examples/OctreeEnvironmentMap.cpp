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
 * \author  Florian Drews
 * \date    2014-06-20
 *
 */
//----------------------------------------------------------------------
#include <gpu_voxels/GpuVoxels.h>

#include <gpu_voxels/logging/logging_gpu_voxels.h>

using namespace gpu_voxels;

int main(int argc, char* argv[])
{
  icl_core::logging::initialize(argc, argv);

  if (argc < 2)
  {
    LOGGING_ERROR(Gpu_voxels, "Name of pcd file missing!" << endl);
    return 1;
  }

  /*!
   * First, we generate an API class, which defines the
   * volume of our space and the resolution.
   * Be careful here! The size is limited by the memory
   * of your GPU. Even if an empty Octree is small, a
   * Voxelmap will always require the full memory.
   */
  GpuVoxels* gvl = new GpuVoxels(200, 200, 200, 0.05);

  /*!
   * Add a static environment map to visualize.
   */
  gvl->addMap(MT_OCTREE, "myEnvironmentMap");

  /*!
   * Or use a probabilistic octree instead
   */
  //gvl->addMap(eGVL_PROBAB_OCTREE, "myEnvironmentMap");
  std::string pcd_file = argv[1];
  LOGGING_INFO(Gpu_voxels, "Creating octree from file '" << pcd_file.c_str() << "' ..." << endl);

  if (!gvl->getMap("myEnvironmentMap")->insertPointcloudFromFile(pcd_file, eVT_SWEPT_VOLUME_START, true, Vector3f(0, 0, 0)))
  {
    LOGGING_WARNING(Gpu_voxels, "Could not insert the PCD file..." << endl);
  }

  bool force_repaint = true;
  while (true)
  {
    gvl->visualizeMap("myEnvironmentMap", force_repaint);
    force_repaint = false; // Repaint of map isn't necessary. Only if the visualizer requested it, since the map data doesn't change

    usleep(100000); // max 10 FPS
  }
  return 0;
}
