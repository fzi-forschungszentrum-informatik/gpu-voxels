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
 * \author  Matthias Wagner
 * \date    2014-01-14
 *
 *  \brief Camera class for the voxel map visualizer on GPU
 *
 */
//----------------------------------------------------------------------
#include <cstdlib>
#include <signal.h>

#include <boost/filesystem/operations.hpp>
#include <boost/filesystem/path.hpp>

#include <gpu_voxels/GpuVoxels.h>
#include <gpu_voxels/helpers/Kinect.h>
#include <gpu_voxels/helpers/MetaPointCloud.h>
#include <gpu_voxels/helpers/file_handling.h>
#include <gpu_voxels/logging/logging_gpu_voxels.h>
#include <gpu_voxels/helpers/binvox_handling.h>

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

  gvl = new GpuVoxels(500, 500, 500, 0.02);

  gvl->addMap(MT_OCTREE, "myFirstMap");
  gvl->addMap(MT_PROBAB_VOXELMAP, "mySeconMap");

  bfs::path model_path;
  if (!file_handling::getGpuVoxelsPath(model_path))
  {
    LOGGING_ERROR(Gpu_voxels, "The environment variable 'GPU_VOXELS_MODEL_PATH' could not be read. Did you set it?" << endl);
    return -1; // exit here.
  }

  bfs::path pc_file_0 = bfs::path(model_path / "hollie_plain_left_arm_2_link.xyz");
  bfs::path pc_file_1 = bfs::path(model_path / "helmet1_3.binvox");

  if (!gvl->getMap("myFirstMap")->insertPointcloudFromFile(pc_file_0.generic_string(), eVT_OCCUPIED, true,
                                                  Vector3f(0.2, 0.2, 0)))
  {
    LOGGING_WARNING(Gpu_voxels, "Could not insert the PCD file..." << endl);
  }
  if (!gvl->getMap("mySeconMap")->insertPointcloudFromFile(pc_file_1.generic_string(), eVT_OCCUPIED, true,
                                                  Vector3f(0.1, 0.1, 0)))
  {
    LOGGING_WARNING(Gpu_voxels, "Could not insert the PCD file..." << endl);
  }

  gvl->visualizeMap("myFirstMap");
  gvl->visualizeMap("mySeconMap");

  size_t num_cols = 0;
  std::string exit = "";
  while (true)
  {
    std::cout << "To quit the program enter \"exit\", to run the test enter \"t\"" << std::endl;
    std::cin >> exit;
    if (exit.compare("exit") == 0)
    {
      break;
    }
    else if (exit.compare("t") == 0)
    {
      size_t i = 0;
      while(i < 50) {
        i++;        
        num_cols = gvl->getMap("myFirstMap")->collideWith(gvl->getMap("mySeconMap"), 0.0f, Vector3ui(i,0,0));
        std::cout << "Num Cols: " << num_cols << std::endl;
        gvl->visualizeMap("myFirstMap");
        gvl->visualizeMap("mySeconMap");
        usleep(100000);
      }
    }
  }
  delete gvl;
}
