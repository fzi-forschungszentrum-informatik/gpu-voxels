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
 * \date    2014-10-20
 *
 */
//----------------------------------------------------------------------
#include <cstdlib>
#include <signal.h>

#include "IbeoSourceWrapper.h"
#include <gpu_voxels/GpuVoxels.h>
#include <gpu_voxels/logging/logging_gpu_voxels.h>
#include <gpu_voxels/helpers/CudaMath.h>
#include <gpu_voxels/helpers/cuda_datatypes.h>
#include <vector>

using namespace gpu_voxels;

GpuVoxels* gvl;
std::string ibeo_uri = "ibeo+file:///home/drews/parking_spaces/20140811-163823.idc?offset=-02:00:16";
std::string ncom_uri = "ncom+file:///home/drews/parking_spaces/140811_141202.ncom";

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

void callback(std::vector<Vector3f>& transformed_point_cloud)
{
    gvl->getMap("myEnvironmentMap")->insertGlobalData(transformed_point_cloud, eVT_OCCUPIED);
    gvl->visualizeMap("myEnvironmentMap");

    // only visualize the current data, since the data examples used for testing are not in sync
    sleep(0.1); // wait a moment to visualize the map before it is cleared
    gvl->clearMap("myEnvironmentMap");
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
  gvl = new GpuVoxels(200, 200, 200, 0.1);

  /*!
   * A second map will represent the environment.
   * As it is captured by a sensor, this map is probabilistic.
   * We also have an a priori static map file in a PCD, so we
   * also load that into the map.
   * The PCD file is (in this example) retrieved via an environment variable
   * to access the directory where model files are stored in:
   * GPU_VOXELS_MODEL_PATH
   * The additional (optional) params shift the map to zero and then add
   * a offset to the loaded pointcloud.
   */
  gvl->addMap(MT_OCTREE, "myEnvironmentMap");

  Vector3f center = gvl->getMap("myEnvironmentMap")->getMetricDimensions() * 0.5f;
  IbeoSourceWrapper ibeo(callback, ibeo_uri, ncom_uri, center);
  ibeo.run();
}
